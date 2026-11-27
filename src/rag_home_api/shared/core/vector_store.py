from functools import lru_cache

import psycopg
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.vectorstores import VectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_postgres import PGVector
from langchain_postgres.vectorstores import DistanceStrategy
from pydantic import SecretStr

from rag_home_api.config import settings


# temperature=1(기본값)만 허용하는 최신/추론 모델 접두사 (gpt-6-luna 포함)
_FIXED_TEMP_PREFIXES = ("gpt-5", "gpt-6", "o1", "o3", "o4")

# 팀 통합 표준 임베딩 (전원 1536차원 통일) 및 도메인별 멀티 컬렉션 레지스트리
UNIFIED_EMBEDDING_MODEL = "text-embedding-3-small"
TEAM_COLLECTIONS: dict[str, dict[str, str]] = {
    "law": {
        "standard": "rag_law_v1",
        "current": "jh_1000",
        "owner": "jh",
        "embedding_model": UNIFIED_EMBEDDING_MODEL,
        "description": "주택임대차 법령·가이드 (1000/200)",
    },
    "public_housing": {
        "standard": "rag_notice_v1",
        "current": "yj_public_rental",
        "owner": "yj",
        "embedding_model": UNIFIED_EMBEDDING_MODEL,
        "description": "공공임대·청년안심주택 모집 공고문 (단일 컬렉션, chunk_version=V2, 6,197청크)",
    },
    "rent_stats": {
        "standard": "rag_rent_stats_v1",
        "current": "housing_rent_kh",
        "owner": "kh",
        "embedding_model": UNIFIED_EMBEDDING_MODEL,
        "description": "서울 25구 전월세 실거래 요약 통계 블록 (6,200청크)",
    },
    "rent_deals": {
        "standard": "rag_rent_deals_v1",
        "current": "course_v0_small",
        "owner": "bc",
        "embedding_model": UNIFIED_EMBEDDING_MODEL,
        "description": "전월세 개별 거래 조건 필터 검색 (1행=1청크, 관악구 31,772청크)",
    },
}


def make_db_id(collection_name: str, chunk_id: str) -> str:
    """PGVector 단일 테이블(langchain_pg_embedding.id) 전역 PK 충돌 방지용 표준 ID 생성기.

    이미 '{collection_name}:' 접두어가 붙어 있으면 중복으로 붙이지 않는다(멱등성 보장).
    """
    prefix = f"{collection_name}:"
    return chunk_id if chunk_id.startswith(prefix) else f"{prefix}{chunk_id}"


class _LocalSentenceTransformerEmbeddings(Embeddings):
    """로컬 GPU/CPU HuggingFace 임베딩 모델(KURE-v1, KURE-v3, BGE-M3 등) LangChain 어댑터."""

    def __init__(self, model_name: str, batch_size: int = 32) -> None:
        try:
            import torch
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                f"로컬 임베딩 모델({model_name})을 사용하려면 torch와 sentence-transformers가 필요합니다."
            ) from exc
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model_kwargs = {"torch_dtype": torch.float16} if self.device == "cuda" else {}
        self._model = SentenceTransformer(model_name, device=self.device, model_kwargs=model_kwargs)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vecs = self._model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [v.tolist() for v in vecs]

    def embed_query(self, text: str) -> list[float]:
        vec = self._model.encode(
            text,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vec.tolist()


def get_llm(for_rerank: bool = False, temperature: float | None = None) -> BaseChatModel:
    """팀 공통 LLM 팩토리 (답변 생성 · 평가셋 초안 · Reranker · RAGAS Judge 공용)."""
    model_name = settings.rerank_model if for_rerank else settings.llm_model
    kwargs: dict = {
        "model": model_name,
        "api_key": SecretStr(settings.openai_api_key),
    }
    if temperature is not None and not model_name.startswith(_FIXED_TEMP_PREFIXES):
        kwargs["temperature"] = temperature
    return ChatOpenAI(**kwargs)


@lru_cache(maxsize=8)
def get_embeddings(model_name: str | None = None) -> Embeddings:
    """텍스트 청크를 벡터로 변환하는 임베딩 객체.

    - OpenAI 모델(`text-embedding-*`): OpenAIEmbeddings 사용
    - 그 외 HuggingFace 모델(`nlpai-lab/KURE-v1`, `nlpai-lab/KURE-v3` 등): 로컬 SentenceTransformer 어댑터 사용
    ※ 주의: 임베딩 모델을 바꾸면 벡터 차원/공간이 달라지므로 반드시 별도 collection_name에 새로 적재해야 합니다!
    """
    chosen = model_name or settings.embedding_model
    if chosen.startswith("text-embedding-"):
        return OpenAIEmbeddings(
            model=chosen,
            api_key=SecretStr(settings.openai_api_key),
        )
    return _LocalSentenceTransformerEmbeddings(chosen)


def get_vector_store(
    collection_name: str | None = None,
    embedding_model: str | None = None,
) -> VectorStore:
    """PGVector 저장소 연결. 첫 호출 시 vector 확장과 테이블 2개(collection, embedding)를 자동 생성한다."""
    return PGVector(
        embeddings=get_embeddings(embedding_model),
        collection_name=collection_name or settings.collection_name,
        connection=settings.database_url,
        distance_strategy=DistanceStrategy.COSINE,
        use_jsonb=True,
    )


def get_raw_db_connection() -> psycopg.Connection:
    """PGVector 테이블에 직접 SQL을 날릴 때 쓰는 순수 psycopg 연결 (+psycopg 제거 필수)."""
    raw_dsn = settings.database_url.replace("postgresql+psycopg://", "postgresql://")
    return psycopg.connect(raw_dsn)