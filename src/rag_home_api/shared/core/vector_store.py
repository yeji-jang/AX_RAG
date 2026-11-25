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


def get_embeddings() -> Embeddings:
    """텍스트 청크를 벡터로 변환하는 임베딩 객체. (나중에 KURE 로컬 GPU 모델 실험 시 여기 1곳만 교체!)"""
    return OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=SecretStr(settings.openai_api_key),
    )


def get_vector_store(collection_name: str | None = None) -> VectorStore:
    """PGVector 저장소 연결. 첫 호출 시 vector 확장과 테이블 2개(collection, embedding)를 자동 생성한다."""
    return PGVector(
        embeddings=get_embeddings(),
        collection_name=collection_name or settings.collection_name,
        connection=settings.database_url,
        distance_strategy=DistanceStrategy.COSINE,
        use_jsonb=True,
    )


def get_raw_db_connection() -> psycopg.Connection:
    """PGVector 테이블에 직접 SQL을 날릴 때 쓰는 순수 psycopg 연결 (+psycopg 제거 필수)."""
    raw_dsn = settings.database_url.replace("postgresql+psycopg://", "postgresql://")
    return psycopg.connect(raw_dsn)