import re
from functools import lru_cache

from kiwipiepy import Kiwi
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from rag_home_api.config import settings
from rag_home_api.shared.core.logger import get_logger
from rag_home_api.shared.core.vector_store import get_raw_db_connection

logger = get_logger(__name__)

# Kiwi 품사 태그: 일반명사(NNG), 고유명사(NNP), 의존명사(NNB), 외국어(SL), 숫자(SN), 어근(XR)
KEEP_POS = {"NNG", "NNP", "NNB", "SL", "SN", "XR"}


@lru_cache(maxsize=1)
def _get_kiwi() -> Kiwi:
    """Kiwi 형태소 분석기는 로딩 비용이 있으므로 프로세스당 1회만 생성한다."""
    return Kiwi()


def tokenize_korean_housing(text: str) -> list[str]:
    """Kiwi 형태소(명사·숫자·영문) + 공고/주택형 코드 원형(sh_306214, 25B)을 함께 보존한다."""
    tokens: list[str] = []

    # ① 영숫자·하이픈·언더스코어 코드 보존 (실습 2의 모델코드 분리 패턴 계승)
    for code in re.findall(r"[a-z0-9][a-z0-9_\-]{1,}", text.lower()):
        tokens.append(code)
        for sep in ("_", "-"):
            if sep in code:
                tokens.extend(part for part in code.split(sep) if len(part) > 1)

    # ② 한국어 형태소 분석 ("대항력이랑" -> "대항력", "확정일자를" -> "확정일자")
    for tok in _get_kiwi().tokenize(text):
        if tok.tag in KEEP_POS and len(tok.form) > 1:
            tokens.append(tok.form.lower())

    return tokens


@lru_cache(maxsize=2)
def _load_corpus_cached(
    collection_name: str,
    index_version: str,
) -> tuple[list[Document], BM25Okapi | None]:
    """PGVector의 동일 청크 코퍼스를 1회 로드해 BM25Okapi 역색인을 메모리에 유지한다."""
    sql = """
        SELECT e.document, e.cmetadata
        FROM langchain_pg_embedding e
        JOIN langchain_pg_collection c ON e.collection_id = c.uuid
        WHERE c.name = %s
    """
    docs: list[Document] = []
    with get_raw_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (collection_name,))
            for content, meta in cur.fetchall():
                docs.append(Document(page_content=content, metadata=meta or {}))

    if not docs:
        return [], None

    logger.info("BM25 코퍼스 로드 완료: %s (%d개 청크)", collection_name, len(docs))
    tokenized_corpus = [tokenize_korean_housing(d.page_content) for d in docs]
    return docs, BM25Okapi(tokenized_corpus)


def _match_filters(meta: dict, filters: dict | None) -> bool:
    """vector_retriever._build_pg_filter와 동일한 필터 조건을 BM25에도 적용한다."""
    if not filters:
        return True
    for key in (
        "doc_type",
        "district",
        "dong",
        "housing_type",
        "lease_type",
        "area_band",
        "source",
        "category",
    ):
        if filters.get(key) and meta.get(key) != filters[key]:
            return False
    if filters.get("max_deposit_krw") is not None:
        dep = meta.get("min_deposit_krw")
        if dep is None or dep > int(filters["max_deposit_krw"]):
            return False
    if filters.get("max_monthly_rent_krw") is not None:
        rent = meta.get("min_monthly_rent_krw")
        if rent is None or rent > int(filters["max_monthly_rent_krw"]):
            return False
    return True


def search_bm25(
    query: str,
    top_k: int = 12,
    filters: dict | None = None,
    collection_name: str | None = None,
) -> list[dict]:
    """rank_bm25 + Kiwi 토크나이저로 키워드 매칭 상위 청크를 검색한다."""
    target_collection = collection_name or settings.collection_name
    docs, bm25 = _load_corpus_cached(target_collection, settings.index_version)
    if not docs or bm25 is None:
        return []

    scores = bm25.get_scores(tokenize_korean_housing(query))
    paired: list[tuple[Document, float]] = [
        (doc, float(sc))
        for doc, sc in zip(docs, scores)
        if sc > 0 and _match_filters(doc.metadata, filters)
    ]
    paired.sort(key=lambda x: x[1], reverse=True)

    return [
        {
            "chunk_id": doc.metadata["chunk_id"],
            "document": doc,
            "bm25_rank": rank,
            "bm25_score": round(sc, 4),
        }
        for rank, (doc, sc) in enumerate(paired[:top_k], start=1)
    ]