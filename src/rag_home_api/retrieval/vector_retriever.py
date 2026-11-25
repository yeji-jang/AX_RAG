from langchain_core.documents import Document

from rag_home_api.config import settings
from rag_home_api.shared.core.vector_store import get_vector_store


def _build_pg_filter(filters: dict | None) -> dict | None:
    """ChunkMetadata의 JSONB 필드에 맞는 PGVector Pre-filter 조건을 조립한다."""
    if not filters:
        return None

    conds: list[dict] = []
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
        if filters.get(key):
            conds.append({key: {"$eq": filters[key]}})
    if filters.get("max_deposit_krw") is not None:
        conds.append({"min_deposit_krw": {"$lte": int(filters["max_deposit_krw"])}})
    if filters.get("max_monthly_rent_krw") is not None:
        conds.append({"min_monthly_rent_krw": {"$lte": int(filters["max_monthly_rent_krw"])}})

    if len(conds) == 1:
        return conds[0]
    return {"$and": conds} if len(conds) > 1 else None


def search_vector(
    query: str,
    top_k: int = 5,
    filters: dict | None = None,
    collection_name: str | None = None,
    embedding_model: str | None = None,
) -> list[dict]:
    """[검색 A: Vector Baseline] PGVector 코사인 유사도 검색.

    나중에 검색 B(Hybrid RRF)·C(Re-ranking)와 같은 형식으로 비교할 수 있게
    표준 딕셔너리 리스트로 반환한다.
    """
    store = get_vector_store(
        collection_name=collection_name or settings.collection_name,
        embedding_model=embedding_model,
    )
    hits: list[tuple[Document, float]] = store.similarity_search_with_relevance_scores(
        query,
        k=top_k,
        filter=_build_pg_filter(filters),
    )
    return [
        {
            "chunk_id": doc.metadata["chunk_id"],
            "document": doc,
            "vector_rank": rank,
            "vector_score": round(float(score), 4),
            "bm25_rank": None,
            "bm25_score": 0.0,
            "combined_score": round(float(score), 4),
            "rank_before": rank,
        }
        for rank, (doc, score) in enumerate(hits, start=1)
    ]