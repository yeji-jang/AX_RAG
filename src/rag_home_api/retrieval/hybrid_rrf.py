from rag_home_api.config import settings
from rag_home_api.retrieval.bm25_retriever import search_bm25
from rag_home_api.retrieval.vector_retriever import search_vector


def search_hybrid_rrf(
    keyword_query: str,
    semantic_query: str,
    top_k: int = 12,
    filters: dict | None = None,
    bm25_weight: float | None = None,
    vector_weight: float | None = None,
    rrf_k: int | None = None,
    collection_name: str | None = None,
) -> list[dict]:
    """[검색 B: Hybrid RRF] BM25 순위와 PGVector 순위를 가중 RRF로 결합·중복 제거한다.

    Note:
        - BM25 점수(비정규 양수)와 코사인 유사도(0~1)는 척도가 달라 단순 합산 금지!
        - 순위(rank)의 역수 1/(k + rank)를 가중합해 공정하게 결합한다. (실습 2 패턴 계승)
        - 추후 Query Rewrite(STEP 5)와 바로 연결되도록 keyword_query / semantic_query를 분리 입력받는다.
    """
    w_bm25 = settings.bm25_weight if bm25_weight is None else bm25_weight
    w_vec = settings.vector_weight if vector_weight is None else vector_weight
    k_const = settings.rrf_k if rrf_k is None else rrf_k
    pool = max(top_k * 2, 20)

    bm25_hits = search_bm25(keyword_query, top_k=pool, filters=filters, collection_name=collection_name)
    vec_hits = search_vector(semantic_query, top_k=pool, filters=filters, collection_name=collection_name)

    registry: dict[str, dict] = {}

    for item in bm25_hits:
        cid = item["chunk_id"]
        registry[cid] = {
            "chunk_id": cid,
            "document": item["document"],
            "bm25_rank": item["bm25_rank"],
            "bm25_score": item["bm25_score"],
            "vector_rank": None,
            "vector_score": 0.0,
            "combined_score": w_bm25 * (1.0 / (k_const + item["bm25_rank"])),
        }

    for item in vec_hits:
        cid = item["chunk_id"]
        add = w_vec * (1.0 / (k_const + item["vector_rank"]))
        if cid in registry:
            registry[cid]["vector_rank"] = item["vector_rank"]
            registry[cid]["vector_score"] = item["vector_score"]
            registry[cid]["combined_score"] += add
        else:
            registry[cid] = {
                "chunk_id": cid,
                "document": item["document"],
                "bm25_rank": None,
                "bm25_score": 0.0,
                "vector_rank": item["vector_rank"],
                "vector_score": item["vector_score"],
                "combined_score": add,
            }

    ranked = sorted(registry.values(), key=lambda x: x["combined_score"], reverse=True)[:top_k]
    for rank, item in enumerate(ranked, start=1):
        item["combined_score"] = round(item["combined_score"], 6)
        item["rank_before"] = rank
    return ranked