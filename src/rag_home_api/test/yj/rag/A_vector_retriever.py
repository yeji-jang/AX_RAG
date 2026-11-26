"""
# A 구성 : 의미 검색기 (Vector Retriever)

(순서)
    1. 기본 컬렉션 및 Top-K 설정
    2. PGVector 코사인 유사도 검색
    3. 단독 실행 테스트
"""

from functools import lru_cache

from langchain_core.documents import Document

from rag_home_api.test.yj.core.logger import get_logger
from rag_home_api.test.yj.core.vector_store import get_vector_store

logger = get_logger(__name__)

#===================================================================
# 1. 기본 설정
#===================================================================
DEFAULT_COLLECTION = "yj_preV2_chunkV1_cs800_ov100"
TOP_K = 5

#===================================================================
# 2. 의미 검색 (A 구성 : PGVector)
#===================================================================
@lru_cache
def get_cached_store(collection : str) :
    """ 
    # 질문할 때마다 DB 연결 객체를 새로 만들지 않도록 캐싱 (검색 전용이므로 reset=False)
    """
    return get_vector_store(
        collection, 
        reset=False)


def search_vector(
    question : str,
    collection : str = DEFAULT_COLLECTION,
    top_k : int = TOP_K,
    filters : dict | None = None,
) -> list[tuple[Document, float]] :
    store = get_cached_store(collection)


    # 1) PGVector에서 코사인 거리(distance) 기준으로 가까운 상위 top_k개 조회
    raw_results = store.similarity_search_with_score(
        query=question, 
        k=top_k, 
        filter=filters,
        )

    # 2) 코사인 거리는 0에 가까울수록 비슷하므로, 직관적인 유사도(1 - 거리)로 바꿔 반환
    scored_docs : list[tuple[Document, float]] = []
    for rank, (doc, distance) in enumerate(raw_results, start=1) : 
        similarity = round(1.0 - distance, 4)
        doc.metadata["vector_rank"] = rank  # 의미 검색 순위 기록
        scored_docs.append((doc, similarity))
    
    return scored_docs


#===================================================================
# 3. 단독 실행 테스트
#===================================================================
if __name__ == "__main__" :

    test_question = "서울삼전 행복주택 26형 신혼부부 임대보증금은 얼마인가요?"

    print("=== [1] 의미 검색 (Vector) ===")
    for rank, (doc, score) in enumerate(search_vector(test_question, top_k=3), start=1) :
        preview = doc.page_content[:80].replace("\n", " ")
        print(f"  {rank}위 | {score:.4f} | {doc.id} | {preview}")