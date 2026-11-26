"""
# B 구성 : 하이브리드 검색기 (BM25 키워드 + Vector 의미 검색 RRF 결합)

(순서)
    1. 한국어 형태소 분석기(Kiwi) 설정 및 토큰 분해
    2. 키워드 검색 (BM25Index)
    3. 하이브리드 검색 (A 의미 검색 + BM25 키워드 검색 순위 결합 RRF)
    4. 단독 실행 테스트
"""

from kiwipiepy import Kiwi
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from rag_home_api.test.yj.core.logger import get_logger
from rag_home_api.test.yj.core.vector_store import get_raw_db_connection
from rag_home_api.test.yj.rag.A_vector_retriever import (
    DEFAULT_CHUNK_VERSION,
    DEFAULT_COLLECTION,
    TOP_K,
    search_vector,
)

logger = get_logger(__name__)


#===================================================================
# 1. RRF 상수 및 한국어 형태소 분해 함수
#===================================================================
RRF_K = 60  # 순위 차이를 완만하게 반영하기 위한 RRF 표준 상수

# Kiwi 형태소 분석기는 생성 비용이 크므로 파일 로드 시 한 번만 만들어 재사용
kiwi = Kiwi()

def tokenize_ko(text : str) -> list[str] :
    """
    한국어 문장을 BM25용 검색 토큰(명사·외국어·숫자 등)으로 분해합니다.

    # BM25가 조사('은/는/이/가') 때문에 다른 단어로 오해하지 않도록 핵심 품사만 뽑음
    # - NN* : 일반명사·고유명사 (예: 행복주택, 보증금, 신혼부부)
    # - SL  : 알파벳 (예: LH, SH)
    # - SN  : 숫자 (예: 26, 36, 120)
    """
    tokens : list[str] = []

    for token in kiwi.tokenize(text) :
        if token.tag.startswith(("NN", "SL", "SN")) :
            tokens.append(token.form.lower())

    return tokens

#===================================================================
# 2. 키워드 검색 (BM25 : Kiwi 형태소 토큰 매칭)
#===================================================================
def matches_filters(metadata : dict, filters : dict | None) -> bool :
    """청크 메타데이터가 filters 조건(예: {'chunk_version': 'V2'}, {'source': 'LH'})을 모두 만족하는지 확인"""
    if not filters :
        return True

    for key, expected in filters.items() :
        actual = metadata.get(key)
        if isinstance(expected, list) :
            if actual not in expected :
                return False
        elif actual != expected :
            return False

    return True


def load_chunks_from_db(
        collection : str = DEFAULT_COLLECTION,
        chunk_version : str | None = None,
    ) -> list[Document] :
    """
    (BM25 사전 구축용)
    PGVector에 적재된 청크 본문과 메타데이터를 읽어옴 (chunk_version 지정 시 해당 버전만 조회)
    """
    if chunk_version :
        query = """
            SELECT e.id, e.document, e.cmetadata
            FROM langchain_pg_embedding e
            JOIN langchain_pg_collection c ON e.collection_id = c.uuid
            WHERE c.name = %s
              AND e.cmetadata->>'chunk_version' = %s
        """
        params = (collection, chunk_version)
    else :
        query = """
            SELECT e.id, e.document, e.cmetadata
            FROM langchain_pg_embedding e
            JOIN langchain_pg_collection c ON e.collection_id = c.uuid
            WHERE c.name = %s
        """
        params = (collection,)

    conn = get_raw_db_connection()

    try :
        cur = conn.cursor()
        cur.execute(query, params)
        rows = cur.fetchall()
    finally :
        conn.close()

    docs : list[Document] = []
    for chunk_id, content, metadata in rows :
        docs.append(Document(
            id=chunk_id,
            page_content=content,
            metadata=metadata or {},
        ))

    return docs


class BM25Index :
    """
    컬렉션(및 특정 chunk_version) 문서를 한 번만 형태소 분해해 두고 여러 질문에 재사용하는 BM25 검색기
    """

    def __init__(self,
                collection : str = DEFAULT_COLLECTION,
                chunk_version : str | None = None) :
        self.collection = collection
        self.chunk_version = chunk_version
        self.docs = load_chunks_from_db(collection, chunk_version=chunk_version)

        # 모든 청크를 미리 토큰 리스트로 분해해 BM25 색인 생성
        tokenized_corpus : list[list[str]] = []
        for doc in self.docs :
            tokenized_corpus.append(
                tokenize_ko(
                    doc.page_content
                )
            )

        self.bm25 = BM25Okapi(tokenized_corpus)
        logger.info(
            f"BM25 인덱스 구축 완료 : {collection} (version={chunk_version or 'ALL'}, {len(self.docs)}건)")


    def search(self,
               question : str,
               top_k : int = TOP_K,
               filters : dict | None = None,
               ) -> list[tuple[Document, float]] :


        # 1) 질문도 같은 방식으로 형태소 분해 후 전체 청크별 BM25 점수 계산
        question_tokens = tokenize_ko(question)
        scores = self.bm25.get_scores(question_tokens)

        # 2) 점수가 높은 순서대로 정렬하되, filters 조건에 맞는 청크만 상위 top_k개 추출
        sorted_indices = sorted(
            range(len(scores)),
            key=lambda idx : scores[idx],
            reverse=True,
        )

        # 3) (Document, 점수) 형태로 묶고 키워드 검색 순위 기록
        scored_docs : list[tuple[Document, float]] = []

        for idx in sorted_indices :
            doc = self.docs[idx]
            if not matches_filters(doc.metadata, filters) :
                continue

            rank = len(scored_docs) + 1
            bm25_score = round(float(scores[idx]), 4)
            doc.metadata["bm25_rank"] = rank
            scored_docs.append((doc, bm25_score))

            if len(scored_docs) >= top_k :
                break

        return scored_docs

#===================================================================
# 3. 하이브리드 검색 (B 구성 : 의미 검색 + 키워드 검색 RRF 결합)
#===================================================================
def search_hybrid(
    question : str,
    bm25_index : BM25Index,
    collection : str = DEFAULT_COLLECTION,
    top_k : int = TOP_K,
    rrf_k : int = RRF_K,
    filters : dict | None = None,
) -> list[tuple[Document, float]] :

    """
    코사인 유사도(0~1)와 BM25 점수(0~수십)는 점수 기준이 달라 단순 합산할 수 없으므로,
    각 검색기에서 몇 등 했는지(순위)만 가지고 RRF 공식 1 / (rrf_k + 순위) 로 합산함
    """

    # 1) 각 검색기에서 최종 개수(top_k)의 2배씩 넉넉히 후보를 가져옴
    candidate_k = top_k * 2
    vector_hits = search_vector(
        question,
        collection=collection,
        top_k=candidate_k,
        filters=filters,
    )
    bm25_hits = bm25_index.search(
        question,
        top_k=candidate_k,
        filters=filters,
    )

    rrf_scores : dict[str, float] = {}
    doc_by_id : dict[str, Document] = {}


    # 2) 의미 검색 순위 점수 누적
    for rank, (doc, _) in enumerate(vector_hits, start=1) :
        chunk_id = doc.metadata["chunk_id"]
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + (1.0 / (rrf_k + rank))
        doc_by_id[chunk_id] = doc


    # 3) 키워드 검색 순위 점수 누적 (양쪽 모두 상위권이면 점수가 더해져 최상위로 올라감)
    for rank, (doc, _) in enumerate(bm25_hits, start=1) :
        chunk_id = doc.metadata["chunk_id"]
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + (1.0 / (rrf_k + rank))
        doc_by_id[chunk_id] = doc

    # 4) 합산 RRF 점수가 높은 상위 top_k개 반환
    sorted_ids = sorted(
        rrf_scores.items(),
        key=lambda item : item[1],
        reverse=True,
    )[:top_k]

    hybrid_results : list[tuple[Document, float]] = []
    
    for chunk_id, score in sorted_ids :
        hybrid_results.append((doc_by_id[chunk_id], round(score, 5)))

    return hybrid_results

#===================================================================
# 4. 단독 실행 테스트
#===================================================================
if __name__ == "__main__" :
    test_question = "서울삼전 행복주택 26형 신혼부부 임대보증금은 얼마인가요?"
    version_filter = {"chunk_version": DEFAULT_CHUNK_VERSION}

    bm25 = BM25Index(DEFAULT_COLLECTION, chunk_version=DEFAULT_CHUNK_VERSION)

    print(f"=== [BM25] 키워드 검색 ({DEFAULT_CHUNK_VERSION}) ===")
    for rank, (doc, score) in enumerate(bm25.search(test_question, top_k=3, filters=version_filter), start=1) :
        preview = doc.page_content[:80].replace("\n", " ")
        print(f"  {rank}위 | {score:.4f} | {doc.metadata.get('chunk_id')} | {preview}")

    print(f"\n=== [B] 하이브리드 검색 (RRF | {DEFAULT_CHUNK_VERSION}) ===")
    for rank, (doc, score) in enumerate(search_hybrid(test_question, bm25, top_k=3, filters=version_filter), start=1) :
        preview = doc.page_content[:80].replace("\n", " ")
        print(f"  {rank}위 | {score:.5f} | {doc.metadata.get('chunk_id')} | {preview}")















