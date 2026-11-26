"""
# C 구성 : LLM 기반 재정렬기 (Re-ranker)

(순서)
    1. 리랭커 시스템 프롬프트 및 기본 설정
    2. LLM 구조화 출력 스키마 (RankedItem, RerankOutput)
    3. Hybrid 후보 청크 재정렬 및 전·후 순위 기록 (rerank_chunks)
    4. 단독 실행 테스트
"""

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from rag_home_api.test.yj.core.schemas.C_reranker_output import RerankOutput
from rag_home_api.test.yj.core.logger import get_logger
from rag_home_api.test.yj.core.vector_store import get_llm
from rag_home_api.test.yj.rag.A_vector_retriever import DEFAULT_COLLECTION, TOP_K
from rag_home_api.test.yj.rag.B_hybrid_retriever import BM25Index, search_hybrid
logger = get_logger(__name__)


#===================================================================
# 1. 기본 설정 및 리랭커 프롬프트
#===================================================================
RERANK_CANDIDATE_K = 15  # B(Hybrid)에서 먼저 넉넉히 뽑아올 후보 개수
RERANK_SYSTEM_PROMPT = """
당신은 서울시 공공임대주택 공고문 검색 결과를 재정렬하는 심사관입니다.

[질문]에 답하는 데 가장 직접적인 근거가 되는 순서대로 [후보 청크]의 점수를 매기세요.

채점 기준 (0~10점):
- 9~10점 : 질문의 공고명·단지·유형·계층이 정확히 일치하고 정답 수치/규정이 들어있음
- 6~8점  : 같은 공고/유형이지만 세부 조건 일부만 관련 있음
- 0~5점  : 다른 공고·다른 계층이거나 질문과 무관함

규칙 :
- 반드시 제공된 후보 번호(candidate_no)만 사용하세요.
- 관련도가 높은 순서대로 정렬해 반환하세요.
""".strip()

#===================================================================
# 2. 후보 청크 재정렬 함수 (C 구성)
#===================================================================
def rerank_chunks(
    question : str,
    candidates : list[tuple[Document, float]],
    top_k : int = TOP_K,
) -> tuple[list[tuple[Document, float]], list[dict]] :
    
    """
    Hybrid 검색 후보들을 LLM이 직접 읽고 관련도 순으로 다시 줄 세움
    - 반환 1 : 재정렬된 상위 top_k개 [(Document, 점수), ...]
    - 반환 2 : 순위 변동 기록 [{"chunk_id", "before_rank", "after_rank", ...}, ...]
    """

    if not candidates :
        return [], []
    
    # 1) 각 후보 청크에 [후보 1], [후보 2] 번호표를 붙여 하나의 문자열로 합침
    candidate_blocks : list[str] = []

    for idx, (doc, _) in enumerate(candidates, start=1) :
        clean_text = doc.page_content.replace("\n", " ")
        candidate_blocks.append(f"[후보 {idx}]\n{clean_text}")
    candidates_prompt = "\n\n".join(candidate_blocks)

    try :
        # 2) LLM에게 질문과 후보 목록을 주고 RerankOutput 양식으로 결과 받기
        llm = get_llm().with_structured_output(RerankOutput)
        user_message = (
            f"[질문]\n{question}\n\n"
            f"[후보 청크 목록 (상위 {top_k}개 선별)]\n{candidates_prompt}"
        )
        output : RerankOutput = llm.invoke([
            SystemMessage(content=RERANK_SYSTEM_PROMPT),
            HumanMessage(content=user_message),
        ])
        reranked_hits : list[tuple[Document, float]] = []
        trace : list[dict] = []
        used_numbers : set[int] = set()

        # 3) LLM이 골라준 순서대로 상위 top_k개를 담고 전·후 순위(before -> after) 기록
        for item in output.ranked_items :
            no = item.candidate_no
            # 없는 후보 번호이거나 이미 담은 번호면 건너뜀
            if no < 1 or no > len(candidates) or no in used_numbers :
                continue
            used_numbers.add(no)
            doc, _ = candidates[no - 1]
            after_rank = len(reranked_hits) + 1
            score = round(item.relevance_score, 2)
            reranked_hits.append((doc, score))
            trace.append({
                "chunk_id": doc.metadata.get("chunk_id"),
                "before_rank": no,
                "after_rank": after_rank,
                "rerank_score": score,
                "reason": item.reason,
            })
            if len(reranked_hits) >= top_k :
                break

        # 4) 만약 LLM이 top_k개보다 적게 골랐다면, 기존 Hybrid 순위대로 빈자리 채움
        for idx, (doc, rrf_score) in enumerate(candidates, start=1) :
            if len(reranked_hits) >= top_k :
                break
            if idx in used_numbers :
                continue

            after_rank = len(reranked_hits) + 1
            reranked_hits.append((doc, round(rrf_score, 5)))
            trace.append({
                "chunk_id": doc.metadata.get("chunk_id"),
                "before_rank": idx,
                "after_rank": after_rank,
                "rerank_score": 0.0,
                "reason": "개수 부족으로 기존 Hybrid 순위 보충",
            })

        return reranked_hits, trace
    
    except Exception as err :
        # 5) API 오류 등으로 실패하더라도 검색이 멈추지 않게 기존 Hybrid 상위 top_k개 반환
        logger.warning(f"Re-ranking 실패 -> 기존 Hybrid 순위 유지 : {err}")
        fallback_hits = candidates[:top_k]
        fallback_trace : list[dict] = []
        for rank, (doc, score) in enumerate(fallback_hits, start=1) :
            fallback_trace.append({
                "chunk_id": doc.metadata.get("chunk_id"),
                "before_rank": rank,
                "after_rank": rank,
                "rerank_score": score,
                "reason": "리랭킹 오류로 기존 순위 유지",
            })
        return fallback_hits, fallback_trace
    
#===================================================================
# 3. 단독 실행 테스트 (B 후보 15개 -> C 재정렬 Top-3 비교)
#===================================================================
if __name__ == "__main__" :
    test_question = "서울삼전 행복주택 26형 신혼부부 임대보증금은 얼마인가요?"

    bm25 = BM25Index(DEFAULT_COLLECTION)
    hybrid_candidates = search_hybrid(
        test_question, 
        bm25, 
        top_k=RERANK_CANDIDATE_K)

    reranked, trace = rerank_chunks(
        test_question, 
        hybrid_candidates, 
        top_k=3)
    
    print("=== [C] Hybrid + Re-ranking 결과 ===")
    for info in trace :
        print(
            f"  {info['after_rank']}위 (기존 {info['before_rank']}위) | "
            f"점수 {info['rerank_score']} | {info['chunk_id']} | {info['reason']}"
        )