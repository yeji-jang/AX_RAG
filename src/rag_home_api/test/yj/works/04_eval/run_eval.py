"""
# 평가 실행기 : A(의미) · B(하이브리드) · C(리랭킹) 비교 평가 및 리포트 저장

(순서)
    1. 평가 대상(dev/test), 컬렉션, 임계치 및 입출력 경로 설정
    2. 각 파이프라인(A/B/C)으로 질문 실행 -> 답변·응답시간·Hit@5 수집
    3. 답변 가능 문항에 대해 RAGAS 4대 지표 채점 및 실패 원인 자동 진단
    4. 요약 비교표(.md)와 전체 원본 기록(.json) 저장
    5. 실행
"""

import json
import math
import statistics
import sys
import time
import types
from pathlib import Path

# [호환성 패치] langchain-community 1.x에서 제거된 vertexai 모듈을 ragas가 찾을 때 나는 에러 방지
if "langchain_community.chat_models.vertexai" not in sys.modules :
    _stub = types.ModuleType("langchain_community.chat_models.vertexai")
    _stub.ChatVertexAI = type("ChatVertexAI", (), {})
    sys.modules["langchain_community.chat_models.vertexai"] = _stub

from datasets import Dataset
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

from rag_home_api.test.yj.core.logger import get_logger
from rag_home_api.test.yj.core.schemas.evaluation import EvalQAItem, SampleEvaluationResult
from rag_home_api.test.yj.core.vector_store import get_embeddings, get_llm
from rag_home_api.test.yj.rag.A_vector_retriever import DEFAULT_COLLECTION, TOP_K, search_vector
from rag_home_api.test.yj.rag.B_hybrid_retriever import BM25Index, search_hybrid
from rag_home_api.test.yj.rag.C_reranker import RERANK_CANDIDATE_K, rerank_chunks
from rag_home_api.test.yj.rag.generator import generate_answer

logger = get_logger(__name__)


#===================================================================
# 1. 평가 설정값 및 경로 지정
#===================================================================
DATASET_VER = "first"         # 평가셋 폴더 버전 (docs/04-1_eval_set/first/)
SPLIT = "dev"                 # 개발/튜닝은 "dev"(10문항), 최종 고정 평가만 "test"(20문항)
FAILURE_THRESHOLD = 0.7       # RAGAS 미달 판정 기준 점수
MODES = ["A_vector", "B_hybrid", "C_hybrid_rerank"]

BASE_DIR = Path(__file__).resolve().parents[2]  # test/yj/
EVAL_SET_PATH = BASE_DIR / "docs" / "04-1_eval_set" / DATASET_VER / f"{SPLIT}.json"
REPORT_DIR = BASE_DIR / "docs" / "04-2_eval_reports"

# 답변 본문에 아래 표현이 들어있거나 인용 번호가 비어있으면 '근거 부족 안내(거절)'로 판정
INSUFFICIENT_PHRASES = ("찾을 수 없습니다", "찾지 못했습니다", "정보가 없", "확인할 수 없")


#===================================================================
# 2. 평가셋 불러오기 & 파이프라인 실행 (검색 + 답변 생성 + Hit@5 계산)
#===================================================================
def load_eval_set(path : Path = EVAL_SET_PATH) -> list[EvalQAItem] :
    raw_list = json.loads(path.read_text(encoding="utf-8"))

    items : list[EvalQAItem] = []
    for row in raw_list :
        items.append(EvalQAItem(**row))

    return items


def to_safe_float(value) -> float :
    """RAGAS가 간혹 NaN이나 빈 값을 돌려줄 때 에러 없이 0.0으로 바꿔주는 보조 함수"""
    try :
        num = float(value)
        if math.isnan(num) :
            return 0.0
        return round(num, 4)
    except Exception :
        return 0.0


def is_refusal_answer(answer : str, cited_numbers : list[int]) -> bool :
    """LLM이 근거 부족을 안내했는지 판별"""
    if not cited_numbers :
        return True
    for phrase in INSUFFICIENT_PHRASES :
        if phrase in answer :
            return True
    return False


def run_pipeline_on_dataset(
    qa_items : list[EvalQAItem],
    pipeline_mode : str,
    bm25 : BM25Index | None = None,
) -> list[dict] :
    records : list[dict] = []

    for item in qa_items :
        start_time = time.perf_counter()
        rerank_trace : list[dict] = []

        try :
            # 1) 모드에 맞는 검색기 실행 (A / B / C)
            if pipeline_mode == "A_vector" :
                hits = search_vector(item.question, collection=DEFAULT_COLLECTION, top_k=TOP_K)

            elif pipeline_mode == "B_hybrid" :
                if bm25 is None :
                    raise ValueError("B_hybrid 모드에는 BM25Index가 필요합니다.")
                hits = search_hybrid(item.question, bm25, collection=DEFAULT_COLLECTION, top_k=TOP_K)

            elif pipeline_mode == "C_hybrid_rerank" :
                if bm25 is None :
                    raise ValueError("C_hybrid_rerank 모드에는 BM25Index가 필요합니다.")
                candidates = search_hybrid(
                    item.question,
                    bm25,
                    collection=DEFAULT_COLLECTION,
                    top_k=RERANK_CANDIDATE_K,
                )
                hits, rerank_trace = rerank_chunks(item.question, candidates, top_k=TOP_K)

            else :
                raise ValueError(f"지원하지 않는 모드입니다 : {pipeline_mode}")

            # 2) 검색된 청크로 답변 생성
            gen = generate_answer(item.question, hits)
            answer = gen["answer"]
            contexts = gen["retrieved_contexts"]
            chunk_ids = gen.get("retrieved_chunk_ids", [])

            # 3) 정상 답변인지 근거 부족 안내(거절)인지 상태 구분
            if is_refusal_answer(answer, gen["cited_numbers"]) :
                status = "insufficient_evidence"
            else :
                status = "answered"

        except Exception as err :
            # API 오류는 '거절'로 잘못 채점되지 않도록 별도 'error' 상태로 분리
            logger.error(f"[{pipeline_mode} | {item.question_id}] 실행 오류 : {err}")
            answer = f"ERROR: {err}"
            contexts = []
            chunk_ids = []
            status = "error"

        elapsed = round(time.perf_counter() - start_time, 2)

        # 4) Hit@5 계산 : 상위 5개 검색 청크 중 정답 허용 청크가 하나라도 있으면 1점 (근거 없음 문항은 제외)
        if item.query_type == "unanswerable" :
            hit_at_5 = None
        else :
            hit_at_5 = 0
            for cid in chunk_ids[:5] :
                if cid in item.allowed_chunk_ids :
                    hit_at_5 = 1
                    break

        records.append({
            "question_id": item.question_id,
            "query_type": item.query_type,
            "pipeline_mode": pipeline_mode,
            "question": item.question,
            "ground_truth": item.ground_truth,
            "allowed_chunk_ids": item.allowed_chunk_ids,
            "retrieved_chunk_ids": chunk_ids,
            "rerank_trace": rerank_trace,
            "contexts": contexts if contexts else ["검색된 문맥 없음"],
            "answer": answer,
            "status": status,
            "hit_at_5": hit_at_5,
            "elapsed_sec": elapsed,
        })
        logger.info(
            f"[{pipeline_mode} | {item.question_id}] status={status} hit@5={hit_at_5} time={elapsed:.2f}s"
        )

    return records