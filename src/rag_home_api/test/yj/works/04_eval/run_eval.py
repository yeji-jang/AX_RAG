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

from ragas.run_config import RunConfig
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
DATASET_VER = "chunk_V1"      # 청크 버전별 폴더 (docs/04-1_eval_set/chunk_V1/)
SPLIT = "dev"                 # 개발/튜닝은 "dev"(10문항), 최종 고정 평가만 "test"(20문항)
FAILURE_THRESHOLD = 0.7       # RAGAS 미달 판정 기준 점수
MODES = ["A_vector", "B_hybrid", "C_hybrid_rerank"]

BASE_DIR = Path(__file__).resolve().parents[2]  # test/yj/
EVAL_SET_PATH = BASE_DIR / "docs" / "04-1_eval_set" / DATASET_VER / f"{SPLIT}.json"
REPORT_DIR = BASE_DIR / "docs" / "04-2_eval_reports" / DATASET_VER    # ← DATASET_VER 추가

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


def to_safe_float(value) -> float | None :
    """RAGAS가 NaN이나 빈 값을 돌려줄 때 0.0으로 왜곡하지 않고 None(결측)으로 반환"""
    if value is None :
        return None
    try :
        num = float(value)
        if math.isnan(num) :
            return None
        return round(num, 4)
    except Exception :
        return None


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

#===================================================================
def diagnose_record(record : dict, scores : dict[str, float | None]) -> tuple[str, str] :
    """문항 상태와 RAGAS 점수를 보고 실패 유형과 개선 처방전을 결정"""
    if record["status"] == "error" :
        return "ERROR", "API 또는 DB 호출 오류 점검 필요"

    if record["query_type"] == "unanswerable" :
        if record["status"] == "insufficient_evidence" :
            return "PASS", "근거 없음 문항을 올바르게 정보 부족 안내함"
        return "HALLUCINATION_FAILURE", "근거 없는 질문에 단정 답변함 -> 프롬프트 거절 기준 강화 필요"

    cp = scores["context_precision"]
    cr = scores["context_recall"]
    faith = scores["faithfulness"]
    rel = scores["answer_relevancy"]

    if (
        record["hit_at_5"] == 0
        or (cr is not None and cr < FAILURE_THRESHOLD)
        or (cp is not None and cp < FAILURE_THRESHOLD)
    ) :
        return "RETRIEVAL_FAILURE", "검색 누락/순위 하락 -> BM25 가중치, 표 청킹(V2), 리랭킹 점검 필요"

    if faith is not None and faith < FAILURE_THRESHOLD :
        return "HALLUCINATION_FAILURE", "문맥에 없는 내용 생성 -> 프롬프트 근거 제한 규칙 강화 필요"

    if rel is not None and rel < FAILURE_THRESHOLD :
        return "IRRELEVANT_ANSWER", "질문 의도 이탈 -> 쿼리 리라이트 또는 답변 프롬프트 개선 필요"

    if None in (cp, cr, faith, rel) :
        return "METRIC_MISSING", "RAGAS 일부 지표 결측(NaN) -> 재평가 필요"

    return "PASS", "검색 및 생성 품질 목표 충족"


def score_with_ragas(records : list[dict]) -> list[SampleEvaluationResult] :
    # 1) 답변 가능 문항 중 정상 실행된 건만 추려서 RAGAS 채점 대상 구성
    answerable : list[dict] = []
    for r in records :
        if r["query_type"] != "unanswerable" and r["status"] != "error" :
            answerable.append(r)

    ragas_scores_by_id : dict[str, dict[str, float | None]] = {}

    if answerable :
        # 2) RAGAS 입력 데이터셋 조립
        ds = Dataset.from_dict({
            "user_input": [r["question"] for r in answerable],
            "retrieved_contexts": [r["contexts"] for r in answerable],
            "response": [r["answer"] for r in answerable],
            "reference": [r["ground_truth"] for r in answerable],
        })

        # 동시 요청 과부하(Connection error) 방지를 위해 워커 수를 16 -> 4로 제한
        ragas_cfg = RunConfig(max_workers=4, max_retries=5, timeout=180)

        # gpt-6-luna(추론 모델)는 temperature/n 강제 변경을 거부하므로 bypass 옵션 필수
        evaluator_llm = LangchainLLMWrapper(
            get_llm(),
            run_config=ragas_cfg,
            bypass_temperature=True,
            bypass_n=True,
        )
        evaluator_emb = LangchainEmbeddingsWrapper(get_embeddings())

        eval_result = evaluate(
            dataset=ds,
            metrics=[context_precision, context_recall, faithfulness, answer_relevancy],
            llm=evaluator_llm,
            embeddings=evaluator_emb,
            run_config=ragas_cfg,
            raise_exceptions=False,
        )

        # 3) 문항 ID별로 4대 지표 점수 매핑
        df = eval_result.to_pandas()
        for idx, rec in enumerate(answerable) :
            row = df.iloc[idx]
            ragas_scores_by_id[rec["question_id"]] = {
                "context_precision": to_safe_float(row.get("context_precision")),
                "context_recall": to_safe_float(row.get("context_recall")),
                "faithfulness": to_safe_float(row.get("faithfulness")),
                "answer_relevancy": to_safe_float(row.get("answer_relevancy")),
            }

    # 4) 전체 문항(근거 없음·오류 포함)에 점수와 실패 진단을 붙여 결과 스키마로 반환
    empty_scores : dict[str, float | None] = {
        "context_precision": None,
        "context_recall": None,
        "faithfulness": None,
        "answer_relevancy": None,
    }

    scored_results : list[SampleEvaluationResult] = []
    for r in records :
        scores = ragas_scores_by_id.get(r["question_id"], empty_scores)
        failure_type, prescription = diagnose_record(r, scores)

        scored_results.append(SampleEvaluationResult(
            question_id=r["question_id"],
            query_type=r["query_type"],
            pipeline_mode=r["pipeline_mode"],
            question=r["question"],
            answer=r["answer"],
            status=r["status"],
            hit_at_5=r["hit_at_5"],
            elapsed_sec=r["elapsed_sec"],
            **scores,
            failure_type=failure_type,
            prescription=prescription,
        ))

    return scored_results


#===================================================================
# 4. 모드별 평균 요약 및 리포트(.json / .md) 저장
#===================================================================
def avg_valid(values : list[float | None]) -> float | None :
    """결측(None)을 제외하고 유효한 점수만 평균 계산"""
    valid = [v for v in values if v is not None]
    if not valid :
        return None
    return round(sum(valid) / len(valid), 4)


def fmt_score(value : float | None) -> str :
    """마크다운 표 출력용 : 결측(None)이면 '-' 표시"""
    if value is None :
        return "-"
    return f"{value:.4f}"


def summarize_mode(results : list[SampleEvaluationResult]) -> dict :
    answerable = [r for r in results if r.query_type != "unanswerable"]
    unanswerable = [r for r in results if r.query_type == "unanswerable"]
    ok_times = [r.elapsed_sec for r in results if r.status != "error"]

    hit_sum = sum(r.hit_at_5 or 0 for r in answerable)
    refusal_sum = sum(1 for r in unanswerable if r.status == "insufficient_evidence")
    ragas_missing = sum(
        1 for r in answerable
        if None in (r.context_precision, r.context_recall, r.faithfulness, r.answer_relevancy)
    )
    n_ans = len(answerable) or 1

    return {
        "n_total": len(results),
        "n_success": len(ok_times),
        "n_error": len(results) - len(ok_times),
        "n_ragas_missing": ragas_missing,
        "hit_at_5": f"{hit_sum}/{len(answerable)} ({hit_sum / n_ans:.1%})",
        "unanswerable_refusal": f"{refusal_sum}/{len(unanswerable)}",
        "avg_context_precision": avg_valid([r.context_precision for r in answerable]),
        "avg_context_recall": avg_valid([r.context_recall for r in answerable]),
        "avg_faithfulness": avg_valid([r.faithfulness for r in answerable]),
        "avg_answer_relevancy": avg_valid([r.answer_relevancy for r in answerable]),
        "avg_sec": round(statistics.mean(ok_times), 2) if ok_times else 0.0,
        "median_sec": round(statistics.median(ok_times), 2) if ok_times else 0.0,
    }


def save_reports(
    all_raw : dict[str, list[dict]],
    all_scored : dict[str, list[SampleEvaluationResult]],
) -> None :
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M")

    # 1) 원본 전체 결과 JSON 저장 (지침서 STEP 6 증빙용)
    json_path = REPORT_DIR / f"eval_{SPLIT}_{timestamp}.json"
    payload = {
        "split": SPLIT,
        "collection": DEFAULT_COLLECTION,
        "top_k": TOP_K,
        "rerank_candidate_k": RERANK_CANDIDATE_K,
        "summary": {mode: summarize_mode(res) for mode, res in all_scored.items()},
        "scored": {mode: [r.model_dump() for r in res] for mode, res in all_scored.items()},
        "raw": all_raw,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    # 2) 사람이 읽기 쉬운 비교 요약표 & 미달 문항 Markdown 저장
    md_path = REPORT_DIR / f"eval_{SPLIT}_{timestamp}.md"
    lines = [
        f"# 평가 리포트 ({SPLIT}, {timestamp})",
        f"- 컬렉션: `{DEFAULT_COLLECTION}` / Top-K: `{TOP_K}` / 임계치: `{FAILURE_THRESHOLD}`",
        "",
        "## 1. 파이프라인 비교 요약",
        "| 모드 | Hit@5 | 근거없음 거절 | Context Precision | Context Recall | Faithfulness | Answer Relevancy | 평균(초) | 중앙값(초) | 오류 | 결측 |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for mode, res in all_scored.items() :
        s = summarize_mode(res)
        lines.append(
            f"| {mode} | {s['hit_at_5']} | {s['unanswerable_refusal']} | "
            f"{fmt_score(s['avg_context_precision'])} | {fmt_score(s['avg_context_recall'])} | "
            f"{fmt_score(s['avg_faithfulness'])} | {fmt_score(s['avg_answer_relevancy'])} | "
            f"{s['avg_sec']} | {s['median_sec']} | {s['n_error']} | {s['n_ragas_missing']} |"
        )

    lines += ["", "## 2. 임계치 미달 / 실패 항목 정리"]
    for mode, res in all_scored.items() :
        failed = [r for r in res if r.failure_type != "PASS"]
        lines.append(f"\n### {mode} (미달 {len(failed)}건 / {len(res)}건)")

        for r in failed :
            lines.append(
                f"- **{r.question_id}** (`{r.query_type}` | `{r.failure_type}` | Hit@5={r.hit_at_5} | "
                f"CP={r.context_precision}, CR={r.context_recall}, F={r.faithfulness}, AR={r.answer_relevancy})"
            )
            lines.append(f"  - 질문: {r.question}")
            lines.append(f"  - 처방: {r.prescription}")

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n리포트 저장 완료!\n  - JSON : {json_path}\n  - MD   : {md_path}")

#===================================================================
# 5. 실행
#===================================================================
if __name__ == "__main__" :
    qa_items = load_eval_set()
    bm25 = BM25Index(DEFAULT_COLLECTION)

    all_raw : dict[str, list[dict]] = {}
    all_scored : dict[str, list[SampleEvaluationResult]] = {}

    for mode in MODES :
        logger.info(f"=== [{mode}] 평가 시작 ({len(qa_items)}문항) ===")
        raw_records = run_pipeline_on_dataset(qa_items, pipeline_mode=mode, bm25=bm25)
        scored_records = score_with_ragas(raw_records)

        all_raw[mode] = raw_records
        all_scored[mode] = scored_records

    save_reports(all_raw, all_scored)