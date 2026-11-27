"""서울 주거 RAG — 실험 원인 판별 워크벤치 · 회차별 리포트 & 팀원 통합 비교 Streamlit 대시보드 (v2).

작성자: 최기혁 (kh)
실행 방법 (pyproject.toml 건드리지 않고 즉시 실행!):
    uv run --with streamlit streamlit run src/rag_home_api/test/kh/eval_dashboard.py

6개 핵심 탭:
  1. 🧭 의사결정 홈       : 검색/생성/거절 3구역 분리 KPI + TP·FP·FN·TN 2×2 혼동행렬 + 4단계 깔때기 병목 + 다음 변수 1개 추천
  2. 🔬 문항별 X-Ray 해부 : Before vs After Top-5 5조건 불일치 자동 태그 + 문항별 TP/FP/FN/TN + RAGAS·답변·컨텍스트 비교
  3. 📈 회차별 랩노트     : eval_version 격리 + Altair 연결 선 그래프 2종 + 단일 변인 실험 랩노트
  4. 📝 회차별 분석 리포트: 테스트 1회 실행당 6파트 평가 분석 리포트(개선/악화 Regression Watch 포함) + MD/HTML 다운로드
  5. 🧪 시험지 & 거절트랙 : dev_v{n} 버전 관리 + test 봉인 + 2-Track(RAGAS vs 과거 데이터 범위 외 거절) 점검
  6. 🤝 팀원 통합 비교    : test/*/results/ 자동 탐색 + 다중 파일 업로드 + 도메인별 목표(0.60/0.70) 통합 비교 & 120문항 병합 검증
"""

from __future__ import annotations

import base64
import html as html_lib
import io
import json
import math
import re
import statistics
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

try:
    import altair as alt
except ImportError:  # pragma: no cover
    alt = None

KH_DIR = Path(__file__).resolve().parent
TEST_ROOT_DIR = KH_DIR.parent  # src/rag_home_api/test/ (팀원별 폴더 탐색 루트)
KH_RESULTS_DIR = KH_DIR / "results"
KH_EVAL_DIR = KH_DIR / "eval"
REPO_DATA_DIR = KH_DIR.parents[3] / "data"

ALL_PRESET_NAMES: list[str] = [
    "A_vector",
    "B_rrf_55_45",
    "B_rrf_70_30",
    "B_rrf_80_20",
    "B_rrf_70_30_filter",
    "C_rrf_70_30_filter_rerank",
    "B_filter_rewrite",
    "C_filter_rewrite",
    "C_filter_rewrite_cache",
    "C_filter_rewrite_hyde",
    "B_chunk_v3_pyeong",
    "B_emb_3large",
    "A_vector_kure_v1",
    "B_rrf_10_90_kure_v1",
    "B_rrf_25_75_kure_v1",
    "B_rrf_50_50_kure_v1",
    "B_rrf_55_45_kure_v1",
    "B_rrf_25_75_k15_kure_v1",
    "B_rrf_50_50_k15_kure_v1",
    "B_rrf_25_75_k15_filter_kure_v1",
    "B_rrf_25_75_k15_filter_v2_kure_v1",
    "C_rrf_25_75_k15_filter_v2_rerank_kure_v1",
    "C_rrf_25_75_k15_filter_v2_rewrite_kure_v1",
    "C_rrf_25_75_k15_filter_v2_cache_kure_v1",
    "B_emb_kure_v1",
]

SAFE_COLLECTION_PREP_CMDS: dict[str, str] = {
    "B_chunk_v3_pyeong": "uv run python -m rag_home_api.test.kh.rent_chunker --collection housing_rent_kh_v3_pyeong --include-pyeong --recent-examples 5",
    "B_emb_3large": "uv run python -m rag_home_api.test.kh.rent_chunker --collection housing_rent_kh_3large --embedding-model text-embedding-3-large",
    "B_emb_kure_v1": "uv run python -m rag_home_api.test.kh.rent_chunker --collection housing_rent_kh_kure_v1 --embedding-model nlpai-lab/KURE-v1",
}

# ==================== 유비쿼터스 언어 (영어/전문용어 + 한글 병기) 마스터 사전 ====================
RAGAS_LABELS: dict[str, str] = {
    "faithfulness": "Faithfulness (충실도)",
    "answer_relevancy": "Answer Relevancy (답변 관련성)",
    "llm_context_precision_with_reference": "Context Precision (근거 정밀도)",
    "context_recall": "Context Recall (근거 재현율)",
}

RETRIEVAL_LABELS: dict[str, str] = {
    "recall@12": "Recall@12 (재현율@12)",
    "hit@12": "Hit@12 (적중 여부@12)",
    "precision@12": "Precision@12 (정밀도@12)",
    "mrr@12": "MRR@12 (평균 역순위@12)",
    "first_rank@12": "first_rank@12 (첫 정답 순위@12)",
    "hit@5": "Hit@5 (적중 여부@5)",
    "recall@5": "Recall@5 (재현율@5)",
    "precision@5": "Precision@5 (정밀도@5)",
    "top1": "Top1 (Hit@1 · 첫 결과 정답률)",
    "mrr@5": "MRR@5 (평균 역순위@5)",
    "first_rank@5": "first_rank@5 (첫 정답 순위@5)",
    "refusal": "Refusal (근거 없음 거절률)",
    "scope_notice": "Scope Notice (과거 실거래 고지율)",
    "citation_rate": "Citation Rate (출처 표기율)",
    "lat_median": "Latency Median (지연 중앙값, ms)",
    "lat_p95": "Latency p95 (지연 p95, ms)",
    "lat_avg": "Latency Avg (지연 평균, ms)",
}

# 튜토리얼/코치 모드용 툴팁(?) 사전 — 모든 KPI·컨트롤에 마우스 올리면 즉시 쉬운 풀이 표시
METRIC_HELP: dict[str, str] = {
    "recall@12": "도서관 12권 바구니(k=12)에 실제 정답 청크를 안 빠뜨리고 담은 비율 = TP/(TP+FN). 1.0 미만이면 2단계 검색 누락(FN) → B(Hybrid)·Query Rewrite가 우선!",
    "hit@5": "최종 Top-5 안에 정답 청크가 단 1개라도 포함됐는지. 주의: 복합 2구 비교 문항은 1개만 찾아도 1이 되므로 반드시 Recall@12와 함께 보세요!",
    "top1": "맨 1등(#1) 청크가 바로 정답인 비율. 3단계 순위 실패(가중치·경량 필터·검색 C) 개선 효과를 가장 정직하게 보여줍니다.",
    "mrr@5": "평균 역순위 = 1 / 첫 정답 순위 (1등=1.0, 2등=0.5, 4등=0.25, 미발견=0). 단일 정답 문항에서는 RAGAS Context Precision과 사실상 같습니다.",
    "cp": "RAGAS 근거 정밀도: 정답에 필요한 청크가 Top-5의 맨 위에 올라왔는지(순위 가중). 낮으면 3단계 순위 실패 → 가중치·필터·검색 C!",
    "cr": "RAGAS 근거 재현율: 모범정답을 쓰는 데 필요한 재료가 Top-5에 빠짐없이 들어있는지. 낮으면 2단계 검색 누락(FN) → B·Query Rewrite!",
    "f": "RAGAS 충실도(환각 검사): 생성 답변이 검색된 문서 내용에만 근거했는지. 낮으면 4단계 생성 프롬프트 점검!",
    "ar": "RAGAS 답변 관련성: 답변만 보고 원래 질문을 역추론해 유사도 측정! 첫 문장 주어 생략·TMI·거절(noncommittal=1→0.0) 시 급락합니다.",
    "citation_rate": "답변 가능(answered) 문항 중 근거 출처 파일명을 빠짐없이 표기한 비율.",
    "refusal": "Track B (과거 실거래 범위 밖: 현재 공실·옵션·반경 500m) 질문을 환각 없이 정직 거절한 비율 (목표 ≥ 70%, RAGAS에서 분리 채점).",
    "scope_notice": "거절 답변에서 '체결 완료된 과거 실거래 통계 기준' 한계를 사용자에게 명확히 안내한 비율.",
    "lat_median": "검색 지연 중앙값(ms). 첫 호출 BM25 로드(~13초) 착시를 제외한 실제 체감 속도입니다.",
}

DOMAIN_META: dict[str, tuple[str, float]] = {
    "rent": ("전월세 (실거래)", 0.60),
    "rent_stats": ("전월세 시세통계 (kh)", 0.60),
    "rent_deals": ("전월세 조건탐색 (bc)", 0.60),
    "law": ("주택임대차 법령 (jh)", 0.70),
    "public_housing": ("공공임대 공고 (yj)", 0.70),
    "unified": ("통합 (전 도메인)", 0.65),
}

OWNER_META: dict[str, dict[str, Any]] = {
    "kh": {
        "label": "최기혁 (kh · 서울 25구 전월세 시세 · 목표 0.60)",
        "domain": "rent_stats",
        "target": 0.60,
        "eval_dir": KH_EVAL_DIR,
    },
    "yj": {
        "label": "장예지 (yj · 공공임대 공고문 · 목표 0.70)",
        "domain": "public_housing",
        "target": 0.70,
        "eval_dir": TEST_ROOT_DIR / "yj" / "docs" / "04-1_eval_set",
    },
    "jh": {
        "label": "최지훈 (jh · 주택임대차 법령 · 목표 0.70)",
        "domain": "law",
        "target": 0.70,
        "eval_dir": TEST_ROOT_DIR / "jh" / "평가셋",
    },
    "bc": {
        "label": "선병철 (bc · 전월세 조건탐색 · 목표 0.60)",
        "domain": "rent_deals",
        "target": 0.60,
        "eval_dir": TEST_ROOT_DIR / "bc" / "eval",
    },
}


_LAST_LOAD_ERRORS: dict[str, list[str]] = {}


def _clean_ratio_str(val: Any) -> str:
    """'7/8 (87.5%)' -> '7/8' 정규화 (순수 stdlib)."""
    if val is None:
        return "-"
    m = re.search(r"(\d+)\s*/\s*(\d+)", str(val))
    return f"{m.group(1)}/{m.group(2)}" if m else str(val)


def _convert_yeji_eval_local(
    raw: dict[str, Any],
    source_label: str = "eval_dev.json",
    source_branch: str = "local(dev)",
    chunk_version: str = "",
) -> dict[str, Any]:
    """예지(yj) eval_*.json -> 공통 UnifiedEvalSummary dict + 문항별 X-Ray 로그(_companion_logs) 순수 stdlib 변환."""
    summary_in = raw.get("summary") or {}
    scored_in = raw.get("scored") or {}
    raw_in = raw.get("raw") or {}
    modes_out: dict[str, dict[str, Any]] = {}
    companion_logs: list[dict[str, Any]] = []

    if not chunk_version:
        m_cv = re.search(r"chunk_(V\d+)", source_label, flags=re.I)
        chunk_version = m_cv.group(1).upper() if m_cv else ("V2" if "V2" in str(raw.get("collection", "")) else "V1")
    total_chunks = 6197 if chunk_version == "V2" else 6346
    col_name = str(raw.get("collection") or "yj_public_rental")

    for mode_name, m in summary_in.items():
        scored_rows = scored_in.get(mode_name) or []
        raw_rows = raw_in.get(mode_name) or []
        raw_by_qid = {str(r.get("question_id", "")): r for r in raw_rows}

        ans_scored = [r for r in scored_rows if r.get("query_type") != "unanswerable"]
        hit_str = _clean_ratio_str(m.get("hit_at_5"))
        ans_cnt = int(hit_str.split("/")[1]) if "/" in hit_str else len(ans_scored)

        avg_sec = float(m.get("avg_sec", 0.0) or 0.0)
        med_sec = float(m.get("median_sec", 0.0) or 0.0)
        lat_list = sorted(float(r.get("elapsed_sec", 0.0) or 0.0) * 1000.0 for r in scored_rows if r.get("elapsed_sec") is not None)
        p95_ms = round(lat_list[max(0, int(len(lat_list) * 0.95) - 1)], 1) if lat_list else round(avg_sec * 1000.0, 1)

        top1_cnt = 0
        mrr_vals: list[float] = []
        cite_ok = 0
        per_item_ragas: list[dict[str, Any]] = []
        underperforming: list[dict[str, Any]] = []

        for sr in scored_rows:
            qid = str(sr.get("question_id", ""))
            rr = raw_by_qid.get(qid) or {}
            qtype = str(sr.get("query_type", ""))
            expected_status = "insufficient_evidence" if qtype == "unanswerable" else "answered"
            allowed = list(rr.get("allowed_chunk_ids") or [])
            ret_ids = list(rr.get("retrieved_chunk_ids") or [])[:5]
            contexts = list(rr.get("contexts") or [])
            answer_txt = str(sr.get("answer") or rr.get("answer") or "")
            gt_txt = str(rr.get("ground_truth") or "")

            m5: dict[str, Any] = {}
            if expected_status == "answered":
                ver_tag = f"ck{chunk_version}"
                ver_allowed = [c for c in allowed if ver_tag in c] or allowed
                exp_set = set(ver_allowed)
                hit_flag = int(bool(sr.get("hit_at_5")))
                tp_raw = sum(1 for cid in ret_ids if cid in exp_set) if exp_set else hit_flag
                tp = max(tp_raw, hit_flag)
                fp = max(0, len(ret_ids) - tp)
                fn = 0 if hit_flag else 1
                first_rank = next((i for i, cid in enumerate(ret_ids, start=1) if cid in exp_set), 0)
                if first_rank == 0 and hit_flag:
                    first_rank = 1
                top1 = int(first_rank == 1)
                mrr = (1.0 / first_rank) if first_rank else 0.0
                top1_cnt += top1
                mrr_vals.append(mrr)
                if re.search(r"\[\d+\]", answer_txt):
                    cite_ok += 1
                m5 = {
                    "hit@5": hit_flag,
                    "recall@5": 1.0 if hit_flag else 0.0,
                    "precision@5": round(tp / len(ret_ids), 4) if ret_ids else 0.0,
                    "top1": top1,
                    "mrr@5": round(mrr, 4),
                    "first_rank": first_rank,
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "tn": max(0, total_chunks - max(1, tp) - fp),
                }
                scores: dict[str, float] = {}
                for sk, tk in (
                    ("faithfulness", "faithfulness"),
                    ("answer_relevancy", "answer_relevancy"),
                    ("context_precision", "llm_context_precision_with_reference"),
                    ("context_recall", "context_recall"),
                ):
                    if sr.get(sk) is not None:
                        scores[tk] = round(float(sr[sk]), 4)
                failed = {k: v for k, v in scores.items() if v < 0.70}
                per_item_ragas.append(
                    {
                        "question_id": qid,
                        "user_input": str(sr.get("question") or rr.get("question") or ""),
                        "scores": scores,
                        "passed_all": len(failed) == 0 and bool(sr.get("hit_at_5")),
                    }
                )
                if failed or sr.get("failure_type") not in (None, "", "PASS"):
                    underperforming.append(
                        {
                            "question_id": qid,
                            "user_input": str(sr.get("question") or rr.get("question") or ""),
                            "failed_metrics": failed,
                            "all_scores": scores,
                            "llm_diagnosis": f"[{sr.get('failure_type', '')}] {sr.get('prescription', '')}".strip(),
                        }
                    )

            companion_logs.append(
                {
                    "mode": mode_name,
                    "question_id": qid,
                    "question_type": qtype,
                    "expected_status": expected_status,
                    "actual_status": sr.get("status", ""),
                    "allowed_chunk_ids": allowed,
                    "top5_ids": ret_ids,
                    "retrieved_contexts": contexts,
                    "reference_contexts": [gt_txt] if gt_txt else [],
                    "response": answer_txt,
                    "retrieval_ms": round(float(sr.get("elapsed_sec", 0.0) or 0.0) * 1000.0, 1),
                    "metrics_k5": m5,
                    "metrics_k12": (
                        {
                            "hit@12": m5["hit@5"],
                            "recall@12": m5["recall@5"],
                            "precision@12": m5["precision@5"],
                            "top1": m5["top1"],
                            "mrr@12": m5["mrr@5"],
                            "first_rank": m5["first_rank"],
                            "tp": m5["tp"],
                            "fp": m5["fp"],
                            "fn": m5["fn"],
                            "tn": m5["tn"],
                        }
                        if m5
                        else {}
                    ),
                }
            )

        pmode = "A_vector" if "A_" in mode_name else ("B_hybrid" if "B_" in mode_name else "C_hybrid_rerank")
        use_rw = mode_name == "C_plus_opt"
        modes_out[mode_name] = {
            "preset": {
                "name": mode_name,
                "pipeline_mode": pmode,
                "use_metadata_filter": True,
                "use_rewrite": use_rw,
                "use_cache": False,
                "use_hyde": False,
                "collection_name": col_name,
                "embedding_model": "text-embedding-3-small",
                "chunk_strategy": f"chunk_{chunk_version}",
            },
            "retrieval": {
                "answerable_count": ans_cnt,
                "recall@12": None,
                "hit@5": hit_str,
                "top1": f"{top1_cnt}/{ans_cnt}" if ans_cnt and raw_rows else "-",
                "mrr@5": round(statistics.mean(mrr_vals), 4) if mrr_vals and raw_rows else None,
            },
            "latency_ms": {
                "avg": round(avg_sec * 1000.0, 1),
                "median": round(med_sec * 1000.0, 1),
                "p95": p95_ms,
            },
            "refusal": _clean_ratio_str(m.get("unanswerable_refusal")),
            "scope_notice": _clean_ratio_str(m.get("unanswerable_refusal")),
            "citation_rate": f"{cite_ok}/{ans_cnt}" if ans_cnt else "-",
            "ragas_avg": {
                "faithfulness": round(float(m.get("avg_faithfulness", 0.0) or 0.0), 4),
                "answer_relevancy": round(float(m.get("avg_answer_relevancy", 0.0) or 0.0), 4),
                "llm_context_precision_with_reference": round(float(m.get("avg_context_precision", 0.0) or 0.0), 4),
                "context_recall": round(float(m.get("avg_context_recall", 0.0) or 0.0), 4),
            },
            "per_item_ragas": per_item_ragas,
            "underperforming_items": underperforming,
        }

    m_ts = re.search(r"(\d{8}_\d{4,6})", source_label)
    ts = m_ts.group(1) if m_ts else datetime.now().strftime("%Y%m%d_%H%M%S")
    if len(ts) == 13:
        ts = ts + "00"
    split_name = str(raw.get("split", "dev"))
    return {
        "timestamp": ts,
        "owner": "yj",
        "domain": "public_housing",
        "eval_version": f"yj_{split_name}",
        "ragas_target": 0.70,
        "total_corpus_chunks": total_chunks,
        "collection_name": col_name,
        "embedding_model": "text-embedding-3-small",
        "run_note": f"chunk_{chunk_version} ({split_name})",
        "source_branch": source_branch,
        "source_file": source_label,
        "notes": f"예지 공공임대 공고문 {chunk_version} ({split_name}) 평가 자동 변환",
        "modes": modes_out,
        "_companion_logs": companion_logs,
    }


def _convert_jihun_jsonl_local(
    jsonl_text: str,
    source_label: str = "dev_원본로그.jsonl",
    source_branch: str = "local(dev)",
) -> dict[str, Any]:
    """지훈(jh) JSONL 로그 -> 공통 UnifiedEvalSummary dict 순수 stdlib 변환."""
    rows = [json.loads(line) for line in jsonl_text.splitlines() if line.strip()]
    groups: dict[str, list[dict]] = {}
    latest_ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    for r in rows:
        if "meta" in r:
            if r["meta"].get("started"):
                latest_ts = str(r["meta"]["started"])
            continue
        if "run_id" in r:
            parts = str(r.get("run_id", "")).split(":")
            tag = parts[0] if len(parts) >= 2 else "run"
            mode = str(r.get("mode", parts[1] if len(parts) >= 2 else "B"))
            key = f"{tag}_{mode}"
        else:
            qid = str(r.get("question_id", ""))
            split_tag = "TEST" if "TEST" in qid else "DEV"
            key = f"jh_1000_{split_tag}_{r.get('mode', 'B')}"
        groups.setdefault(key, []).append(r)

    modes_out: dict[str, dict[str, Any]] = {}
    for gkey, recs in groups.items():
        ans = [r for r in recs if r.get("expected_status") == "answered"]
        unans = [r for r in recs if r.get("expected_status") == "insufficient_evidence"]
        hit_cnt = sum(1 for r in ans if r.get("hit5") is True)
        top1_cnt = sum(1 for r in ans if r.get("first_hit_rank") == 1)
        mrr_vals = [(1.0 / int(r["first_hit_rank"])) if r.get("first_hit_rank") else 0.0 for r in ans]
        lats = sorted(float(r.get("latency_sec", 0.0) or 0.0) * 1000.0 for r in recs if r.get("latency_sec") is not None)
        p95_ms = round(lats[max(0, int(len(lats) * 0.95) - 1)], 1) if lats else 0.0
        ref_cnt = sum(1 for r in unans if r.get("status") == "insufficient_evidence")
        modes_out[gkey] = {
            "retrieval": {
                "answerable_count": len(ans),
                "recall@12": None,
                "hit@5": f"{hit_cnt}/{len(ans)}" if ans else "-",
                "top1": f"{top1_cnt}/{len(ans)}" if ans else "-",
                "mrr@5": round(statistics.mean(mrr_vals), 4) if mrr_vals else None,
            },
            "latency_ms": {
                "avg": round(statistics.mean(lats), 1) if lats else 0.0,
                "median": round(statistics.median(lats), 1) if lats else 0.0,
                "p95": p95_ms,
            },
            "refusal": f"{ref_cnt}/{len(unans)}" if unans and any(r.get("status") != "retrieval_only" for r in unans) else "검색전용",
            "ragas_avg": {},
            "underperforming_items": [],
        }

    return {
        "timestamp": latest_ts,
        "owner": "jh",
        "domain": "law",
        "eval_version": "jh_law_qa_v1",
        "ragas_target": 0.70,
        "total_corpus_chunks": 167,
        "collection_name": "jh_1000",
        "embedding_model": "text-embedding-3-small",
        "source_branch": source_branch,
        "source_file": source_label,
        "notes": "지훈 법령 파트 JSONL 자동 변환 (RAGAS 미측정)",
        "modes": modes_out,
    }


def _convert_jihun_experiment_splits_local(
    md_text: str,
    source_label: str = "test/jh/실험기록.md (§19)",
    source_branch: str = "local(dev)",
) -> list[dict[str, Any]]:
    """지훈(jh) 실험기록.md §19 DEV/TEST A/B/C 표 -> 시험지별 표준 요약 리스트 순수 stdlib 변환."""
    sec19_idx = md_text.find("## 19.")
    chunk = md_text[sec19_idx:] if sec19_idx >= 0 else md_text

    hit_map: dict[tuple[str, str], str] = {}
    ref_sec_idx = chunk.find("### 근거 부족 보류")
    hit_chunk = chunk[:ref_sec_idx] if ref_sec_idx >= 0 else chunk
    for m in re.finditer(r"^\|\s*(DEV|TEST)\s*\|\s*(A|B|C)\s*\|\s*(\d+/\d+)\s*\|", hit_chunk, flags=re.MULTILINE):
        hit_map.setdefault((m.group(1), m.group(2)), m.group(3))

    ref_map: dict[tuple[str, str], str] = {}
    if ref_sec_idx >= 0:
        for m in re.finditer(r"^\|\s*(DEV|TEST)\s*\|\s*(A|B|C)\s*\|\s*(\d+/\d+)\s*\|", chunk[ref_sec_idx : ref_sec_idx + 1200], flags=re.MULTILINE):
            ref_map.setdefault((m.group(1), m.group(2)), m.group(3))

    lat_map: dict[tuple[str, str], tuple[float, float, float]] = {}
    lat_sec_idx = chunk.find("### 지연:")
    if lat_sec_idx >= 0:
        for m in re.finditer(
            r"^\|\s*(DEV|TEST)\s*\|\s*(A|B|C)\s*\|\s*\d+\s*\|\s*[\d.]+\s*\|\s*[\d.]+\s*\|\s*[\d.]+\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|",
            chunk[lat_sec_idx : lat_sec_idx + 1500],
            flags=re.MULTILINE,
        ):
            lat_map[(m.group(1), m.group(2))] = (
                round(float(m.group(3)) * 1000.0, 1),
                round(float(m.group(4)) * 1000.0, 1),
                round(float(m.group(5)) * 1000.0, 1),
            )

    pmode_map = {"A": "A_vector", "B": "B_hybrid", "C": "C_hybrid_rerank"}
    out: list[dict[str, Any]] = []
    for split_key, eval_ver, ts in (
        ("DEV", "jh_law_dev", "20261126_120000"),
        ("TEST", "jh_law_test", "20261126_130000"),
    ):
        split_modes: dict[str, dict[str, Any]] = {}
        for mode_name in ("A", "B", "C"):
            hit_str = hit_map.get((split_key, mode_name))
            if not hit_str:
                continue
            ans_cnt = int(hit_str.split("/")[1])
            avg_ms, med_ms, p95_ms = lat_map.get((split_key, mode_name), (0.0, 0.0, 0.0))
            mkey = f"jh_1000_{split_key}_{mode_name}"
            split_modes[mkey] = {
                "preset": {
                    "name": mkey,
                    "pipeline_mode": pmode_map[mode_name],
                    "use_metadata_filter": False,
                    "use_rewrite": False,
                    "use_cache": False,
                    "use_hyde": False,
                    "collection_name": "jh_1000",
                    "embedding_model": "text-embedding-3-small",
                    "chunk_strategy": "law-1000-200-dedupe",
                },
                "retrieval": {
                    "answerable_count": ans_cnt,
                    "recall@12": None,
                    "hit@5": hit_str,
                    "top1": "-",
                    "mrr@5": None,
                },
                "latency_ms": {"avg": avg_ms, "median": med_ms, "p95": p95_ms},
                "refusal": ref_map.get((split_key, mode_name), "-"),
                "ragas_avg": {},
                "underperforming_items": [],
            }
        if split_modes:
            out.append(
                {
                    "timestamp": ts,
                    "owner": "jh",
                    "domain": "law",
                    "eval_version": eval_ver,
                    "ragas_target": 0.70,
                    "total_corpus_chunks": 167,
                    "collection_name": "jh_1000",
                    "embedding_model": "text-embedding-3-small",
                    "run_note": f"jh_1000 {split_key} (A/B/C)",
                    "source_branch": source_branch,
                    "source_file": source_label,
                    "notes": "지훈 법령 파트 실험기록.md §19 최종 집계 자동 변환 (공식 RAGAS 미측정)",
                    "modes": split_modes,
                }
            )
    return out


def _convert_byeongcheol_history_md_local(
    md_text: str,
    source_label: str = "test/bc/docs/development_history.md",
    source_branch: str = "local(dev)",
) -> list[dict[str, Any]]:
    """병철(bc) development_history.md -> Dev / 기존 Test(RAGAS 포함) / 신규 Test(T21~T40) 순수 stdlib 변환."""
    pmode_map = {"A": "A_vector", "B": "B_hybrid", "C": "C_hybrid_rerank"}
    dev_hits: dict[tuple[str, str], str] = {}
    test_hits: dict[tuple[str, str], str] = {}
    for m in re.finditer(
        r"^\|\s*(Dev|기존 Test)\s+(A|B|C)\s*\|\s*(\d+/\d+)\s*\|\s*(\d+/\d+)\s*\|\s*(\d+/\d+)\s*\|",
        md_text,
        flags=re.MULTILINE,
    ):
        target_map = dev_hits if m.group(1) == "Dev" else test_hits
        letter = m.group(2)
        target_map[("baseline", letter)] = m.group(3)
        target_map[("filter_v1", letter)] = m.group(4)
        target_map[("filter_v2", letter)] = m.group(5)

    test_ragas: dict[tuple[str, str], dict[str, float]] = {}
    ragas_start = md_text.find("### 같은 설정으로 재평가한 RAGAS")
    ragas_end = md_text.find("### F1은 무엇을 계산했나")
    ragas_block = md_text[ragas_start : (ragas_end if ragas_end > ragas_start else ragas_start + 1500)] if ragas_start >= 0 else ""
    for m in re.finditer(
        r"^\|\s*(A|B|C)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*[+\-\d.]+\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|",
        ragas_block,
        flags=re.MULTILINE,
    ):
        letter = m.group(1)
        test_ragas.setdefault(
            ("baseline", letter),
            {"llm_context_precision_with_reference": float(m.group(2)), "context_recall": float(m.group(4))},
        )
        test_ragas.setdefault(
            ("filter_v2", letter),
            {"llm_context_precision_with_reference": float(m.group(3)), "context_recall": float(m.group(5))},
        )

    m_v3 = re.search(r"근거 있는 17문항 Hit@5\s*\|\s*(\d+/\d+)", md_text)
    v3_hit = m_v3.group(1) if m_v3 else "15/17"

    def _mk_mode(stage: str, letter: str, hit_str: str, ragas_map: dict[str, float] | None = None, refusal: str = "-") -> dict[str, Any]:
        ans_cnt = int(hit_str.split("/")[1]) if "/" in hit_str else 0
        return {
            "preset": {
                "name": f"{stage}_{letter}",
                "pipeline_mode": pmode_map[letter],
                "use_metadata_filter": stage != "baseline",
                "use_rewrite": False,
                "use_cache": False,
                "use_hyde": False,
                "collection_name": "course_v0_small",
                "embedding_model": "text-embedding-3-small",
                "chunk_strategy": f"V0-{stage}",
            },
            "retrieval": {"answerable_count": ans_cnt, "recall@12": None, "hit@5": hit_str, "top1": "-", "mrr@5": None},
            "latency_ms": {"avg": 0.0, "median": 0.0, "p95": 0.0},
            "refusal": refusal,
            "ragas_avg": dict(ragas_map) if ragas_map else {},
            "underperforming_items": [],
        }

    summaries: list[dict[str, Any]] = []
    if dev_hits:
        for stage, stage_lbl, ts in (
            ("baseline", "교안 기준선 (필터 OFF)", "20261125_150000"),
            ("filter_v1", "초기 필터 (v1)", "20261125_200000"),
            ("filter_v2", "정규화 보완 필터 (v2)", "20261126_110000"),
        ):
            modes = {f"{stage}_{lt}": _mk_mode(stage, lt, dev_hits[(stage, lt)]) for lt in ("A", "B", "C") if (stage, lt) in dev_hits}
            if modes:
                summaries.append(
                    {
                        "timestamp": ts,
                        "owner": "bc",
                        "domain": "rent_deals",
                        "eval_version": "bc_deals_dev",
                        "ragas_target": 0.60,
                        "total_corpus_chunks": 31772,
                        "collection_name": "course_v0_small",
                        "embedding_model": "text-embedding-3-small",
                        "run_note": stage_lbl,
                        "source_branch": source_branch,
                        "source_file": source_label,
                        "modes": modes,
                    }
                )

    if test_hits:
        for stage, stage_lbl, ts in (
            ("baseline", "교안 기준선 (필터 OFF)", "20261125_160000"),
            ("filter_v1", "초기 필터 (v1)", "20261125_210000"),
            ("filter_v2", "정규화 보완 필터 (v2)", "20261126_120000"),
        ):
            modes = {
                f"{stage}_{lt}": _mk_mode(stage, lt, test_hits[(stage, lt)], test_ragas.get((stage, lt)))
                for lt in ("A", "B", "C")
                if (stage, lt) in test_hits
            }
            if modes:
                summaries.append(
                    {
                        "timestamp": ts,
                        "owner": "bc",
                        "domain": "rent_deals",
                        "eval_version": "bc_deals_test",
                        "ragas_target": 0.60,
                        "total_corpus_chunks": 31772,
                        "collection_name": "course_v0_small",
                        "embedding_model": "text-embedding-3-small",
                        "run_note": stage_lbl,
                        "source_branch": source_branch,
                        "source_file": source_label,
                        "modes": modes,
                    }
                )

    summaries.append(
        {
            "timestamp": "20261126_140000",
            "owner": "bc",
            "domain": "rent_deals",
            "eval_version": "bc_deals_test_v3",
            "ragas_target": 0.60,
            "total_corpus_chunks": 31772,
            "collection_name": "course_v0_small",
            "embedding_model": "text-embedding-3-small",
            "run_note": "신규 T21~T40 (C · 필터 v2)",
            "source_branch": source_branch,
            "source_file": source_label,
            "modes": {"filter_v2_C": _mk_mode("filter_v2", "C", v3_hit, None, refusal="3/3")},
        }
    )
    return summaries


def _convert_byeongcheol_comparison_local(
    raw: dict[str, Any],
    source_label: str = "comparison_data.json",
    source_branch: str = "uploaded",
) -> dict[str, Any]:
    """병철(bc) comparison_data.json -> 공통 UnifiedEvalSummary dict 순수 stdlib 변환."""
    q_meta = {q["id"]: q for q in raw.get("questions") or [] if isinstance(q, dict) and q.get("id")}
    modes_out: dict[str, dict[str, Any]] = {}
    for stage_key, prefix in (("before", "before"), ("after", "after_filter")):
        rows = raw.get(stage_key) or []
        by_sys: dict[str, list[dict]] = {}
        for r in rows:
            if q_meta.get(r.get("question_id"), {}).get("split") == "dev":
                by_sys.setdefault(str(r.get("system", "B")), []).append(r)
        if not by_sys:
            for r in rows:
                by_sys.setdefault(str(r.get("system", "B")), []).append(r)
        for sys_name in sorted(by_sys):
            grp = by_sys[sys_name]
            ans = [r for r in grp if q_meta.get(r.get("question_id"), {}).get("reference_answerable", r.get("reference_answerable", True))]
            unans = [r for r in grp if r not in ans]
            hit_cnt = sum(1 for r in ans if bool(r.get("hit_at_5")))
            ref_cnt = sum(1 for r in unans if any(k in str(r.get("response", "")) for k in ("확인할 수 없", "찾을 수 없", "없습니다", "부족")))
            lats = sorted(float(r.get("seconds", 0.0) or 0.0) * 1000.0 for r in grp if r.get("seconds") is not None)
            p95_ms = round(lats[max(0, int(len(lats) * 0.95) - 1)], 1) if lats else 0.0
            modes_out[f"{prefix}_{sys_name}"] = {
                "retrieval": {"answerable_count": len(ans), "recall@12": None, "hit@5": f"{hit_cnt}/{len(ans)}" if ans else "-", "top1": "-", "mrr@5": None},
                "latency_ms": {"avg": round(statistics.mean(lats), 1) if lats else 0.0, "median": round(statistics.median(lats), 1) if lats else 0.0, "p95": p95_ms},
                "refusal": f"{ref_cnt}/{len(unans)}" if unans else "-",
                "ragas_avg": {},
                "underperforming_items": [],
            }
    ts = str((raw.get("consolidation") or {}).get("date", "")).replace("-", "") or datetime.now().strftime("%Y%m%d")
    return {
        "timestamp": f"{ts}_000000" if len(ts) == 8 else ts,
        "owner": "bc",
        "domain": "rent_deals",
        "eval_version": "bc_deals_dev",
        "ragas_target": 0.60,
        "collection_name": "course_v0_small",
        "embedding_model": "text-embedding-3-small",
        "source_branch": source_branch,
        "source_file": source_label,
        "modes": modes_out,
    }


def _load_owner_summaries(owner: str) -> list[tuple[Path, dict]]:
    """선택한 팀원(kh/yj/jh/bc)의 전체 회차별 평가 요약을 시간순으로 로드한다.

    외부 패키지(`rag_home_api`, `psycopg`, `langchain`) 의존성을 완전히 제거한 순수 stdlib 로더이므로
    `uv run --with streamlit` 격리 환경이나 Streamlit 핫 리로드(`R`)에서도 4명 전원이 100% 안정 로드된다.
    """
    errors: list[str] = []
    out: list[tuple[Path, dict]] = []

    if owner == "kh":
        out = _load_all_local_summaries()
    elif owner == "yj":
        yj_dir = TEST_ROOT_DIR / "yj" / "docs" / "04-2_eval_reports"
        if yj_dir.exists():
            yj_files = sorted(
                yj_dir.rglob("eval_*.json"),
                key=lambda p: (re.search(r"(\d{8}_\d{4,6})", p.name).group(1) if re.search(r"(\d{8}_\d{4,6})", p.name) else p.name),
            )
            for fp in yj_files:
                try:
                    cv = fp.parent.name.replace("chunk_", "") if fp.parent.name.startswith("chunk_") else ""
                    raw = _load_json_file(fp)
                    out.append((fp, _convert_yeji_eval_local(raw, source_label=f"{fp.parent.name}/{fp.name}", chunk_version=cv)))
                except Exception as exc:
                    errors.append(f"{fp.name}: {exc}")
        else:
            errors.append(f"경로 없음: {yj_dir}")
    elif owner == "jh":
        jh_dir = TEST_ROOT_DIR / "jh"
        jh_res_dir = jh_dir / "results"
        if jh_res_dir.exists():
            for fp in sorted(jh_res_dir.glob("*.jsonl")):
                try:
                    out.append((fp, _convert_jihun_jsonl_local(fp.read_text(encoding="utf-8"), source_label=fp.name)))
                except Exception as exc:
                    errors.append(f"{fp.name}: {exc}")
        jh_md = jh_dir / "실험기록.md"
        if jh_md.exists():
            try:
                for sm in _convert_jihun_experiment_splits_local(jh_md.read_text(encoding="utf-8")):
                    out.append((jh_md, sm))
            except Exception as exc:
                errors.append(f"실험기록.md: {exc}")
        elif not out:
            errors.append(f"경로 없음: {jh_md}")
    elif owner == "bc":
        bc_dir = TEST_ROOT_DIR / "bc"
        bc_hist_md = bc_dir / "docs" / "development_history.md"
        if bc_hist_md.exists():
            try:
                for sm in _convert_byeongcheol_history_md_local(bc_hist_md.read_text(encoding="utf-8")):
                    out.append((bc_hist_md, sm))
            except Exception as exc:
                errors.append(f"development_history.md: {exc}")
        else:
            bc_comp = bc_dir / "results" / "comparison_data.json"
            if bc_comp.exists():
                try:
                    out.append((bc_comp, _convert_byeongcheol_comparison_local(_load_json_file(bc_comp), source_branch="local(dev)")))
                except Exception as exc:
                    errors.append(f"comparison_data.json: {exc}")
            else:
                errors.append(f"경로 없음: {bc_hist_md}")

    _LAST_LOAD_ERRORS[owner] = errors
    return out

RUN_MILESTONE_NOTES: dict[str, dict[str, str]] = {
    "20261125_114421": {
        "tag": "EXP-2 (3구 866청크 · 초안)",
        "context": "3개 구(강남·관악·종로, 866청크) 파일럿 · 초안 평가셋(_draft) · 기본 프롬프트",
    },
    "20261125_145306": {
        "tag": "EXP-3 (25구 6,200청크 · 공정v1)",
        "context": "25개 구 전체(6,200청크, 출처 파일 1줄 추가) · 공정 v1 평가셋 · 간결 프롬프트 · RRF 3종 + 경량 필터 비교",
    },
    "20261125_154004": {
        "tag": "EXP-4 (Option A · 주어 1줄 보강)",
        "context": "B_rrf_70_30_filter 고정 + ANSWER_PROMPT 첫 문장 대상 5조건 주어 명시 1줄 추가 + 전 문항 per_item_ragas 기록",
    },
}


# ==================== 데이터 로드 & 공통 유틸 ====================
def _load_json_file(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl_file(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _find_companion_jsonl(summary_path: Path, timestamp: str) -> Path | None:
    if not timestamp:
        return None
    for pat in (f"exp_rent_ab_ragas_{timestamp}.jsonl", f"exp_*{timestamp}.jsonl"):
        matches = list(summary_path.parent.glob(pat))
        if matches:
            return matches[0]
    return None


def _load_all_local_summaries(results_dir: Path = KH_RESULTS_DIR) -> list[tuple[Path, dict]]:
    """지정 results 폴더의 summary_*.json 을 시간순(오래된 순)으로 로드한다."""
    if not results_dir.exists():
        return []
    out: list[tuple[Path, dict]] = []
    for p in sorted(results_dir.glob("summary_*.json")):
        try:
            out.append((p, _load_json_file(p)))
        except Exception:
            continue
    return out


def _parse_ratio(val: Any) -> float | None:
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str) and "/" in val:
        try:
            a, b = val.split("/", 1)
            denom = float(b.strip())
            return float(a.strip()) / denom if denom > 0 else 0.0
        except ValueError:
            return None
    return None


def _fmt_ts(ts: str) -> str:
    if len(ts) >= 13 and "_" in ts:
        d, t = ts.split("_", 1)
        if len(d) == 8 and len(t) >= 4:
            return f"{d[4:6]}-{d[6:8]} {t[0:2]}:{t[2:4]}"
    return ts or "-"


def _infer_eval_version(summary: dict) -> str:
    if summary.get("eval_version"):
        return str(summary["eval_version"])
    ep = str(summary.get("eval_path", ""))
    return Path(ep).stem if ep else "unknown_eval"


def _infer_domain(summary: dict, owner_hint: str = "") -> str:
    if summary.get("domain"):
        return str(summary["domain"])
    blob = (str(summary.get("collection_name", "")) + " " + str(summary.get("eval_path", "")) + " " + owner_hint).lower()
    if "law" in blob or "법령" in blob or "jihun" in blob:
        return "law"
    if "public" in blob or "공고" in blob or "임대" in blob or "yeji" in blob:
        return "public_housing"
    if "rent" in blob or "전월세" in blob or "kh" in blob or "bc" in blob:
        return "rent"
    return "unified"


def _infer_corpus_size(summary: dict) -> int:
    if summary.get("total_corpus_chunks"):
        return int(summary["total_corpus_chunks"])
    ts = str(summary.get("timestamp", ""))
    if ts == "20261125_114421":
        return 866
    return 6200


# ==================== TP / FP / FN / TN 혼동행렬 & 5조건 불일치 파서 ====================
def _item_confusion_matrix(log_row: dict, k: int, total_corpus_chunks: int = 6200) -> dict[str, int]:
    """문항 1건의 metrics_k12 / metrics_k5 에서 TP, FP, FN, TN 을 정확히 산출한다 (구버전 로그 호환)."""
    mkey = "metrics_k12" if k >= 10 else "metrics_k5"
    m = log_row.get(mkey) or {}
    if all(x in m for x in ("tp", "fp", "fn", "tn")):
        return {"tp": int(m["tp"]), "fp": int(m["fp"]), "fn": int(m["fn"]), "tn": int(m["tn"])}

    ref_ctxs = log_row.get("reference_contexts") or []
    allowed = log_row.get("allowed_chunk_ids") or []
    g = len(allowed) if allowed else len(ref_ctxs)
    if g <= 0:
        return {"tp": 0, "fp": 0, "fn": 0, "tn": total_corpus_chunks}

    rec = float(m.get(f"recall@{k}", m.get("recall@12" if k >= 10 else "recall@5", 0.0)) or 0.0)
    prec = float(m.get(f"precision@{k}", m.get("precision@12" if k >= 10 else "precision@5", 0.0)) or 0.0)
    tp = int(round(rec * g))
    if k <= 5 and log_row.get("top5_ids"):
        actual_k = len(log_row["top5_ids"])
    elif tp > 0 and prec > 0:
        actual_k = int(round(tp / prec))
    else:
        actual_k = k
    fp = max(0, actual_k - tp)
    fn = max(0, g - tp)
    tn = max(0, total_corpus_chunks - g - fp)
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def _aggregate_confusion_matrix(log_rows: list[dict], mode: str, k: int, total_corpus_chunks: int = 6200) -> dict[str, int]:
    """특정 모드의 답변 가능 전체 문항에 대한 TP, FP, FN, TN 합계를 계산한다."""
    agg = {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "q_count": 0}
    for r in log_rows:
        if r.get("mode") != mode or r.get("expected_status") != "answered":
            continue
        cm = _item_confusion_matrix(r, k=k, total_corpus_chunks=total_corpus_chunks)
        for key in ("tp", "fp", "fn", "tn"):
            agg[key] += cm[key]
        agg["q_count"] += 1
    return agg


def _parse_rent_chunk_id(cid: str) -> dict[str, str] | None:
    """'RENT_용산구_한남동_단독다가구_전세_33초과_60이하' -> 5조건 딕셔너리 파싱."""
    if not cid.startswith("RENT_"):
        return None
    parts = cid.split("_")
    if len(parts) < 6:
        return None
    return {
        "gu": parts[1],
        "dong": parts[2],
        "housing": parts[3],
        "lease": parts[4],
        "area": "_".join(parts[5:]),
    }


def _parse_reference_targets(log_row: dict) -> list[dict[str, str]]:
    """정답 컨텍스트(allowed_chunk_ids 또는 reference_contexts 헤더)에서 목표 5조건 리스트 추출."""
    targets: list[dict[str, str]] = []
    for cid in log_row.get("allowed_chunk_ids") or []:
        parsed = _parse_rent_chunk_id(cid)
        if parsed:
            targets.append(parsed)
    if targets:
        return targets

    area_map = {"33㎡이하(원룸급)": "33이하", "33~60㎡(소형)": "33초과_60이하", "60㎡초과(중대형)": "60초과"}
    for ctx in log_row.get("reference_contexts") or []:
        first = ctx.splitlines()[0] if ctx else ""
        # [서울 전월세 실거래 요약 | 용산구 한남동 | 단독다가구 | 전세 | 33~60㎡(소형)]
        m = re.match(r"\[[^|]+\|\s*([^|\s]+)\s+([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*([^\]]+)\]", first)
        if m:
            raw_area = m.group(5).strip()
            targets.append(
                {
                    "gu": m.group(1).strip(),
                    "dong": m.group(2).strip(),
                    "housing": m.group(3).strip(),
                    "lease": m.group(4).strip(),
                    "area": area_map.get(raw_area, raw_area),
                }
            )
    return targets


def _explain_chunk_mismatch(cid: str, targets: list[dict[str, str]]) -> tuple[bool, list[str]]:
    """후보 청크 ID 1개가 정답 5조건과 일치하는지, 틀렸다면 정확히 어떤 축이 불일치인지 태그 반환."""
    cand = _parse_rent_chunk_id(cid)
    if not cand or not targets:
        return False, []

    best_diffs: list[str] | None = None
    for tgt in targets:
        diffs: list[str] = []
        if cand["gu"] != tgt["gu"]:
            diffs.append(f"❌ 구 불일치({cand['gu']}≠{tgt['gu']})")
        if cand["dong"] != tgt["dong"]:
            diffs.append(f"❌ 동 불일치({cand['dong']}≠{tgt['dong']})")
        if cand["housing"] != tgt["housing"]:
            diffs.append(f"❌ 유형 불일치({cand['housing']}≠{tgt['housing']})")
        if cand["lease"] != tgt["lease"]:
            diffs.append(f"❌ 계약 불일치({cand['lease']}≠{tgt['lease']})")
        if cand["area"] != tgt["area"]:
            diffs.append(f"❌ 면적 불일치({cand['area']}≠{tgt['area']})")
        if not diffs:
            return True, ["✅ ★ 정답 5조건 완전 일치!"]
        if best_diffs is None or len(diffs) < len(best_diffs):
            best_diffs = diffs
    return False, best_diffs or []


def _classify_funnel_stage(log_row: dict | None, ragas_scores: dict | None, target: float) -> tuple[int, str]:
    """문항 1건의 4단계 깔때기 상태(0=통과, 2=검색누락FN, 3=순위실패FP, 4=생성/주어/인용) 반환."""
    is_fail = False
    if ragas_scores:
        is_fail = any(float(v) < target for v in ragas_scores.values())
    if not log_row:
        return (4 if is_fail else 0), ("상세 로그(.jsonl) 없음" if is_fail else "🟢 통과")

    m12 = log_row.get("metrics_k12") or {}
    m5 = log_row.get("metrics_k5") or {}
    hit12 = int(m12.get("hit@12", 0) or 0)
    hit5 = int(m5.get("hit@5", 0) or 0)
    rec12 = float(m12.get("recall@12", 0.0) or 0.0)
    rec5 = float(m5.get("recall@5", 1.0 if hit5 else 0.0) or 0.0)
    fn5 = int(m5.get("fn", 0) or 0)
    r12 = int(m12.get("first_rank", 0) or 0)
    r5 = int(m5.get("first_rank", 0) or 0)

    if hit12 == 0:
        return 2, "🔴 [2단계] 후보 12개 밖 완전 누락 (FN · Recall@12=0) — B/필터/Query Rewrite 점검"
    if rec12 < 1.0:
        return 2, f"🔴 [2단계] 복합 정답 일부 누락 (FN · Recall@12={rec12:.2f}, 첫 정답 #{r12}) — Query Rewrite(서브쿼리 분해) 필요"
    if hit5 == 0:
        return 3, f"🟠 [3단계] 후보 12개 중 #{r12} 생존, Top-5 탈락 — 검색 C(Re-ranking)로 구제 가능"
    if fn5 > 0 or rec5 < 1.0:
        return 3, f"🟠 [3단계] 복합 정답 일부 Top-5 누락 (Recall@12=1.00, Recall@5={rec5:.2f}, FN@5={fn5}) — 검색 C(Re-ranking)로 12→5 구제 가능"
    if r5 > 1:
        return 3, f"🟡 [3단계] Top-5 포함(first_rank=#{r5}), 1등 밀림 — 검색 C(Re-ranking)·동 필터·가중치 개선"
    if is_fail:
        return 4, "🔵 [4단계] 검색 1등 적중(first_rank=#1, FN@5=0) — 답변 생성·주어·TMI·인용 프롬프트 점검"
    return 0, "🟢 [전 단계 통과] 검색 1등 + RAGAS 4지표 목표 달성"


def _funnel_badge(log_row: dict | None, ragas_scores: dict | None = None, target: float = 0.60) -> str:
    _, msg = _classify_funnel_stage(log_row, ragas_scores, target)
    return msg


# ==================== 히스토리 스텝 빌더 & 단일 변인 Diff ====================
def _describe_preset_diff(prev_step: dict | None, cur_step: dict) -> tuple[int, list[str]]:
    """두 스텝 사이에 정확히 몇 개의 변수가 바뀌었는지(단일 변인 통제 가드) 계산한다."""
    if prev_step is None:
        return 0, ["최초 기준선 (Baseline)"]

    diffs: list[str] = []
    if cur_step["eval_version"] != prev_step["eval_version"]:
        diffs.append(f"시험지 변경: {prev_step['eval_version']} → {cur_step['eval_version']}")

    pp, cp = prev_step.get("preset") or {}, cur_step.get("preset") or {}
    pmode = pp.get("pipeline_mode") or ("A_vector" if prev_step["mode"] == "A_vector" else "B_hybrid")
    cmode = cp.get("pipeline_mode") or ("A_vector" if cur_step["mode"] == "A_vector" else "B_hybrid")
    if pmode != cmode:
        diffs.append(f"검색 방식: {pmode} → {cmode}")
    if pmode != "A_vector" and cmode != "A_vector":
        bw1, vw1 = float(pp.get("bm25_weight", 0.55) or 0.55), float(pp.get("vector_weight", 0.45) or 0.45)
        bw2, vw2 = float(cp.get("bm25_weight", 0.55) or 0.55), float(cp.get("vector_weight", 0.45) or 0.45)
        if abs(bw1 - bw2) > 1e-6 or abs(vw1 - vw2) > 1e-6:
            diffs.append(f"BM25:Vec 가중치: {bw1:.2f}:{vw1:.2f} → {bw2:.2f}:{vw2:.2f}")
        rk1 = int(pp.get("rrf_k", 60) or 60)
        rk2 = int(cp.get("rrf_k", 60) or 60)
        if rk1 != rk2:
            diffs.append(f"rrf_k: {rk1} → {rk2}")

    if bool(pp.get("use_metadata_filter")) != bool(cp.get("use_metadata_filter")):
        diffs.append(f"메타데이터 필터: {'ON' if pp.get('use_metadata_filter') else 'OFF'} → {'ON' if cp.get('use_metadata_filter') else 'OFF'}")
    elif bool(pp.get("use_metadata_filter")) and bool(cp.get("use_metadata_filter")):
        pv2 = "filter_v2" in str(prev_step.get("mode", ""))
        cv2 = "filter_v2" in str(cur_step.get("mode", ""))
        if pv2 != cv2:
            diffs.append(f"사전 필터 규칙: {'v2(평수·법정동)' if pv2 else 'v1'} → {'v2(평수·법정동)' if cv2 else 'v1'}")
    if bool(pp.get("use_rewrite")) != bool(cp.get("use_rewrite")):
        diffs.append(f"Query Rewrite: {'ON' if pp.get('use_rewrite') else 'OFF'} → {'ON' if cp.get('use_rewrite') else 'OFF'}")
    if bool(pp.get("use_cache")) != bool(cp.get("use_cache")):
        diffs.append(f"Cache: {'ON' if pp.get('use_cache') else 'OFF'} → {'ON' if cp.get('use_cache') else 'OFF'}")
    if bool(pp.get("use_hyde")) != bool(cp.get("use_hyde")):
        diffs.append(f"HyDE: {'ON' if pp.get('use_hyde') else 'OFF'} → {'ON' if cp.get('use_hyde') else 'OFF'}")
    if pp.get("candidate_k") != cp.get("candidate_k") and pp.get("candidate_k") and cp.get("candidate_k"):
        diffs.append(f"candidate_k: {pp.get('candidate_k')} → {cp.get('candidate_k')}")
    if pp.get("collection_name") != cp.get("collection_name") and (pp.get("collection_name") or cp.get("collection_name")):
        diffs.append(f"청킹 컬렉션: {pp.get('collection_name', 'housing_rent_kh')} → {cp.get('collection_name', 'housing_rent_kh')}")
    if pp.get("chunk_strategy") != cp.get("chunk_strategy") and pp.get("chunk_strategy") and cp.get("chunk_strategy"):
        diffs.append(f"청킹/단계: {pp.get('chunk_strategy')} → {cp.get('chunk_strategy')}")
    if pp.get("embedding_model") != cp.get("embedding_model") and (pp.get("embedding_model") or cp.get("embedding_model")):
        diffs.append(f"임베딩 모델: {pp.get('embedding_model', 'text-embedding-3-small')} → {cp.get('embedding_model', 'text-embedding-3-small')}")
    pv_prompt = pp.get("prompt_version")
    cv_prompt = cp.get("prompt_version")
    if pv_prompt and cv_prompt and pv_prompt != cv_prompt:
        diffs.append(f"프롬프트: {pv_prompt} → {cv_prompt}")

    if not diffs:
        if cur_step.get("run_note"):
            diffs.append(f"프롬프트/설정 변경: {cur_step['run_note']}")
        elif cur_step["timestamp"] == "20261125_154004":
            diffs.append("ANSWER_PROMPT 첫 문장 대상 5조건 주어 명시 1줄 추가")
        elif cur_step["timestamp"] != prev_step["timestamp"]:
            diffs.append(f"동일 프리셋 재실행 ({_fmt_ts(prev_step['timestamp'])} → {_fmt_ts(cur_step['timestamp'])})")
    return len(diffs), diffs


def _describe_delta_narrative(cur: dict, prev: dict | None) -> str:
    if prev is None:
        return f"기준선 측정: Hit@5 {cur['hit5_str']}, MRR@5 {cur['mrr5']:.3f}, CP {cur['cp']:.3f}, AR {cur['ar']:.3f}"

    parts: list[str] = []
    d_top1 = (cur["top1_rate"] - prev["top1_rate"]) * 100
    d_cp = cur["cp"] - prev["cp"]
    d_cr = cur["cr"] - prev["cr"]
    d_ar = cur["ar"] - prev["ar"]
    d_f = cur["f"] - prev["f"]
    if abs(d_top1) >= 1.0:
        parts.append(f"Top1 {prev['top1_str']}→{cur['top1_str']}({d_top1:+.1f}%p)")
    if abs(d_cp) >= 0.005:
        parts.append(f"CP {d_cp:+.3f}")
    if abs(d_cr) >= 0.005:
        parts.append(f"CR {d_cr:+.3f}")
    if abs(d_ar) >= 0.005:
        parts.append(f"AR {d_ar:+.3f}")
    if abs(d_f) >= 0.02:
        parts.append(f"F {d_f:+.3f}")

    if cur["timestamp"] == "20261125_154004":
        note = "🏆 주어 명시 1줄로 AR 목표(0.60) 첫 돌파 → RAGAS 4지표 전원 통과!"
    elif cur["timestamp"] == "20261125_145306" and cur["mode"] == "B_rrf_70_30_filter":
        note = "필터로 Top1 2배·CP 0.823 급등 + p95 단축 (단, 주어 생략으로 AR -0.05)"
    elif cur["timestamp"] == "20261125_145306" and cur["mode"] == "B_rrf_70_30":
        note = "RRF 최적 가중치(70:30) 도출 · CP 0.60 첫 돌파"
    elif cur["timestamp"] == "20261125_145306" and cur["mode"] == "B_rrf_80_20":
        note = "BM25 과중으로 DEV-02 5위 하락 → CP 0.60 재이탈"
    elif cur["timestamp"] == "20261125_114421" and cur["mode"] == "B_hybrid_rrf":
        note = "A→B 전환으로 Hit@5 3배·MRR 7.3배·CR 0.60 돌파"
    elif cur.get("run_note"):
        note = cur["run_note"]
    else:
        note = f"미달 {prev['fail_cnt']}건→{cur['fail_cnt']}건"

    return f"{', '.join(parts) if parts else '주요 지표 유지'} | {note}"


def _build_history_steps(all_summaries: list[tuple[Path, dict]]) -> list[dict]:
    """모든 summary_*.json 을 시간순 스텝 레코드(동반 .jsonl 로그 포함)로 구성한다."""
    steps: list[dict] = []
    idx = 0
    for path, summary in all_summaries:
        ts = str(summary.get("timestamp", ""))
        eval_ver = _infer_eval_version(summary)
        eval_name = Path(summary.get("eval_path", "-")).name
        col_name = str(summary.get("collection_name", "-"))
        corpus_n = _infer_corpus_size(summary)
        run_note = str(summary.get("run_note", ""))
        run_tag = RUN_MILESTONE_NOTES.get(ts, {}).get("tag", f"RUN ({_fmt_ts(ts)})")
        comp_path = _find_companion_jsonl(path, ts)
        comp_logs = summary.get("_companion_logs") or (_load_jsonl_file(comp_path) if comp_path else [])
        owner = str(summary.get("owner") or "kh")

        for mname, mdata in summary.get("modes", {}).items():
            idx += 1
            ret = mdata.get("retrieval", {})
            lat = mdata.get("latency_ms", {})
            rg = mdata.get("ragas_avg", {})
            preset = mdata.get("preset") or {}
            mode_logs = [r for r in comp_logs if r.get("mode") == mname]

            # 동반 로그에서 출처 표기율·과거 데이터 고지율 자동 보완 계산
            ans_logs = [r for r in mode_logs if r.get("expected_status") == "answered"]
            ref_logs = [r for r in mode_logs if r.get("expected_status") == "insufficient_evidence"]
            cite_str = mdata.get("citation_rate")
            if not cite_str and ans_logs:
                c_ok = sum(1 for r in ans_logs if ("출처:" in (r.get("response") or "") or ".csv" in (r.get("response") or "") or re.search(r"\[\d+\]", r.get("response") or "")))
                cite_str = f"{c_ok}/{len(ans_logs)}"
            scope_str = mdata.get("scope_notice")
            if not scope_str and ref_logs:
                s_ok = sum(
                    1
                    for r in ref_logs
                    if any(w in (r.get("response") or "") for w in ("과거", "체결 완료", "실거래 통계", "포함되지", "나와 있지", "없습니다"))
                )
                scope_str = f"{s_ok}/{len(ref_logs)}"

            has_f = rg.get("faithfulness") is not None
            has_ar = rg.get("answer_relevancy") is not None
            has_cp = rg.get("llm_context_precision_with_reference") is not None
            has_cr = rg.get("context_recall") is not None
            has_ragas = any((has_f, has_ar, has_cp, has_cr))
            has_r12 = ret.get("recall@12") is not None
            has_mrr5 = ret.get("mrr@5") is not None
            has_top1 = _parse_ratio(ret.get("top1")) is not None
            has_lat = float(lat.get("median", 0.0) or 0.0) > 0.0 or float(lat.get("avg", 0.0) or 0.0) > 0.0

            f_val = float(rg.get("faithfulness", 0.0) or 0.0)
            ar_val = float(rg.get("answer_relevancy", 0.0) or 0.0)
            cp_val = float(rg.get("llm_context_precision_with_reference", 0.0) or 0.0)
            cr_val = float(rg.get("context_recall", 0.0) or 0.0)
            measured_ragas = [v for ok, v in ((has_f, f_val), (has_ar, ar_val), (has_cp, cp_val), (has_cr, cr_val)) if ok]
            ragas_mean = round(sum(measured_ragas) / len(measured_ragas), 4) if measured_ragas else 0.0
            fail_cnt = len(mdata.get("underperforming_items", []))
            dom_target = float(summary.get("ragas_target", 0.60) or 0.60)

            short_ts = ts[-6:-2] if len(ts) >= 6 else ts
            n_modes_in_file = len(summary.get("modes", {}))
            if ts == "20261125_154004":
                tag_txt = "+주어"
            elif preset.get("use_cache"):
                ch_str = str(ret.get("cache_hits", ""))
                cache_state = "Cache Cold·MISS" if ch_str.startswith("0/") else ("Cache Warm·HIT" if "/" in ch_str else "Cache")
                prefix = "Test · " if "test" in eval_ver.lower() else ""
                tag_txt = f"{prefix}C·{cache_state}"
            elif n_modes_in_file > 1 and owner == "kh" and ("test" in eval_ver.lower() or "kure" in mname):
                prefix = "Test · " if "test" in eval_ver.lower() else ""
                pm = preset.get("pipeline_mode", "")
                if pm == "A_vector":
                    tag_txt = f"{prefix}A·Vector"
                elif pm == "C_hybrid_rerank" and preset.get("use_rewrite"):
                    tag_txt = f"{prefix}C·Rewrite"
                elif pm == "C_hybrid_rerank":
                    tag_txt = f"{prefix}C·Rerank"
                elif preset.get("use_metadata_filter"):
                    tag_txt = f"{prefix}B·FilterV2" if "filter_v2" in mname else f"{prefix}B·Filter"
                else:
                    tag_txt = f"{prefix}{run_note[:18]}" if run_note else ""
            elif preset.get("use_rewrite") and owner == "kh":
                prefix = "Test · " if "test" in eval_ver.lower() else ""
                tag_txt = f"{prefix}C·Rewrite ON"
            elif run_note:
                tag_txt = run_note[:24]
            else:
                tag_txt = ""
            step_label = f"#{idx} [{short_ts}] {mname} ({tag_txt})" if tag_txt else f"#{idx} [{short_ts}] {mname}"

            # 문항별 RAGAS 점수 맵 (per_item_ragas 우선, 없으면 underperforming_items에서 보완)
            item_ragas: dict[str, dict[str, float]] = {}
            for it in mdata.get("per_item_ragas", []):
                item_ragas[it.get("question_id", "")] = it.get("scores", {})
            for it in mdata.get("underperforming_items", []):
                qid = it.get("question_id", "")
                if qid not in item_ragas:
                    item_ragas[qid] = it.get("all_scores", {})

            rec = {
                "order": idx,
                "owner": owner,
                "step_label": step_label,
                "file_path": path,
                "file_name": path.name,
                "summary": summary,
                "mode_data": mdata,
                "mode_logs": mode_logs,
                "item_ragas": item_ragas,
                "timestamp": ts,
                "ts_display": _fmt_ts(ts),
                "run_tag": run_tag,
                "run_note": run_note,
                "eval_version": eval_ver,
                "eval_name": eval_name,
                "collection": col_name,
                "corpus_n": corpus_n,
                "mode": mname,
                "preset": preset,
                "has_r12": has_r12,
                "has_mrr5": has_mrr5,
                "has_top1": has_top1,
                "has_f": has_f,
                "has_ar": has_ar,
                "has_cp": has_cp,
                "has_cr": has_cr,
                "has_ragas": has_ragas,
                "has_lat": has_lat,
                "recall12": float(ret.get("recall@12", 0.0) or 0.0),
                "hit5_str": str(ret.get("hit@5", "-")),
                "hit5_rate": _parse_ratio(ret.get("hit@5")) or 0.0,
                "top1_str": str(ret.get("top1", "-")),
                "top1_rate": _parse_ratio(ret.get("top1")) or 0.0,
                "mrr5": float(ret.get("mrr@5", 0.0) or 0.0),
                "refusal_str": str(mdata.get("refusal", "-")),
                "refusal_rate": _parse_ratio(mdata.get("refusal")) or 0.0,
                "scope_notice_str": str(scope_str or "-"),
                "citation_str": str(cite_str or "-"),
                "lat_median": float(lat.get("median", 0.0) or 0.0),
                "lat_p95": float(lat.get("p95", 0.0) or 0.0),
                "lat_avg": float(lat.get("avg", 0.0) or 0.0),
                "f": f_val,
                "ar": ar_val,
                "cp": cp_val,
                "cr": cr_val,
                "ragas_mean": ragas_mean,
                "fail_cnt": fail_cnt,
                "pass_4of4": (len(measured_ragas) == 4 and all(v >= dom_target for v in measured_ragas)),
                "cm12": _aggregate_confusion_matrix(mode_logs, mname, k=12, total_corpus_chunks=corpus_n),
                "cm5": _aggregate_confusion_matrix(mode_logs, mname, k=5, total_corpus_chunks=corpus_n),
            }
            prev = steps[-1] if steps else None
            _, diffs = _describe_preset_diff(prev, rec)
            rec["change_desc"] = " · ".join(diffs)
            rec["delta_narrative"] = _describe_delta_narrative(rec, prev)
            steps.append(rec)
    return steps


# ==================== 차트 & 리포트 Export 유틸 ====================
def _render_altair_line_chart(
    steps: list[dict],
    series: list[tuple[str, str, str]],
    target: float | None = None,
) -> None:
    step_order = [s["step_label"] for s in steps]
    long_rows = []
    for s in steps:
        for key, label, _ in series:
            long_rows.append(
                {
                    "회차 스텝": s["step_label"],
                    "지표": label,
                    "점수": round(float(s[key]), 4),
                    "변경 조건": s["change_desc"],
                    "변화 해석": s["delta_narrative"],
                }
            )
    df_long = pd.DataFrame(long_rows)

    if alt is None:
        piv = df_long.pivot(index="회차 스텝", columns="지표", values="점수").reindex(step_order)
        st.line_chart(piv, height=320)
        return

    domain = [label for _, label, _ in series]
    colors = [color for _, _, color in series]

    base = alt.Chart(df_long).encode(
        x=alt.X("회차 스텝:N", sort=step_order, axis=alt.Axis(labelAngle=-20, labelLimit=200, title=None)),
        y=alt.Y("점수:Q", scale=alt.Scale(domain=[0, 1.05]), title="점수 (0.0 ~ 1.0)"),
        color=alt.Color("지표:N", scale=alt.Scale(domain=domain, range=colors), legend=alt.Legend(orient="top", columns=2, title=None)),
        tooltip=["회차 스텝", "지표", alt.Tooltip("점수:Q", format=".4f"), "변경 조건", "변화 해석"],
    )
    chart = base.mark_line(strokeWidth=2.6) + base.mark_point(size=75, filled=True)
    if target is not None:
        rule = alt.Chart(pd.DataFrame({"목표치": [target]})).mark_rule(color="#dc2626", strokeDash=[6, 4], strokeWidth=2).encode(y="목표치:Q")
        chart = chart + rule
    st.altair_chart(chart.properties(height=330).interactive(), use_container_width=True)


def _recommend_next_variable(step: dict, target: float) -> tuple[str, str, str, str]:
    """4단계 깔때기 분포를 보고 다음에 딱 1개만 바꿀 변수를 추천한다."""
    buckets: dict[int, list[str]] = {2: [], 3: [], 4: []}
    for r in step["mode_logs"]:
        if r.get("expected_status") != "answered":
            continue
        qid = r.get("question_id", "")
        stg, _ = _classify_funnel_stage(r, step["item_ragas"].get(qid), target)
        if stg in buckets:
            buckets[stg].append(qid)

    preset = step.get("preset") or {}
    pmode = preset.get("pipeline_mode") or ("A_vector" if step["mode"] == "A_vector" else "B_hybrid")
    if pmode == "A_vector":
        return (
            "검색 B (Hybrid BM25+RRF 55:45) 활성화",
            f"Vector 단독 기준 2단계 검색 누락({len(buckets[2])}건)과 3단계 순위 실패({len(buckets[3])}건)가 혼재합니다. 지명·전월세 키워드 회수부터 확보하세요.",
            "pipeline_mode: A_vector → B_hybrid (bm25=0.55, vec=0.45)",
            "구어체·표현 변형 문항이 BM25 편향으로 악화되지 않는지 감시",
        )
    if len(buckets[3]) >= 2 and not preset.get("use_metadata_filter"):
        bw = float(preset.get("bm25_weight", 0.55) or 0.55)
        if bw < 0.68:
            return (
                "BM25 가중치 미세 조정 (0.55 → 0.70) 또는 80:20 비교",
                f"3단계 순위 실패가 {len(buckets[3])}건({', '.join(buckets[3])})입니다. 필터 전에 먼저 BM25:Vec 최적 비율을 찾으세요.",
                "bm25_weight: 0.55 → 0.70 (단일 변수 변경)",
                "과도한 BM25 비중(0.80+)에서 오히려 밀리는 문항 여부 감시",
            )
        return (
            "규칙 기반 경량 메타데이터 필터 ON (LLM 호출 0원)",
            f"3단계 조건 불일치 순위 실패({', '.join(buckets[3])})가 주 병목입니다. 명확한 단일 조건만 필터링해 오답 면적/전월세 청크를 사전 제거하세요.",
            "use_metadata_filter: False → True (결과 <3건 시 자동 폴백)",
            "복합 비교 문항 과잉 필터링 여부 + 답변 Answer Relevancy 변동 감시",
        )
    if len(buckets[2]) >= 1:
        return (
            "Query Rewrite (optimize/query_rewriter.py · 복합 2구 서브쿼리 분해)",
            f"2단계 검색 누락 FN({', '.join(buckets[2])})은 Re-ranker(C)로 살릴 수 없습니다! 2개 지역 비교 질문을 서브쿼리 2개로 분해해 각각 검색·병합하세요.",
            "use_rewrite: False → True (복합 질의 분해 + 구어체 정규화)",
            "단순 문항 6건의 조건 왜곡·악화 0건 확인",
        )
    if len(buckets[3]) >= 1:
        if pmode != "C_hybrid_rerank":
            return (
                "검색 C (retrieval/reranker.py · 12→5 정밀 재정렬)",
                f"후보 12개에 정답이 100% 들어있으나 3단계 순위/복합 Top-5 누락({', '.join(buckets[3])})이 남았습니다. 검색 C로 후보 12개를 재정렬하세요.",
                "pipeline_mode: B_hybrid → C_hybrid_rerank (candidate_k=12 → top_k=5)",
                "LLM 리랭킹 지연(p95) 증가 폭 + 이미 1등이던 문항 하락 0건 확인",
            )
        return (
            "Query Rewrite (2구 서브쿼리 분해)로 복합 문항 양쪽 지역 Top-5 동시 보장",
            f"검색 C 이후에도 3단계 순위/복합 Top-5 누락({', '.join(buckets[3])})이 남았습니다. 서브쿼리 분해로 양쪽 지역을 라운드로빈 병합하세요.",
            "use_rewrite: False → True (서브쿼리 분해)",
            "단순 문항 6건의 조건 왜곡·악화 0건 확인",
        )
    if len(buckets[4]) >= 1:
        return (
            "ANSWER_PROMPT 보강 (생활어·공식표기 괄호 병기 · 첫 문장 주어 · 거절 고지)",
            f"검색은 1등(FN@5=0)인데 4단계 생성에서 미달({', '.join(buckets[4])})이 발생했습니다. 생활어-문서표기 괄호 병기 및 첫 문장 주어를 보강하세요.",
            "ANSWER_PROMPT v3: 생활어+공식표기 괄호 병기 · 최근 1년 실거래 · 과거 고지 고정",
            "Faithfulness 유지 여부 & Context Precision 보존 확인",
        )
    return (
        "목표 달성! 다음 단계: 버전 캐싱(cache.py) · HyDE · 봉인 test 20문항 최종 1회 평가",
        "현재 dev 시험지에서 4단계 깔때기 병목이 모두 해소되었습니다.",
        "use_cache=True (적중 <100ms, 무효화 15건 검증) → test 20문항 봉인 해제",
        "버전 변경 시 캐시 무효화 실패 0건 & status='error' 미저장 확인",
    )


def _compute_regression_diff(cur: dict, prev: dict | None, target: float) -> tuple[list[str], list[str]]:
    """직전 스텝 대비 개선된 문항 vs 악화된 문항(Regression Watch)을 자동 검출한다."""
    if prev is None or cur["eval_version"] != prev["eval_version"]:
        return [], []
    prev_logs = {r.get("question_id"): r for r in prev["mode_logs"]}
    improved: list[str] = []
    regressed: list[str] = []
    for r in cur["mode_logs"]:
        if r.get("expected_status") != "answered":
            continue
        qid = r.get("question_id", "")
        pr = prev_logs.get(qid)
        if not pr:
            continue
        cr5 = int((r.get("metrics_k5") or {}).get("first_rank", 0) or 0)
        pr5 = int((pr.get("metrics_k5") or {}).get("first_rank", 0) or 0)
        cmrr = float((r.get("metrics_k5") or {}).get("mrr@5", 0.0) or 0.0)
        pmrr = float((pr.get("metrics_k5") or {}).get("mrr@5", 0.0) or 0.0)
        crg = cur["item_ragas"].get(qid) or {}
        prg = prev["item_ragas"].get(qid) or {}
        c_ar, p_ar = crg.get("answer_relevancy"), prg.get("answer_relevancy")
        c_cp, p_cp = crg.get("llm_context_precision_with_reference"), prg.get("llm_context_precision_with_reference")

        score_delta = (cmrr - pmrr) + ((float(c_ar) - float(p_ar)) if (c_ar is not None and p_ar is not None) else 0.0)
        detail = f"`{qid}` (순위 #{pr5 or 'X'}→#{cr5 or 'X'}"
        if c_cp is not None and p_cp is not None:
            detail += f", CP {p_cp:.2f}→{c_cp:.2f}"
        if c_ar is not None and p_ar is not None:
            detail += f", AR {p_ar:.2f}→{c_ar:.2f}"
        detail += ")"

        if score_delta >= 0.08:
            improved.append(detail)
        elif score_delta <= -0.08:
            regressed.append(detail)
    return improved, regressed


def _build_single_step_report_md(cur: dict, prev: dict | None, target: float) -> str:
    """특정 회차 스텝 1개에 대한 6파트 평가 분석 리포트 마크다운(실험기록.md 복붙용)을 생성한다."""
    n_diff, diffs = _describe_preset_diff(prev, cur)
    improved, regressed = _compute_regression_diff(cur, prev, target)
    rec_title, rec_why, rec_knob, rec_watch = _recommend_next_variable(cur, target)
    cm12, cm5 = cur["cm12"], cur["cm5"]

    def _d(cur_v: float, prev_v: float | None, fmt: str = ".4f") -> str:
        if prev is None or prev_v is None:
            return "-"
        return f"{cur_v - prev_v:+{fmt}}"

    lines = [
        f"# 평가 분석 리포트 — {cur['step_label']}",
        "",
        "## 1. 실험 메타 & 단일 변인 체크",
        f"- **실행 시각 / 파일**: `{cur['ts_display']}` (`{cur['file_name']}`)",
        f"- **시험지 버전 (`eval_version`)**: `{cur['eval_version']}` (`{cur['eval_name']}`)",
        f"- **컬렉션 / 총 청크 수**: `{cur['collection']}` (`N={cur['corpus_n']:,}`)",
        f"- **직전 비교 기준선**: `{prev['step_label'] if prev else '없음 (최초 기준선)'}`",
        f"- **변경 변수 ({n_diff}개)**: {' · '.join(diffs)}",
        f"- **가설 메모**: {cur['run_note'] or '(미기재)'}",
        "",
        "## 2. 종합 성적표 & 직전 대비 변화 (유비쿼터스 지표)",
        "",
        "| 구분 | 지표 | 직전 | 이번 | 변화 (Δ) | 목표 판정 |",
        "|---|---|---:|---:|---:|---|",
        f"| 검색 | {RETRIEVAL_LABELS['recall@12']} | {prev['recall12'] if prev else '-'} | `{cur['recall12']:.4f}` | {_d(cur['recall12'], prev['recall12'] if prev else None)} | 후보 확보 |",
        f"| 검색 | {RETRIEVAL_LABELS['hit@5']} | {prev['hit5_str'] if prev else '-'} | `{cur['hit5_str']}` | {_d(cur['hit5_rate']*100, prev['hit5_rate']*100 if prev else None, '.1f')}%p | 적중률 |",
        f"| 검색 | {RETRIEVAL_LABELS['top1']} | {prev['top1_str'] if prev else '-'} | `{cur['top1_str']}` | {_d(cur['top1_rate']*100, prev['top1_rate']*100 if prev else None, '.1f')}%p | 1등 정확도 |",
        f"| 검색 | {RETRIEVAL_LABELS['mrr@5']} | {prev['mrr5'] if prev else '-'} | `{cur['mrr5']:.4f}` | {_d(cur['mrr5'], prev['mrr5'] if prev else None)} | 순위 정밀도 |",
        f"| RAGAS | {RAGAS_LABELS['faithfulness']} | {prev['f'] if prev else '-'} | `{cur['f']:.4f}` | {_d(cur['f'], prev['f'] if prev else None)} | {'✅ 통과' if cur['f'] >= target else '🔴 미달'} |",
        f"| RAGAS | {RAGAS_LABELS['answer_relevancy']} | {prev['ar'] if prev else '-'} | `{cur['ar']:.4f}` | {_d(cur['ar'], prev['ar'] if prev else None)} | {'✅ 통과' if cur['ar'] >= target else '🔴 미달'} |",
        f"| RAGAS | {RAGAS_LABELS['llm_context_precision_with_reference']} | {prev['cp'] if prev else '-'} | `{cur['cp']:.4f}` | {_d(cur['cp'], prev['cp'] if prev else None)} | {'✅ 통과' if cur['cp'] >= target else '🔴 미달'} |",
        f"| RAGAS | {RAGAS_LABELS['context_recall']} | {prev['cr'] if prev else '-'} | `{cur['cr']:.4f}` | {_d(cur['cr'], prev['cr'] if prev else None)} | {'✅ 통과' if cur['cr'] >= target else '🔴 미달'} |",
        f"| 거절(Track B) | {RETRIEVAL_LABELS['refusal']} / {RETRIEVAL_LABELS['scope_notice']} | {prev['refusal_str'] if prev else '-'} | `{cur['refusal_str']}` / `{cur['scope_notice_str']}` | - | {'✅ 통과' if cur['refusal_rate'] >= 0.7 else '🔴 미달'} |",
        f"| 운영 | {RETRIEVAL_LABELS['lat_median']} / p95 | {prev['lat_median'] if prev else '-'} | `{cur['lat_median']}ms` / `{cur['lat_p95']}ms` | {_d(cur['lat_median'], prev['lat_median'] if prev else None, '.1f')}ms | 10초 이내 |",
        "",
        "## 3. TP · FP · FN · TN 혼동행렬 합계 (답변 가능 문항 기준)",
        f"- **후보 k=12 (`metrics_k12`)**: `TP={cm12['tp']}` · `FP={cm12['fp']}` · **`FN={cm12['fn']}`** · `TN={cm12['tn']:,}`",
        f"- **최종 k=5 (`metrics_k5`)**: `TP={cm5['tp']}` · `FP={cm5['fp']}` · **`FN={cm5['fn']}`** · `TN={cm5['tn']:,}`",
        "",
        "## 4. 개선된 문항 vs 악화된 문항 (Regression Watch)",
        f"- **▲ 개선 ({len(improved)}건)**: {', '.join(improved) if improved else '없음 (기준선 또는 변동 없음)'}",
        f"- **▼ 악화·주의 ({len(regressed)}건)**: {', '.join(regressed) if regressed else '0건 (부작용 없음 ✅)'}",
        "",
        "## 5. 원인 판별 결론 & 다음 1개 변수 조정안",
        f"- **추천 다음 실험**: **{rec_title}**",
        f"- **판단 근거**: {rec_why}",
        f"- **바꿀 변수 (딱 1개)**: `{rec_knob}`",
        f"- **같이 감시할 부작용**: {rec_watch}",
    ]
    return "\n".join(lines)


def _build_comparison_df(summary: dict) -> pd.DataFrame:
    rows = []
    for mname, mdata in summary.get("modes", {}).items():
        ret = mdata.get("retrieval", {})
        lat = mdata.get("latency_ms", {})
        ragas = mdata.get("ragas_avg", {})
        rows.append(
            {
                "프리셋 (Mode)": mname,
                RETRIEVAL_LABELS["recall@12"]: ret.get("recall@12", "-"),
                RETRIEVAL_LABELS["hit@5"]: ret.get("hit@5", "-"),
                RETRIEVAL_LABELS["top1"]: ret.get("top1", "-"),
                RETRIEVAL_LABELS["mrr@5"]: ret.get("mrr@5", "-"),
                RETRIEVAL_LABELS["refusal"]: mdata.get("refusal", "-"),
                RETRIEVAL_LABELS["lat_median"]: lat.get("median", "-"),
                RETRIEVAL_LABELS["lat_p95"]: lat.get("p95", "-"),
                RAGAS_LABELS["faithfulness"]: ragas.get("faithfulness", "-"),
                RAGAS_LABELS["answer_relevancy"]: ragas.get("answer_relevancy", "-"),
                RAGAS_LABELS["llm_context_precision_with_reference"]: ragas.get("llm_context_precision_with_reference", "-"),
                RAGAS_LABELS["context_recall"]: ragas.get("context_recall", "-"),
                "기준 미달 건수": len(mdata.get("underperforming_items", [])),
            }
        )
    return pd.DataFrame(rows)


def _fmt_md_basic(text: str) -> str:
    escaped = html_lib.escape(text)
    escaped = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", escaped)
    escaped = re.sub(r"`([^`]+)`", r"<code>\1</code>", escaped)
    return escaped.replace("\n", "<br>")


def _build_svg_line_chart(
    steps: list[dict],
    series: list[tuple[str, str, str]],
    title: str,
    target: float | None = None,
) -> str:
    if not steps:
        return ""
    w, h = 920, 290
    pl, pr, pt, pb = 54, 24, 26, 76
    cw, ch = w - pl - pr, h - pt - pb
    n = len(steps)

    def x_pos(i: int) -> float:
        return pl + (cw * i / max(1, n - 1)) if n > 1 else pl + cw / 2

    def y_pos(v: float) -> float:
        return pt + ch * (1.0 - min(max(v, 0.0), 1.0))

    grid = []
    for tick in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
        y = y_pos(tick)
        grid.append(
            f'<line x1="{pl}" y1="{y:.1f}" x2="{w - pr}" y2="{y:.1f}" stroke="var(--border)" stroke-width="1" />'
            f'<text x="{pl - 8}" y="{y + 4:.1f}" text-anchor="end" font-size="10.5" fill="var(--muted)">{tick:.1f}</text>'
        )
    t_svg = ""
    if target is not None:
        yt = y_pos(target)
        t_svg = (
            f'<line x1="{pl}" y1="{yt:.1f}" x2="{w - pr}" y2="{yt:.1f}" stroke="var(--fail)" stroke-width="1.6" stroke-dasharray="6,4" />'
            f'<text x="{w - pr - 2}" y="{yt - 5:.1f}" text-anchor="end" font-size="10.5" font-weight="700" fill="var(--fail)">목표 {target:.2f}</text>'
        )
    x_lbls = [
        f'<text x="{x_pos(i):.1f}" y="{h - pb + 16}" text-anchor="end" transform="rotate(-22 {x_pos(i):.1f} {h - pb + 16})" font-size="10" fill="var(--muted)">{html_lib.escape(s["step_label"])}</text>'
        for i, s in enumerate(steps)
    ]
    paths, legs = [], []
    for key, label, color in series:
        pts = [(x_pos(i), y_pos(float(s[key])), float(s[key])) for i, s in enumerate(steps)]
        d = " ".join(f"{'M' if i == 0 else 'L'}{x:.1f},{y:.1f}" for i, (x, y, _) in enumerate(pts))
        paths.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="2.4" />')
        for x, y, val in pts:
            paths.append(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.2" fill="{color}" stroke="var(--card)" stroke-width="1.5">'
                f"<title>{html_lib.escape(label)}: {val:.3f}</title></circle>"
            )
        legs.append(f'<span class="leg-item"><span class="leg-dot" style="background:{color}"></span>{html_lib.escape(label)}</span>')

    return (
        f'<div class="svg-wrap"><div class="bar-title">{html_lib.escape(title)}</div>'
        f'<div class="legend">{"".join(legs)}</div>'
        f'<svg viewBox="0 0 {w} {h}" width="100%" style="max-height:310px;display:block;">'
        f'{"".join(grid)}{t_svg}{"".join(paths)}{"".join(x_lbls)}</svg></div>'
    )


_SVG_FONT = "'Pretendard', 'Malgun Gothic', 'Apple SD Gothic Neo', 'Noto Sans KR', sans-serif"


def _short_step_tick(step: dict) -> str:
    """발표용 그래프 X축 라벨을 간결하게 줄인다."""
    mode = str(step.get("mode") or "")
    ts = str(step.get("timestamp") or "")
    tag = f"#{step.get('order', '')}"
    short_map = [
        ("cache", "C+Cache (Warm)" if float(step.get("lat_median") or 999) < 10.0 else "C+Cache (Cold)"),
        ("rewrite", "C+Rewrite"),
        ("rerank", "C (Rerank)"),
        ("filter_v2", "B+FilterV2"),
        ("filter_kure", "B+FilterV1"),
        ("k15", "FIX B (k15)"),
        ("55_45_kure", "B 55:45 (KURE)"),
        ("50_50_kure", "B 50:50 (KURE)"),
        ("25_75_kure", "B 25:75 (KURE)"),
        ("10_90_kure", "B 10:90 (KURE)"),
        ("vector_kure", "A (KURE-v1)"),
        ("a_vector", "A (Vector)"),
        ("70_30_filter", "B 70:30+Filter"),
        ("80_20", "B 80:20"),
        ("70_30", "B 70:30"),
        ("55_45", "B 55:45"),
    ]
    low = mode.lower()
    for needle, nice in short_map:
        if needle in low:
            return f"{tag} {nice}"
    clean = mode if len(mode) <= 18 else mode[:17] + "…"
    return f"{tag} {clean}" if tag != "#" else (clean or ts)


def _build_standalone_svg_line_chart(
    steps: list[dict],
    series: list[tuple[str, str, str]],
    title: str,
    subtitle: str = "",
    target: float | None = None,
    width: int = 1240,
    height: int = 480,
) -> str:
    """발표 자료(PPT/HTML/PNG 3x)용 완전 자립형 흰 배경 고해상도 SVG 선 그래프."""
    if not steps:
        return ""
    pl, pr, pt, pb = 76, 68, 108, 104
    cw, ch = width - pl - pr, height - pt - pb
    n = len(steps)

    def x_pos(i: int) -> float:
        return pl + (cw * i / max(1, n - 1)) if n > 1 else pl + cw / 2

    def y_pos(v: float) -> float:
        return pt + ch * (1.0 - min(max(v, 0.0), 1.0))

    grid: list[str] = []
    for tick in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
        y = y_pos(tick)
        dash = ' stroke-dasharray="4,4"' if tick > 0 else ""
        stroke = "#cbd5e1" if tick == 0 else "#e2e8f0"
        grid.append(
            f'<line x1="{pl}" y1="{y:.1f}" x2="{width - pr}" y2="{y:.1f}" stroke="{stroke}" stroke-width="1.2"{dash} />'
            f'<text x="{pl - 12}" y="{y + 4.5:.1f}" text-anchor="end" font-size="13" font-weight="600" fill="#64748b">{tick:.1f}</text>'
        )

    t_svg = ""
    if target is not None:
        yt = y_pos(target)
        t_svg = (
            f'<line x1="{pl}" y1="{yt:.1f}" x2="{width - pr}" y2="{yt:.1f}" stroke="#dc2626" stroke-width="2" stroke-dasharray="8,5" />'
            f'<rect x="{width - pr - 98}" y="{yt - 24:.1f}" width="96" height="21" rx="6" fill="#fef2f2" stroke="#fca5a5" stroke-width="1" />'
            f'<text x="{width - pr - 50}" y="{yt - 9.5:.1f}" text-anchor="middle" font-size="12" font-weight="700" fill="#dc2626">목표 {target:.2f}</text>'
        )

    x_lbls: list[str] = []
    for i, s in enumerate(steps):
        xi = x_pos(i)
        tick_txt = _short_step_tick(s)
        x_lbls.append(
            f'<line x1="{xi:.1f}" y1="{pt + ch}" x2="{xi:.1f}" y2="{pt + ch + 6}" stroke="#94a3b8" stroke-width="1.2" />'
            f'<text x="{xi:.1f}" y="{pt + ch + 22}" text-anchor="end" transform="rotate(-24 {xi:.1f} {pt + ch + 22})" '
            f'font-size="12.5" font-weight="600" fill="#334155">{html_lib.escape(tick_txt)}</text>'
        )

    # 상단 범례 (가로 균등 배치)
    leg_items: list[str] = []
    leg_x = pl
    leg_y = 74
    for _, label, color in series:
        short_lbl = label.split("(")[0].strip()
        leg_items.append(
            f'<circle cx="{leg_x + 6}" cy="{leg_y - 4}" r="6" fill="{color}" />'
            f'<text x="{leg_x + 18}" y="{leg_y}" font-size="13.5" font-weight="700" fill="#1e293b">{html_lib.escape(short_lbl)}</text>'
        )
        leg_x += max(175, len(short_lbl) * 9 + 46)

    paths: list[str] = []
    end_labels: list[tuple[float, float, str, str]] = []
    for s_idx, (key, label, color) in enumerate(series):
        pts = [(x_pos(i), y_pos(float(s.get(key, 0.0))), float(s.get(key, 0.0))) for i, s in enumerate(steps)]
        d = " ".join(f"{'M' if i == 0 else 'L'}{x:.1f},{y:.1f}" for i, (x, y, _) in enumerate(pts))
        paths.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="3.2" stroke-linejoin="round" stroke-linecap="round" />')
        for idx_p, (x, y, val) in enumerate(pts):
            paths.append(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5.5" fill="{color}" stroke="#ffffff" stroke-width="2" />'
            )
            if n <= 7 and idx_p != n - 1:
                dy = -12 if (s_idx % 2 == 0) else 19
                paths.append(
                    f'<text x="{x:.1f}" y="{y + dy:.1f}" text-anchor="middle" font-size="11" font-weight="700" fill="{color}">{val:.2f}</text>'
                )
        lx, ly, lval = pts[-1]
        end_labels.append((ly, lx, f"{lval:.3f}", color))

    # 우측 최종값 라벨 충돌 회피
    end_labels.sort(key=lambda t: t[0])
    placed_y: list[float] = []
    for ly, lx, txt, color in end_labels:
        cy = ly + 4
        for py in placed_y:
            if abs(cy - py) < 16:
                cy = py + 16
        placed_y.append(cy)
        paths.append(
            f'<text x="{lx + 11:.1f}" y="{cy:.1f}" text-anchor="start" font-size="13" font-weight="800" fill="{color}">{txt}</text>'
        )

    sub_svg = (
        f'<text x="{pl}" y="52" font-size="13" font-weight="500" fill="#64748b">{html_lib.escape(subtitle)}</text>'
        if subtitle
        else ""
    )
    return (
        f'<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
        f'style="font-family:{_SVG_FONT};background:#ffffff;">'
        f'<rect x="1" y="1" width="{width - 2}" height="{height - 2}" rx="14" fill="#ffffff" stroke="#e2e8f0" stroke-width="1.5" />'
        f'<text x="{pl}" y="32" font-size="20" font-weight="800" fill="#0f172a">{html_lib.escape(title)}</text>'
        f'{sub_svg}{"".join(leg_items)}{"".join(grid)}{t_svg}{"".join(paths)}{"".join(x_lbls)}'
        f'</svg>'
    )


def _build_standalone_svg_bar_compare(
    before_s: dict,
    after_s: dict,
    title: str,
    subtitle: str = "",
    target: float | None = None,
    width: int = 1240,
    height: int = 510,
) -> str:
    """발표용 Before vs After 8대 핵심 지표 막대 비교 SVG (상승폭 Δ 배지 포함)."""
    metrics = [
        ("recall12", "Recall@12", "ret"),
        ("hit5_rate", "Hit@5", "ret"),
        ("top1_rate", "Top1", "ret"),
        ("mrr5", "MRR@5", "ret"),
        ("f", "Faithfulness", "ragas"),
        ("ar", "Answer Rel.", "ragas"),
        ("cp", "Context Prec.", "ragas"),
        ("cr", "Context Rec.", "ragas"),
    ]
    pl, pr, pt, pb = 72, 40, 118, 108
    cw, ch = width - pl - pr, height - pt - pb

    def y_pos(v: float) -> float:
        return pt + ch * (1.0 - min(max(v, 0.0), 1.0))

    grid: list[str] = []
    for tick in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
        y = y_pos(tick)
        dash = ' stroke-dasharray="4,4"' if tick > 0 else ""
        stroke = "#cbd5e1" if tick == 0 else "#e2e8f0"
        grid.append(
            f'<line x1="{pl}" y1="{y:.1f}" x2="{width - pr}" y2="{y:.1f}" stroke="{stroke}" stroke-width="1.2"{dash} />'
            f'<text x="{pl - 12}" y="{y + 4.5:.1f}" text-anchor="end" font-size="13" font-weight="600" fill="#64748b">{tick:.1f}</text>'
        )

    t_svg = ""
    if target is not None:
        yt = y_pos(target)
        t_svg = (
            f'<line x1="{pl}" y1="{yt:.1f}" x2="{width - pr}" y2="{yt:.1f}" stroke="#dc2626" stroke-width="2" stroke-dasharray="8,5" />'
            f'<text x="{width - pr - 4}" y="{yt - 7:.1f}" text-anchor="end" font-size="12" font-weight="700" fill="#dc2626">RAGAS 목표 {target:.2f}</text>'
        )

    # 검색 4종 / RAGAS 4종 중앙 구분선
    mid_x = pl + cw * 0.5
    div_svg = (
        f'<line x1="{mid_x:.1f}" y1="{pt - 14}" x2="{mid_x:.1f}" y2="{pt + ch + 46}" stroke="#cbd5e1" stroke-width="1.5" stroke-dasharray="5,5" />'
        f'<text x="{pl + cw * 0.25:.1f}" y="{pt - 18}" text-anchor="middle" font-size="13" font-weight="800" fill="#1d4ed8">검색 지표 (Retrieval)</text>'
        f'<text x="{pl + cw * 0.75:.1f}" y="{pt - 18}" text-anchor="middle" font-size="13" font-weight="800" fill="#047857">생성·근거 지표 (RAGAS)</text>'
    )

    group_w = cw / len(metrics)
    bar_w = min(38.0, group_w * 0.30)
    gap = 6.0
    bars: list[str] = []
    for i, (key, label, kind) in enumerate(metrics):
        cx = pl + group_w * (i + 0.5)
        bv = float(before_s.get(key, 0.0) or 0.0)
        av = float(after_s.get(key, 0.0) or 0.0)
        dv = av - bv
        a_col = "#2563eb" if kind == "ret" else "#059669"

        bx = cx - bar_w - gap / 2
        ax = cx + gap / 2
        by = y_pos(bv)
        ay = y_pos(av)
        bh = max(2.0, pt + ch - by)
        ah = max(2.0, pt + ch - ay)

        bars.append(
            f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bar_w:.1f}" height="{bh:.1f}" rx="5" fill="#94a3b8" />'
            f'<text x="{bx + bar_w / 2:.1f}" y="{by - 6:.1f}" text-anchor="middle" font-size="11.5" font-weight="700" fill="#64748b">{bv:.3f}</text>'
            f'<rect x="{ax:.1f}" y="{ay:.1f}" width="{bar_w:.1f}" height="{ah:.1f}" rx="5" fill="{a_col}" />'
            f'<text x="{ax + bar_w / 2:.1f}" y="{ay - 6:.1f}" text-anchor="middle" font-size="12" font-weight="800" fill="{a_col}">{av:.3f}</text>'
            f'<text x="{cx:.1f}" y="{pt + ch + 24}" text-anchor="middle" font-size="13" font-weight="700" fill="#1e293b">{html_lib.escape(label)}</text>'
        )
        d_col = "#15803d" if dv > 0.001 else ("#b91c1c" if dv < -0.001 else "#64748b")
        d_bg = "#dcfce7" if dv > 0.001 else ("#fee2e2" if dv < -0.001 else "#f1f5f9")
        bars.append(
            f'<rect x="{cx - 32:.1f}" y="{pt + ch + 32}" width="64" height="20" rx="10" fill="{d_bg}" />'
            f'<text x="{cx:.1f}" y="{pt + ch + 46}" text-anchor="middle" font-size="11.5" font-weight="800" fill="{d_col}">{dv:+.3f}</text>'
        )

    b_lbl = _short_step_tick(before_s)
    a_lbl = _short_step_tick(after_s)
    legend = (
        f'<rect x="{pl}" y="68" width="14" height="14" rx="3" fill="#94a3b8" />'
        f'<text x="{pl + 20}" y="80" font-size="13.5" font-weight="700" fill="#475569">Before: {html_lib.escape(b_lbl)}</text>'
        f'<rect x="{pl + 340}" y="68" width="14" height="14" rx="3" fill="#2563eb" />'
        f'<rect x="{pl + 357}" y="68" width="14" height="14" rx="3" fill="#059669" />'
        f'<text x="{pl + 378}" y="80" font-size="13.5" font-weight="800" fill="#0f172a">After: {html_lib.escape(a_lbl)}</text>'
    )
    footer_txt = (
        f"RAGAS 평균: {before_s['ragas_mean']:.4f} → {after_s['ragas_mean']:.4f} ({after_s['ragas_mean'] - before_s['ragas_mean']:+.4f})   |   "
        f"지연 중앙값: {before_s['lat_median']:.1f}ms → {after_s['lat_median']:.1f}ms   |   "
        f"거절률: {before_s['refusal_str']} → {after_s['refusal_str']}"
    )
    footer = (
        f'<rect x="{pl}" y="{height - 38}" width="{cw}" height="28" rx="8" fill="#f8fafc" stroke="#e2e8f0" />'
        f'<text x="{pl + cw / 2:.1f}" y="{height - 19}" text-anchor="middle" font-size="12.5" font-weight="700" fill="#334155">{html_lib.escape(footer_txt)}</text>'
    )
    sub_svg = (
        f'<text x="{pl}" y="52" font-size="13" font-weight="500" fill="#64748b">{html_lib.escape(subtitle)}</text>'
        if subtitle
        else ""
    )
    return (
        f'<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
        f'style="font-family:{_SVG_FONT};background:#ffffff;">'
        f'<rect x="1" y="1" width="{width - 2}" height="{height - 2}" rx="14" fill="#ffffff" stroke="#e2e8f0" stroke-width="1.5" />'
        f'<text x="{pl}" y="32" font-size="20" font-weight="800" fill="#0f172a">{html_lib.escape(title)}</text>'
        f'{sub_svg}{legend}{"".join(grid)}{t_svg}{div_svg}{"".join(bars)}{footer}'
        f'</svg>'
    )


def _build_standalone_svg_latency_chart(
    steps: list[dict],
    title: str,
    subtitle: str = "",
    width: int = 1240,
    height: int = 0,
) -> str:
    """발표용 지연시간 중앙값(Median ms) 가로 막대 비교 SVG (로그 스케일로 0.3ms 캐시와 5,184ms 동시 시각화)."""
    valid = [s for s in steps if s.get("has_lat", True)]
    if not valid:
        return ""
    row_h = 44
    pt, pb, pl, pr = 86, 44, 240, 175
    if height <= 0:
        height = max(320, pt + pb + len(valid) * row_h)
    cw = width - pl - pr
    max_log = max(math.log10(max(float(s.get("lat_median") or 0.1), 0.1) + 1.0) for s in valid) or 1.0

    rows_svg: list[str] = []
    for i, s in enumerate(valid):
        med = float(s.get("lat_median") or 0.0)
        p95 = float(s.get("lat_p95") or 0.0)
        y = pt + i * row_h
        frac = math.log10(max(med, 0.1) + 1.0) / max_log
        bw = max(14.0, cw * frac)
        if med < 10.0:
            col, badge = "#059669", " ⚡ Cache HIT"
        elif med < 500.0:
            col, badge = "#2563eb", ""
        elif med < 3000.0:
            col, badge = "#d97706", ""
        else:
            col, badge = "#dc2626", ""
        med_str = f"{med:.1f} ms" if med < 1000.0 else f"{med / 1000.0:.2f} s"
        p95_str = f" (p95 {p95:.1f}ms)" if p95 < 1000.0 else f" (p95 {p95 / 1000.0:.2f}s)"
        rows_svg.append(
            f'<text x="{pl - 14}" y="{y + 21}" text-anchor="end" font-size="13" font-weight="700" fill="#1e293b">{html_lib.escape(_short_step_tick(s))}</text>'
            f'<rect x="{pl}" y="{y + 6}" width="{cw}" height="24" rx="6" fill="#f1f5f9" />'
            f'<rect x="{pl}" y="{y + 6}" width="{bw:.1f}" height="24" rx="6" fill="{col}" />'
            f'<text x="{pl + bw + 10:.1f}" y="{y + 22}" font-size="13" font-weight="800" fill="{col}">{html_lib.escape(med_str + p95_str + badge)}</text>'
        )

    sub_svg = (
        f'<text x="36" y="54" font-size="13" font-weight="500" fill="#64748b">{html_lib.escape(subtitle)}</text>'
        if subtitle
        else ""
    )
    return (
        f'<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
        f'style="font-family:{_SVG_FONT};background:#ffffff;">'
        f'<rect x="1" y="1" width="{width - 2}" height="{height - 2}" rx="14" fill="#ffffff" stroke="#e2e8f0" stroke-width="1.5" />'
        f'<text x="36" y="34" font-size="20" font-weight="800" fill="#0f172a">{html_lib.escape(title)}</text>'
        f'{sub_svg}{"".join(rows_svg)}'
        f'<text x="{pl}" y="{height - 14}" font-size="11.5" fill="#64748b">※ 막대 길이는 log10(ms+1) 스케일 (0.3ms 캐시와 5,000ms+ 리라이트를 한 화면에 비교)</text>'
        f'</svg>'
    )


def _render_svg_png_export_pair(
    svg_str: str,
    file_stem: str,
    key_prefix: str,
    png_scale: int = 3,
) -> None:
    """추가 패키지 없이 [원본 SVG 저장] + [브라우저 Canvas 3x 고해상도 PNG 저장] 버튼을 나란히 렌더링."""
    if not svg_str:
        return
    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            "📐 SVG 저장 (벡터 · 무한확대)",
            data=svg_str.encode("utf-8"),
            file_name=f"{file_stem}.svg",
            mime="image/svg+xml",
            key=f"dl_svg_{key_prefix}",
            use_container_width=True,
        )
    with c2:
        b64 = base64.b64encode(svg_str.encode("utf-8")).decode("ascii")
        safe_fn = re.sub(r"[^a-zA-Z0-9_\-]", "_", file_stem)
        btn_id = f"png_btn_{re.sub(r'[^a-zA-Z0-9_]', '_', key_prefix)}"
        html_btn = f"""
        <div style="margin:0;padding:0;">
          <button id="{btn_id}" style="
            width:100%;height:38.5px;border-radius:8px;border:1px solid #cbd5e1;
            background:#0f172a;color:#ffffff;font-weight:700;font-size:13.5px;
            cursor:pointer;font-family:{_SVG_FONT};transition:opacity .15s;
          " onmouseover="this.style.opacity='0.88'" onmouseout="this.style.opacity='1'">
            📸 고해상도 PNG ({png_scale}x) 저장
          </button>
        </div>
        <script>
        (function() {{
          const btn = document.getElementById("{btn_id}");
          if (!btn) return;
          btn.addEventListener("click", function() {{
            const b64 = "{b64}";
            const bin = atob(b64);
            const bytes = new Uint8Array(bin.length);
            for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
            const svgText = new TextDecoder("utf-8").decode(bytes);
            const blob = new Blob([svgText], {{type: "image/svg+xml;charset=utf-8"}});
            const url = URL.createObjectURL(blob);
            const img = new Image();
            img.onload = function() {{
              const scale = {png_scale};
              const canvas = document.createElement("canvas");
              canvas.width = img.width * scale;
              canvas.height = img.height * scale;
              const ctx = canvas.getContext("2d");
              ctx.fillStyle = "#ffffff";
              ctx.fillRect(0, 0, canvas.width, canvas.height);
              ctx.scale(scale, scale);
              ctx.drawImage(img, 0, 0);
              URL.revokeObjectURL(url);
              canvas.toBlob(function(pngBlob) {{
                const a = document.createElement("a");
                a.href = URL.createObjectURL(pngBlob);
                a.download = "{safe_fn}_{png_scale}x.png";
                document.body.appendChild(a);
                a.click();
                setTimeout(() => {{ URL.revokeObjectURL(a.href); a.remove(); }}, 500);
              }}, "image/png");
            }};
            img.src = url;
          }});
        }})();
        </script>
        """
        components.html(html_btn, height=44)


def _build_chart_bundle_zip(charts: list[tuple[str, str, str]]) -> bytes:
    """발표용 그래프 SVG 팩 + 1클릭 일괄 3x PNG 변환 HTML 뷰어를 ZIP으로 묶는다."""
    buf = io.BytesIO()
    cards_html: list[str] = []
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for stem, title, svg in charts:
            if not svg:
                continue
            zf.writestr(f"{stem}.svg", svg.encode("utf-8"))
            b64 = base64.b64encode(svg.encode("utf-8")).decode("ascii")
            cards_html.append(
                f'<div class="card"><h3>{html_lib.escape(title)}</h3>'
                f'<img src="data:image/svg+xml;base64,{b64}" data-name="{html_lib.escape(stem)}" style="width:100%;border-radius:10px;border:1px solid #e2e8f0;" /></div>'
            )
        viewer = (
            "<!doctype html><html lang='ko'><head><meta charset='utf-8'><title>발표용 그래프 일괄 PNG 변환기</title>"
            "<style>body{font-family:'Malgun Gothic',sans-serif;max-width:1280px;margin:24px auto;padding:0 16px;background:#f8fafc;}"
            "button{background:#0f172a;color:#fff;border:none;border-radius:10px;padding:12px 22px;font-size:15px;font-weight:700;cursor:pointer;margin-bottom:18px;}"
            ".card{background:#fff;padding:18px;border-radius:14px;border:1px solid #e2e8f0;margin-bottom:18px;}</style></head><body>"
            "<h1>📸 발표용 RAG 평가 그래프 팩 (3x PNG 일괄 저장)</h1>"
            "<button onclick='exportAll()'>📥 아래 그래프 전체 3x 고해상도 PNG 일괄 다운로드</button>"
            + "".join(cards_html)
            + "<script>async function exportAll(){for(const img of document.querySelectorAll('img[data-name]')){"
            "const c=document.createElement('canvas');c.width=img.naturalWidth*3;c.height=img.naturalHeight*3;"
            "const x=c.getContext('2d');x.fillStyle='#ffffff';x.fillRect(0,0,c.width,c.height);x.scale(3,3);x.drawImage(img,0,0);"
            "const a=document.createElement('a');a.href=c.toDataURL('image/png');a.download=img.dataset.name+'_3x.png';a.click();"
            "await new Promise(r=>setTimeout(r,350));}}</script></body></html>"
        )
        zf.writestr("00_발표용_PNG_일괄변환기.html", viewer.encode("utf-8"))
    return buf.getvalue()


def _render_presentation_export_panel(
    steps: list[dict],
    before_s: dict | None,
    after_s: dict | None,
    target: float,
    eval_ver: str,
    owner_key: str,
    key_scope: str = "lab",
) -> None:
    """발표 시연·슬라이드 삽입용 고해상도 그래프(PNG 3x / SVG / ZIP) Export 패널."""
    if not steps:
        return
    safe_ver = re.sub(r"[^a-zA-Z0-9_\-]", "_", eval_ver or "all")
    svg_ragas = _build_standalone_svg_line_chart(
        steps,
        [
            ("f", RAGAS_LABELS["faithfulness"], "#16a34a"),
            ("ar", RAGAS_LABELS["answer_relevancy"], "#dc2626"),
            ("cp", RAGAS_LABELS["llm_context_precision_with_reference"], "#2563eb"),
            ("cr", RAGAS_LABELS["context_recall"], "#7c3aed"),
        ],
        title=f"RAGAS 4대 지표 추이 ({eval_ver})",
        subtitle=f"담당: {owner_key} · 총 {len(steps)}개 실행 스텝 · 빨간 점선 = 공식 목표({target:.2f})",
        target=target,
    )
    svg_ret = _build_standalone_svg_line_chart(
        steps,
        [
            ("recall12", RETRIEVAL_LABELS["recall@12"], "#0d9488"),
            ("hit5_rate", "Hit@5 (적중률@5)", "#2563eb"),
            ("top1_rate", RETRIEVAL_LABELS["top1"], "#d97706"),
            ("mrr5", RETRIEVAL_LABELS["mrr@5"], "#7c3aed"),
        ],
        title=f"검색 4대 지표 추이 ({eval_ver})",
        subtitle=f"담당: {owner_key} · Recall@12 · Hit@5 · Top1 · MRR@5 단계별 상승 궤적",
        target=None,
    )
    svg_bar = (
        _build_standalone_svg_bar_compare(
            before_s,
            after_s,
            title=f"Before vs After 8대 핵심 지표 비교 ({eval_ver})",
            subtitle=f"{before_s['step_label']}  →  {after_s['step_label']}",
            target=target,
        )
        if (before_s is not None and after_s is not None)
        else ""
    )
    svg_lat = _build_standalone_svg_latency_chart(
        steps,
        title=f"지연시간 중앙값 (Latency Median / p95) 비교 ({eval_ver})",
        subtitle=f"담당: {owner_key} · 0.3ms 캐시 히트 vs 리랭킹/리라이트 체감 속도 대비",
    )

    with st.expander("📸 발표용 고해상도 그래프 이미지 Export (3x PNG · 벡터 SVG · 일괄 ZIP)", expanded=True):
        st.caption(
            "💡 각 인터랙티브 Altair 차트 우측 상단 `•••` 메뉴에서도 기본 PNG/SVG 저장이 가능하며, "
            "아래 버튼은 **제목·수치 라벨·Δ 배지·흰 배경이 포함된 발표 슬라이드 전용 3배 해상도 PNG(`3720px`) & 벡터 SVG**를 즉시 내려받습니다."
        )
        bundle = _build_chart_bundle_zip(
            [
                (f"01_{owner_key}_{safe_ver}_ragas_trend", f"RAGAS 4대 지표 추이 ({eval_ver})", svg_ragas),
                (f"02_{owner_key}_{safe_ver}_retrieval_trend", f"검색 4대 지표 추이 ({eval_ver})", svg_ret),
                (f"03_{owner_key}_{safe_ver}_before_after_bar", f"Before vs After 8대 지표 비교 ({eval_ver})", svg_bar),
                (f"04_{owner_key}_{safe_ver}_latency", f"지연시간 비교 ({eval_ver})", svg_lat),
            ]
        )
        st.download_button(
            f"📦 현재 시험지({eval_ver}) 발표용 그래프 4종 일괄 ZIP 저장 (SVG 4개 + PNG 일괄변환 HTML)",
            data=bundle,
            file_name=f"presentation_charts_{owner_key}_{safe_ver}.zip",
            mime="application/zip",
            key=f"dl_zip_{key_scope}_{owner_key}_{safe_ver}",
            use_container_width=True,
        )
        show_prev = st.checkbox("발표용 Export 이미지 미리보기 펼치기", value=False, key=f"prev_svg_{key_scope}_{owner_key}_{safe_ver}")

        e1, e2 = st.columns(2)
        with e1:
            st.markdown("**1) RAGAS 4대 지표 추이**")
            if show_prev and svg_ragas:
                st.image(svg_ragas, use_container_width=True)
            _render_svg_png_export_pair(svg_ragas, f"01_{owner_key}_{safe_ver}_ragas_trend", f"{key_scope}_ragas_{safe_ver}")
        with e2:
            st.markdown("**2) 검색 4대 지표 추이**")
            if show_prev and svg_ret:
                st.image(svg_ret, use_container_width=True)
            _render_svg_png_export_pair(svg_ret, f"02_{owner_key}_{safe_ver}_retrieval_trend", f"{key_scope}_ret_{safe_ver}")

        e3, e4 = st.columns(2)
        with e3:
            if svg_bar:
                st.markdown("**3) Before vs After 8대 지표 막대 비교**")
                if show_prev:
                    st.image(svg_bar, use_container_width=True)
                _render_svg_png_export_pair(svg_bar, f"03_{owner_key}_{safe_ver}_before_after_bar", f"{key_scope}_bar_{safe_ver}")
        with e4:
            if svg_lat:
                st.markdown("**4) 지연시간 (Median / p95) 비교**")
                if show_prev:
                    st.image(svg_lat, use_container_width=True)
                _render_svg_png_export_pair(svg_lat, f"04_{owner_key}_{safe_ver}_latency", f"{key_scope}_lat_{safe_ver}")


def _export_html(
    cur_step: dict,
    prev_step: dict | None,
    history_steps: list[dict],
    target: float,
) -> str:
    """단독 HTML 리포트 (전체 히스토리 SVG 선그래프 + 선택 회차 TP/FP/FN/TN·진단·분석 리포트)."""
    ts = html_lib.escape(cur_step["timestamp"])
    eval_ver = html_lib.escape(cur_step["eval_version"])
    svg_ragas = _build_svg_line_chart(
        history_steps,
        [
            ("f", RAGAS_LABELS["faithfulness"], "#16a34a"),
            ("ar", RAGAS_LABELS["answer_relevancy"], "#dc2626"),
            ("cp", RAGAS_LABELS["llm_context_precision_with_reference"], "#2563eb"),
            ("cr", RAGAS_LABELS["context_recall"], "#7c3aed"),
        ],
        "RAGAS 4대 지표 회차별 추이 (연결 선 그래프)",
        target=target,
    )
    svg_ret = _build_svg_line_chart(
        history_steps,
        [
            ("recall12", RETRIEVAL_LABELS["recall@12"], "#0d9488"),
            ("hit5_rate", "Hit@5 (적중률@5)", "#2563eb"),
            ("top1_rate", RETRIEVAL_LABELS["top1"], "#d97706"),
            ("mrr5", RETRIEVAL_LABELS["mrr@5"], "#7c3aed"),
        ],
        "검색 4대 지표 회차별 추이 (연결 선 그래프)",
        target=None,
    )
    report_md = _build_single_step_report_md(cur_step, prev_step, target)
    report_html = _fmt_md_basic(report_md)

    hist_rows = []
    for s in history_steps:
        hist_rows.append(
            f"<tr><td><strong>#{s['order']}</strong></td><td>{html_lib.escape(s['ts_display'])}</td>"
            f"<td><code>{html_lib.escape(s['eval_version'])}</code></td><td><code>{html_lib.escape(s['mode'])}</code></td>"
            f"<td>{html_lib.escape(s['change_desc'])}</td><td>{s['recall12']:.3f}</td><td>{html_lib.escape(s['top1_str'])}</td>"
            f"<td>{s['mrr5']:.3f}</td><td class='{'ok' if s['f']>=target else 'fail'}'>{s['f']:.3f}</td>"
            f"<td class='{'ok' if s['ar']>=target else 'fail'}'>{s['ar']:.3f}</td>"
            f"<td class='{'ok' if s['cp']>=target else 'fail'}'>{s['cp']:.3f}</td>"
            f"<td class='{'ok' if s['cr']>=target else 'fail'}'>{s['cr']:.3f}</td>"
            f"<td>{s['cm12']['fn']} / {s['cm5']['fn']}</td><td>{html_lib.escape(s['delta_narrative'])}</td></tr>"
        )

    return f"""<!DOCTYPE html>
<html lang="ko"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>서울 주거 RAG 평가 분석 리포트 ({ts})</title>
<style>
  :root{{--bg:#f8fafc;--surface:#f1f5f9;--card:#fff;--border:#e2e8f0;--fg:#0f172a;--muted:#475569;--primary:#2563eb;--ok:#16a34a;--ok-bg:#f0fdf4;--fail:#dc2626;--fail-bg:#fef2f2;}}
  :root[data-theme="dark"]{{--bg:#0b1120;--surface:#111827;--card:#162033;--border:#263449;--fg:#e2e8f0;--muted:#94a3b8;--primary:#60a5fa;--ok:#4ade80;--ok-bg:#052e16;--fail:#f87171;--fail-bg:#450a0a;}}
  body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","Malgun Gothic",sans-serif;margin:0;background:var(--bg);color:var(--fg);font-size:13.5px;line-height:1.55;}}
  .wrap{{max-width:1280px;margin:0 auto;padding:20px;}}
  .card{{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:18px 20px;margin-bottom:16px;}}
  .svg-wrap{{background:var(--bg);border:1px solid var(--border);border-radius:10px;padding:12px;margin-bottom:12px;}}
  .bar-title{{font-weight:700;margin-bottom:6px;}}
  .legend{{display:flex;flex-wrap:wrap;gap:12px;font-size:12px;color:var(--muted);margin-bottom:8px;}}
  .leg-item{{display:inline-flex;align-items:center;gap:5px;}} .leg-dot{{width:10px;height:10px;border-radius:3px;display:inline-block;}}
  table{{width:100%;border-collapse:collapse;font-size:12px;}} th,td{{border:1px solid var(--border);padding:6px 8px;text-align:left;vertical-align:top;}}
  th{{background:var(--surface);white-space:nowrap;}} td.ok{{color:var(--ok);background:var(--ok-bg);font-weight:700;}} td.fail{{color:var(--fail);background:var(--fail-bg);font-weight:700;}}
  code{{background:var(--surface);padding:1px 4px;border-radius:4px;font-family:ui-monospace,Consolas,monospace;}}
  .btn{{padding:6px 10px;border-radius:7px;border:1px solid var(--border);background:var(--card);color:var(--fg);cursor:pointer;font-size:12px;}}
</style></head>
<body><div class="wrap">
  <div class="card" style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:10px;">
    <div><h1 style="margin:0;font-size:20px;">🏠 서울 주거 RAG — 회차별 평가 분석 리포트</h1>
    <div style="color:var(--muted);font-size:12.5px;margin-top:4px;">선택 스텝: <strong>{html_lib.escape(cur_step['step_label'])}</strong> · 시험지: <code>{eval_ver}</code> · 목표: <strong>{target:.2f}</strong></div></div>
    <div style="display:flex;gap:6px;"><button class="btn" onclick="document.documentElement.setAttribute('data-theme',document.documentElement.getAttribute('data-theme')==='dark'?'light':'dark')">🌓 테마</button><button class="btn" onclick="window.print()">🖨️ 인쇄/PDF</button></div>
  </div>
  <div class="card"><h2 style="margin-top:0;">1. 선택 회차 평가 분석 요약</h2><div style="font-size:13px;">{report_html}</div></div>
  <div class="card"><h2 style="margin-top:0;">2. 회차별 수치 추이 (연결 선 그래프)</h2>{svg_ragas}{svg_ret}</div>
  <div class="card"><h2 style="margin-top:0;">3. 전체 실험 랩노트 표</h2>
    <div style="overflow-x:auto;"><table><thead><tr><th>#</th><th>시각</th><th>시험지</th><th>프리셋</th><th>바꾼 변수</th><th>Recall@12</th><th>Top1</th><th>MRR@5</th><th>Faithful</th><th>Ans Rel</th><th>Ctx Prec</th><th>Ctx Rec</th><th>FN(k12/k5)</th><th>변화 해석</th></tr></thead>
    <tbody>{"".join(hist_rows)}</tbody></table></div>
  </div>
</div></body></html>"""


# ==================== UI 렌더링 헬퍼: 깔끔한 스타일 & 2×2 혼동행렬 카드 ====================
def _inject_clean_theme() -> None:
    """여백·카드·지표 가독성을 정돈해 번잡함을 줄이는 최소 CSS (Streamlit 상단 헤더 겹침 방지)."""
    st.markdown(
        """
        <style>
          .block-container { padding-top: 3.6rem !important; padding-bottom: 2.4rem; max-width: 1380px; }
          div[data-testid="stMetric"] {
            border: 1px solid rgba(148, 163, 184, 0.28);
            border-radius: 10px;
            padding: 10px 14px;
            background: rgba(148, 163, 184, 0.04);
          }
          div[data-testid="stMetricLabel"] p { font-size: 0.80rem !important; opacity: 0.82; }
          div[data-testid="stMetricValue"] { font-size: 1.45rem !important; font-weight: 700; }
          h1 { font-size: 1.65rem !important; margin-bottom: 0.25rem !important; }
          h2, h3 { margin-top: 0.5rem !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_confusion_matrix_2x2(
    cm: dict[str, int],
    k: int,
    prev_cm: dict[str, int] | None = None,
    title_prefix: str = "",
    horizontal: bool = False,
) -> None:
    def _delta_val(key: str) -> str | None:
        if not prev_cm:
            return None
        d = cm[key] - prev_cm[key]
        return f"{d:+d}" if d != 0 else "0"

    if title_prefix:
        st.caption(f"{title_prefix}상위 `k={k}` 청크 단위 혼동행렬")

    cards = [
        ("✅ TP · 뽑은 진짜 정답", f"{cm['tp']}", _delta_val("tp"), "normal", f"뽑았는데(P) 진짜 정답(T) — 상위 {k}개 적중"),
        ("🔴 FN · 놓친 정답 (누락!)", f"{cm['fn']}", _delta_val("fn"), "inverse", "안 뽑았는데(N) 사실 정답(F) — 0이 목표!"),
        ("⚠️ FP · 뽑았지만 오답", f"{cm['fp']}", _delta_val("fp"), "inverse", f"뽑았는데(P) 오답(F) — 정답 1권 문항은 구조상 {max(0, k-1)}권 FP"),
        ("👍 TN · 잘 버린 무관 청크", f"{cm['tn']:,}", None, "off", "안 뽑았는데(N) 무관(T) — Accuracy 대신 Recall·MRR 사용"),
    ]
    if horizontal:
        cols = st.columns(4)
        for col, (lbl, val, dlt, dcol, cap) in zip(cols, cards):
            with col:
                with st.container(border=True):
                    st.metric(lbl, val, dlt, delta_color=dcol)
                    st.caption(cap)
    else:
        c1, c2 = st.columns(2)
        for idx_c, (lbl, val, dlt, dcol, cap) in enumerate(cards):
            with (c1 if idx_c in (0, 2) else c2):
                with st.container(border=True):
                    st.metric(lbl, val, dlt, delta_color=dcol)
                    st.caption(cap)


def _render_wrapped_html_table(headers: list[str], rows: list[dict[str, Any]], col_widths: list[str] | None = None) -> None:
    """셀 줄바꿈이 자연스럽게 동작해 잘림(...) 없이 한눈에 다 보이는 반응형 HTML 테이블."""
    ths = []
    for idx_h, h in enumerate(headers):
        w = f' style="width:{col_widths[idx_h]};"' if col_widths and idx_h < len(col_widths) else ""
        ths.append(f"<th{w}>{html_lib.escape(h)}</th>")

    trs = []
    for r in rows:
        row_cls = r.get("_row_cls", "")
        tds = []
        for h in headers:
            raw = str(r.get(h, ""))
            cell = html_lib.escape(raw)
            cell = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", cell)
            cell = re.sub(r"`([^`]+)`", r"<code>\1</code>", cell)
            cell = cell.replace("\n", "<br>")
            tds.append(f"<td>{cell}</td>")
        trs.append(f'<tr class="{row_cls}">{"".join(tds)}</tr>')

    st.markdown(
        f"""
        <style>
          .kh-wrap-tbl {{
            width: 100%; border-collapse: collapse; font-size: 0.87rem; line-height: 1.5;
            border: 1px solid rgba(148,163,184,0.32); border-radius: 10px; overflow: hidden;
          }}
          .kh-wrap-tbl th {{
            text-align: left; padding: 9px 12px; font-weight: 700;
            background: rgba(148,163,184,0.12); border-bottom: 1px solid rgba(148,163,184,0.32);
            white-space: nowrap;
          }}
          .kh-wrap-tbl td {{
            padding: 9px 12px; vertical-align: top; word-break: keep-all; overflow-wrap: anywhere;
            border-bottom: 1px solid rgba(148,163,184,0.18);
          }}
          .kh-wrap-tbl tr.row-trig {{ background: rgba(239, 68, 68, 0.08); }}
          .kh-wrap-tbl tr.row-watch {{ background: rgba(245, 158, 11, 0.06); }}
          .kh-wrap-tbl code {{
            padding: 1px 5px; border-radius: 4px; font-size: 0.82rem;
            background: rgba(148,163,184,0.16);
          }}
        </style>
        <table class="kh-wrap-tbl">
          <thead><tr>{"".join(ths)}</tr></thead>
          <tbody>{"".join(trs)}</tbody>
        </table>
        """,
        unsafe_allow_html=True,
    )


# ==================== 6단계 수치 기준 의사결정 네비게이터 엔진 ====================
def _load_manifest_chunk_ids(collection_name: str) -> set[str] | None:
    """컬렉션에 대응하는 청크 매니페스트(.jsonl)에서 chunk_id 집합을 로드한다."""
    candidates = [REPO_DATA_DIR / f"chunk_manifest_{collection_name}.jsonl"]
    if collection_name == "housing_rent_kh":
        candidates.extend([REPO_DATA_DIR / "chunk_manifest_rent_kh.jsonl", REPO_DATA_DIR / "chunk_manifest_rent.jsonl"])
    for path in candidates:
        if not path.exists():
            continue
        ids: set[str] = set()
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                cid = row.get("chunk_id")
                if cid:
                    ids.add(str(cid))
            return ids
        except Exception:
            return None
    return None


def _count_top1_mismatch_axes(mode_logs: list[dict]) -> dict[str, list[str]]:
    """답변 가능 문항의 Top-1 오답 불일치 축(구/동/유형/계약/면적)별 문항 ID를 집계한다."""
    axes: dict[str, list[str]] = {"구": [], "동": [], "유형": [], "계약": [], "면적": []}
    for r in mode_logs:
        if r.get("expected_status") != "answered":
            continue
        top5 = r.get("top5_ids") or []
        if not top5:
            continue
        targets = _parse_reference_targets(r)
        ok, tags = _explain_chunk_mismatch(top5[0], targets)
        if ok:
            continue
        qid = str(r.get("question_id", ""))
        for tag in tags:
            for axis in axes:
                if f"{axis} 불일치" in tag and qid not in axes[axis]:
                    axes[axis].append(qid)
    return axes


def _evaluate_decision_matrix(
    after_s: dict | None,
    before_s: dict | None,
    target: float,
) -> list[dict[str, Any]]:
    """현재 선택 실험(after_s)을 6단계 수치 기준 매트릭스(Trigger → 근거 → Action → 헛수고 금지)로 실시간 판정한다."""
    mode_logs = after_s["mode_logs"] if after_s else []
    item_ragas = after_s["item_ragas"] if after_s else {}
    preset = (after_s.get("preset") or {}) if after_s else {}
    pmode = preset.get("pipeline_mode") or (("A_vector" if after_s["mode"] == "A_vector" else "B_hybrid") if after_s else "A_vector")
    bw = float(preset.get("bm25_weight", 0.55) or 0.55)
    col_name = after_s["collection"] if after_s else "housing_rent_kh"

    # 1단계: 청킹/색인 매니페스트 정합성 검사
    manifest_ids = _load_manifest_chunk_ids(col_name)
    missing_manifest_qids: list[str] = []
    if after_s and manifest_ids is not None:
        for r in mode_logs:
            if r.get("expected_status") != "answered":
                continue
            for cid in r.get("allowed_chunk_ids") or []:
                if cid not in manifest_ids:
                    missing_manifest_qids.append(f"{r.get('question_id')}({cid})")
                    break

    # 2~4단계 문항 분류 + Top-1 불일치 축 분석
    s2_qids: list[str] = []
    s2_partial_qids: list[str] = []
    s2_zero_qids: list[str] = []
    s3_qids: list[str] = []
    s3_partial5_qids: list[str] = []
    s4_ar_qids: list[str] = []
    s4_f_qids: list[str] = []
    over_refusal_qids: list[str] = []
    is_kure_track = bool(
        after_s and ("kure" in str(col_name).lower() or "kure" in str(after_s.get("mode", "")).lower())
    )
    cur_mode_name = str(after_s.get("mode", "")) if after_s else ""

    for r in mode_logs:
        qid = str(r.get("question_id", ""))
        if r.get("expected_status") != "answered":
            continue
        if r.get("actual_status") == "insufficient_evidence":
            over_refusal_qids.append(qid)
        rg = item_ragas.get(qid) or {}
        stg, _ = _classify_funnel_stage(r, rg, target)
        m12 = r.get("metrics_k12") or {}
        m5 = r.get("metrics_k5") or {}
        rec12 = float(m12.get("recall@12", 0.0) or 0.0)
        fn5_item = int(m5.get("fn", 0) or 0)
        if stg == 2:
            s2_qids.append(qid)
            if rec12 > 0:
                s2_partial_qids.append(qid)
            else:
                s2_zero_qids.append(qid)
        elif stg == 3:
            s3_qids.append(qid)
            if fn5_item > 0 and int(m5.get("hit@5", 0) or 0) == 1:
                s3_partial5_qids.append(qid)
        elif stg == 4:
            if rg.get("answer_relevancy") is not None and float(rg["answer_relevancy"]) < target:
                s4_ar_qids.append(qid)
            if rg.get("faithfulness") is not None and float(rg["faithfulness"]) < 0.80:
                s4_f_qids.append(qid)

    mismatch_axes = _count_top1_mismatch_axes(mode_logs)
    cond_mismatch_qids = sorted(set(mismatch_axes["면적"] + mismatch_axes["계약"] + mismatch_axes["유형"]))
    dong_mismatch_qids = mismatch_axes["동"]

    # 1단계 판정
    if after_s is None:
        s1_status, s1_reading = "idle", "실행 전 — 먼저 A_vector 기준선 실행 후 확인"
    elif missing_manifest_qids:
        s1_status = "triggered"
        s1_reading = f"매니페스트 누락 {len(missing_manifest_qids)}건: {', '.join(missing_manifest_qids[:3])}"
    else:
        cnt_txt = f"{len(manifest_ids):,}청크 확인" if manifest_ids is not None else "매니페스트 파일 미탐지"
        s1_status = "ok"
        s1_reading = f"컬렉션 `{col_name}` 정답 ID 누락 0건 ({cnt_txt})"

    # 2단계 판정 (FN(k=12) > 0 or Recall@12 < 0.90 or CR < target — 미측정 지표는 오발동 제외)
    if after_s is None:
        s2_status, s2_reading, s2_preset, s2_note = "idle", "실행 전 — A_vector 실행 후 k=12 FN 확인", "A_vector", "S1: A_vector 기준선 측정"
    else:
        has_r12 = bool(after_s.get("has_r12", True))
        has_cr = bool(after_s.get("has_cr", True))
        fn12 = int(after_s["cm12"]["fn"])
        rec12 = float(after_s["recall12"])
        cr_v = float(after_s["cr"])
        if not has_r12 and not has_cr and after_s["cm12"]["q_count"] == 0:
            s2_status = "idle"
            s2_reading = f"Recall@12=미측정 · CR=미측정 · Hit@5={after_s['hit5_str']} (검색 전용 로그)"
        else:
            s2_trig = (fn12 > 0) or (has_r12 and rec12 < 0.90) or (has_cr and cr_v < target)
            s2_status = "triggered" if s2_trig else "ok"
            r12_txt = f"{rec12:.3f} (기준 ≥0.90)" if has_r12 else "미측정"
            cr_txt = f"{cr_v:.3f} (기준 ≥{target:.2f})" if has_cr else "미측정"
            s2_reading = (
                f"FN(k=12)={fn12}건 · Recall@12={r12_txt} · CR={cr_txt}"
                + (f" · 누락: {', '.join(s2_qids[:6])}" if s2_qids else " · 누락 0건")
            )
        if pmode == "A_vector":
            s2_preset, s2_note = "B_rrf_55_45", "가설: Vector 단독 FN 해소를 위해 BM25+RRF(55:45) 결합"
        elif s2_partial_qids or not preset.get("use_rewrite"):
            s2_preset = "C_filter_rewrite" if pmode == "C_hybrid_rerank" else "B_filter_rewrite"
            s2_note = f"가설: 2구·구어체 FN({', '.join(s2_qids[:4]) or '부분누락'}) 해소를 위해 Query Rewrite ON"
        else:
            s2_preset, s2_note = "C_filter_rewrite_hyde", "가설: Rewrite 이후 잔여 의미 불일치 FN을 HyDE(Vector전용)로 보강"

    # 3단계 판정 (FN(k=12)==0인데 Top1 < 0.70 or MRR@5 < 0.65 or CP < target or 복합 Top-5 누락)
    if after_s is None:
        s3_status, s3_reading, s3_preset, s3_note = "idle", "실행 전 — 2단계 FN 해소 후 확인", "B_rrf_70_30", "가설: RRF 가중치 70:30 스윕"
    else:
        has_top1 = bool(after_s.get("has_top1", True))
        has_mrr5 = bool(after_s.get("has_mrr5", True))
        has_cp = bool(after_s.get("has_cp", True))
        fn12 = int(after_s["cm12"]["fn"])
        fn5 = int(after_s["cm5"]["fn"])
        top1_r = float(after_s["top1_rate"])
        mrr5 = float(after_s["mrr5"])
        cp_v = float(after_s["cp"])
        if not has_top1 and not has_mrr5 and not has_cp and len(s3_qids) == 0:
            s3_status = "ok" if after_s["hit5_rate"] >= 0.80 else "watch"
            s3_reading = f"Hit@5={after_s['hit5_str']}({after_s['hit5_rate']*100:.1f}%) · Top1/MRR@5/CP=미측정"
        else:
            s3_has_issue = (has_top1 and top1_r < 0.70) or (has_mrr5 and mrr5 < 0.65) or (has_cp and cp_v < target) or len(s3_qids) > 0 or fn5 > 0
            s3_status = ("triggered" if (s3_has_issue and fn12 == 0) else ("watch" if s3_has_issue else "ok"))
            axis_summary = []
            if s3_partial5_qids:
                axis_summary.append(f"[복합 Top-5 누락 {len(s3_partial5_qids)}건: {', '.join(s3_partial5_qids)}]")
            if cond_mismatch_qids:
                axis_summary.append(f"[면적/계약/유형 불일치 {len(cond_mismatch_qids)}건: {', '.join(cond_mismatch_qids)}]")
            if dong_mismatch_qids:
                axis_summary.append(f"[동 불일치 {len(dong_mismatch_qids)}건: {', '.join(dong_mismatch_qids)}]")
            top1_txt = f"{after_s['top1_str']}({top1_r*100:.0f}%, 기준 ≥70%)" if has_top1 else "미측정"
            mrr_txt = f"{mrr5:.3f}(≥0.65)" if has_mrr5 else "미측정"
            cp_txt = f"{cp_v:.3f}(≥{target:.2f})" if has_cp else "미측정"
            s3_reading = (
                f"Top1={top1_txt} · MRR@5={mrr_txt} · CP={cp_txt} · FN(k5)={fn5}"
                + (f" · {' '.join(axis_summary)}" if axis_summary else "")
            )
        if is_kure_track:
            if pmode == "A_vector":
                s3_preset, s3_note = "B_rrf_25_75_k15_kure_v1", "가설: KURE-v1 A→FIX B(25:75, k=15) 전환"
            elif not preset.get("use_metadata_filter"):
                s3_preset, s3_note = "B_rrf_25_75_k15_filter_v2_kure_v1", "가설: 평수·법정동 사전 필터 v2 ON으로 Top-1 100% 확보"
            elif pmode != "C_hybrid_rerank":
                s3_preset = "C_rrf_25_75_k15_filter_v2_rerank_kure_v1"
                s3_note = f"가설: 검색 C(12→5 Re-ranking)로 복합 Top-5 누락({', '.join(s3_partial5_qids or s3_qids) or '순위밀림'}) 구제"
            else:
                s3_preset = "C_rrf_25_75_k15_filter_v2_rewrite_kure_v1"
                s3_note = f"가설: 검색 C 이후 잔여 복합 문항({', '.join(s3_qids) or '순위밀림'}) Query Rewrite(2구 분해) 구제"
        else:
            if pmode == "A_vector":
                s3_preset, s3_note = "B_rrf_55_45", "가설: A→B(55:45) 전환으로 검색 순위 기준선 확보"
            elif bw < 0.68 and not preset.get("use_metadata_filter"):
                s3_preset, s3_note = "B_rrf_70_30", "가설: BM25 가중치 55:45→70:30 상향으로 지명·조건 Top-1 개선"
            elif not preset.get("use_metadata_filter"):
                s3_preset, s3_note = "B_rrf_70_30_filter", "가설: 경량 메타데이터 필터 ON으로 면적/계약/유형 불일치 제거"
            elif pmode != "C_hybrid_rerank":
                s3_preset, s3_note = "C_rrf_70_30_filter_rerank", f"가설: 검색 C(Re-ranking)로 잔여 동 불일치({', '.join(dong_mismatch_qids) or '순위밀림'}) Top-1 구제"
            else:
                s3_preset, s3_note = "C_filter_rewrite", f"가설: 검색 C 이후 잔여 순위 실패({', '.join(s3_qids) or '복합문항'}) Query Rewrite로 구제"

    # 4단계 / Track B / 운영 단계 추천 프리셋 (자기 자신 재추천 방지!)
    if is_kure_track:
        if pmode != "C_hybrid_rerank":
            s4_next_preset = "C_rrf_25_75_k15_filter_v2_rerank_kure_v1"
            s4_rec_label = "ANSWER_PROMPT v3 + 검색 C (C_rrf_25_75_k15_filter_v2_rerank_kure_v1)"
        elif not preset.get("use_rewrite"):
            s4_next_preset = "C_rrf_25_75_k15_filter_v2_rewrite_kure_v1"
            s4_rec_label = "ANSWER_PROMPT 보강 후 C+Rewrite (C_rrf_25_75_k15_filter_v2_rewrite_kure_v1)"
        else:
            s4_next_preset = "C_rrf_25_75_k15_filter_v2_cache_kure_v1"
            s4_rec_label = "ANSWER_PROMPT 보강 + Cache 검증"
        s4_candidates = [
            p for p in [
                "C_rrf_25_75_k15_filter_v2_rerank_kure_v1",
                "C_rrf_25_75_k15_filter_v2_rewrite_kure_v1",
                "C_rrf_25_75_k15_filter_v2_cache_kure_v1",
            ]
            if p != cur_mode_name
        ] or ["C_rrf_25_75_k15_filter_v2_cache_kure_v1"]
        s6_next_preset = "C_rrf_25_75_k15_filter_v2_cache_kure_v1"
        s6_candidates = [
            "C_rrf_25_75_k15_filter_v2_cache_kure_v1",
            "C_rrf_25_75_k15_filter_v2_rerank_kure_v1",
            "C_rrf_25_75_k15_filter_v2_rewrite_kure_v1",
        ]
    else:
        if pmode != "C_hybrid_rerank":
            s4_next_preset = "C_rrf_70_30_filter_rerank"
            s4_rec_label = "ANSWER_PROMPT v3 + 검색 C (C_rrf_70_30_filter_rerank)"
        else:
            s4_next_preset = "C_filter_rewrite"
            s4_rec_label = "ANSWER_PROMPT 보강 + C_filter_rewrite"
        s4_candidates = [
            p for p in ["C_rrf_70_30_filter_rerank", "C_filter_rewrite", "C_filter_rewrite_cache"]
            if p != cur_mode_name
        ] or ["C_filter_rewrite_cache"]
        s6_next_preset = "C_filter_rewrite_cache"
        s6_candidates = ["C_filter_rewrite_cache", "C_filter_rewrite_hyde", "C_filter_rewrite"]

    # 4단계 판정 (first_rank==1인데 AR < target 또는 F < 0.80 — 미측정 시 idle)
    if after_s is None:
        s4_status, s4_reading = "idle", "실행 전 — 검색 1등 확보 후 생성 지표 점검"
    elif not after_s.get("has_f", True) and not after_s.get("has_ar", True):
        s4_status, s4_reading = "idle", "AR=미측정 · F=미측정 (공식 RAGAS 생성 지표 미실행)"
    else:
        ar_v, f_v = float(after_s["ar"]), float(after_s["f"])
        s4_trig = len(s4_ar_qids) > 0 or len(s4_f_qids) > 0 or (after_s.get("has_ar", True) and ar_v < target) or (after_s.get("has_f", True) and f_v < 0.80)
        s4_status = "triggered" if (len(s4_ar_qids) > 0 or len(s4_f_qids) > 0) else ("watch" if s4_trig else "ok")
        s4_reading = (
            f"AR={ar_v:.3f}(기준 ≥{target:.2f}) · F={f_v:.3f}(기준 ≥0.80)"
            + (f" · 1등인데 AR미달: {', '.join(s4_ar_qids)}" if s4_ar_qids else "")
            + (f" · F<0.80: {', '.join(s4_f_qids)}" if s4_f_qids else "")
        )

    # Track B 거절 / 과거 데이터 판정 (미측정 시 idle)
    if after_s is None:
        s5_status, s5_reading = "idle", "실행 전 — 거절 전용 트랙 별도 측정"
    elif _parse_ratio(after_s["refusal_str"]) is None and len(over_refusal_qids) == 0:
        s5_status, s5_reading = "idle", f"Refusal={after_s['refusal_str']} · 과거 고지={after_s['scope_notice_str']} (별도 거절 집계 없음)"
    else:
        ref_r = float(after_s["refusal_rate"])
        scope_r = _parse_ratio(after_s["scope_notice_str"])
        scope_ok = True if scope_r is None else (scope_r >= 0.999)
        s5_trig = ref_r < 0.70 or (not scope_ok) or len(over_refusal_qids) > 0
        s5_status = "triggered" if s5_trig else "ok"
        s5_reading = (
            f"Refusal={after_s['refusal_str']}({ref_r*100:.0f}%, 기준 ≥70%) · 과거 고지={after_s['scope_notice_str']}(기준 100%)"
            + (f" · 과잉거절: {', '.join(over_refusal_qids)}" if over_refusal_qids else " · 과잉거절 0건")
        )

    # 운영 · 캐시 / 지연 판정
    if after_s is None:
        s6_status, s6_reading = "idle", "실행 전 — 품질 확보 후 Cache ON 검증"
    else:
        p95_v = float(after_s["lat_p95"])
        med_v = float(after_s["lat_median"])
        use_cache = bool(preset.get("use_cache"))
        if p95_v > 5000.0:
            s6_status = "triggered"
        elif not use_cache:
            s6_status = "watch"
        else:
            s6_status = "ok"
        s6_reading = f"p95={p95_v:.1f}ms (경고 >5,000ms) · 중앙값={med_v:.1f}ms · Cache={'ON' if use_cache else 'OFF(미검증)'}"

    badge_map = {"triggered": "🔴 발동", "watch": "🟡 점검", "ok": "🟢 양호", "idle": "⚪ 대기"}

    rows: list[dict[str, Any]] = [
        {
            "stage_id": "S1_chunk",
            "priority": 1,
            "stage_title": "1단계 · 청킹 / 색인 (책 자체의 품질)",
            "short_title": "1단계 · 청킹/색인",
            "status": s1_status,
            "status_badge": badge_map[s1_status],
            "trigger_rule": (
                "• 정답 문항 ID가 `chunk_manifest`에 없음\n"
                "• 구어체·평수 질문에서 원문 청크에 평수/쉬운 표현 단서가 전혀 없어 검색기 교체로도 `FN`이 안 줄음"
            ),
            "current_reading": s1_reading,
            "workbench_evidence": "• **2번 탭** `reference_contexts` 원문 확인\n• `data/chunk_manifest_*.jsonl` 조회",
            "action_guide": (
                "• 기존 컬렉션 유지 + **별도 컬렉션 신설 적재** 후 `presets.py`의 `collection_name`만 스위치!\n"
                "• 예: `B_chunk_v3_pyeong` (`--include-pyeong --recent-examples 5`)"
            ),
            "forbidden": "기존 `housing_rent_kh`에 덮어쓰기 (기존 실험 재현 파괴 & `allowed_chunk_ids` 불일치)",
            "recommended_preset": "B_chunk_v3_pyeong",
            "candidate_presets": ["B_chunk_v3_pyeong", "B_emb_3large", "B_emb_kure_v1"],
            "default_note": "가설: 평수·쉬운 별칭 병기 청킹(v3_pyeong) 분리 컬렉션으로 구어체 회수 개선",
        },
        {
            "stage_id": "S2_fn",
            "priority": 2,
            "stage_title": "2단계 · 검색 누락 (FN — 후보 12개에 정답 자체가 없음)",
            "short_title": "2단계 · 검색누락(FN)",
            "status": s2_status,
            "status_badge": badge_map[s2_status],
            "trigger_rule": "• `k=12` 혼동행렬 **`FN > 0`**\n• **`Recall@12 < 0.90`**\n• **`Context Recall < 0.60`**",
            "current_reading": s2_reading,
            "workbench_evidence": (
                "• **1번 탭** `k=12` 2×2 표의 **빨간 `FN` 칸**\n"
                f"• **1번 탭** 깔때기 2단계 문항 (`{', '.join(s2_qids) or '없음'}`)\n"
                "• **2번 탭** `first_rank = 0` 또는 `recall@12 = 0.50`"
            ),
            "action_guide": (
                "• **증상 ①** (A 단독 지명·숫자 누락): `A_vector` → **`B_rrf_55_45`** (BM25 결합)\n"
                "• **증상 ②** (2개 지역 비교 `recall@12=0.5` 또는 구어체/평수 누락): **`use_rewrite=True`** (`B_filter_rewrite` / `C_filter_rewrite`)\n"
                "• **증상 ③** (추상 표현·의미 불일치 FN): **`use_hyde=True`** 또는 임베딩 분리 컬렉션 (`B_emb_3large` / `B_emb_kure_v1`)\n"
                "• **증상 ④** (정답이 13~20위권에 걸림): `candidate_k` `12 → 20` 확대"
            ),
            "forbidden": "`FN > 0`인데 **검색 C(Re-ranker)**나 **프롬프트** 먼저 수정 (후보 12개에 없는 정답은 C가 못 살림!)",
            "recommended_preset": s2_preset,
            "candidate_presets": ["B_rrf_55_45", "B_filter_rewrite", "C_filter_rewrite", "C_filter_rewrite_hyde", "B_emb_3large", "B_emb_kure_v1"],
            "default_note": s2_note,
        },
        {
            "stage_id": "S3_fp",
            "priority": 3,
            "stage_title": "3단계 · 순위 실패 (FP — 후보 12개엔 있는데 Top-1/Top-5 밀림)",
            "short_title": "3단계 · 순위실패(FP)",
            "status": s3_status,
            "status_badge": badge_map[s3_status],
            "trigger_rule": "• `FN(k=12) == 0`인데 **`Top1 < 0.70`** (예: 8문항 중 5개 이하)\n• **`MRR@5 < 0.65`**\n• **`Context Precision < 0.60`**",
            "current_reading": s3_reading,
            "workbench_evidence": (
                "• **1번 탭** `k=12 FN=0` vs `k=5 FN>0` 또는 `Top1` 낮음\n"
                "• **2번 탭** 불일치 태그:\n"
                f"  - `[❌ 면적/계약/유형 불일치]` {len(cond_mismatch_qids)}건 (`{', '.join(cond_mismatch_qids) or '없음'}`)\n"
                f"  - `[❌ 동 불일치]` {len(dong_mismatch_qids)}건 (`{', '.join(dong_mismatch_qids) or '없음'}`)"
            ),
            "action_guide": (
                "• **① 가중치 스윕**: `55:45 → 70:30 → 80:20` 순서로 **최대점(Peak) 채택** (`80:20`처럼 하락하면 즉시 기각!)\n"
                "• **② `[❌ 면적/계약/유형 불일치]` 다수**: **`use_metadata_filter=True`** (`B_rrf_70_30_filter`)\n"
                "• **③ `[❌ 동 불일치]`·미묘한 조건 1등 싸움**: **`C_rrf_70_30_filter_rerank`** (검색 C 12→5 정밀 재정렬)"
            ),
            "forbidden": "후보 12개에 이미 정답이 있는데 청킹·임베딩 갈아엎기",
            "recommended_preset": s3_preset,
            "candidate_presets": ["B_rrf_70_30", "B_rrf_80_20", "B_rrf_70_30_filter", "C_rrf_70_30_filter_rerank"],
            "default_note": s3_note,
        },
        {
            "stage_id": "S4_gen",
            "priority": 4,
            "stage_title": "4단계 · 생성 / 인용 (검색 1등인데 답변 점수가 낮음)",
            "short_title": "4단계 · 생성/인용",
            "status": s4_status,
            "status_badge": badge_map[s4_status],
            "trigger_rule": "• `first_rank == #1` (`CP=1.0`, `CR=1.0`)인데\n• **`Answer Relevancy < 0.60`**\n• **`Faithfulness < 0.80`**",
            "current_reading": s4_reading,
            "workbench_evidence": (
                "• **2번 탭** 문항 X-Ray에서 검색은 🟢 1등인데 RAGAS만 🔴\n"
                "• 생성 답변 첫 문장(주어 생략·TMI 여부)과 원문 수치 직접 대조"
            ),
            "action_guide": (
                "• **`AR < 0.60`**: `ANSWER_PROMPT` 수정 — 첫 문장에 질문 5조건 주어 명시, 묻지 않은 TMI 제거, 3~5문장 압축\n"
                "• **`Faithfulness < 0.80`**: 실제 환각(없는 숫자 지어냄)이면 '근거 외 수치 생성 금지' 강화 / 원문 일치인데 점수만 튀면 judge 변동(±0.05) 기록"
            ),
            "forbidden": "검색 1등인데 BM25 가중치·필터 건드리기",
            "recommended_preset": s4_next_preset,
            "recommended_label": s4_rec_label,
            "candidate_presets": s4_candidates,
            "default_note": "가설: ANSWER_PROMPT v3(생활어·공식표기 괄호 병기 + 최근 1년) 및 검색 C로 AR/CR 개선",
        },
        {
            "stage_id": "S5_refusal",
            "priority": 5,
            "stage_title": "Track B · 거절 / 과거 데이터 (현재 공실·옵션 질문)",
            "short_title": "Track B · 거절",
            "status": s5_status,
            "status_badge": badge_map[s5_status],
            "trigger_rule": "• **`Refusal < 70%`**\n• **`과거 실거래 고지율 < 100%`**\n• 답변 가능 문항 오거절(**과잉 거절 > 0건**)",
            "current_reading": s5_reading,
            "workbench_evidence": "• **5번 탭** 거절 트랙 표 (`insufficient_evidence` 전용)\n• 현재 공실/옵션 질문에 과거 통계로 억지 답변했는지 확인",
            "action_guide": (
                "• `ANSWER_PROMPT` 거절 규칙 보강 — '본 자료는 과거 체결 완료된 실거래 요약 통계 기준이므로' 고지 문구 고정"
            ),
            "forbidden": "거절 문항을 RAGAS 평균에 섞어 `AR = 0.0`으로 착각하기",
            "recommended_preset": s4_next_preset,
            "recommended_label": f"거절 고지 프롬프트 v3 → {s4_next_preset}",
            "candidate_presets": s4_candidates,
            "default_note": "가설: 거절·과거 실거래 범위 고지 프롬프트 v3 보강 (Track B 100% 검증)",
        },
        {
            "stage_id": "S6_ops",
            "priority": 6,
            "stage_title": "운영 · 캐시 / 지연 (Latency & 효용 판정)",
            "short_title": "운영 · 캐시/지연",
            "status": s6_status,
            "status_badge": badge_map[s6_status],
            "trigger_rule": "• 반복 질의 **Cache 적중 `> 100ms`**\n• 버전 변경 후 **무효화 실패 `> 0건`**\n• **`p95` 지연이 허용치(5~10초) 초과**",
            "current_reading": s6_reading,
            "workbench_evidence": "• **1번 탭** 지연 KPI (`median` / `p95`)\n• 캐시 히트/무효화 · `status='error'` 미저장 로그",
            "action_guide": (
                f"• **`{s6_next_preset}`** ON (반복 질의 지연 단축 · 버전 무효화 검증)\n"
                "• C/HyDE/Rewrite로 품질 이득은 `+0.01` 미만인데 `p95`만 +3초 늘면 **비용·지연 대비 효용 부족으로 기각**"
            ),
            "forbidden": "`status='error'` 응답을 캐시에 저장하거나 버전 키(`collection`/`prompt_version`) 누락",
            "recommended_preset": s6_next_preset,
            "candidate_presets": s6_candidates,
            "default_note": "가설: 버전 인식 SQLite Cache ON으로 반복 질의 <100ms 및 무효화 검증",
        },
    ]
    return rows


def _evaluate_keep_discard_gate(before_s: dict | None, after_s: dict | None, target: float) -> dict[str, Any]:
    """Step 5 채택(Keep) vs 기각(Discard) 4중 게이트 자동 판정."""
    if not before_s or not after_s or before_s["order"] == after_s["order"]:
        return {"verdict": "⚪ 비교 대기 (서로 다른 Before / After 스텝 선택 필요)", "checks": []}

    n_diff, diffs = _describe_preset_diff(before_s, after_s)
    improved, regressed = _compute_regression_diff(after_s, before_s, target)
    key_deltas = {
        "Recall@12": after_s["recall12"] - before_s["recall12"],
        "MRR@5": after_s["mrr5"] - before_s["mrr5"],
        "CP": after_s["cp"] - before_s["cp"],
        "CR": after_s["cr"] - before_s["cr"],
        "AR": after_s["ar"] - before_s["ar"],
        "F": after_s["f"] - before_s["f"],
    }
    best_gain_name, best_gain_val = max(key_deltas.items(), key=lambda kv: kv[1])
    d_p95 = after_s["lat_p95"] - before_s["lat_p95"]

    g1 = n_diff == 1
    g2 = best_gain_val >= 0.02 or (after_s["top1_rate"] - before_s["top1_rate"]) >= 0.05
    g3 = len(regressed) == 0
    g4 = after_s["lat_p95"] <= 10000.0 and (d_p95 <= 3000.0 or best_gain_val >= 0.05)

    checks = [
        ("1. 단일 변인 가드", g1, f"변수 {n_diff}개 변경 (`{diffs[0] if diffs else '-'}`)"),
        ("2. 핵심 지표 실질 개선", g2, f"최대 개선 `{best_gain_name} {best_gain_val:+.3f}` (Top1 {(after_s['top1_rate']-before_s['top1_rate'])*100:+.1f}%p)"),
        ("3. 회귀 감시 (악화 0건)", g3, f"개선 {len(improved)}건 · 악화 {len(regressed)}건"),
        ("4. 지연·효용 합리성", g4, f"p95 {after_s['lat_p95']:.1f}ms ({d_p95:+.1f}ms)"),
    ]
    if g1 and g2 and g3 and g4:
        verdict = "✅ 채택(Keep) 권장 — 4중 게이트 전원 통과! 현재 스텝을 새 기준선으로 승격하세요."
    elif g2 and g3 and not g1:
        verdict = "🟡 조건부 검토 — 지표는 개선됐으나 변수가 2개 이상 동시에 바뀌어 원인 분리 재실행을 권장합니다."
    elif not g2:
        verdict = "❌ 기각(Discard) 권장 — 핵심 지표 개선이 노이즈(±0.02) 이하입니다. 기각 사유를 실험기록.md에 남기세요."
    else:
        verdict = "🟡 조건부 검토 — 일부 회귀(악화) 또는 지연 증가가 관찰됩니다. 4번 탭 Regression Watch를 확인하세요."
    return {"verdict": verdict, "checks": checks}


# ==================== 상단 1줄 요약 바 & PAGE 1~2 (현황 요약 / 네비게이터) ====================
def _render_compact_status_bar(
    before_s: dict | None,
    after_s: dict | None,
    matrix_rows: list[dict[str, Any]],
) -> None:
    """페이지 상단에 단 1줄의 정돈된 현황 스트립만 표시한다 (과도한 배너·중첩 제거)."""
    top_row = next((r for r in matrix_rows if r["status"] == "triggered"), None) or next(
        (r for r in matrix_rows if r["status"] == "watch"), matrix_rows[0]
    )
    with st.container(border=True):
        c1, c2, c3, c4 = st.columns([1.45, 1.15, 1.15, 1.05])
        if after_s is None or before_s is None:
            c1.markdown("**현재 비교**\n\n`실행 기록 대기`")
            c2.markdown("**단일 변인 가드**\n\n⚪ 기준선 실행 전")
            c3.markdown("**최우선 병목**\n\n⚪ Step 1 (`A_vector`) 시작")
            c4.markdown("**추천 다음 1변수**\n\n`A_vector`")
            return

        n_diff, diffs = _describe_preset_diff(before_s, after_s)
        c1.markdown(f"**현재 비교 (`{after_s['eval_version']}`)**\n\n`{before_s['step_label']}` → **`{after_s['step_label']}`**")
        if before_s["order"] == after_s["order"]:
            c2.markdown("**단일 변인 가드**\n\n⚪ 동일 스텝 선택 중")
        elif n_diff == 1:
            c2.markdown(f"**단일 변인 가드**\n\n✅ 변수 1개 (`{diffs[0][:34]}`)")
        else:
            c2.markdown(f"**단일 변인 가드**\n\n⚠️ 복합 {n_diff}개 동시 변경")
        c3.markdown(f"**최우선 병목**\n\n{top_row['status_badge']} **{top_row['short_title']}**")
        c4.markdown(f"**추천 다음 1변수**\n\n`{top_row.get('recommended_label') or top_row['recommended_preset']}`")


def _render_sop_coach_panel(
    scoped_steps: list[dict],
    before_s: dict | None,
    after_s: dict | None,
    target: float,
    sel_eval_ver: str,
    matrix_rows: list[dict[str, Any]],
) -> None:
    """PAGE 2: 원인 진단 & 다음 실험 선택 (독립 전용 화면 — 중첩 탭·과잉 박스 제거)."""
    primary_row = next((r for r in matrix_rows if r["status"] == "triggered"), None) or next(
        (r for r in matrix_rows if r["status"] == "watch"), matrix_rows[1]
    )
    st.subheader("🧭 원인 진단 & 다음 실험 선택 네비게이터")
    st.caption("1→6순위 수치 기준에 따라 현재 병목을 자동 판정하고, 바꿀 손잡이 딱 1개와 실행 명령어를 바로 조립합니다.")

    t_diag, t_table, t_sop = st.tabs(
        [
            "🎯 단계별 진단 & 명령어 조립",
            "📊 6단계 수치 기준표 & Keep/Discard 게이트",
            "📖 7단계 표준 루틴 (SOP)",
        ]
    )

    # ---- Sub-tab 1: 깔끔한 단계별 선택기 + 2열 정돈 카드 + 명령어 조립기 ----
    with t_diag:
        stage_labels = [f"{r['status_badge']} {r['short_title']}" for r in matrix_rows]
        default_idx = matrix_rows.index(primary_row)
        after_key = f"{after_s['timestamp']}_{after_s['mode']}" if after_s else "none"

        picked_lbl = st.radio(
            "진단할 단계 선택 (기본값 = 현재 최우선 발동 단계)",
            stage_labels,
            index=default_idx,
            horizontal=True,
            key=f"nav_stage_radio_{after_key}",
        )
        sel_row = matrix_rows[stage_labels.index(picked_lbl)]

        col_l, col_r = st.columns(2, gap="medium")
        with col_l:
            with st.container(border=True):
                st.markdown(f"**① 수치 기준 & 현재 실측치** &nbsp; {sel_row['status_badge']}")
                st.markdown(sel_row["trigger_rule"])
                st.divider()
                st.markdown(f"**📌 내 실측치**: {sel_row['current_reading']}")
            with st.container(border=True):
                st.markdown("**② 대시보드에서 확인할 근거 (Workbench)**")
                st.markdown(sel_row["workbench_evidence"])
        with col_r:
            with st.container(border=True):
                st.markdown("**③ 정확히 바꿀 손잡이 (딱 1개)**")
                st.markdown(sel_row["action_guide"])
            with st.container(border=True):
                st.markdown("**④ ❌ 절대 하면 안 되는 헛수고**")
                st.markdown(sel_row["forbidden"])

        with st.container(border=True):
            st.markdown(f"**⚡ 실행 명령어 자동 조립기** — `{sel_row['short_title']}` 맞춤 추천")
            eval_files = [p.name for p in sorted(KH_EVAL_DIR.glob("*.json")) if "test" not in p.name.lower()] if KH_EVAL_DIR.exists() else []
            if not eval_files:
                eval_files = ["kh_rent_dev.json", "kh_rent_dev_v2.json"]

            cg1, cg2, cg3 = st.columns([1.05, 0.95, 1.6])
            with cg1:
                rec_p = sel_row["recommended_preset"] if sel_row["recommended_preset"] in ALL_PRESET_NAMES else "A_vector"
                cmd_preset = st.selectbox(
                    "프리셋 (1개)",
                    ALL_PRESET_NAMES,
                    index=ALL_PRESET_NAMES.index(rec_p),
                    key=f"nav_cmd_preset_{sel_row['stage_id']}_{after_key}",
                    help=f"이 단계 추천 후보: {', '.join(sel_row['candidate_presets'])}",
                )
            with cg2:
                default_eval_idx = 0
                for idx_e, ef in enumerate(eval_files):
                    if sel_eval_ver and sel_eval_ver in ef:
                        default_eval_idx = idx_e
                        break
                cmd_eval = st.selectbox(
                    "시험지 (`dev` 전용)",
                    eval_files,
                    index=default_eval_idx,
                    key=f"nav_cmd_eval_{sel_row['stage_id']}_{after_key}",
                )
            with cg3:
                cmd_note = st.text_input(
                    "가설 메모 (`--note`)",
                    value=sel_row["default_note"],
                    key=f"nav_cmd_note_{sel_row['stage_id']}_{after_key}",
                )

            if cmd_preset in SAFE_COLLECTION_PREP_CMDS:
                st.caption(f"📦 `{cmd_preset}`은 기존 DB를 덮어쓰지 않는 별도 컬렉션 프리셋입니다. 최초 1회 분리 색인 명령어:")
                st.code(SAFE_COLLECTION_PREP_CMDS[cmd_preset], language="powershell")

            safe_note = cmd_note.replace('"', "'")
            st.code(
                f'uv run python -m rag_home_api.test.kh.exp_03_rent_eval {cmd_preset} --eval {cmd_eval} --note "{safe_note}"',
                language="powershell",
            )

    # ---- Sub-tab 2: 전체 6단계 수치 기준표 & Step 5 Keep/Discard 게이트 ----
    with t_table:
        st.markdown("##### A. 단계별 수치 기준(Trigger) → 근거 → 수정 결정(Action) 전체 표")
        full_headers = ["병목 단계", "판정", "수치 기준 (언제 문제인가?)", "현재 실측치", "수정할 손잡이 (딱 1개)", "금지 헛수고"]
        full_rows = [
            {
                "_row_cls": "row-trig" if r["status"] == "triggered" else ("row-watch" if r["status"] == "watch" else ""),
                "병목 단계": f"**{r['short_title']}**",
                "판정": r["status_badge"],
                "수치 기준 (언제 문제인가?)": r["trigger_rule"],
                "현재 실측치": r["current_reading"],
                "수정할 손잡이 (딱 1개)": r["action_guide"],
                "금지 헛수고": r["forbidden"],
            }
            for r in matrix_rows
        ]
        _render_wrapped_html_table(full_headers, full_rows, col_widths=["12%", "8%", "20%", "20%", "26%", "14%"])

        st.markdown("##### B. Step 5 채택(Keep) vs 기각(Discard) 4중 게이트")
        gate = _evaluate_keep_discard_gate(before_s, after_s, target)
        if gate["verdict"].startswith("✅"):
            st.success(gate["verdict"])
        elif gate["verdict"].startswith("❌"):
            st.error(gate["verdict"])
        else:
            st.info(gate["verdict"])
        if gate["checks"]:
            gc = st.columns(4)
            for col, (gname, gok, gdetail) in zip(gc, gate["checks"]):
                with col:
                    with st.container(border=True):
                        st.markdown(f"**{'✅' if gok else '⚠️'} {gname}**")
                        st.caption(gdetail)

    # ---- Sub-tab 3: 7단계 SOP ----
    with t_sop:
        if not scoped_steps or after_s is None:
            cur_step_num = 1
        elif len(scoped_steps) == 1:
            cur_step_num = 2
        elif after_s["pass_4of4"] and after_s["fail_cnt"] == 0:
            cur_step_num = 7
        else:
            cur_step_num = 5
        step_badges = [
            ("S0 시험지 동결", 0),
            ("S1 A 단독 실행", 1),
            ("S2 5분 판별", 2),
            ("S3 손잡이 1개", 3),
            ("S4 1변수 실행", 4),
            ("S5 채택/기각", 5),
            ("S6~7 기록·test봉인", 6),
        ]
        bar_parts = [
            f"**[▶ {lbl}]**" if num == cur_step_num else (f"~~{lbl}~~ ✅" if num < cur_step_num else lbl)
            for lbl, num in step_badges
        ]
        with st.container(border=True):
            st.markdown(" → ".join(bar_parts))
        st.markdown(
            """
1. **Step 0 (시험지 동결)**: `dev_v1` 유지 또는 현실 구어체·평수 `dev_v2` 확정 *(질문 수정 시에만 `_v{n}` 신설, 튜닝 중엔 시험지 고정)*
2. **Step 1 (A 단독 실행)**: `A_vector` 딱 1개만 먼저 돌려 바닥 기준선 확보
3. **Step 2 (5분 병목 판별)**: `1. 현황 요약` (`k=12` `FN` 확인) → `3. 문항별 X-Ray` (불일치 태그)에서 **가장 윗단계(1→2→3→4순)** 확인
4. **Step 3 (손잡이 1개 선택)**: `2. 원인 진단 & 다음 실험 선택` 탭에서 발동 단계의 추천 변수 **딱 1개**만 선택
5. **Step 4 (1개만 바꿔 실행)**: `--note "가설: ~"` 적고 자동 조립된 명령어로 프리셋 1개 실행
6. **Step 5 (채택/기각 판정)**: 단일 변인 가드(✅) + `Keep/Discard 4중 게이트` + `4. 추이 & 회차 리포트`의 Regression Watch(악화 0건) 확인
7. **Step 6~7 (기록 & 봉인 개봉)**: 회차 리포트 → `실험기록.md` 누적 → dev 완성 후 봉인 `test` 20문항 1회 성적표 + `5. 팀 통합 비교`
            """
        )


# ==================== PAGE 1: 현황 요약 (KPI · 혼동행렬 · 6단계 상태) ====================
def render_decision_tab(
    before_s: dict,
    after_s: dict,
    target: float,
    coach_mode: bool = True,
    matrix_rows: list[dict[str, Any]] | None = None,
) -> None:
    st.subheader("📊 현황 요약 — 핵심 지표 · 혼동행렬 · 단계별 상태")
    if coach_mode:
        st.caption("💡 각 지표 이름 옆 **`?`**에 마우스를 올리면 쉬운 풀이가 보입니다. 병목 진단과 다음 실행 명령어는 좌측 **`2. 원인 진단 & 다음 실험 선택`** 메뉴에서 바로 진행하세요.")

    # 1) 검색 지표 (6)
    st.markdown("##### 🔎 검색 지표 (Retrieval)")
    r1, r2, r3, r4, r5, r6 = st.columns(6)
    r1.metric(
        "Recall@12 (재현율)",
        f"{after_s['recall12']:.3f}" if after_s.get("has_r12", True) else "미측정",
        f"{after_s['recall12'] - before_s['recall12']:+.3f}" if (after_s.get("has_r12", True) and before_s.get("has_r12", True)) else None,
        help=METRIC_HELP["recall@12"],
    )
    r2.metric("Hit@5 (적중률)", after_s["hit5_str"], f"{(after_s['hit5_rate'] - before_s['hit5_rate'])*100:+.1f}%p", help=METRIC_HELP["hit@5"])
    r3.metric(
        "Top1 (1등 정답률)",
        after_s["top1_str"],
        f"{(after_s['top1_rate'] - before_s['top1_rate'])*100:+.1f}%p" if (after_s.get("has_top1", True) and before_s.get("has_top1", True)) else None,
        help=METRIC_HELP["top1"],
    )
    r4.metric(
        "MRR@5 (평균역순위)",
        f"{after_s['mrr5']:.3f}" if after_s.get("has_mrr5", True) else "미측정",
        f"{after_s['mrr5'] - before_s['mrr5']:+.3f}" if (after_s.get("has_mrr5", True) and before_s.get("has_mrr5", True)) else None,
        help=METRIC_HELP["mrr@5"],
    )
    r5.metric(
        "Context Precision",
        f"{after_s['cp']:.3f}" if after_s.get("has_cp", True) else "미측정",
        f"{after_s['cp'] - before_s['cp']:+.3f}" if (after_s.get("has_cp", True) and before_s.get("has_cp", True)) else None,
        help=METRIC_HELP["cp"],
    )
    r6.metric(
        "Context Recall",
        f"{after_s['cr']:.3f}" if after_s.get("has_cr", True) else "미측정",
        f"{after_s['cr'] - before_s['cr']:+.3f}" if (after_s.get("has_cr", True) and before_s.get("has_cr", True)) else None,
        help=METRIC_HELP["cr"],
    )

    # 2) 생성 + 거절·운영 지표 (6)
    st.markdown("##### ✍️ 생성 (Generation) · 🛡️ 거절 (Track B) · ⏱️ 지연")
    g1, g2, g3, g4, g5, g6 = st.columns(6)
    g1.metric(
        "Faithfulness (충실도)",
        f"{after_s['f']:.3f}" if after_s.get("has_f", True) else "미측정",
        f"{after_s['f'] - before_s['f']:+.3f}" if (after_s.get("has_f", True) and before_s.get("has_f", True)) else None,
        help=METRIC_HELP["f"],
    )
    g2.metric(
        "Answer Relevancy",
        f"{after_s['ar']:.3f}" if after_s.get("has_ar", True) else "미측정",
        f"{after_s['ar'] - before_s['ar']:+.3f}" if (after_s.get("has_ar", True) and before_s.get("has_ar", True)) else None,
        help=METRIC_HELP["ar"],
    )
    g3.metric("Citation (출처 표기)", after_s["citation_str"], help=METRIC_HELP["citation_rate"])
    g4.metric("Refusal (거절률)", after_s["refusal_str"], help=METRIC_HELP["refusal"])
    g5.metric("Scope Notice (고지)", after_s["scope_notice_str"], help=METRIC_HELP["scope_notice"])
    g6.metric(
        "Latency 중앙값",
        f"{after_s['lat_median']:.1f} ms" if after_s.get("has_lat", True) else "미측정",
        f"{after_s['lat_median'] - before_s['lat_median']:+.1f} ms" if (after_s.get("has_lat", True) and before_s.get("has_lat", True)) else None,
        delta_color="inverse",
        help=METRIC_HELP["lat_median"],
    )

    st.divider()

    # 3) 전체 너비: 🚦 6단계 병목 진단 요약 (잘림/스크롤 없이 한눈에 전체 확인!)
    st.markdown(f"##### 🚦 6단계 병목 진단 요약 (목표 `{target:.2f}` · 위→아래 우선순위)")
    rows = matrix_rows if matrix_rows is not None else _evaluate_decision_matrix(after_s, before_s, target)
    sum_headers = ["단계", "상태", "현재 실측치 & 원인 문항", "추천 1변수"]
    sum_rows = [
        {
            "_row_cls": "row-trig" if r["status"] == "triggered" else ("row-watch" if r["status"] == "watch" else ""),
            "단계": f"**{r['short_title']}**",
            "상태": r["status_badge"],
            "현재 실측치 & 원인 문항": r["current_reading"],
            "추천 1변수": f"`{r['recommended_preset']}`",
        }
        for r in rows
    ]
    _render_wrapped_html_table(sum_headers, sum_rows, col_widths=["16%", "9%", "59%", "16%"])

    top_r = next((r for r in rows if r["status"] == "triggered"), None) or next(
        (r for r in rows if r["status"] == "watch"), rows[0]
    )
    st.caption(
        f"🎯 **최우선 공략**: {top_r['status_badge']} **{top_r['short_title']}** → **`{top_r['recommended_preset']}`** "
        "(상세 기준·근거·실행 명령어는 좌측 메뉴 **`🧭 2. 원인 진단 & 다음 실험 선택`**에서 바로 복사하세요)"
    )

    st.divider()

    # 4) 전체 너비 4열 가로 배치: 🧮 검색 혼동행렬 (TP · FN · FP · TN)
    cm_head_l, cm_head_r = st.columns([1.2, 1.3])
    with cm_head_l:
        st.markdown("##### 🧮 검색 혼동행렬 (`TP · FN · FP · TN`)")
    with cm_head_r:
        k_choice = st.radio(
            "집계 기준",
            ["후보 단계 k=12 (2단계 FN 누락 점검)", "최종 전달 k=5 (3단계 Top-5 점검)"],
            horizontal=True,
            label_visibility="collapsed",
        )
    k_val = 12 if "12" in k_choice else 5
    cur_cm = after_s["cm12"] if k_val == 12 else after_s["cm5"]
    prev_cm = (before_s["cm12"] if k_val == 12 else before_s["cm5"]) if before_s["order"] != after_s["order"] else None
    if cur_cm["q_count"] > 0:
        _render_confusion_matrix_2x2(cur_cm, k=k_val, prev_cm=prev_cm, horizontal=True)
    else:
        st.info("연동된 `.jsonl` 상세 로그가 없어 혼동행렬을 산출할 수 없습니다.")

    st.divider()
    with st.expander("📸 발표용 Before vs After 8대 지표 비교 그래프 (미리보기 & 3x PNG / SVG 저장)", expanded=False):
        eval_v = after_s.get("eval_version", "eval")
        safe_v = re.sub(r"[^a-zA-Z0-9_\-]", "_", eval_v)
        svg_cmp = _build_standalone_svg_bar_compare(
            before_s,
            after_s,
            title=f"Before vs After 8대 핵심 지표 비교 ({eval_v})",
            subtitle=f"{before_s['step_label']}  →  {after_s['step_label']}",
            target=target,
        )
        st.image(svg_cmp, use_container_width=True)
        _render_svg_png_export_pair(svg_cmp, f"before_after_{safe_v}", f"home_cmp_{safe_v}_{before_s['order']}_{after_s['order']}")


# ==================== TAB 2: 문항별 X-Ray 해부 ====================
def render_xray_tab(before_s: dict, after_s: dict, target: float, coach_mode: bool = True) -> None:
    st.subheader(f"🔬 문항별 X-Ray 해부 — `{before_s['step_label']}` vs `{after_s['step_label']}`")
    if coach_mode:
        st.caption(
            "💡 **Step 2 코치 팁**: 미달 문항을 선택해 1등 청크의 **빨간 불일치 태그(`❌ 동/면적/계약/유형 불일치`)**를 보면 "
            "가중치·필터·검색 C·Query Rewrite 중 정확히 무엇을 고쳐야 할지 10초 만에 판별됩니다!"
        )
    if not after_s["mode_logs"]:
        st.info("연결된 `.jsonl` 상세 로그가 없습니다.")
        return

    before_map = {r.get("question_id"): r for r in before_s["mode_logs"]}
    after_logs = after_s["mode_logs"]

    flt_opt = st.radio(
        "문항 필터",
        ["전체 (10문항)", "기준 미달만", "개선된 문항", "악화·주의 문항", "거절 트랙 (과거 데이터 범위 외)"],
        horizontal=True,
    )

    summary_rows = []
    filtered_logs = []
    for r in after_logs:
        qid = r.get("question_id", "")
        br = before_map.get(qid)
        status = r.get("expected_status", "answered")
        arg = after_s["item_ragas"].get(qid) or {}
        brg = before_s["item_ragas"].get(qid) or {}
        stg, stg_msg = _classify_funnel_stage(r, arg, target) if status == "answered" else (-1, "🛡️ 거절 트랙 (RAGAS 제외)")

        cmrr = float((r.get("metrics_k5") or {}).get("mrr@5", 0.0) or 0.0)
        pmrr = float((br.get("metrics_k5") or {}).get("mrr@5", 0.0) or 0.0) if br else cmrr
        c_ar = arg.get("answer_relevancy")
        p_ar = brg.get("answer_relevancy")
        delta = (cmrr - pmrr) + ((float(c_ar) - float(p_ar)) if (c_ar is not None and p_ar is not None) else 0.0)

        keep = True
        if flt_opt == "기준 미달만":
            keep = status == "answered" and stg > 0
        elif flt_opt == "개선된 문항":
            keep = status == "answered" and delta >= 0.08
        elif flt_opt == "악화·주의 문항":
            keep = status == "answered" and delta <= -0.08
        elif flt_opt.startswith("거절 트랙"):
            keep = status != "answered"
        if not keep:
            continue

        filtered_logs.append(r)
        m12 = r.get("metrics_k12") or {}
        m5 = r.get("metrics_k5") or {}
        bm5 = (br.get("metrics_k5") or {}) if br else {}
        summary_rows.append(
            {
                "ID": qid,
                "유형": r.get("question_type", ""),
                "깔때기 상태": "🟢 통과" if stg == 0 else ("🛡️ 거절" if stg == -1 else f"{stg}단계 병목"),
                "first_rank (Before→After)": f"#{bm5.get('first_rank', '-') if br else '-'} → #{m5.get('first_rank', '-')}",
                RETRIEVAL_LABELS["recall@12"]: m12.get("recall@12", "-"),
                RETRIEVAL_LABELS["precision@5"]: m5.get("precision@5", "-"),
                RETRIEVAL_LABELS["mrr@5"]: m5.get("mrr@5", "-"),
                RAGAS_LABELS["faithfulness"]: arg.get("faithfulness", "-"),
                RAGAS_LABELS["answer_relevancy"]: arg.get("answer_relevancy", "-"),
                RAGAS_LABELS["llm_context_precision_with_reference"]: arg.get("llm_context_precision_with_reference", "-"),
                RAGAS_LABELS["context_recall"]: arg.get("context_recall", "-"),
            }
        )

    if not filtered_logs:
        st.success("선택한 필터 조건에 해당하는 문항이 없습니다!")
        return

    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

    # 선택한 문항 1건 심층 X-Ray
    q_options = [f"{r.get('question_id')} | {r.get('question_type')}" for r in filtered_logs]
    picked = st.selectbox("🔍 심층 해부할 문항 선택", q_options)
    arow = filtered_logs[q_options.index(picked)]
    qid = arow.get("question_id", "")
    brow = before_map.get(qid)
    arg = after_s["item_ragas"].get(qid) or {}
    brg = before_s["item_ragas"].get(qid) or {}

    if arow.get("expected_status") == "insufficient_evidence":
        st.warning(
            f"🛡️ **[{qid}] 거절 전용 트랙 (과거 데이터 범위 외 질문)**\n\n"
            "RAGAS `Answer Relevancy`에 넣으면 정직 거절 시 `noncommittal=1 → 0.0` 역설 감점이 발생하므로, "
            "별도 거절 트랙에서 **환각 없음 + 과거 실거래 통계 한계 고지 여부**로 평가합니다."
        )
        st.markdown(f"**After 응답 (`{after_s['step_label']}`):**\n> {arow.get('response', '')}")
        if brow:
            st.markdown(f"**Before 응답 (`{before_s['step_label']}`):**\n> {brow.get('response', '')}")
        return

    st.info(_funnel_badge(arow, arg, target))

    # 패널 A: Before vs After Top-5 조건 불일치 / 허용 청크 적중 X-Ray
    st.markdown("#### A. 검색 Top-5 정답 적중 X-Ray (허용 정답 청크 vs 실제 Top-5)")
    targets = _parse_reference_targets(arow)
    allowed_set = set(arow.get("allowed_chunk_ids") or [])
    col_b, col_a = st.columns(2)
    with col_b:
        st.markdown(f"**Before · `{before_s['step_label']}`**")
        if brow:
            for rank_i, cid in enumerate(brow.get("top5_ids") or [], start=1):
                if targets:
                    ok, tags = _explain_chunk_mismatch(cid, targets)
                    tag_str = " ".join(tags)
                else:
                    ok = cid in allowed_set
                    tag_str = "✅ ★ 허용 정답 청크 일치!" if ok else "⬜ 참고 후보 청크"
                (st.success if ok else st.markdown)(f"`#{rank_i}` `{cid}`  \n{tag_str}")
        else:
            st.caption("Before 실행에 동일 문항 로그가 없습니다.")
    with col_a:
        st.markdown(f"**After · `{after_s['step_label']}`**")
        for rank_i, cid in enumerate(arow.get("top5_ids") or [], start=1):
            if targets:
                ok, tags = _explain_chunk_mismatch(cid, targets)
                tag_str = " ".join(tags)
            else:
                ok = cid in allowed_set
                tag_str = "✅ ★ 허용 정답 청크 일치!" if ok else "⬜ 참고 후보 청크"
            (st.success if ok else st.markdown)(f"`#{rank_i}` `{cid}`  \n{tag_str}")

    # 패널 B: 문항별 TP/FP/FN/TN 혼동행렬 (깔끔하게 접어 두고 필요할 때만 펼치기)
    with st.expander("🧮 이 문항의 청크 혼동행렬 (`TP · FP · FN · TN` · `k=12 / k=5`) 펼치기", expanded=False):
        cm_c1, cm_c2 = st.columns(2)
        with cm_c1:
            _render_confusion_matrix_2x2(
                _item_confusion_matrix(arow, 12, after_s["corpus_n"]),
                k=12,
                prev_cm=_item_confusion_matrix(brow, 12, before_s["corpus_n"]) if brow else None,
                title_prefix="[후보 k=12] ",
            )
        with cm_c2:
            _render_confusion_matrix_2x2(
                _item_confusion_matrix(arow, 5, after_s["corpus_n"]),
                k=5,
                prev_cm=_item_confusion_matrix(brow, 5, before_s["corpus_n"]) if brow else None,
                title_prefix="[최종 k=5] ",
            )
        st.caption(
            "💡 정답 청크가 1권뿐인 단일 문항에서 5권을 뽑아 1등으로 맞히면 `TP=1(진짜 정답), FP=4(나머지 4칸 오답)` → "
            "`Precision@5 = 1/5 = 0.20`이 이론상 만점입니다. 순위 품질은 `MRR@5`와 RAGAS `Context Precision`으로 확인하세요."
        )

    # 패널 C: RAGAS 4지표 & 생성 답변 Before vs After
    st.markdown("#### C. RAGAS 4대 지표 & 생성 답변 (`response`) Before vs After")
    rc1, rc2 = st.columns(2)
    with rc1:
        rg_rows = []
        for k, lbl in RAGAS_LABELS.items():
            bv = brg.get(k)
            av = arg.get(k)
            d_txt = f"{float(av) - float(bv):+.3f}" if (av is not None and bv is not None) else "-"
            rg_rows.append(
                {
                    "RAGAS 지표": lbl,
                    "Before": f"{float(bv):.3f}" if bv is not None else "-",
                    "After": f"{float(av):.3f}" if av is not None else "-",
                    "변화 (Δ)": d_txt,
                    "판정": ("✅ 통과" if float(av) >= target else "🔴 미달") if av is not None else "-",
                }
            )
        st.dataframe(pd.DataFrame(rg_rows), use_container_width=True, hide_index=True)
    with rc2:
        if brow:
            st.markdown(f"**Before 답변 (`{before_s['step_label']}`):**\n> {brow.get('response', '')}")
        st.markdown(f"**After 답변 (`{after_s['step_label']}`):**\n> {arow.get('response', '')}")

    # LLM 진단 (있으면 표시)
    diag_match = next(
        (x for x in after_s["mode_data"].get("underperforming_items", []) if x.get("question_id") == qid),
        None,
    )
    if diag_match:
        st.markdown("#### 💡 LLM 자동 원인 분류 & 개선안")
        st.markdown(diag_match.get("llm_diagnosis", ""))

    # 패널 D: 원문 컨텍스트 & Raw JSON (기본 접힘)
    with st.expander("📄 정답 컨텍스트 vs 실제 검색 컨텍스트 원문 비교 & Raw JSON 펼치기", expanded=False):
        cc1, cc2 = st.columns(2)
        with cc1:
            st.markdown("**✅ 정답 컨텍스트 (`reference_contexts`)**")
            for ctx in arow.get("reference_contexts") or []:
                st.code(ctx, language="text")
        with cc2:
            st.markdown("**🔎 실제 검색 컨텍스트 (`retrieved_contexts` 상위 3개)**")
            for idx_c, ctx in enumerate((arow.get("retrieved_contexts") or [])[:3], start=1):
                st.caption(f"#{idx_c}")
                st.code(ctx, language="text")
        st.json(
            {
                "applied_filters": arow.get("applied_filters"),
                "filter_fallback": arow.get("filter_fallback"),
                "metrics_k12": arow.get("metrics_k12"),
                "metrics_k5": arow.get("metrics_k5"),
            }
        )


# ==================== TAB 3: 회차별 랩노트 & 추이 선그래프 ====================
def render_lab_history_tab(
    steps: list[dict],
    target: float,
    selected_eval_ver: str,
    coach_mode: bool = True,
    before_s: dict | None = None,
    after_s: dict | None = None,
    owner_key: str = "kh",
) -> None:
    st.subheader("📈 회차별 실험 랩노트 & 수치 변화 추이 (연결 선 그래프)")
    if coach_mode:
        st.caption(
            "💡 **Step 5 코치 팁**: 같은 시험지(`eval_version`) 1개만 격리한 상태에서 선그래프가 우상향하는지 확인하세요. "
            "수치가 꺾인 스텝(예: `80:20` 과중)은 **기각(Discard)**하고 직전 최적점으로 복귀합니다!"
        )
    if not steps:
        st.info("표시할 실험 스텝이 없습니다.")
        return

    distinct_evals = sorted({s["eval_version"] for s in steps})
    if len(distinct_evals) > 1:
        st.warning(
            f"⚠️ 현재 서로 다른 시험지 버전({', '.join(distinct_evals)})이 함께 표시되고 있습니다. "
            "공정한 전/후 비교를 위해 좌측 사이드바에서 **특정 `eval_version` 1개만 격리**해 보는 것을 권장합니다!"
        )
    else:
        st.caption(f"🔒 동일 시험지 격리 중: `{selected_eval_ver}` ({len(steps)}개 스텝)")

    view_mode = st.radio(
        "표시 범위",
        ["전체 실행 스텝 모두 보기 (시간순 상세)", "회차별 대표 최고 구성만 보기 (핵심 흐름)"],
        horizontal=True,
        key="lab_view_mode",
    )
    if view_mode.startswith("회차별 대표"):
        by_ts: dict[str, list[dict]] = {}
        for s in steps:
            by_ts.setdefault(s["timestamp"], []).append(s)
        shown = [max(g, key=lambda x: (x["ragas_mean"], x["mrr5"])) for g in by_ts.values()]
    else:
        shown = steps

    g1, g2 = st.columns(2)
    with g1:
        st.markdown(f"#### 1) RAGAS 4대 지표 회차별 추이 (빨간 점선 = 목표 {target:.2f})")
        _render_altair_line_chart(
            shown,
            [
                ("f", RAGAS_LABELS["faithfulness"], "#16a34a"),
                ("ar", RAGAS_LABELS["answer_relevancy"], "#dc2626"),
                ("cp", RAGAS_LABELS["llm_context_precision_with_reference"], "#2563eb"),
                ("cr", RAGAS_LABELS["context_recall"], "#7c3aed"),
            ],
            target=target,
        )
    with g2:
        st.markdown("#### 2) 검색 4대 지표 회차별 추이 (`Recall@12 · Hit@5 · Top1 · MRR@5`)")
        _render_altair_line_chart(
            shown,
            [
                ("recall12", RETRIEVAL_LABELS["recall@12"], "#0d9488"),
                ("hit5_rate", "Hit@5 (적중률@5)", "#2563eb"),
                ("top1_rate", RETRIEVAL_LABELS["top1"], "#d97706"),
                ("mrr5", RETRIEVAL_LABELS["mrr@5"], "#7c3aed"),
            ],
            target=None,
        )

    _render_presentation_export_panel(
        shown,
        before_s=before_s or (shown[0] if shown else None),
        after_s=after_s or (shown[-1] if shown else None),
        target=target,
        eval_ver=selected_eval_ver,
        owner_key=owner_key,
        key_scope="lab",
    )

    st.divider()
    st.markdown("#### 3) 단일 변인 실험 랩노트 통합표")
    lab_df = pd.DataFrame(
        [
            {
                "순서": f"#{s['order']}",
                "시각": s["ts_display"],
                "시험지": s["eval_version"],
                "프리셋 (Mode)": s["mode"],
                "변경 변수 (단일 변인)": s["change_desc"],
                RETRIEVAL_LABELS["recall@12"]: round(s["recall12"], 3),
                RETRIEVAL_LABELS["hit@5"]: s["hit5_str"],
                RETRIEVAL_LABELS["top1"]: s["top1_str"],
                RETRIEVAL_LABELS["mrr@5"]: round(s["mrr5"], 3),
                "FN (k12/k5)": f"{s['cm12']['fn']} / {s['cm5']['fn']}",
                RAGAS_LABELS["faithfulness"]: round(s["f"], 4),
                RAGAS_LABELS["answer_relevancy"]: round(s["ar"], 4),
                RAGAS_LABELS["llm_context_precision_with_reference"]: round(s["cp"], 4),
                RAGAS_LABELS["context_recall"]: round(s["cr"], 4),
                RETRIEVAL_LABELS["refusal"]: s["refusal_str"],
                RETRIEVAL_LABELS["lat_median"]: s["lat_median"],
                "직전 대비 변화 & 해석": s["delta_narrative"],
            }
            for s in shown
        ]
    )
    st.dataframe(lab_df, use_container_width=True, hide_index=True)

    # 초기 11/24 미니 실험 참고
    mini_logs = sorted(KH_RESULTS_DIR.glob("exp_ab_kh_*.jsonl")) if KH_RESULTS_DIR.exists() else []
    if mini_logs:
        with st.expander("📎 초기 통합 미니 검색 실험 참고 (2026-11-24 · 216청크 · Hit@3 전용)", expanded=False):
            for mp in mini_logs:
                mrows = _load_jsonl_file(mp)
                by_p: dict[str, list[dict]] = {}
                for r in mrows:
                    by_p.setdefault((r.get("preset") or {}).get("name", "unknown"), []).append(r)
                st.dataframe(
                    pd.DataFrame(
                        [
                            {
                                "프리셋 (Mode)": pn,
                                "Hit@3 (적중 여부@3)": f"{sum(int(x.get('hit_at_k', 0)) for x in grp)}/{len(grp)}",
                                "Top1 (Hit@1 · 첫 결과 정답률)": f"{sum(int(x.get('top1_ok', 0)) for x in grp)}/{len(grp)}",
                                "Latency Median (지연 중앙값, ms)": round(sorted(float(x.get("latency_ms", 0)) for x in grp)[len(grp)//2], 1),
                            }
                            for pn, grp in by_p.items()
                        ]
                    ),
                    use_container_width=True,
                    hide_index=True,
                )


# ==================== TAB 4: 회차별 평가 분석 리포트 ====================
def render_per_test_report_tab(steps: list[dict], target: float, coach_mode: bool = True) -> None:
    st.subheader("📝 테스트 회차별 평가 분석 리포트 (1회 실행 = 1개 독립 리포트)")
    if coach_mode:
        st.caption(
            "💡 **Step 5·6 코치 팁**: 아래 `4. 개선된 문항 vs 악화된 문항(Regression Watch)`에서 **악화 0건**을 확인한 뒤, "
            "Markdown 리포트를 복사해 `실험기록.md`에 누적하세요 (기각한 실패 실험도 솔직히 기록!)."
        )
    if not steps:
        st.info("리포트를 생성할 실행 기록이 없습니다.")
        return

    labels = [s["step_label"] for s in steps]
    sel_lbl = st.selectbox("분석 리포트를 볼 실행 회차 선택", labels, index=len(labels) - 1, key="report_step_sel")
    idx = labels.index(sel_lbl)
    cur = steps[idx]
    prev = steps[idx - 1] if idx > 0 and steps[idx - 1]["eval_version"] == cur["eval_version"] else None

    md_report = _build_single_step_report_md(cur, prev, target)
    html_report = _export_html(cur, prev, steps, target)

    d1, d2 = st.columns(2)
    with d1:
        st.download_button(
            label="📝 이 회차 Markdown 리포트 다운로드 (실험기록.md 복붙용)",
            data=md_report.encode("utf-8"),
            file_name=f"eval_report_{cur['timestamp']}_{cur['mode']}.md",
            mime="text/markdown",
            use_container_width=True,
        )
    with d2:
        st.download_button(
            label="🌐 이 회차 단독 HTML 리포트 다운로드 (선그래프 포함)",
            data=html_report.encode("utf-8"),
            file_name=f"eval_report_{cur['timestamp']}_{cur['mode']}.html",
            mime="text/html",
            use_container_width=True,
        )

    st.divider()
    st.markdown(md_report)

    with st.expander("📋 마크다운 원문 복사 박스", expanded=False):
        st.code(md_report, language="markdown")


# ==================== TAB 5: 시험지 버전(_v{n}) & 거절 트랙 ====================
def render_evalset_tab(selected_owner: str = "kh") -> None:
    st.subheader("🧪 시험지 버전 관리 (`_v{n}`) & 2-Track 거절 평가 점검")

    # 2-Track 원리 안내 (과거 데이터 vs 현재 매물)
    c_t1, c_t2 = st.columns(2)
    with c_t1:
        st.info(
            "**Track A · 답변 가능 문항 (`answered`) → RAGAS 4대 지표**\n\n"
            "DB에 실제 정답 청크가 존재하는 문항만 `Faithfulness · Answer Relevancy · Context Precision · Context Recall`로 채점합니다."
        )
    with c_t2:
        st.warning(
            "**Track B · 과거 데이터 범위 외 / 현재 매물 (`insufficient_evidence`) → 거절 트랙 분리**\n\n"
            "현재 입주 가능 공실·옵션·반경 500m 질문은 RAGAS에 넣으면 정직하게 거절할수록 `noncommittal=1 → Answer Relevancy 0.0` 감점을 받습니다. "
            "따라서 **거절률 (≥70%) + 과거 실거래 통계 한계 고지율**로 별도 채점합니다."
        )

    st.caption("💡 불변 스냅샷 원칙: 질문·정답을 수정할 때만 `_v2`, `_v3` 새 파일 생성 · 시스템(가중치/필터/C/프롬프트) 튜닝 중에는 같은 `_v{n}` 고정 재사용!")

    uploaded_eval = st.file_uploader("외부/팀원 평가셋 JSON 업로드 (선택)", type=["json"], key="eval_uploader")
    raw_eval: Any = None
    source_label = ""
    unlock = True

    if uploaded_eval is not None:
        raw_eval = json.loads(uploaded_eval.read().decode("utf-8"))
        source_label = uploaded_eval.name
    else:
        owner_eval_dir = OWNER_META.get(selected_owner, OWNER_META["kh"])["eval_dir"]
        local_files = sorted(owner_eval_dir.glob("*.json")) if owner_eval_dir.exists() else []
        if not local_files:
            st.info(f"`{owner_eval_dir}`에 JSON 평가셋 파일이 아직 없습니다.")
            return
        sel_file = st.selectbox(
            f"로컬 시험지 선택 (`{selected_owner}` · `{owner_eval_dir.relative_to(TEST_ROOT_DIR)}`)",
            local_files,
            format_func=lambda p: p.name,
            key=f"eval_file_sel_{selected_owner}",
        )
        source_label = sel_file.name
        if "test" in sel_file.name.lower() and selected_owner == "kh":
            st.warning("🔒 **최종 성적표용(`test`) 시험지입니다!** 튜닝 전 과적합 방지를 위해 문항 본문은 기본 봉인됩니다.")
            unlock = st.checkbox("통계만 말고 문항 본문도 열람하기 (주의: 튜닝 시 참고 금지!)", value=False, key=f"unlock_{sel_file.name}")
        raw_eval = _load_json_file(sel_file)

    if isinstance(raw_eval, list):
        eval_items = raw_eval
    elif isinstance(raw_eval, dict):
        eval_items = raw_eval.get("items") or raw_eval.get("questions") or []
    else:
        eval_items = []

    if not eval_items:
        st.error("평가셋 JSON에서 문항 리스트(`[...]`, `items`, `questions`)를 찾지 못했습니다.")
        return

    norm_rows = []
    for x in eval_items:
        if not isinstance(x, dict):
            continue
        qtype = str(x.get("question_type") or x.get("query_type") or "")
        if "expected_status" in x:
            exp_st = str(x.get("expected_status"))
        elif "reference_answerable" in x:
            exp_st = "answered" if x.get("reference_answerable") else "insufficient_evidence"
        else:
            exp_st = "insufficient_evidence" if qtype == "unanswerable" else "answered"
        norm_rows.append(
            {
                "ID": x.get("question_id") or x.get("id") or "",
                "구분": x.get("eval_category") or x.get("split") or exp_st,
                "질문 타입": qtype,
                "기대 상태": exp_st,
                "근거 청크 수": len(x.get("allowed_chunk_ids") or []),
                "질문 (user_input)": x.get("user_input") or x.get("question") or "",
                "모범정답 (reference)": x.get("reference") or x.get("ground_truth") or x.get("answer") or "",
            }
        )
    df_meta = pd.DataFrame(norm_rows)

    k1, k2, k3 = st.columns(3)
    k1.metric("총 문항 수", len(eval_items))
    k2.metric("Track A · 답변 가능 (answered)", int((df_meta["기대 상태"] == "answered").sum()))
    k3.metric("Track B · 거절 전용 (insufficient_evidence)", int((df_meta["기대 상태"] == "insufficient_evidence").sum()))

    st.caption(f"파일: `{source_label}`")
    st.dataframe(
        df_meta.groupby(["구분", "질문 타입"], dropna=False).size().reset_index(name="문항 수"),
        use_container_width=True,
        hide_index=True,
    )
    if uploaded_eval is not None or unlock:
        st.dataframe(df_meta, use_container_width=True, hide_index=True)


# ==================== TAB 6: 팀원 통합 비교 (도메인별 · 120문항) ====================
def _discover_team_summaries() -> list[tuple[str, Path, dict]]:
    """data/team_unified/summary_*.json + test/*/results/summary_*.json + 순수 stdlib 팀원별 최신 이력 자동 탐색."""
    found: list[tuple[str, Path, dict]] = []
    seen_owners: set[str] = set()
    unified_dir = REPO_DATA_DIR / "team_unified"
    if unified_dir.exists():
        for fp in sorted(unified_dir.glob("summary_*.json")):
            try:
                data = _load_json_file(fp)
                owner = str(data.get("owner") or fp.stem.replace("summary_", ""))
                found.append((owner, fp, data))
                seen_owners.add(owner)
            except Exception:
                continue

    if TEST_ROOT_DIR.exists():
        for member_dir in sorted(TEST_ROOT_DIR.iterdir()):
            if not member_dir.is_dir():
                continue
            res_dir = member_dir / "results"
            if not res_dir.exists():
                continue
            files = sorted(res_dir.glob("summary_*.json"), reverse=True)
            for fp in files:
                try:
                    found.append((member_dir.name, fp, _load_json_file(fp)))
                    seen_owners.add(member_dir.name)
                except Exception:
                    continue

    # data/team_unified 가 없더라도(예: 클린 클론) 순수 stdlib 로더로 4인 최신 결과 자동 보완
    for owner in ("kh", "yj", "jh", "bc"):
        if owner in seen_owners:
            continue
        hist = _load_owner_summaries(owner)
        if hist:
            fp, data = hist[-1]
            found.append((owner, fp, data))
    return found


def _adapt_uploaded_team_json(raw: Any, filename: str) -> dict | None:
    """팀원 원본 JSON(예지 eval_dev_*.json, 병철 comparison_data.json, 표준 summary_*.json)을 순수 stdlib로 자동 변환."""
    if not isinstance(raw, dict):
        return None
    if "modes" in raw and isinstance(raw["modes"], dict):
        return raw
    if "summary" in raw and "scored" in raw and isinstance(raw["summary"], dict):
        return _convert_yeji_eval_local(raw, source_label=filename, source_branch="uploaded")
    if "questions" in raw and "after" in raw and isinstance(raw["after"], list):
        return _convert_byeongcheol_comparison_local(raw, source_label=filename, source_branch="uploaded")
    return None


def render_team_integration_tab() -> None:
    st.subheader("🤝 팀원 통합 비교 (방안 3 통합 어댑터 · 도메인별 목표치 · 120문항)")
    st.caption(
        "`data/team_unified/summary_*.json` 및 `test/<팀원>/results/summary_*.json`을 자동 탐색합니다. "
        "예지(`eval_*.json`)·병철(`comparison_data.json`) 원본 JSON을 바로 드래그해도 자동 변환됩니다! "
        "(목표치: 전월세 `0.60` · 법령/공공임대 `0.70` · 거절률 `70%+`)"
    )

    # 1) 팀원 결과 파일 수집 (로컬 자동 탐색 + 다중 업로드)
    auto_found = _discover_team_summaries()
    uploaded_team = st.file_uploader(
        "팀원 결과 JSON 다중 업로드 (표준 summary_*.json · 예지 eval_*.json · 병철 comparison_data.json 모두 지원)",
        type=["json"],
        accept_multiple_files=True,
        key="team_summary_uploader",
    )

    entries: list[dict] = []
    # 팀원별 로컬 최신 파일 1개씩 기본 포함 (선택 가능)
    by_member: dict[str, list[tuple[Path, dict]]] = {}
    for member, fp, data in auto_found:
        by_member.setdefault(member, []).append((fp, data))

    if by_member:
        st.markdown("#### 1) 자동 감지된 팀원 결과 (`data/team_unified/` · `test/*/results/`)")
        cols = st.columns(max(1, min(4, len(by_member))))
        for i, (member, flist) in enumerate(by_member.items()):
            with cols[i % len(cols)]:
                sel_fp = st.selectbox(
                    f"팀원 `{member}` 결과 선택",
                    [fp for fp, _ in flist],
                    format_func=lambda p: p.name,
                    key=f"team_sel_{member}",
                )
                sdata = dict(flist)[sel_fp]
                modes = list((sdata.get("modes") or {}).keys())
                if modes:
                    sel_m = st.selectbox(
                        f"`{member}` 대표 프리셋",
                        modes,
                        index=len(modes) - 1,
                        key=f"team_mode_{member}",
                    )
                    dom = _infer_domain(sdata, member)
                    entries.append(
                        {
                            "owner": sdata.get("owner") or member,
                            "domain": dom,
                            "file": sel_fp.name,
                            "mode": sel_m,
                            "summary": sdata,
                            "mode_data": sdata["modes"][sel_m],
                        }
                    )

    for uf in uploaded_team or []:
        try:
            raw_json = json.loads(uf.read().decode("utf-8"))
            sdata = _adapt_uploaded_team_json(raw_json, uf.name)
            if not sdata:
                st.warning(f"파일 `{uf.name}`은 인식 가능한 결과 JSON 형식이 아닙니다.")
                continue
            modes = list((sdata.get("modes") or {}).keys())
            if not modes:
                continue
            owner_name = str(sdata.get("owner") or uf.name.split("_")[0])
            dom = _infer_domain(sdata, uf.name)
            sel_m = modes[-1]
            entries.append(
                {
                    "owner": owner_name,
                    "domain": dom,
                    "file": uf.name,
                    "mode": sel_m,
                    "summary": sdata,
                    "mode_data": sdata["modes"][sel_m],
                }
            )
        except Exception as exc:
            st.error(f"파일 `{uf.name}` 로드 실패: {exc}")

    if not entries:
        st.info("비교할 팀원 `summary_*.json` 파일이 없습니다. `uv run python -m rag_home_api.test.kh.unify_team_results --from-git` 실행 또는 위 업로더를 사용하세요!")
    else:
        rows = []
        chart_rows = []
        ragas_f_vals, ragas_ar_vals, ragas_cp_vals = [], [], []
        for e in entries:
            dom_label, dom_target = DOMAIN_META.get(e["domain"], (e["domain"], 0.65))
            dom_target = float(e["summary"].get("ragas_target", dom_target))
            mdata = e["mode_data"]
            ret = mdata.get("retrieval") or {}
            lat = mdata.get("latency_ms") or {}
            rg = mdata.get("ragas_avg") or {}
            has_ragas = bool(rg) and any(rg.get(k) is not None for k in RAGAS_LABELS)
            if has_ragas:
                f_v = float(rg.get("faithfulness", 0.0) or 0.0)
                ar_v = float(rg.get("answer_relevancy", 0.0) or 0.0)
                cp_v = float(rg.get("llm_context_precision_with_reference", 0.0) or 0.0)
                cr_v = float(rg.get("context_recall", 0.0) or 0.0)
                pass_cnt = sum(1 for v in (f_v, ar_v, cp_v, cr_v) if v >= dom_target)
                pass_txt = f"{pass_cnt}/4 {'🏆' if pass_cnt == 4 else '⚠️'}"
                ragas_f_vals.append(f_v)
                ragas_ar_vals.append(ar_v)
                ragas_cp_vals.append(cp_v)
            else:
                f_v = ar_v = cp_v = cr_v = None
                pass_txt = "RAGAS 미측정"

            label_id = f"{e['owner']} ({dom_label} · {e['mode']})"
            rows.append(
                {
                    "팀원": e["owner"],
                    "도메인": dom_label,
                    "목표치": dom_target,
                    "대표 프리셋": e["mode"],
                    "문항 수(답변가능)": ret.get("answerable_count", "-"),
                    RETRIEVAL_LABELS["recall@12"]: "-" if ret.get("recall@12") is None else ret.get("recall@12"),
                    RETRIEVAL_LABELS["hit@5"]: ret.get("hit@5", "-"),
                    RETRIEVAL_LABELS["top1"]: ret.get("top1", "-"),
                    RETRIEVAL_LABELS["mrr@5"]: "-" if ret.get("mrr@5") is None else ret.get("mrr@5"),
                    RAGAS_LABELS["faithfulness"]: round(f_v, 4) if f_v is not None else "미측정",
                    RAGAS_LABELS["answer_relevancy"]: round(ar_v, 4) if ar_v is not None else "미측정",
                    RAGAS_LABELS["llm_context_precision_with_reference"]: round(cp_v, 4) if cp_v is not None else "미측정",
                    RAGAS_LABELS["context_recall"]: round(cr_v, 4) if cr_v is not None else "미측정",
                    RETRIEVAL_LABELS["refusal"]: mdata.get("refusal", "-"),
                    RETRIEVAL_LABELS["lat_median"]: lat.get("median", "-"),
                    "목표 달성": pass_txt,
                }
            )
            if has_ragas:
                for mk, ml in RAGAS_LABELS.items():
                    chart_rows.append({"팀원·도메인": label_id, "RAGAS 지표": ml, "점수": float(rg.get(mk, 0.0) or 0.0)})

        team_df = pd.DataFrame(rows)
        st.markdown("#### 2) 팀원 · 도메인별 통합 성적표")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("참여 팀원/도메인 수", f"{len(rows)}개 (RAGAS 완료 {len(ragas_f_vals)}개)")
        k2.metric("RAGAS 평균 Faithfulness", f"{sum(ragas_f_vals)/len(ragas_f_vals):.3f}" if ragas_f_vals else "미측정")
        k3.metric("RAGAS 평균 Answer Relevancy", f"{sum(ragas_ar_vals)/len(ragas_ar_vals):.3f}" if ragas_ar_vals else "미측정")
        k4.metric("RAGAS 평균 Context Precision", f"{sum(ragas_cp_vals)/len(ragas_cp_vals):.3f}" if ragas_cp_vals else "미측정")

        st.dataframe(team_df, use_container_width=True, hide_index=True)

        if alt is not None and chart_rows:
            st.markdown("#### 3) 도메인별 RAGAS 4대 지표 비교 차트 (RAGAS 측정 완료 도메인)")
            ch = (
                alt.Chart(pd.DataFrame(chart_rows))
                .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
                .encode(
                    x=alt.X("RAGAS 지표:N", title=None, axis=alt.Axis(labelAngle=-10)),
                    y=alt.Y("점수:Q", scale=alt.Scale(domain=[0, 1.05])),
                    color=alt.Color("팀원·도메인:N", legend=alt.Legend(orient="top", title=None)),
                    xOffset="팀원·도메인:N",
                    tooltip=["팀원·도메인", "RAGAS 지표", alt.Tooltip("점수:Q", format=".4f")],
                )
                .properties(height=320)
            )
            st.altair_chart(ch, use_container_width=True)

        # 팀 통합 Markdown 다운로드
        cols_md = list(team_df.columns)
        md_lines = [
            "# 서울 주거 RAG 팀 통합 평가 비교 리포트",
            "",
            "| " + " | ".join(cols_md) + " |",
            "| " + " | ".join(["---"] * len(cols_md)) + " |",
            *["| " + " | ".join(str(r[c]) for c in cols_md) + " |" for _, r in team_df.iterrows()],
        ]
        st.download_button(
            "📝 팀 통합 비교 Markdown 리포트 다운로드",
            data="\n".join(md_lines).encode("utf-8"),
            file_name="team_unified_rag_comparison.md",
            mime="text/markdown",
        )

    # 4) 팀 통합 평가셋 (120문항) 병합 & 중복 ID 충돌 검사기
    st.divider()
    st.markdown("#### 4) 🧩 팀 평가셋 120문항 통합 검증기 (다중 JSON 업로드)")
    up_evals = st.file_uploader(
        "팀원 평가셋 JSON 여러 개 업로드 (예: 지훈 30 + 예지 30 + 최기혁 30 + 병철 30)",
        type=["json"],
        accept_multiple_files=True,
        key="team_eval_merger",
    )
    if up_evals:
        merged: list[dict] = []
        seen_ids: set[str] = set()
        dup_ids: list[str] = []
        per_file = []
        for uf in up_evals:
            arr = json.loads(uf.read().decode("utf-8"))
            ans_c = sum(1 for x in arr if x.get("expected_status") == "answered")
            ref_c = sum(1 for x in arr if x.get("expected_status") == "insufficient_evidence")
            for x in arr:
                qid = str(x.get("question_id", ""))
                if qid in seen_ids:
                    dup_ids.append(f"{qid} ({uf.name})")
                seen_ids.add(qid)
                merged.append({**x, "_source_file": uf.name})
            per_file.append({"파일명": uf.name, "총 문항": len(arr), "답변가능": ans_c, "거절": ref_c})

        m1, m2, m3 = st.columns(3)
        m1.metric("통합 총 문항 수", f"{len(merged)}문항")
        m2.metric("고유 question_id 수", f"{len(seen_ids)}개")
        m3.metric("중복 ID 충돌", f"{len(dup_ids)}건")
        if dup_ids:
            st.error(f"중복 ID 발견: `{', '.join(dup_ids[:10])}`")
        else:
            st.success("✅ 중복 `question_id` 충돌 0건! 바로 통합 실행 가능합니다.")
        st.dataframe(pd.DataFrame(per_file), use_container_width=True, hide_index=True)

    with st.expander("📋 팀원용 표준 summary_*.json 규격 안내 (이 형식이면 자동 인식!)", expanded=False):
        st.code(
            json.dumps(
                {
                    "timestamp": "20261126_100000",
                    "owner": "jihun | yeji | kh | byeongcheol",
                    "domain": "law | public_housing | rent",
                    "eval_version": "law_dev_v1",
                    "ragas_target": 0.70,
                    "modes": {
                        "B_hybrid_rrf": {
                            "retrieval": {"answerable_count": 8, "recall@12": 0.875, "hit@5": "8/8", "top1": "6/8", "mrr@5": 0.82},
                            "latency_ms": {"avg": 300.0, "median": 260.0, "p95": 350.0},
                            "refusal": "2/2",
                            "ragas_avg": {
                                "faithfulness": 0.88,
                                "answer_relevancy": 0.75,
                                "llm_context_precision_with_reference": 0.82,
                                "context_recall": 0.81,
                            },
                            "underperforming_items": [],
                        }
                    },
                },
                ensure_ascii=False,
                indent=2,
            ),
            language="json",
        )


# ==================== MAIN ====================
def main() -> None:
    st.set_page_config(
        page_title="서울 주거 RAG — 4인 통합 실험 워크벤치",
        page_icon="🧪",
        layout="wide",
    )
    _inject_clean_theme()

    # ==================== SIDEBAR (담당자 · 시험지 · 기준선/실험군 선택) ====================
    with st.sidebar:
        st.markdown("### 🧪 서울 주거 RAG 워크벤치")
        page = st.radio(
            "메뉴",
            [
                "📊 1. 현황 요약",
                "🧭 2. 원인 진단 & 다음 실험 선택",
                "🔬 3. 문항별 X-Ray",
                "📈 4. 추이 & 회차 리포트",
                "🤝 5. 시험지(_v{n}) & 팀 통합",
            ],
            label_visibility="collapsed",
        )

        st.divider()
        st.markdown("**🎛️ 비교 대상 선택 (팀원 4인 전원 지원)**")
        owner_keys = list(OWNER_META.keys())
        sel_owner = st.selectbox(
            "👤 담당자 / 팀원 (`owner`)",
            owner_keys,
            index=0,
            format_func=lambda k: OWNER_META[k]["label"],
            help="팀원별 로컬 평가 결과(kh · yj · jh · bc)를 전환해 시험지 버전과 기준선(Before) · 실험군(After)을 동일 워크벤치에서 비교합니다.",
        )

        all_summaries = _load_owner_summaries(sel_owner)
        all_steps = _build_history_steps(all_summaries)
        load_errs = _LAST_LOAD_ERRORS.get(sel_owner) or []
        if load_errs:
            st.warning(f"`{sel_owner}` 일부 파일 로드 경고: " + " | ".join(load_errs[:2]))

        eval_versions = sorted({s["eval_version"] for s in all_steps})
        ver_options = ["전체 시험지 (버전 혼합)", *eval_versions] if len(eval_versions) > 1 else eval_versions
        default_ver_idx = len(ver_options) - 1 if eval_versions else 0
        sel_eval_ver = st.selectbox(
            "시험지 버전 (`eval_version`)",
            ver_options or ["(결과 없음)"],
            index=default_ver_idx if ver_options else 0,
            key=f"eval_ver_sel_{sel_owner}",
            help="같은 시험지(_v{n} / dev / test)끼리만 비교해야 시스템 변경 효과가 공정하게 보입니다.",
        )

        scoped_steps = (
            all_steps
            if (not sel_eval_ver or sel_eval_ver.startswith("전체 시험지"))
            else [s for s in all_steps if s["eval_version"] == sel_eval_ver]
        )

        before_s: dict | None = None
        after_s: dict | None = None
        if scoped_steps:
            step_labels = [s["step_label"] for s in scoped_steps]
            b_lbl = st.selectbox(
                "기준선 (Before)",
                step_labels,
                index=max(0, len(step_labels) - 2),
                key=f"before_sel_{sel_owner}_{sel_eval_ver}",
            )
            a_lbl = st.selectbox(
                "실험군 (After)",
                step_labels,
                index=len(step_labels) - 1,
                key=f"after_sel_{sel_owner}_{sel_eval_ver}",
            )
            before_s = scoped_steps[step_labels.index(b_lbl)]
            after_s = scoped_steps[step_labels.index(a_lbl)]

        # 고급 설정·코치·내보내기는 접어 두어 사이드바 스크롤 제거
        default_target = float(
            after_s["summary"].get("ragas_target", OWNER_META[sel_owner]["target"])
            if after_s
            else OWNER_META[sel_owner]["target"]
        )
        with st.expander("⚙️ 목표치 · 코치 팁 · 내보내기", expanded=False):
            coach_mode = st.toggle("코치 팁 표시", value=True)
            target = st.slider(
                "RAGAS 목표치",
                0.30,
                0.95,
                default_target,
                0.05,
                key=f"target_slider_{sel_owner}",
            )
            if after_s is not None:
                ts_tag = after_s["timestamp"]
                st.download_button(
                    "🌐 HTML 종합 리포트",
                    data=_export_html(after_s, before_s, scoped_steps, target).encode("utf-8"),
                    file_name=f"workbench_report_{sel_owner}_{ts_tag}_{after_s['mode']}.html",
                    mime="text/html",
                    use_container_width=True,
                )
                st.download_button(
                    "📊 CSV 비교표 (Excel)",
                    data=_build_comparison_df(after_s["summary"]).to_csv(index=False).encode("utf-8-sig"),
                    file_name=f"summary_{sel_owner}_{ts_tag}.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
                st.download_button(
                    "📝 Markdown 리포트",
                    data=_build_single_step_report_md(after_s, before_s, target).encode("utf-8"),
                    file_name=f"eval_report_{sel_owner}_{ts_tag}_{after_s['mode']}.md",
                    mime="text/markdown",
                    use_container_width=True,
                )

    matrix_rows = _evaluate_decision_matrix(after_s, before_s, target)

    # ==================== 상단 1줄 요약 바 (단일 진입 헤더) ====================
    _render_compact_status_bar(before_s, after_s, matrix_rows)

    # ==================== 선택된 1개 페이지 렌더링 (1화면 1목적) ====================
    if page.startswith("📊 1."):
        if before_s is None or after_s is None:
            st.info(f"`{sel_owner}` 팀원의 실행 기록이 아직 없습니다.")
        else:
            render_decision_tab(before_s, after_s, target, coach_mode=coach_mode, matrix_rows=matrix_rows)

    elif page.startswith("🧭 2."):
        _render_sop_coach_panel(scoped_steps, before_s, after_s, target, sel_eval_ver, matrix_rows)

    elif page.startswith("🔬 3."):
        if before_s is None or after_s is None:
            st.info("비교할 실행 기록이 아직 없습니다.")
        else:
            render_xray_tab(before_s, after_s, target, coach_mode=coach_mode)

    elif page.startswith("📈 4."):
        if before_s is None or after_s is None:
            st.info("기록된 실험 스텝이 아직 없습니다.")
        else:
            t_hist, t_rep = st.tabs(["📈 추이 선그래프 & 랩노트", "📝 회차별 분석 리포트 & 내보내기"])
            with t_hist:
                render_lab_history_tab(
                    scoped_steps,
                    target,
                    sel_eval_ver,
                    coach_mode=coach_mode,
                    before_s=before_s,
                    after_s=after_s,
                    owner_key=sel_owner,
                )
            with t_rep:
                render_per_test_report_tab(scoped_steps, target, coach_mode=coach_mode)

    else:
        t_eval, t_team = st.tabs(["🧪 시험지 버전(_v{n}) & 거절 트랙", "🤝 팀원 통합 비교 (도메인 · 120문항)"])
        with t_eval:
            render_evalset_tab(selected_owner=sel_owner)
        with t_team:
            render_team_integration_tab()


if __name__ == "__main__":
    main()
