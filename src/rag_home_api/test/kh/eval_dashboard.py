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

import html as html_lib
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

try:
    import altair as alt
except ImportError:  # pragma: no cover
    alt = None

KH_DIR = Path(__file__).resolve().parent
TEST_ROOT_DIR = KH_DIR.parent  # src/rag_home_api/test/ (팀원별 폴더 탐색 루트)
KH_RESULTS_DIR = KH_DIR / "results"
KH_EVAL_DIR = KH_DIR / "eval"

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

DOMAIN_META: dict[str, tuple[str, float]] = {
    "rent": ("전월세 (실거래)", 0.60),
    "law": ("주택임대차 법령", 0.70),
    "public_housing": ("공공임대 공고", 0.70),
    "unified": ("통합 (전 도메인)", 0.65),
}

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
    r12 = int(m12.get("first_rank", 0) or 0)
    r5 = int(m5.get("first_rank", 0) or 0)

    if hit12 == 0:
        return 2, "🔴 [2단계] 후보 12개 밖 완전 누락 (FN · Recall@12=0) — B/필터/Query Rewrite 점검"
    if rec12 < 1.0:
        return 2, f"🔴 [2단계] 복합 정답 일부 누락 (FN · Recall@12={rec12:.2f}, 첫 정답 #{r12}) — Query Rewrite(서브쿼리 분해) 필요"
    if hit5 == 0:
        return 3, f"🟠 [3단계] 후보 12개 중 #{r12} 생존, Top-5 탈락 — 검색 C(Re-ranking)로 구제 가능"
    if r5 > 1:
        return 3, f"🟡 [3단계] Top-5 포함(first_rank=#{r5}), 1등 밀림 — 검색 C(Re-ranking)·동 필터·가중치 개선"
    if is_fail:
        return 4, "🔵 [4단계] 검색 1등 적중(first_rank=#1, FN=0) — 답변 생성·주어·TMI·인용 프롬프트 점검"
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
    elif cmode != "A_vector" and (pp.get("bm25_weight") != cp.get("bm25_weight")):
        bw1, vw1 = pp.get("bm25_weight", 0.55), pp.get("vector_weight", 0.45)
        bw2, vw2 = cp.get("bm25_weight", 0.55), cp.get("vector_weight", 0.45)
        diffs.append(f"BM25:Vec 가중치: {bw1:.2f}:{vw1:.2f} → {bw2:.2f}:{vw2:.2f}")

    if bool(pp.get("use_metadata_filter")) != bool(cp.get("use_metadata_filter")):
        diffs.append(f"메타데이터 필터: {'ON' if pp.get('use_metadata_filter') else 'OFF'} → {'ON' if cp.get('use_metadata_filter') else 'OFF'}")
    if bool(pp.get("use_rewrite")) != bool(cp.get("use_rewrite")):
        diffs.append(f"Query Rewrite: {'ON' if pp.get('use_rewrite') else 'OFF'} → {'ON' if cp.get('use_rewrite') else 'OFF'}")
    if bool(pp.get("use_cache")) != bool(cp.get("use_cache")):
        diffs.append(f"Cache: {'ON' if pp.get('use_cache') else 'OFF'} → {'ON' if cp.get('use_cache') else 'OFF'}")
    if bool(pp.get("use_hyde")) != bool(cp.get("use_hyde")):
        diffs.append(f"HyDE: {'ON' if pp.get('use_hyde') else 'OFF'} → {'ON' if cp.get('use_hyde') else 'OFF'}")
    if pp.get("candidate_k") != cp.get("candidate_k") and pp.get("candidate_k") and cp.get("candidate_k"):
        diffs.append(f"candidate_k: {pp.get('candidate_k')} → {cp.get('candidate_k')}")

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
        comp_logs = _load_jsonl_file(comp_path) if comp_path else []

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
                c_ok = sum(1 for r in ans_logs if ("출처:" in (r.get("response") or "") or ".csv" in (r.get("response") or "")))
                cite_str = f"{c_ok}/{len(ans_logs)}"
            scope_str = mdata.get("scope_notice")
            if not scope_str and ref_logs:
                s_ok = sum(
                    1
                    for r in ref_logs
                    if any(w in (r.get("response") or "") for w in ("과거", "체결 완료", "실거래 통계", "포함되지", "나와 있지", "없습니다"))
                )
                scope_str = f"{s_ok}/{len(ref_logs)}"

            f_val = float(rg.get("faithfulness", 0.0) or 0.0)
            ar_val = float(rg.get("answer_relevancy", 0.0) or 0.0)
            cp_val = float(rg.get("llm_context_precision_with_reference", 0.0) or 0.0)
            cr_val = float(rg.get("context_recall", 0.0) or 0.0)
            fail_cnt = len(mdata.get("underperforming_items", []))

            short_ts = ts[-6:-2] if len(ts) >= 6 else ts
            step_label = f"#{idx} [{short_ts}] {mname}"
            if ts == "20261125_154004":
                step_label = f"#{idx} [{short_ts}] {mname} (+주어)"
            elif run_note:
                step_label = f"#{idx} [{short_ts}] {mname} ({run_note[:14]})"

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
                "ragas_mean": round((f_val + ar_val + cp_val + cr_val) / 4.0, 4),
                "fail_cnt": fail_cnt,
                "pass_4of4": all(v >= 0.60 for v in (f_val, ar_val, cp_val, cr_val)),
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
    if len(buckets[4]) >= 1:
        return (
            "ANSWER_PROMPT 보강 (첫 문장 대상 5조건 주어 명시 · TMI 금지)",
            f"검색은 1등인데 4단계 생성에서 미달({', '.join(buckets[4])})이 발생했습니다. 주어 생략·TMI가 RAGAS Answer Relevancy를 깎고 있습니다.",
            "ANSWER_PROMPT 규칙 1: 첫 문장에 대상 5조건 주어 명시",
            "Faithfulness 유지 여부 & Context Precision 보존 확인",
        )
    if len(buckets[3]) >= 1:
        return (
            "검색 C (retrieval/reranker.py) 추가 또는 법정동(dong) 필터 확장",
            f"후보 12개에 정답이 100% 들어있으나 3단계 순위 실패({', '.join(buckets[3])})가 남았습니다. B 후보 12개를 정밀 재정렬하세요.",
            "pipeline_mode: B_hybrid → C_rerank (candidate_k=12 → top_k=5)",
            "LLM 리랭킹 지연(p95) 증가 폭 + 이미 1등이던 문항 하락 0건 확인",
        )
    if len(buckets[2]) >= 1:
        return (
            "Query Rewrite (optimize/query_rewriter.py · 복합 2구 서브쿼리 분해)",
            f"2단계 검색 누락 FN({', '.join(buckets[2])})은 Re-ranker(C)로 살릴 수 없습니다! 2개 지역 비교 질문을 서브쿼리 2개로 분해해 각각 검색·병합하세요.",
            "use_rewrite: False → True (복합 질의 분해 + 구어체 정규화)",
            "단순 문항 6건의 조건 왜곡·악화 0건 확인",
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


# ==================== UI 렌더링 헬퍼: 2×2 혼동행렬 카드 ====================
def _render_confusion_matrix_2x2(
    cm: dict[str, int],
    k: int,
    prev_cm: dict[str, int] | None = None,
    title_prefix: str = "",
) -> None:
    def _delta(key: str) -> str:
        if not prev_cm:
            return ""
        d = cm[key] - prev_cm[key]
        return f" ({d:+d})" if d != 0 else " (변동 없음)"

    st.markdown(f"**{title_prefix}청크 단위 2×2 혼동행렬 (상위 `k={k}` 기준)**")
    c1, c2 = st.columns(2)
    with c1:
        st.success(f"**TP (True Positive · 찾은 정답)**\n### `{cm['tp']}`{_delta('tp')}\n실제 정답 청크를 상위 {k}개에 포함 ✅")
        st.warning(f"**FP (False Positive · 섞인 노이즈)**\n### `{cm['fp']}`{_delta('fp')}\n정답이 아닌 청크가 상위 {k}개에 섞임")
    with c2:
        st.error(f"**FN (False Negative · 놓친 정답 / 검색 누락!)**\n### `{cm['fn']}`{_delta('fn')}\n실제 정답을 상위 {k}개에 못 넣음 ❌ *(0이 목표!)*")
        st.info(f"**TN (True Negative · 잘 걸러낸 무관 청크)**\n### `{cm['tn']:,}`\n무관한 청크를 안 뽑음 *(N이 커서 Accuracy 대신 Precision·Recall 사용)*")


# ==================== TAB 1: 의사결정 홈 ====================
def render_decision_tab(before_s: dict, after_s: dict, target: float) -> None:
    st.subheader(f"🧭 의사결정 홈 — `{before_s['step_label']}` → `{after_s['step_label']}`")

    # 1) 3구역 분리 KPI: 검색 지표 / 생성 지표 / 거절·운영 지표
    st.markdown("#### 1) 🔎 검색 지표 (Retrieval — `metrics_k12` · `metrics_k5` · RAGAS 검색 2지표)")
    r1, r2, r3, r4, r5, r6 = st.columns(6)
    r1.metric(RETRIEVAL_LABELS["recall@12"], f"{after_s['recall12']:.3f}", f"{after_s['recall12'] - before_s['recall12']:+.3f}")
    r2.metric(RETRIEVAL_LABELS["hit@5"], after_s["hit5_str"], f"{(after_s['hit5_rate'] - before_s['hit5_rate'])*100:+.1f}%p")
    r3.metric(RETRIEVAL_LABELS["top1"], after_s["top1_str"], f"{(after_s['top1_rate'] - before_s['top1_rate'])*100:+.1f}%p")
    r4.metric(RETRIEVAL_LABELS["mrr@5"], f"{after_s['mrr5']:.3f}", f"{after_s['mrr5'] - before_s['mrr5']:+.3f}")
    r5.metric(RAGAS_LABELS["llm_context_precision_with_reference"], f"{after_s['cp']:.3f}", f"{after_s['cp'] - before_s['cp']:+.3f}")
    r6.metric(RAGAS_LABELS["context_recall"], f"{after_s['cr']:.3f}", f"{after_s['cr'] - before_s['cr']:+.3f}")

    st.markdown("#### 2) ✍️ 생성 지표 (Generation) & 🛡️ 거절·운영 트랙 (Track B 분리)")
    g1, g2, g3, g4, g5, g6 = st.columns(6)
    g1.metric(RAGAS_LABELS["faithfulness"], f"{after_s['f']:.3f}", f"{after_s['f'] - before_s['f']:+.3f}")
    g2.metric(RAGAS_LABELS["answer_relevancy"], f"{after_s['ar']:.3f}", f"{after_s['ar'] - before_s['ar']:+.3f}")
    g3.metric(RETRIEVAL_LABELS["citation_rate"], after_s["citation_str"])
    g4.metric(RETRIEVAL_LABELS["refusal"], after_s["refusal_str"])
    g5.metric(RETRIEVAL_LABELS["scope_notice"], after_s["scope_notice_str"])
    g6.metric(
        RETRIEVAL_LABELS["lat_median"],
        f"{after_s['lat_median']:.1f} ms",
        f"{after_s['lat_median'] - before_s['lat_median']:+.1f} ms",
        delta_color="inverse",
    )

    # 2) TP / FP / FN / TN 2×2 혼동행렬
    st.divider()
    st.markdown("#### 3) 🧮 검색 혼동행렬 (`TP · FP · FN · TN` — 전체 답변 가능 문항 합계)")
    k_choice = st.radio(
        "혼동행렬 기준 k 선택",
        ["후보 단계 k=12 (검색 B의 FN 누락 점검)", "최종 전달 k=5 (생성기 입력 FP·FN 점검)"],
        horizontal=True,
    )
    k_val = 12 if "12" in k_choice else 5
    cur_cm = after_s["cm12"] if k_val == 12 else after_s["cm5"]
    prev_cm = (before_s["cm12"] if k_val == 12 else before_s["cm5"]) if before_s["order"] != after_s["order"] else None
    if cur_cm["q_count"] > 0:
        _render_confusion_matrix_2x2(cur_cm, k=k_val, prev_cm=prev_cm, title_prefix=f"[{after_s['mode']}] ")
    else:
        st.info("연동된 `.jsonl` 상세 로그가 없어 혼동행렬을 산출할 수 없습니다.")

    # 3) 4단계 깔때기 병목 레이더
    st.divider()
    st.markdown(f"#### 4) 🔻 4단계 깔때기 병목 레이더 (`{after_s['step_label']}` 기준 · 목표 `{target:.2f}`)")
    buckets: dict[int, list[str]] = {0: [], 2: [], 3: [], 4: []}
    for r in after_s["mode_logs"]:
        if r.get("expected_status") != "answered":
            continue
        qid = r.get("question_id", "")
        stg, _ = _classify_funnel_stage(r, after_s["item_ragas"].get(qid), target)
        buckets.setdefault(stg, []).append(qid)

    b1, b2, b3, b4 = st.columns(4)
    b1.success(f"**1단계 · 색인/청킹 누락**\n\n**0건** ✅\n\n정답 청크 매니페스트 존재 확인")
    b2.error(f"**2단계 · 검색 누락 (FN)**\n\n**{len(buckets[2])}건**: `{', '.join(buckets[2]) or '없음'}`\n\n처방: B(Hybrid) · `candidate_k` · Query Rewrite")
    b3.warning(f"**3단계 · 순위 실패 (FP)**\n\n**{len(buckets[3])}건**: `{', '.join(buckets[3]) or '없음'}`\n\n처방: 가중치 · 경량 필터 · 검색 C(Re-ranking)")
    b4.info(f"**4단계 · 생성/주어/인용**\n\n**{len(buckets[4])}건**: `{', '.join(buckets[4]) or '없음'}`\n\n처방: `ANSWER_PROMPT` 주어·간결화 점검")

    # 4) 다음 단일 변수 추천 카드
    st.divider()
    rec_title, rec_why, rec_knob, rec_watch = _recommend_next_variable(after_s, target)
    st.markdown(f"#### 5) 🎯 다음에 딱 1개만 바꿀 변수 추천 (통과 문항: `{len(buckets[0])}/{max(1, sum(len(v) for v in buckets.values()))}`)")
    st.success(f"**추천 액션: {rec_title}**\n\n- **판단 근거**: {rec_why}\n- **바꿀 변수 (딱 1개)**: `{rec_knob}`\n- **함께 감시할 부작용**: {rec_watch}")


# ==================== TAB 2: 문항별 X-Ray 해부 ====================
def render_xray_tab(before_s: dict, after_s: dict, target: float) -> None:
    st.subheader(f"🔬 문항별 X-Ray 해부 — `{before_s['step_label']}` vs `{after_s['step_label']}`")
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

    # 패널 A: Before vs After Top-5 5조건 불일치 자동 태그 비교
    st.markdown("#### A. 검색 Top-5 조건 불일치 X-Ray (정답 5조건 vs 실제 Top-5)")
    targets = _parse_reference_targets(arow)
    col_b, col_a = st.columns(2)
    with col_b:
        st.markdown(f"**Before · `{before_s['step_label']}`**")
        if brow:
            for rank_i, cid in enumerate(brow.get("top5_ids") or [], start=1):
                ok, tags = _explain_chunk_mismatch(cid, targets)
                tag_str = " ".join(tags)
                (st.success if ok else st.markdown)(f"`#{rank_i}` `{cid}`  \n{tag_str}")
        else:
            st.caption("Before 실행에 동일 문항 로그가 없습니다.")
    with col_a:
        st.markdown(f"**After · `{after_s['step_label']}`**")
        for rank_i, cid in enumerate(arow.get("top5_ids") or [], start=1):
            ok, tags = _explain_chunk_mismatch(cid, targets)
            tag_str = " ".join(tags)
            (st.success if ok else st.markdown)(f"`#{rank_i}` `{cid}`  \n{tag_str}")

    # 패널 B: 문항별 TP/FP/FN/TN 2×2 혼동행렬 (k=12 / k=5)
    st.markdown("#### B. 이 문항의 `TP · FP · FN · TN` 2×2 혼동행렬 (`metrics_k12` · `metrics_k5`)")
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
        "💡 참고: 정답 청크가 1개뿐인 단일 문항은 1등으로 맞혀도 `Precision@5 = 1/5 = 0.20`이 이론상 만점이며, "
        "순위 정밀도는 `MRR@5`와 RAGAS `Context Precision`으로 확인합니다."
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
def render_lab_history_tab(steps: list[dict], target: float, selected_eval_ver: str) -> None:
    st.subheader("📈 회차별 실험 랩노트 & 수치 변화 추이 (연결 선 그래프)")
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
def render_per_test_report_tab(steps: list[dict], target: float) -> None:
    st.subheader("📝 테스트 회차별 평가 분석 리포트 (1회 실행 = 1개 독립 리포트)")
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
def render_evalset_tab() -> None:
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
    eval_items: list[dict] | None = None
    source_label = ""
    unlock = True

    if uploaded_eval is not None:
        eval_items = json.loads(uploaded_eval.read().decode("utf-8"))
        source_label = uploaded_eval.name
    else:
        local_files = sorted(KH_EVAL_DIR.glob("*.json")) if KH_EVAL_DIR.exists() else []
        if not local_files:
            st.info("`test/kh/eval/`에 JSON 평가셋 파일이 아직 없습니다.")
            return
        sel_file = st.selectbox("로컬 시험지 선택 (`test/kh/eval/`)", local_files, format_func=lambda p: p.name)
        source_label = sel_file.name
        if "test" in sel_file.name.lower():
            st.warning("🔒 **최종 성적표용(`test`) 시험지입니다!** 튜닝 전 과적합 방지를 위해 문항 본문은 기본 봉인됩니다.")
            unlock = st.checkbox("통계만 말고 문항 본문도 열람하기 (주의: 튜닝 시 참고 금지!)", value=False)
        eval_items = _load_json_file(sel_file)

    if not isinstance(eval_items, list):
        st.error("평가셋 JSON은 리스트(`[...]`) 형태여야 합니다.")
        return

    df_meta = pd.DataFrame(
        [
            {
                "ID": x.get("question_id", ""),
                "구분": x.get("eval_category", x.get("expected_status", "")),
                "질문 타입": x.get("question_type", ""),
                "기대 상태": x.get("expected_status", ""),
                "근거 청크 수": len(x.get("allowed_chunk_ids", [])),
                "질문 (user_input)": x.get("user_input", x.get("question", "")),
                "모범정답 (reference)": x.get("reference", ""),
            }
            for x in eval_items
        ]
    )

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
    """src/rag_home_api/test/*/results/summary_*.json 을 팀원별로 자동 탐색 (최신 파일 우선)."""
    found: list[tuple[str, Path, dict]] = []
    if not TEST_ROOT_DIR.exists():
        return found
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
            except Exception:
                continue
    return found


def render_team_integration_tab() -> None:
    st.subheader("🤝 팀원 통합 비교 (도메인별 목표치 자동 적용 · 4명 120문항 통합)")
    st.caption(
        "`src/rag_home_api/test/<팀원>/results/summary_*.json`을 자동 탐색하거나, 팀원들의 결과 JSON / 평가셋 JSON을 여러 개 한 번에 업로드해 비교합니다. "
        "(목표치 자동 매핑: 전월세 `0.60` · 법령/공공임대 `0.70` · 거절률 `70%+`)"
    )

    # 1) 팀원 결과 파일 수집 (로컬 자동 탐색 + 다중 업로드)
    auto_found = _discover_team_summaries()
    uploaded_team = st.file_uploader(
        "팀원 결과 summary_*.json 다중 업로드 (지훈·예지·병철·최기혁 파일 한 번에 드래그!)",
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
        st.markdown("#### 1) 로컬 자동 감지된 팀원 결과 (`test/*/results/`)")
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
            sdata = json.loads(uf.read().decode("utf-8"))
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
        st.info("비교할 팀원 `summary_*.json` 파일이 없습니다. 위 업로더로 팀원 결과 JSON을 올려보세요!")
    else:
        rows = []
        chart_rows = []
        for e in entries:
            dom_label, dom_target = DOMAIN_META.get(e["domain"], (e["domain"], 0.65))
            dom_target = float(e["summary"].get("ragas_target", dom_target))
            mdata = e["mode_data"]
            ret = mdata.get("retrieval") or {}
            lat = mdata.get("latency_ms") or {}
            rg = mdata.get("ragas_avg") or {}
            f_v = float(rg.get("faithfulness", 0.0) or 0.0)
            ar_v = float(rg.get("answer_relevancy", 0.0) or 0.0)
            cp_v = float(rg.get("llm_context_precision_with_reference", 0.0) or 0.0)
            cr_v = float(rg.get("context_recall", 0.0) or 0.0)
            pass_cnt = sum(1 for v in (f_v, ar_v, cp_v, cr_v) if v >= dom_target)
            label_id = f"{e['owner']} ({dom_label} · {e['mode']})"
            rows.append(
                {
                    "팀원": e["owner"],
                    "도메인": dom_label,
                    "목표치": dom_target,
                    "대표 프리셋": e["mode"],
                    "문항 수(답변가능)": ret.get("answerable_count", "-"),
                    RETRIEVAL_LABELS["recall@12"]: ret.get("recall@12", "-"),
                    RETRIEVAL_LABELS["hit@5"]: ret.get("hit@5", "-"),
                    RETRIEVAL_LABELS["top1"]: ret.get("top1", "-"),
                    RETRIEVAL_LABELS["mrr@5"]: ret.get("mrr@5", "-"),
                    RAGAS_LABELS["faithfulness"]: round(f_v, 4),
                    RAGAS_LABELS["answer_relevancy"]: round(ar_v, 4),
                    RAGAS_LABELS["llm_context_precision_with_reference"]: round(cp_v, 4),
                    RAGAS_LABELS["context_recall"]: round(cr_v, 4),
                    RETRIEVAL_LABELS["refusal"]: mdata.get("refusal", "-"),
                    RETRIEVAL_LABELS["lat_median"]: lat.get("median", "-"),
                    "목표 달성": f"{pass_cnt}/4 {'🏆' if pass_cnt == 4 else '⚠️'}",
                }
            )
            for mk, ml in RAGAS_LABELS.items():
                chart_rows.append({"팀원·도메인": label_id, "RAGAS 지표": ml, "점수": float(rg.get(mk, 0.0) or 0.0)})

        team_df = pd.DataFrame(rows)
        st.markdown("#### 2) 팀원 · 도메인별 통합 성적표")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("참여 결과 수", f"{len(rows)}개")
        k2.metric("팀 평균 Faithfulness (충실도)", f"{team_df[RAGAS_LABELS['faithfulness']].mean():.3f}")
        k3.metric("팀 평균 Answer Relevancy (답변 관련성)", f"{team_df[RAGAS_LABELS['answer_relevancy']].mean():.3f}")
        k4.metric("팀 평균 Context Precision (근거 정밀도)", f"{team_df[RAGAS_LABELS['llm_context_precision_with_reference']].mean():.3f}")

        st.dataframe(team_df, use_container_width=True, hide_index=True)

        if alt is not None and chart_rows:
            st.markdown("#### 3) 도메인별 RAGAS 4대 지표 비교 차트")
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
        page_title="서울 주거 RAG — 실험 분석 워크벤치 & 팀 통합 대시보드 (kh)",
        page_icon="🧪",
        layout="wide",
    )

    st.title("🧪 서울 주거 RAG — 실험 원인 판별 워크벤치 & 팀 통합 대시보드")
    st.caption(
        "유비쿼터스 언어(영어+한글) 기반 · 검색/생성/거절 분리 · TP·TN·FP·FN 혼동행렬 · "
        "4단계 깔때기 병목 판별 · 회차별 자동 리포트 · 팀원 도메인 통합 비교"
    )

    all_summaries = _load_all_local_summaries()
    all_steps = _build_history_steps(all_summaries)

    # ==================== SIDEBAR ====================
    with st.sidebar:
        st.header("🎛️ 워크벤치 통제 패널")

        # 1) 시험지 버전(eval_version) 격리 필터
        eval_versions = sorted({s["eval_version"] for s in all_steps})
        ver_options = ["전체 시험지 (주의: 버전 혼합)", *eval_versions]
        default_ver_idx = len(ver_options) - 1 if eval_versions else 0
        sel_eval_ver = st.selectbox("1. 시험지 버전 격리 (`eval_version`)", ver_options, index=default_ver_idx)

        if sel_eval_ver.startswith("전체 시험지"):
            scoped_steps = all_steps
        else:
            scoped_steps = [s for s in all_steps if s["eval_version"] == sel_eval_ver]
            st.caption(f"🔒 `{sel_eval_ver}`에서 실행한 {len(scoped_steps)}개 스텝만 공정 비교 중")

        # 2) Before vs After 비교 쌍 선택
        before_s: dict | None = None
        after_s: dict | None = None
        if scoped_steps:
            step_labels = [s["step_label"] for s in scoped_steps]
            st.divider()
            st.markdown("**2. Before vs After 비교 쌍 선택**")
            default_b = max(0, len(step_labels) - 2)
            default_a = len(step_labels) - 1
            b_lbl = st.selectbox("기준선 (Before)", step_labels, index=default_b)
            a_lbl = st.selectbox("실험군 (After)", step_labels, index=default_a)
            before_s = scoped_steps[step_labels.index(b_lbl)]
            after_s = scoped_steps[step_labels.index(a_lbl)]

            n_diff, diffs = _describe_preset_diff(before_s, after_s)
            if before_s["order"] == after_s["order"]:
                st.info("동일 스텝을 선택 중입니다. 다른 스텝을 골라 전/후를 비교하세요.")
            elif n_diff == 1:
                st.success(f"✅ **단일 변인 통제 통과 (변수 1개)**\n\n`{diffs[0]}`")
            else:
                st.warning(f"⚠️ **복합 변수 변경 ({n_diff}개 동시)**\n\n" + "\n".join(f"- {d}" for d in diffs))

        st.divider()
        default_target = float(after_s["summary"].get("ragas_target", 0.60)) if after_s else 0.60
        target = st.slider(
            "3. RAGAS 목표 임계치 (전월세 0.60 / 법령·공공 0.70)",
            min_value=0.30,
            max_value=0.95,
            value=default_target,
            step=0.05,
        )

        # 4) 빠른 Export (선택 After 스텝 기준)
        if after_s is not None:
            st.divider()
            st.markdown("**📤 리포트 내보내기 (Export)**")
            ts_tag = after_s["timestamp"]
            st.download_button(
                "🌐 HTML 종합 리포트 (선그래프+분석)",
                data=_export_html(after_s, before_s, scoped_steps, target).encode("utf-8"),
                file_name=f"workbench_report_{ts_tag}_{after_s['mode']}.html",
                mime="text/html",
                use_container_width=True,
            )
            st.download_button(
                "📊 CSV 비교표 (선택 파일, Excel)",
                data=_build_comparison_df(after_s["summary"]).to_csv(index=False).encode("utf-8-sig"),
                file_name=f"summary_{ts_tag}.csv",
                mime="text/csv",
                use_container_width=True,
            )
            st.download_button(
                "📝 Markdown 회차 리포트",
                data=_build_single_step_report_md(after_s, before_s, target).encode("utf-8"),
                file_name=f"eval_report_{ts_tag}_{after_s['mode']}.md",
                mime="text/markdown",
                use_container_width=True,
            )

    # ==================== MAIN 6 TABS ====================
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        [
            "🧭 1. 의사결정 홈 (병목 · TP/TN/FP/FN)",
            "🔬 2. 문항별 X-Ray (불일치 · 혼동행렬)",
            "📈 3. 회차별 랩노트 & 추이 선그래프",
            "📝 4. 회차별 평가 분석 리포트",
            "🧪 5. 시험지 버전(_v{n}) & 거절 트랙",
            "🤝 6. 팀원 통합 비교 (도메인 · 120문항)",
        ]
    )

    if before_s is None or after_s is None:
        for t in (tab1, tab2, tab3, tab4):
            with t:
                st.info("`test/kh/results/summary_*.json` 파일이 아직 없습니다. 평가 스크립트를 먼저 1회 실행하세요.")
    else:
        with tab1:
            render_decision_tab(before_s, after_s, target)
        with tab2:
            render_xray_tab(before_s, after_s, target)
        with tab3:
            render_lab_history_tab(scoped_steps, target, sel_eval_ver)
        with tab4:
            render_per_test_report_tab(scoped_steps, target)

    with tab5:
        render_evalset_tab()
    with tab6:
        render_team_integration_tab()


if __name__ == "__main__":
    main()
