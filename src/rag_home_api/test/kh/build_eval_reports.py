"""[전월세 RAG 맞춤형 평가 리포트 생성기] results/summary_*.json + exp_*.jsonl -> reports/<split>/*.md 자동 생성.

작성자: 최기혁 (kh)
기능:
  - 추가 API 호출 0원(100% 오프라인)으로 이미 저장된 요약 JSON · 원본 JSONL · 시험지 JSON을 결합해
    사람/PR/발표 열람용 개별 Markdown 평가 리포트(6개 섹션)와 reports/README.md 종합 목차를 생성한다.
  - exp_03_rent_eval.py 실행 종료 시 자동으로 호출되어 향후 모든 실험(C, Rewrite, Cache, HyDE, 봉인 test)의
    개별 리포트와 목차를 자동 누적한다.

단독 실행 (전체 11회차 일괄 생성 / 재동기화):
    uv run python -m rag_home_api.test.kh.build_eval_reports
    uv run python -m rag_home_api.test.kh.build_eval_reports --only summary_rent_ab_ragas_20261126_144905.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from rag_home_api.test.kh.rent_chunker import KH_EVAL_DIR, KH_REPORTS_DIR, KH_RESULTS_DIR

RAGAS_KEYS: tuple[tuple[str, str], ...] = (
    ("faithfulness", "F (충실도)"),
    ("answer_relevancy", "AR (답변관련성)"),
    ("llm_context_precision_with_reference", "CP (근거정밀도)"),
    ("context_recall", "CR (근거재현율)"),
)

PRESET_COMPARE_KEYS: tuple[tuple[str, str], ...] = (
    ("pipeline_mode", "파이프라인"),
    ("collection_name", "컬렉션"),
    ("embedding_model", "임베딩"),
    ("chunk_strategy", "청킹"),
    ("prompt_version", "프롬프트"),
    ("bm25_weight", "BM25 비중"),
    ("vector_weight", "Vector 비중"),
    ("rrf_k", "RRF k"),
    ("candidate_k", "후보 k"),
    ("top_k", "Top-k"),
    ("use_metadata_filter", "사전필터"),
    ("use_rewrite", "Query Rewrite"),
    ("use_cache", "캐시"),
    ("use_hyde", "HyDE"),
)


def _infer_eval_version(summary: dict[str, Any]) -> str:
    ev = str(summary.get("eval_version") or "").strip()
    if ev:
        return ev
    ep = str(summary.get("eval_path") or "").strip()
    if ep:
        return Path(ep).stem
    return "unknown_eval"


def _infer_split_dir(eval_version: str) -> str:
    low = eval_version.lower()
    if "test" in low:
        return "test"
    if "dev_v2" in low or "dev2" in low:
        return "dev_v2"
    if "dev" in low:
        return "dev_v1"
    return re.sub(r"[^a-zA-Z0-9_\-]", "_", low) or "misc"


def _fmt_ts(ts: str) -> str:
    if len(ts) == 15 and "_" in ts:
        return f"{ts[0:4]}-{ts[4:6]}-{ts[6:8]} {ts[9:11]}:{ts[11:13]}:{ts[13:15]}"
    return ts


def _parse_ratio(val: Any) -> tuple[int, int] | None:
    if not val:
        return None
    m = re.search(r"(\d+)\s*/\s*(\d+)", str(val))
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def _ratio_rate(val: Any) -> float | None:
    pair = _parse_ratio(val)
    if not pair or pair[1] == 0:
        return None
    return pair[0] / pair[1]


def _fmt_float(val: float | None, digits: int = 4) -> str:
    if val is None:
        return "-"
    return f"{val:.{digits}f}"


def _fmt_delta(cur: float | None, prev: float | None, digits: int = 4, suffix: str = "") -> str:
    if cur is None or prev is None:
        return "-"
    diff = cur - prev
    if abs(diff) < 10 ** (-(digits + 1)):
        return f"0{suffix}"
    return f"{diff:+.{digits}f}{suffix}"


def _clean_inline(text: str, max_len: int | None = None) -> str:
    cleaned = re.sub(r"\s+", " ", str(text or "")).strip()
    if max_len is not None and len(cleaned) > max_len:
        cleaned = cleaned[: max_len - 1].rstrip() + "…"
    return cleaned


def _escape_cell(text: str, max_len: int | None = None) -> str:
    return _clean_inline(text, max_len=max_len).replace("|", "\\|")


def _extract_mode_preset(summary: dict[str, Any], mode_name: str, mode_data: dict[str, Any]) -> dict[str, Any]:
    raw_p = dict(mode_data.get("preset") or {})
    if "pipeline_mode" not in raw_p:
        if mode_name.startswith("A_"):
            raw_p["pipeline_mode"] = "A_vector"
        elif mode_name.startswith("C_"):
            raw_p["pipeline_mode"] = "C_rerank"
        else:
            raw_p["pipeline_mode"] = "B_hybrid"
    raw_p.setdefault("collection_name", summary.get("collection_name") or "housing_rent_kh")
    raw_p.setdefault("embedding_model", summary.get("embedding_model") or "text-embedding-3-small")
    raw_p.setdefault("use_metadata_filter", "filter" in mode_name)
    raw_p.setdefault("use_rewrite", "rewrite" in mode_name)
    raw_p.setdefault("use_cache", "cache" in mode_name)
    raw_p.setdefault("use_hyde", "hyde" in mode_name)
    raw_p.setdefault("top_k", 5)
    raw_p.setdefault("candidate_k", 12)
    return raw_p


def _describe_preset_diff(prev_p: dict[str, Any] | None, cur_p: dict[str, Any]) -> list[str]:
    if prev_p is None:
        return ["최초 기준선 (비교 대상 없음)"]
    diffs: list[str] = []
    for key, label in PRESET_COMPARE_KEYS:
        pv = prev_p.get(key)
        cv = cur_p.get(key)
        if pv != cv:
            diffs.append(f"{label}: `{pv}` → `{cv}`")
    if not diffs:
        p_name = prev_p.get("name")
        c_name = cur_p.get("name")
        if p_name != c_name:
            diffs.append(f"규칙/프리셋 세부분기: `{p_name}` → `{c_name}`")
        else:
            diffs.append("프리셋 메타 동일 (프롬프트/규칙/코드 보강 재측정)")
    return diffs


def _load_eval_items_map(summary: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], bool]:
    """시험지 JSON을 로드한다. (question_id -> item dict, 정확 일치 파일 존재 여부)"""
    candidates: list[Path] = []
    ep = str(summary.get("eval_path") or "").strip()
    if ep:
        p = Path(ep)
        candidates.append(p)
        candidates.append(KH_EVAL_DIR / p.name)
    ev = _infer_eval_version(summary)
    candidates.append(KH_EVAL_DIR / f"{ev}.json")

    for cand in candidates:
        if cand.exists() and cand.is_file():
            try:
                rows = json.loads(cand.read_text(encoding="utf-8"))
                if isinstance(rows, list):
                    return {str(r["question_id"]): r for r in rows if isinstance(r, dict) and r.get("question_id")}, True
            except Exception:
                pass

    # 구 draft 파일 등이 삭제된 경우 빈 맵 반환(오매칭 방지)
    return {}, False


def _load_jsonl_by_mode(summary_path: Path) -> tuple[dict[str, list[dict[str, Any]]], Path | None]:
    ts = summary_path.stem.replace("summary_rent_ab_ragas_", "")
    jp = summary_path.parent / f"exp_rent_ab_ragas_{ts}.jsonl"
    if not jp.exists():
        return {}, None
    by_mode: dict[str, list[dict[str, Any]]] = {}
    for line in jp.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except Exception:
            continue
        by_mode.setdefault(str(row.get("mode") or ""), []).append(row)
    return by_mode, jp


def _compute_cm_totals(mode_logs: list[dict[str, Any]], k_field: str) -> tuple[int, int, int, int] | None:
    ans_rows = [r for r in mode_logs if r.get("expected_status") == "answered"]
    if not ans_rows:
        return None
    tp = fp = fn = tn = 0
    has_any = False
    for r in ans_rows:
        m = r.get(k_field) or {}
        if "tp" in m:
            has_any = True
            tp += int(m.get("tp", 0))
            fp += int(m.get("fp", 0))
            fn += int(m.get("fn", 0))
            tn += int(m.get("tn", 0))
        elif m:
            # 구버전 로그 (tp/fp/fn 필드 도입 전): hit/recall 기반 근사치
            has_any = True
            expected_n = max(1, len(r.get("allowed_chunk_ids") or [1]))
            rec_key = next((k for k in m if k.startswith("recall@")), None)
            if rec_key is not None:
                est_tp = int(round(float(m[rec_key]) * expected_n))
                k_val = int(rec_key.split("@")[1])
                tp += est_tp
                fn += max(0, expected_n - est_tp)
                fp += max(0, k_val - est_tp)
    return (tp, fp, fn, tn) if has_any else None


def _ragas_mean(mode_data: dict[str, Any]) -> float | None:
    ra = mode_data.get("ragas_avg") or {}
    vals = [float(ra[k]) for k, _ in RAGAS_KEYS if k in ra and ra[k] is not None]
    return (sum(vals) / len(vals)) if vals else None


def _pick_representative_mode(summary: dict[str, Any]) -> str:
    modes = summary.get("modes") or {}
    if len(modes) <= 1:
        return next(iter(modes), "")

    def _score(item: tuple[str, dict[str, Any]]) -> tuple[float, float, float, float]:
        _, m = item
        ret = m.get("retrieval") or {}
        rec12 = float(ret.get("recall@12") or 0.0)
        mrr5 = float(ret.get("mrr@5") or 0.0)
        rm = _ragas_mean(m) or 0.0
        hit5 = _ratio_rate(ret.get("hit@5")) or 0.0
        return (rm, mrr5, rec12, hit5)

    return max(modes.items(), key=_score)[0]


def _make_slug(summary: dict[str, Any]) -> str:
    modes = list((summary.get("modes") or {}).keys())
    if len(modes) == 1:
        raw = modes[0]
    elif len(modes) == 2 and modes == ["A_vector", "B_hybrid_rrf"]:
        raw = "3gu_A_B"
    elif len(modes) == 5:
        raw = "25gu_5modes_AB_filter"
    elif len(modes) == 3 and all("kure_v1" in m for m in modes):
        raw = "kure_B_weight_sweep_3modes"
    elif len(modes) == 2 and all("k15" in m for m in modes):
        raw = "kure_B_fix_k15_2modes"
    else:
        raw = f"{len(modes)}modes_{modes[0]}" if modes else "empty"
    return re.sub(r"[^a-zA-Z0-9_\-]", "_", raw)[:48].strip("_")


def _find_best_baseline(
    cur_entry: dict[str, Any],
    cur_mode: str,
    prior_entries: list[dict[str, Any]],
) -> tuple[dict[str, Any], str] | None:
    """동일 시험지(eval_version) 내 이전 실행 중 프리셋 차이가 가장 적은(단일 변인) 직전 모드를 찾는다."""
    cur_ev = cur_entry["eval_version"]
    cur_p = _extract_mode_preset(cur_entry["summary"], cur_mode, cur_entry["summary"]["modes"][cur_mode])
    candidates: list[tuple[int, int, int, dict[str, Any], str]] = []

    for idx, prev in enumerate(prior_entries):
        if prev["eval_version"] != cur_ev:
            continue
        for pm_name, pm_data in (prev["summary"].get("modes") or {}).items():
            prev_p = _extract_mode_preset(prev["summary"], pm_name, pm_data)
            diff_cnt = sum(1 for k, _ in PRESET_COMPARE_KEYS if prev_p.get(k) != cur_p.get(k))
            # 같은 collection + 더 최근 회차 + 더 적은 변수 차이를 우선
            same_col_pen = 0 if prev_p.get("collection_name") == cur_p.get("collection_name") else 1
            # 직전 회차 우선 가중: diff_cnt가 같으면 최신(idx 큰 것) 선택
            candidates.append((diff_cnt, same_col_pen, -idx, prev, pm_name))

    if not candidates:
        return None
    candidates.sort(key=lambda x: (x[0], x[1], x[2]))
    _, _, _, best_prev, best_mode = candidates[0]
    return best_prev, best_mode


def _judge_verdict(
    cur_m: dict[str, Any],
    prev_m: dict[str, Any] | None,
    target: float,
) -> tuple[str, str]:
    """4중 게이트 기준 1차 Keep / Conditional / Discard / Baseline 자동 판정."""
    ret = cur_m.get("retrieval") or {}
    ra = cur_m.get("ragas_avg") or {}
    ref_rate = _ratio_rate(cur_m.get("refusal"))
    lat_med = float((cur_m.get("latency_ms") or {}).get("median") or 0.0)
    passed_ragas = sum(1 for k, _ in RAGAS_KEYS if k in ra and float(ra[k]) >= target)
    has_ragas = bool(ra)

    if prev_m is None:
        return "📌 기준선 (Baseline)", "해당 시험지 최초 측정 기준선으로 보존합니다."

    p_ret = prev_m.get("retrieval") or {}
    d_rec12 = float(ret.get("recall@12") or 0.0) - float(p_ret.get("recall@12") or 0.0)
    d_mrr5 = float(ret.get("mrr@5") or 0.0) - float(p_ret.get("mrr@5") or 0.0)
    cur_rm = _ragas_mean(cur_m)
    prev_rm = _ragas_mean(prev_m)
    d_rm = (cur_rm - prev_rm) if (cur_rm is not None and prev_rm is not None) else 0.0

    ref_ok = (ref_rate is None) or (ref_rate >= 0.70)
    lat_ok = lat_med < 10_000.0

    if not ref_ok or not lat_ok:
        return "❌ 기각 (Discard)", f"안전 게이트 미달 (거절률={cur_m.get('refusal')}, 지연={lat_med:.1f}ms)"

    if d_rec12 < -0.05 and d_mrr5 <= 0.02 and d_rm <= 0.02:
        return "❌ 기각 (Discard)", f"후보 회수율 하락 (Recall@12 {d_rec12:+.4f}) 대비 순위·RAGAS 이득 부족"

    if (d_rec12 >= 0 and (d_mrr5 >= 0.02 or d_rm >= 0.02)) or (has_ragas and passed_ragas == 4):
        return "✅ 채택 (Keep)", f"기준선 대비 핵심 지표 상승 (Recall@12 {d_rec12:+.4f}, MRR@5 {d_mrr5:+.4f}, RAGAS 평균 {d_rm:+.4f})"

    if d_mrr5 > 0.05 or d_rm > 0.05:
        return "⚖️ 조건부 채택 (Trade-off)", f"순위·RAGAS 상승(MRR@5 {d_mrr5:+.4f}, RAGAS {d_rm:+.4f}) vs Recall@12 변화({d_rec12:+.4f})"

    return "❌ 기각 (Discard)", f"기준선 대비 유의미한 개선 없음 (Recall@12 {d_rec12:+.4f}, MRR@5 {d_mrr5:+.4f}, RAGAS {d_rm:+.4f})"


def _suggest_next_variable(cur_m: dict[str, Any], cur_p: dict[str, Any], target: float) -> tuple[str, str]:
    ret = cur_m.get("retrieval") or {}
    ra = cur_m.get("ragas_avg") or {}
    rec12 = float(ret.get("recall@12") or 0.0)
    mrr5 = float(ret.get("mrr@5") or 0.0)
    top1_rate = _ratio_rate(ret.get("top1")) or 0.0
    ar = float(ra["answer_relevancy"]) if "answer_relevancy" in ra else None
    f_val = float(ra["faithfulness"]) if "faithfulness" in ra else None
    cr = float(ra["context_recall"]) if "context_recall" in ra else None

    if rec12 < 0.85:
        if cur_p.get("pipeline_mode") == "A_vector" and "kure" not in str(cur_p.get("embedding_model", "")).lower():
            return "임베딩 모델 교체 (`KURE-v1`) 또는 Hybrid(`B_hybrid`) 전환", f"Recall@12={rec12:.4f}로 후보 누락이 최우선 병목"
        if not cur_p.get("use_metadata_filter"):
            return "사전 메타데이터 필터링 (`use_metadata_filter=True`)", f"Recall@12={rec12:.4f} 개선 및 지역·유형·면적 오탐 제거"
        if not cur_p.get("use_rewrite"):
            return "Query Rewrite (2개 지역 복합 비교 서브쿼리 분해)", f"단일 검색 후보 한계(Recall@12={rec12:.4f}) 해소"
        return "청킹·어휘 정규화 점검", f"Recall@12={rec12:.4f} 잔여 누락 원인 분석"

    if rec12 < 1.0 or (cr is not None and cr < 1.0):
        if not cur_p.get("use_metadata_filter"):
            return "사전 메타데이터 필터링 (`use_metadata_filter=True`)", f"Recall@12={rec12:.4f} / Top1={ret.get('top1')} 정밀도 도약"
        if not cur_p.get("use_rewrite"):
            return "Query Rewrite (복합 비교 2구 분해 + 구어체 정규화)", f"Recall@12={rec12:.4f}, CR={_fmt_float(cr)} — 2개 구 비교 문항 양쪽 Top-5 확보"

    if top1_rate < 0.85 or mrr5 < 0.85:
        if not cur_p.get("use_metadata_filter"):
            return "사전 메타데이터 필터링 (`use_metadata_filter=True`)", f"후보는 확보됐으나 Top1={ret.get('top1')} / MRR@5={mrr5:.4f} 순위 병목"
        if cur_p.get("pipeline_mode") != "C_rerank":
            return "검색 C (`C_rerank`: Top-12 → Top-5 재정렬)", f"후보 확보 후 상위 순위(MRR@5={mrr5:.4f}) 정밀화"

    if ar is not None and ar < target:
        if not cur_p.get("use_rewrite"):
            return "Query Rewrite (2구 비교 분해) + 생성 프롬프트 1줄 보강", f"검색은 만점급(MRR@5={mrr5:.4f})이나 AR={ar:.4f} < {target} — 미완 비교·질문 조건(평수/거래빈도) 누락 해소"
        return "생성 프롬프트(`ANSWER_PROMPT`) 질문 항목 누락 방지 보강", f"검색은 충분하나 AR={ar:.4f} < {target} 경계 미달"

    if f_val is not None and f_val < target:
        return "생성 프롬프트 근거 엄격화 (추정·환산 금지)", f"Faithfulness={f_val:.4f} < {target}"

    return "봉인 평가셋(`kh_rent_test.json` 20문항) 최종 1회 고정 평가 또는 캐시 실측", "개발셋 핵심 검색·RAGAS 목표 달성 완료"


def _format_filters(flt: Any, fallback: bool = False) -> str:
    if not flt:
        return "없음" + (" (fallback)" if fallback else "")
    if isinstance(flt, dict):
        parts = [f"{k}={v}" for k, v in flt.items()]
        s = ", ".join(parts)
        return f"`{s}`" + (" ⚠️fallback" if fallback else "")
    return f"`{flt}`"


def _format_rank_cell(
    cur_row: dict[str, Any] | None,
    prev_row: dict[str, Any] | None,
) -> str:
    if not cur_row:
        return "-"
    m12 = cur_row.get("metrics_k12") or {}
    m5 = cur_row.get("metrics_k5") or {}
    r12 = int(m12.get("first_rank") or 0)
    r5 = int(m5.get("first_rank") or 0)
    cur_str = f"#{r12}" if r12 else "X"
    cur_str += f" / #{r5}" if r5 else " / X"

    if prev_row:
        pm12 = prev_row.get("metrics_k12") or {}
        pr12 = int(pm12.get("first_rank") or 0)
        if pr12 != r12:
            p_str = f"#{pr12}" if pr12 else "X"
            c_str = f"#{r12}" if r12 else "X"
            if (pr12 == 0 and r12 > 0) or (pr12 > 0 and r12 > 0 and r12 < pr12):
                return f"{cur_str} (▲{p_str}→{c_str})"
            return f"{cur_str} (▼{p_str}→{c_str})"
    return cur_str


def render_report_markdown(
    cur_entry: dict[str, Any],
    prior_entries: list[dict[str, Any]],
) -> str:
    summary: dict[str, Any] = cur_entry["summary"]
    summary_path: Path = cur_entry["summary_path"]
    ts = str(summary.get("timestamp") or summary_path.stem.replace("summary_rent_ab_ragas_", ""))
    eval_ver = cur_entry["eval_version"]
    split_dir = cur_entry["split_dir"]
    target = float(summary.get("ragas_target") or 0.60)
    run_note = str(summary.get("run_note") or "").strip()
    modes: dict[str, dict[str, Any]] = summary.get("modes") or {}
    mode_names = list(modes.keys())
    rep_mode = _pick_representative_mode(summary)

    eval_map, eval_exact = _load_eval_items_map(summary)
    jsonl_by_mode, jsonl_path = _load_jsonl_by_mode(summary_path)

    # 대표 모드 기준 직전 기준선 탐색
    baseline_pair = _find_best_baseline(cur_entry, rep_mode, prior_entries) if rep_mode else None
    prev_entry = baseline_pair[0] if baseline_pair else None
    prev_mode_name = baseline_pair[1] if baseline_pair else None
    prev_mode_data = prev_entry["summary"]["modes"][prev_mode_name] if (prev_entry and prev_mode_name) else None
    prev_jsonl_by_mode, _ = _load_jsonl_by_mode(prev_entry["summary_path"]) if prev_entry else ({}, None)
    prev_logs_map = (
        {str(r.get("question_id")): r for r in prev_jsonl_by_mode.get(prev_mode_name or "", [])}
        if prev_entry
        else {}
    )

    rep_data = modes[rep_mode] if rep_mode else {}
    rep_preset = _extract_mode_preset(summary, rep_mode, rep_data) if rep_mode else {}
    prev_preset = (
        _extract_mode_preset(prev_entry["summary"], prev_mode_name, prev_mode_data)
        if (prev_entry and prev_mode_name and prev_mode_data)
        else None
    )
    preset_diffs = _describe_preset_diff(prev_preset, rep_preset) if rep_mode else []

    # 문항 수 계산
    first_mode_logs = jsonl_by_mode.get(rep_mode, [])
    ans_cnt = int((rep_data.get("retrieval") or {}).get("answerable_count") or 0)
    ref_pair = _parse_ratio(rep_data.get("refusal"))
    unans_cnt = ref_pair[1] if ref_pair else sum(1 for r in first_mode_logs if r.get("expected_status") == "insufficient_evidence")
    total_q = ans_cnt + unans_cnt if (ans_cnt or unans_cnt) else len(eval_map)

    lines: list[str] = []
    title_mode = rep_mode if len(mode_names) == 1 else f"{len(mode_names)}개 구성 비교 (대표: {rep_mode})"
    lines.append(f"# 평가 리포트 — `{eval_ver}` · `{title_mode}` (`{ts}`)")
    lines.append("")
    lines.append(f"> 자동 생성(API 호출 0원) · [전체 리포트 목차](../README.md) · [종합 실험기록](../../실험기록.md)")
    lines.append("")

    # ── 1. 실험 메타 & 단일 변인 체크 ──
    lines.append("## 1. 실험 메타 & 단일 변인 체크")
    lines.append("")
    lines.append("| 항목 | 값 |")
    lines.append("| :--- | :--- |")
    lines.append(f"| 실행 시각 | `{_fmt_ts(ts)}` |")
    lines.append(f"| 시험지 (`eval_version`) | `{eval_ver}` (총 {total_q}문항 = 답변 가능 {ans_cnt} + 거절 {unans_cnt}) |")
    lines.append(f"| 실행 구성 ({len(mode_names)}종) | {', '.join(f'`{m}`' for m in mode_names)} |")
    corpus_n = int(summary.get("total_corpus_chunks") or 0)
    corpus_str = f"코퍼스 {corpus_n:,}청크" if corpus_n else "코퍼스 수 미기록"
    lines.append(
        f"| 컬렉션 / 임베딩 | `{rep_preset.get('collection_name', '-')}` ({corpus_str}) / `{rep_preset.get('embedding_model', '-')}` |"
    )
    lines.append(
        f"| 청킹 / 프롬프트 | `{rep_preset.get('chunk_strategy', 'rent-v2-11cols-src')}` / `{rep_preset.get('prompt_version', '기본')}` |"
    )
    lines.append(f"| 가설 메모 (`run_note`) | {_escape_cell(run_note) if run_note else '(미기재)'} |")
    if prev_entry and prev_mode_name:
        prev_rel = f"./{prev_entry['report_filename']}" if prev_entry["split_dir"] == split_dir else f"../{prev_entry['split_dir']}/{prev_entry['report_filename']}"
        lines.append(f"| 직전 비교 기준선 | [`{prev_entry['timestamp']}` · `{prev_mode_name}`]({prev_rel}) |")
    else:
        lines.append("| 직전 비교 기준선 | 없음 (해당 시험지 최초 기준선) |")
    lines.append(f"| 변경 변수 ({len(preset_diffs)}항목) | {' · '.join(preset_diffs)} |")
    rel_sum = f"../../results/{summary_path.name}"
    rel_log = f"[`{jsonl_path.name}`](../../results/{jsonl_path.name})" if jsonl_path else "없음 (요약 JSON만 보존)"
    lines.append(f"| 원본 근거 파일 | [`{summary_path.name}`]({rel_sum}) · {rel_log} |")
    lines.append("")

    # ── 2. 종합 성적표 & 기준선 대비 변화 (Δ) ──
    lines.append(f"## 2. 종합 성적표 & 기준선 대비 변화 (목표 RAGAS ≥ `{target:.2f}` · 거절 ≥ `70%`)")
    lines.append("")

    if len(mode_names) == 1:
        m = rep_data
        ret = m.get("retrieval") or {}
        ra = m.get("ragas_avg") or {}
        lat = m.get("latency_ms") or {}
        p_ret = (prev_mode_data or {}).get("retrieval") or {}
        p_ra = (prev_mode_data or {}).get("ragas_avg") or {}
        p_lat = (prev_mode_data or {}).get("latency_ms") or {}

        cm12 = _compute_cm_totals(jsonl_by_mode.get(rep_mode, []), "metrics_k12")
        cm5 = _compute_cm_totals(jsonl_by_mode.get(rep_mode, []), "metrics_k5")
        p_cm12 = _compute_cm_totals(prev_jsonl_by_mode.get(prev_mode_name or "", []), "metrics_k12") if prev_entry else None

        cur_rec12 = float(ret["recall@12"]) if "recall@12" in ret and ret["recall@12"] is not None else None
        prev_rec12 = float(p_ret["recall@12"]) if "recall@12" in p_ret and p_ret["recall@12"] is not None else None
        cur_mrr5 = float(ret["mrr@5"]) if "mrr@5" in ret and ret["mrr@5"] is not None else None
        prev_mrr5 = float(p_ret["mrr@5"]) if "mrr@5" in p_ret and p_ret["mrr@5"] is not None else None
        cur_hit5_r = _ratio_rate(ret.get("hit@5"))
        prev_hit5_r = _ratio_rate(p_ret.get("hit@5"))
        cur_top1_r = _ratio_rate(ret.get("top1"))
        prev_top1_r = _ratio_rate(p_ret.get("top1"))

        lines.append("| 구분 | 지표 | 기준선 | 이번 결과 | 변화 (Δ) | 상태 |")
        lines.append("| :--- | :--- | ---: | ---: | ---: | :---: |")
        lines.append(
            f"| 검색 | `Recall@12` (후보 확보) | `{_fmt_float(prev_rec12)}` | **`{_fmt_float(cur_rec12)}`** | `{_fmt_delta(cur_rec12, prev_rec12)}` | {'✅ 만점' if cur_rec12 == 1.0 else '🔎 점검'} |"
        )
        lines.append(
            f"| 검색 | `Hit@5` (Top-5 적중) | `{p_ret.get('hit@5', '-')}` | **`{ret.get('hit@5', '-')}`** | `{_fmt_delta(cur_hit5_r * 100 if cur_hit5_r is not None else None, prev_hit5_r * 100 if prev_hit5_r is not None else None, 1, '%p')}` | {'✅ 만점' if cur_hit5_r == 1.0 else '🔎 점검'} |"
        )
        lines.append(
            f"| 검색 | `Top1` (1위 적중) | `{p_ret.get('top1', '-')}` | **`{ret.get('top1', '-')}`** | `{_fmt_delta(cur_top1_r * 100 if cur_top1_r is not None else None, prev_top1_r * 100 if prev_top1_r is not None else None, 1, '%p')}` | {'✅ 만점' if cur_top1_r == 1.0 else '🔎 점검'} |"
        )
        lines.append(
            f"| 검색 | `MRR@5` (순위 정밀도) | `{_fmt_float(prev_mrr5)}` | **`{_fmt_float(cur_mrr5)}`** | `{_fmt_delta(cur_mrr5, prev_mrr5)}` | {'✅ 만점' if cur_mrr5 == 1.0 else '🔎 점검'} |"
        )
        if cm12:
            p_cm_str = f"`TP={p_cm12[0]} / FN={p_cm12[2]}`" if p_cm12 else "-"
            lines.append(
                f"| 검색 | `k=12` 혼동행렬 (`TP / FN`) | {p_cm_str} | **`TP={cm12[0]} / FN={cm12[2]}`** (FP={cm12[1]}) | `{_fmt_delta(float(cm12[0]), float(p_cm12[0]) if p_cm12 else None, 0, '건')}` | {'✅ 누락 0' if cm12[2] == 0 else f'⚠️ 누락 {cm12[2]}건'} |"
            )
        if cm5:
            lines.append(
                f"| 검색 | `k=5` 혼동행렬 (`TP / FN`) | - | `TP={cm5[0]} / FN={cm5[2]}` (FP={cm5[1]}) | - | {'✅ 만점' if cm5[2] == 0 else f'⚠️ {cm5[2]}건'} |"
            )
        if ra:
            for rkey, rlabel in RAGAS_KEYS:
                cv = float(ra[rkey]) if rkey in ra and ra[rkey] is not None else None
                pv = float(p_ra[rkey]) if rkey in p_ra and p_ra[rkey] is not None else None
                status = "✅ 통과" if (cv is not None and cv >= target) else "❌ 미달"
                lines.append(
                    f"| RAGAS | `{rlabel}` | `{_fmt_float(pv)}` | **`{_fmt_float(cv)}`** | `{_fmt_delta(cv, pv)}` | {status} |"
                )
        else:
            lines.append("| RAGAS | 4대 지표 | - | `미측정 (--no-ragas)` | - | ⚪ 생략 |")
        lines.append(
            f"| 안전·출처 | 거절 / 과거고지 / 출처율 | `{prev_mode_data.get('refusal', '-') if prev_mode_data else '-'}` | `{m.get('refusal', '-')}` / `{m.get('scope_notice', '-')}` / `{m.get('citation_rate', '-')}` | - | {'✅ 통과' if (_ratio_rate(m.get('refusal')) or 0.0) >= 0.7 else '⚪ 참고'} |"
        )
        cur_med = float(lat["median"]) if "median" in lat and lat["median"] is not None else None
        prev_med = float(p_lat["median"]) if "median" in p_lat and p_lat["median"] is not None else None
        lines.append(
            f"| 운영 | 검색 지연 (`median / p95`) | `{_fmt_float(prev_med, 1)}ms` | `{_fmt_float(cur_med, 1)}ms` / `{_fmt_float(lat.get('p95'), 1)}ms` *(avg {_fmt_float(lat.get('avg'), 1)}ms)* | `{_fmt_delta(cur_med, prev_med, 1, 'ms')}` | ✅ 10초 이내 |"
        )
        if "filter_applied" in ret or "cache_hits" in ret:
            lines.append(
                f"| 파이프라인 | 사전필터 / 캐시 / HyDE격리 | - | 필터 `{ret.get('filter_applied', '-')}` · 캐시 `{ret.get('cache_hits', '-')}` · HyDE `{ret.get('hyde_isolated', '-')}` | - | ℹ️ 참고 |"
            )
        lines.append(
            f"| 진단 | 기준 미달 문항 수 (`<{target:.2f}`) | `{len(prev_mode_data.get('underperforming_items') or []) if prev_mode_data else '-'}` | **`{len(m.get('underperforming_items') or [])}건`** | `{_fmt_delta(float(len(m.get('underperforming_items') or [])), float(len(prev_mode_data.get('underperforming_items') or [])) if prev_mode_data else None, 0, '건')}` | {'🏆 0건' if not (m.get('underperforming_items')) else '🔎 §4 참조'} |"
        )
        lines.append("")
    else:
        lines.append(
            "| 모드 | `Recall@12` | `Hit@5` | `Top1` | `MRR@5` | `F` | `AR` | `CP` | `CR` | 거절 / 고지 | 지연(중앙/p95) | 미달 |"
        )
        lines.append("| :--- | ---: | :---: | :---: | ---: | ---: | ---: | ---: | ---: | :---: | ---: | ---: |")
        if prev_mode_data and prev_entry:
            pr = prev_mode_data.get("retrieval") or {}
            pra = prev_mode_data.get("ragas_avg") or {}
            pl = prev_mode_data.get("latency_ms") or {}
            lines.append(
                f"| *(기준선)* `{prev_mode_name}` (`{prev_entry['timestamp'][9:]}`) | `{_fmt_float(pr.get('recall@12'))}` | `{pr.get('hit@5', '-')}` | `{pr.get('top1', '-')}` | `{_fmt_float(pr.get('mrr@5'))}` | `{_fmt_float(pra.get('faithfulness'))}` | `{_fmt_float(pra.get('answer_relevancy'))}` | `{_fmt_float(pra.get('llm_context_precision_with_reference'))}` | `{_fmt_float(pra.get('context_recall'))}` | `{prev_mode_data.get('refusal', '-')}` / `{prev_mode_data.get('scope_notice', '-')}` | `{_fmt_float(pl.get('median'), 1)}ms` / `{_fmt_float(pl.get('p95'), 1)}ms` | `{len(prev_mode_data.get('underperforming_items') or [])}건` |"
            )
        for mname in mode_names:
            m = modes[mname]
            r = m.get("retrieval") or {}
            ra = m.get("ragas_avg") or {}
            lat = m.get("latency_ms") or {}
            crown = " 🏆" if mname == rep_mode else ""
            f_s = f"{_fmt_float(ra.get('faithfulness'))}{'✅' if (ra.get('faithfulness') or 0) >= target else '❌'}" if "faithfulness" in ra else "-"
            ar_s = f"{_fmt_float(ra.get('answer_relevancy'))}{'✅' if (ra.get('answer_relevancy') or 0) >= target else '❌'}" if "answer_relevancy" in ra else "-"
            cp_s = f"{_fmt_float(ra.get('llm_context_precision_with_reference'))}{'✅' if (ra.get('llm_context_precision_with_reference') or 0) >= target else '❌'}" if "llm_context_precision_with_reference" in ra else "-"
            cr_s = f"{_fmt_float(ra.get('context_recall'))}{'✅' if (ra.get('context_recall') or 0) >= target else '❌'}" if "context_recall" in ra else "-"
            lines.append(
                f"| **`{mname}`**{crown} | `{_fmt_float(r.get('recall@12'))}` | `{r.get('hit@5', '-')}` | `{r.get('top1', '-')}` | `{_fmt_float(r.get('mrr@5'))}` | {f_s} | {ar_s} | {cp_s} | {cr_s} | `{m.get('refusal', '-')}` / `{m.get('scope_notice', '-')}` | `{_fmt_float(lat.get('median'), 1)}ms` / `{_fmt_float(lat.get('p95'), 1)}ms` | `{len(m.get('underperforming_items') or [])}건` |"
            )
        lines.append("")

    # 동일 회차 내 타 모드에 기록된 question_id -> user_input 공유 맵 (초기 v1 통과 문항 보완)
    shared_q_text: dict[str, str] = {}
    for m_obj in modes.values():
        for src_list in (m_obj.get("per_item_ragas") or [], m_obj.get("underperforming_items") or []):
            for item_obj in src_list:
                q_id = str(item_obj.get("question_id") or "")
                q_in = str(item_obj.get("user_input") or "").strip()
                if q_id and q_in and q_id not in shared_q_text:
                    shared_q_text[q_id] = q_in

    # ── 3. 문항별 한눈표 (전 문항) ──
    lines.append("## 3. 문항별 한눈표 (전 문항 검색 순위 · 적용 필터 · RAGAS 4지표)")
    lines.append("")
    for mname in mode_names:
        m = modes[mname]
        if len(mode_names) > 1:
            lines.append(f"### 3-{mode_names.index(mname) + 1}. `{mname}`")
            lines.append("")
        mlogs = jsonl_by_mode.get(mname, [])
        log_by_qid = {str(r.get("question_id")): r for r in mlogs}
        per_map = {str(x["question_id"]): x for x in (m.get("per_item_ragas") or [])}
        under_map = {str(x["question_id"]): x for x in (m.get("underperforming_items") or [])}

        # 문항 ID 순서 구성 (jsonl > eval_map > per_item > underperforming)
        ordered_qids: list[str] = []
        for src in (
            [str(r.get("question_id")) for r in mlogs],
            list(eval_map.keys()) if eval_exact else [],
            list(per_map.keys()),
            list(under_map.keys()),
        ):
            for qid in src:
                if qid and qid not in ordered_qids:
                    ordered_qids.append(qid)

        if not ordered_qids:
            lines.append("_문항별 세부 로그가 기록되지 않았습니다._")
            lines.append("")
            continue

        lines.append("| ID | 구분 | 질문 요약 | 적용 필터 | 순위 (`k=12 / k=5`) | `F` | `AR` | `CP` | `CR` | 판정 |")
        lines.append("| :--- | :--- | :--- | :--- | :---: | ---: | ---: | ---: | ---: | :--- |")

        for qid in ordered_qids:
            lrow = log_by_qid.get(qid)
            prow = per_map.get(qid)
            urow = under_map.get(qid)
            ev_item = eval_map.get(qid) if eval_exact else None

            expected_status = (
                (lrow or {}).get("expected_status")
                or (ev_item or {}).get("expected_status")
                or ("answered" if (prow or urow) else "unknown")
            )
            q_text = (
                (prow or {}).get("user_input")
                or (urow or {}).get("user_input")
                or (ev_item or {}).get("user_input")
                or shared_q_text.get(qid)
                or "(로그 미기록)"
            )
            cat = (lrow or {}).get("eval_category") or (ev_item or {}).get("eval_category") or ("거절" if expected_status == "insufficient_evidence" else "-")
            flt_str = _format_filters((lrow or {}).get("applied_filters"), bool((lrow or {}).get("filter_fallback"))) if lrow else "-"
            rw = (lrow or {}).get("rewrite_info")
            if isinstance(rw, dict) and rw.get("rewritten"):
                sub_cnt = len(rw.get("sub_queries") or [])
                flt_str += f" +RW({sub_cnt})"

            if expected_status == "insufficient_evidence":
                resp = str((lrow or {}).get("response") or "")
                ref_ok = ("확인할 수 없음" in resp or "정보가 없" in resp) if resp else None
                sc_ok = any(w in resp for w in ("과거", "체결 완료", "실거래 통계", "포함되지", "나와 있지")) if resp else None
                if ref_ok is None:
                    verdict = "🛡️ 거절(로그없음)"
                elif ref_ok and sc_ok:
                    verdict = "🛡️ 거절+고지 ✅"
                elif ref_ok:
                    verdict = "🛡️ 거절✅ (고지누락)"
                else:
                    verdict = "❌ 거절 실패"
                lines.append(
                    f"| `{qid}` | {_escape_cell(cat, 14)} | {_escape_cell(q_text, 44)} | {flt_str} | - | - | - | - | - | {verdict} |"
                )
                continue

            # answered item
            rank_str = _format_rank_cell(lrow, prev_logs_map.get(qid) if mname == rep_mode else None)
            scores = (prow or {}).get("scores") or (urow or {}).get("all_scores") or {}

            def _sc(k: str) -> str:
                if k not in scores:
                    return "✅(미기록)" if (not urow and not per_map and m.get("ragas_avg")) else "-"
                v = float(scores[k])
                return f"**{v:.3f}**❌" if v < target else f"{v:.3f}"

            if urow:
                f_short = ",".join(
                    {"faithfulness": "F", "answer_relevancy": "AR", "llm_context_precision_with_reference": "CP", "context_recall": "CR"}.get(k, k)
                    for k in (urow.get("failed_metrics") or {})
                )
                verdict = f"❌ 미달(`{f_short}`)"
            elif prow and prow.get("passed_all"):
                verdict = "✅ ALL PASS"
            elif m.get("ragas_avg"):
                verdict = "✅ PASS"
            else:
                verdict = "⚪ 검색전용"

            lines.append(
                f"| `{qid}` | {_escape_cell(cat, 14)} | {_escape_cell(q_text, 44)} | {flt_str} | {rank_str} | {_sc('faithfulness')} | {_sc('answer_relevancy')} | {_sc('llm_context_precision_with_reference')} | {_sc('context_recall')} | {verdict} |"
            )
        lines.append("")

    # ── 4. 기준 미달 문항 정밀 진단 (LLM Diagnosis + 답변 vs 모범정답 대조) ──
    lines.append(f"## 4. 기준 미달 문항 정밀 진단 (`<{target:.2f}` · LLM 원인 분류 & 생성 답변 대조)")
    lines.append("")
    for mname in mode_names:
        m = modes[mname]
        under = m.get("underperforming_items") or []
        if len(mode_names) > 1:
            lines.append(f"### 4-{mode_names.index(mname) + 1}. `{mname}` (미달 {len(under)}건)")
            lines.append("")
        if not under:
            if m.get("ragas_avg"):
                lines.append(f"🏆 **기준 미달 0건 — 전 문항 목표(`{target:.2f}`) 통과!**")
            else:
                lines.append("_RAGAS 미실행(`--no-ragas`)으로 기준 미달 진단이 생략되었습니다._")
            lines.append("")
            continue

        log_by_qid = {str(r.get("question_id")): r for r in jsonl_by_mode.get(mname, [])}
        for u in under:
            qid = str(u.get("question_id"))
            u_q = str(u.get("user_input") or "")
            failed = u.get("failed_metrics") or {}
            all_sc = u.get("all_scores") or {}
            diag_txt = str(u.get("llm_diagnosis") or "").strip()
            lrow = log_by_qid.get(qid)
            ev_item = eval_map.get(qid) if eval_exact else None
            # draft 시험지 오매칭 방지: user_input 일치 시에만 reference 연결
            ref_ans = ""
            if ev_item and (not u_q or ev_item.get("user_input") == u_q):
                ref_ans = str(ev_item.get("reference") or "").strip()

            failed_str = ", ".join(f"`{k}={v}`" for k, v in failed.items())
            all_str = " · ".join(
                f"{lbl.split()[0]}=`{all_sc[k]:.3f}`" for k, lbl in RAGAS_KEYS if k in all_sc
            )
            lines.append(f"#### ⚠️ `{qid}` — 미달: {failed_str}")
            lines.append(f"- **질문**: {u_q}")
            lines.append(f"- **RAGAS 4지표**: {all_str}")
            if lrow:
                rk = _format_rank_cell(lrow, prev_logs_map.get(qid) if mname == rep_mode else None)
                allowed = lrow.get("allowed_chunk_ids") or []
                top5 = lrow.get("top5_ids") or []
                flt = _format_filters(lrow.get("applied_filters"), bool(lrow.get("filter_fallback")))
                lines.append(f"- **검색 진단**: 순위 `{rk}` · 적용 필터 {flt}")
                if allowed:
                    lines.append(f"  - 정답 청크: {', '.join(f'`{c}`' for c in allowed)}")
                if top5:
                    lines.append(f"  - 실제 Top-5: {', '.join(f'`{c}`' for c in top5[:5])}")
                resp_txt = str(lrow.get("response") or "").strip()
                if resp_txt:
                    lines.append(f"- **실제 생성 답변**: {_clean_inline(resp_txt)}")
            else:
                lines.append("- **검색·답변 원문**: _(해당 회차 `.jsonl` 미보존으로 생략)_")
            if ref_ans:
                lines.append(f"- **모범정답 (`reference`)**: {_clean_inline(ref_ans)}")
            if diag_txt:
                lines.append("- **LLM 정밀 원인 분류 & 개선안**:")
                for dl in diag_txt.splitlines():
                    lines.append(f"  > {dl}" if dl.strip() else "  >")
            lines.append("")

    # ── 5. 거절·범위 고지 (Track B) 실제 응답 검증 ──
    lines.append("## 5. 거절·범위 고지 (Track B) 실제 응답 검증")
    lines.append("")
    if not jsonl_path:
        lines.append(
            f"_원본 `.jsonl` 로그가 보존되지 않아 거절 응답 전문은 생략합니다. (요약 집계: 거절 `{rep_data.get('refusal', '-')}` · 과거 거래 고지 `{rep_data.get('scope_notice', '-')}`)_"
        )
        lines.append("")
    else:
        for mname in mode_names:
            unans_rows = [r for r in jsonl_by_mode.get(mname, []) if r.get("expected_status") == "insufficient_evidence"]
            if not unans_rows:
                continue
            if len(mode_names) > 1:
                lines.append(f"### 5-{mode_names.index(mname) + 1}. `{mname}`")
                lines.append("")
            for r in unans_rows:
                qid = str(r.get("question_id"))
                ev_item = eval_map.get(qid) if eval_exact else None
                q_txt = str((ev_item or {}).get("user_input") or "(시험지 문항)")
                abs_reason = str((ev_item or {}).get("absence_reason") or "").strip()
                resp = str(r.get("response") or "").strip()
                ref_ok = "✅ 거절 통과" if ("확인할 수 없음" in resp or "정보가 없" in resp) else "❌ 거절 실패"
                sc_ok = (
                    "✅ 과거 거래 고지 포함"
                    if any(w in resp for w in ("과거", "체결 완료", "실거래 통계", "포함되지", "나와 있지"))
                    else "⚠️ 과거 거래 고지 미명시"
                )
                flt = _format_filters(r.get("applied_filters"), bool(r.get("filter_fallback")))
                lines.append(f"- **`{qid}`** ({ref_ok} · {sc_ok} · 필터 {flt})")
                lines.append(f"  - 질문: {q_txt}")
                if abs_reason:
                    lines.append(f"  - 근거 부재 사유: {abs_reason}")
                lines.append("  - 실제 응답:")
                lines.append(f"    > {_clean_inline(resp) if resp else '(응답 미생성)'}")
            lines.append("")

    # ── 6. 채택(Keep) / 기각(Discard) 판정 & 다음 1변수 제안 ──
    lines.append("## 6. 채택(Keep) / 기각(Discard) 자동 판정 & 다음 1변수 제안")
    lines.append("")
    verdict_badge, verdict_reason = _judge_verdict(rep_data, prev_mode_data, target)
    next_var, next_why = _suggest_next_variable(rep_data, rep_preset, target)
    lines.append(f"- **대표 구성 (`{rep_mode}`) 판정**: **{verdict_badge}**")
    lines.append(f"- **판단 근거**: {verdict_reason}")
    lines.append(f"- **추천 다음 1변수**: **{next_var}**")
    lines.append(f"  - 이유: {next_why}")
    lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def _scan_all_summaries(results_dir: Path = KH_RESULTS_DIR) -> list[dict[str, Any]]:
    summaries = sorted(results_dir.glob("summary_*.json"))
    entries: list[dict[str, Any]] = []
    seq_by_split: dict[str, int] = {}
    for sp in summaries:
        summary = json.loads(sp.read_text(encoding="utf-8"))
        ts = str(summary.get("timestamp") or sp.stem.replace("summary_rent_ab_ragas_", ""))
        eval_ver = _infer_eval_version(summary)
        split_dir = _infer_split_dir(eval_ver)
        seq_by_split[split_dir] = seq_by_split.get(split_dir, 0) + 1
        seq = seq_by_split[split_dir]
        slug = _make_slug(summary)
        report_filename = f"{seq:02d}_{ts}_{slug}.md"
        entries.append(
            {
                "summary_path": sp,
                "summary": summary,
                "timestamp": ts,
                "eval_version": eval_ver,
                "split_dir": split_dir,
                "seq": seq,
                "report_filename": report_filename,
                "report_path": KH_REPORTS_DIR / split_dir / report_filename,
            }
        )
    return entries


def _write_reports_index(entries: list[dict[str, Any]], reports_dir: Path = KH_REPORTS_DIR) -> Path:
    reports_dir.mkdir(parents=True, exist_ok=True)
    index_path = reports_dir / "README.md"

    by_split: dict[str, list[dict[str, Any]]] = {}
    for e in entries:
        by_split.setdefault(e["split_dir"], []).append(e)

    split_order = [s for s in ("test", "dev_v2", "dev_v1") if s in by_split] + [
        s for s in sorted(by_split) if s not in ("test", "dev_v2", "dev_v1")
    ]
    split_titles = {
        "test": "🏆 최종 봉인 평가 (`test` · 20문항)",
        "dev_v2": "🧪 현실 구어체·평수 시험지 (`dev_v2` · 10문항)",
        "dev_v1": "📚 초기 개발 시험지 (`dev_v1` · 3구/25구)",
    }

    lines: list[str] = [
        "# 서울 전월세 RAG 개별 평가 리포트 인덱스 (`test/kh`)",
        "",
        "> 종합 연구 스토리와 포렌식 분석은 [`../실험기록.md`](../실험기록.md)를, 아래 각 링크는 회차별 **문항 상세 · 적용 필터 · LLM 미달 진단(`llm_diagnosis`) · 거절 응답 전문**을 담고 있습니다.",
        f"> 총 **{len(entries)}회차** 자동 집계 · 재생성 명령: `uv run python -m rag_home_api.test.kh.build_eval_reports`",
        "",
    ]

    if "test" not in by_split:
        lines.append("> ℹ️ `test/` (봉인 20문항 `kh_rent_test.json`) 최종 평가 실행 시 이 상단에 자동 추가됩니다.")
        lines.append("")

    for split in split_order:
        lines.append(f"## {split_titles.get(split, split)}")
        lines.append("")
        lines.append(
            "| 회차 | 리포트 (클릭) | 대표 구성 | `Recall@12` | `Hit@5` | `Top1` | `MRR@5` | `F` | `AR` | `CP` | `CR` | 거절 | 지연(중앙) | 미달 | 판정 |"
        )
        lines.append("| :---: | :--- | :--- | ---: | :---: | :---: | ---: | ---: | ---: | ---: | ---: | :---: | ---: | ---: | :--- |")

        split_entries = by_split[split]
        prior_all: list[dict[str, Any]] = []
        for e in entries:
            if e["split_dir"] != split:
                continue
            # e보다 이전의 동일 split 항목들만 기준선 후보로 전달
            prior_same = [x for x in entries if x["timestamp"] < e["timestamp"]]
            summary = e["summary"]
            rep_mode = _pick_representative_mode(summary)
            m = (summary.get("modes") or {}).get(rep_mode) or {}
            ret = m.get("retrieval") or {}
            ra = m.get("ragas_avg") or {}
            lat = m.get("latency_ms") or {}
            target = float(summary.get("ragas_target") or 0.60)
            bp = _find_best_baseline(e, rep_mode, prior_same)
            prev_m = bp[0]["summary"]["modes"][bp[1]] if bp else None
            verdict, _ = _judge_verdict(m, prev_m, target)
            rel_link = f"./{e['split_dir']}/{e['report_filename']}"
            fallback_lbl = _make_slug(summary)
            note_short = _escape_cell(str(summary.get("run_note") or fallback_lbl), 38)
            lines.append(
                f"| `#{e['seq']:02d}` | [{e['timestamp']} — {note_short}]({rel_link}) | `{rep_mode}` | "
                f"`{_fmt_float(ret.get('recall@12'))}` | `{ret.get('hit@5', '-')}` | `{ret.get('top1', '-')}` | `{_fmt_float(ret.get('mrr@5'))}` | "
                f"`{_fmt_float(ra.get('faithfulness'))}` | `{_fmt_float(ra.get('answer_relevancy'))}` | `{_fmt_float(ra.get('llm_context_precision_with_reference'))}` | `{_fmt_float(ra.get('context_recall'))}` | "
                f"`{m.get('refusal', '-')}` | `{_fmt_float(lat.get('median'), 1)}ms` | `{len(m.get('underperforming_items') or [])}건` | {verdict} |"
            )
            prior_all.append(e)
        lines.append("")

    index_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return index_path


def generate_all_reports(
    results_dir: Path = KH_RESULTS_DIR,
    reports_dir: Path = KH_REPORTS_DIR,
    only_summary_name: str = "",
) -> tuple[list[Path], Path]:
    """results/summary_*.json을 읽어 reports/<split>/*.md 및 reports/README.md를 생성한다."""
    entries = _scan_all_summaries(results_dir)
    for e in entries:
        e["report_path"] = reports_dir / e["split_dir"] / e["report_filename"]

    written: list[Path] = []
    for i, cur in enumerate(entries):
        if only_summary_name and cur["summary_path"].name != only_summary_name:
            continue
        prior = entries[:i]
        md = render_report_markdown(cur, prior)
        out_path: Path = cur["report_path"]
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(md, encoding="utf-8")
        written.append(out_path)

    index_path = _write_reports_index(entries, reports_dir)
    return written, index_path


def generate_report_for_summary(
    summary_path: Path,
    reports_dir: Path = KH_REPORTS_DIR,
) -> tuple[Path | None, Path]:
    """방금 실행된 summary_*.json 1건의 Markdown 리포트를 생성하고 reports/README.md를 갱신한다."""
    written, index_path = generate_all_reports(
        results_dir=summary_path.parent,
        reports_dir=reports_dir,
        only_summary_name=summary_path.name,
    )
    return (written[0] if written else None), index_path


def main() -> None:
    parser = argparse.ArgumentParser(description="전월세 RAG 개별 Markdown 평가 리포트 오프라인 생성기")
    parser.add_argument("--only", default="", help="특정 summary_*.json 파일명 1개만 생성")
    args = parser.parse_args()

    written, index_path = generate_all_reports(only_summary_name=args.only)
    print(f"개별 평가 리포트 {len(written)}건 생성 완료:")
    for p in written:
        print(f"  - {p.relative_to(KH_REPORTS_DIR.parent)}")
    print(f"종합 목차 갱신 완료: {index_path.relative_to(KH_REPORTS_DIR.parent)}")


if __name__ == "__main__":
    main()
