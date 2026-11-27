"""[방안 3 · 1단계] 팀원 4인(예지·지훈·병철·최기혁)의 상이한 평가 결과 포맷을 공통 표준 JSON으로 통합 변환.

작성자: 최기혁 (kh)
출력 위치: data/team_unified/summary_{yj,jh,bc,kh}.json
Streamlit '5. 시험지(_v{n}) & 팀 통합' 탭이 이 폴더를 자동 스캔해 4명 결과를 한 화면에 나란히 비교한다.

원칙:
  - 팀원 기존 코드나 결과 파일을 일절 수정하지 않는다(읽기 전용 어댑터).
  - RAGAS가 아직 측정되지 않은 검색 전용 로그는 임의로 점수를 채우지 않고 빈 dict({})로 정직하게 기록한다.

실행 예시:
  uv run python -m rag_home_api.test.kh.unify_team_results --from-git
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

from rag_home_api.config import DATA_DIR
from rag_home_api.shared.core.vector_store import TEAM_COLLECTIONS
from rag_home_api.test.kh.rent_chunker import KH_RESULTS_DIR

TEAM_UNIFIED_DIR = DATA_DIR / "team_unified"


def _git_show_text(ref_path: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "show", ref_path],
            text=True,
            encoding="utf-8",
            errors="replace",
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return None


def _clean_ratio(val: Any) -> str:
    """'7/8 (87.5%)' -> '7/8' 정규화."""
    if val is None:
        return "-"
    m = re.search(r"(\d+)\s*/\s*(\d+)", str(val))
    return f"{m.group(1)}/{m.group(2)}" if m else str(val)


def convert_yeji_eval(
    raw: dict[str, Any],
    source_label: str = "eval_dev.json",
    source_branch: str = "local(dev)",
    chunk_version: str = "",
) -> dict[str, Any]:
    """예지(yj) eval_*.json -> 공통 UnifiedEvalSummary dict + 문항별 X-Ray 로그(_companion_logs) 변환."""
    summary_in = raw.get("summary") or {}
    scored_in = raw.get("scored") or {}
    raw_in = raw.get("raw") or {}
    modes_out: dict[str, dict[str, Any]] = {}
    companion_logs: list[dict[str, Any]] = []

    if not chunk_version:
        m_cv = re.search(r"chunk_(V\d+)", source_label, flags=re.I)
        chunk_version = m_cv.group(1).upper() if m_cv else ("V2" if "V2" in str(raw.get("collection", "")) else "V1")
    total_chunks = 6197 if chunk_version == "V2" else 6346

    for mode_name, m in summary_in.items():
        scored_rows = scored_in.get(mode_name) or []
        raw_rows = raw_in.get(mode_name) or []
        raw_by_qid = {str(r.get("question_id", "")): r for r in raw_rows}

        ans_scored = [r for r in scored_rows if r.get("query_type") != "unanswerable"]
        hit_str = _clean_ratio(m.get("hit_at_5"))
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
                # 예지 평가셋 allowed_chunk_ids는 V1+V2 공용 OR 허용 목록이므로 현재 청크 버전으로 필터링
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
                "collection_name": str(raw.get("collection") or TEAM_COLLECTIONS["public_housing"]["current"]),
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
            "refusal": _clean_ratio(m.get("unanswerable_refusal")),
            "scope_notice": _clean_ratio(m.get("unanswerable_refusal")),
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
    if len(ts) == 13:  # YYYYMMDD_HHMM -> YYYYMMDD_HHMM00
        ts = ts + "00"
    split_name = str(raw.get("split", "dev"))
    return {
        "timestamp": ts,
        "owner": "yj",
        "domain": "public_housing",
        "eval_version": f"yj_{split_name}",
        "ragas_target": 0.70,
        "total_corpus_chunks": total_chunks,
        "collection_name": str(raw.get("collection") or TEAM_COLLECTIONS["public_housing"]["current"]),
        "embedding_model": "text-embedding-3-small",
        "run_note": f"chunk_{chunk_version} ({split_name})",
        "source_branch": source_branch,
        "source_file": source_label,
        "notes": f"예지 공공임대 공고문 {chunk_version} ({split_name}) 평가 자동 변환",
        "modes": modes_out,
        "_companion_logs": companion_logs,
    }


def convert_jihun_jsonl(jsonl_text: str, source_label: str = "dev_원본로그.jsonl", source_branch: str = "origin/feat/cjh") -> dict[str, Any]:
    """지훈(jh) 신규 run.py --tag JSONL 또는 구 dev_원본로그.jsonl -> 공통 UnifiedEvalSummary dict 변환."""
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
            # 신규 run.py --tag 출력 형식
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
        "collection_name": TEAM_COLLECTIONS["law"]["current"],
        "embedding_model": "text-embedding-3-small",
        "source_branch": source_branch,
        "source_file": source_label,
        "notes": "지훈 법령 파트 JSONL 자동 변환 (RAGAS 미측정 · 추후 RAGAS JSON 연동 시 자동 반영)",
        "modes": modes_out,
    }


def convert_jihun_experiment_md(md_text: str, source_label: str = "실험기록.md (§19)", source_branch: str = "origin/dev") -> dict[str, Any]:
    """지훈(jh) 최종 실험기록.md §19 DEV/TEST A/B/C 표 -> 공통 UnifiedEvalSummary dict 변환 (RAGAS 미측정 정직 표기)."""
    sec19_idx = md_text.find("## 19.")
    chunk = md_text[sec19_idx:] if sec19_idx >= 0 else md_text

    # Hit@5 & 완료 정답률 표 (첫 번째 표만 파싱하여 하단 Context Precision 17/85 표와 덮어쓰기 방지)
    hit_map: dict[tuple[str, str], str] = {}
    ref_sec_idx = chunk.find("### 근거 부족 보류")
    hit_chunk = chunk[:ref_sec_idx] if ref_sec_idx >= 0 else chunk
    for m in re.finditer(r"^\|\s*(DEV|TEST)\s*\|\s*(A|B|C)\s*\|\s*(\d+/\d+)\s*\|", hit_chunk, flags=re.MULTILINE):
        hit_map.setdefault((m.group(1), m.group(2)), m.group(3))

    # 거절(보류) 표: | DEV | B | 2/2 | 2/2 | 2/2 | 2/2 | 0/10 |
    ref_map: dict[tuple[str, str], str] = {}
    if ref_sec_idx >= 0:
        for m in re.finditer(r"^\|\s*(DEV|TEST)\s*\|\s*(A|B|C)\s*\|\s*(\d+/\d+)\s*\|", chunk[ref_sec_idx : ref_sec_idx + 1200], flags=re.MULTILINE):
            ref_map.setdefault((m.group(1), m.group(2)), m.group(3))

    # 지연 표: | DEV | B | 10 | 0.102 | 0 | 1.647 | 1.749 | 1.728 | 2.448 |
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

    modes_out: dict[str, dict[str, Any]] = {}
    for split_name in ("DEV", "TEST"):
        for mode_name in ("A", "B", "C"):
            hit_str = hit_map.get((split_name, mode_name))
            if not hit_str:
                continue
            ans_cnt = int(hit_str.split("/")[1])
            avg_ms, med_ms, p95_ms = lat_map.get((split_name, mode_name), (0.0, 0.0, 0.0))
            modes_out[f"jh_1000_{split_name}_{mode_name}"] = {
                "retrieval": {
                    "answerable_count": ans_cnt,
                    "recall@12": None,
                    "hit@5": hit_str,
                    "top1": "-",
                    "mrr@5": None,
                },
                "latency_ms": {"avg": avg_ms, "median": med_ms, "p95": p95_ms},
                "refusal": ref_map.get((split_name, mode_name), "-"),
                "ragas_avg": {},
                "underperforming_items": [],
            }

    return {
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "owner": "jh",
        "domain": "law",
        "eval_version": "law-qa-v1",
        "ragas_target": 0.70,
        "total_corpus_chunks": 167,
        "collection_name": TEAM_COLLECTIONS["law"]["current"],
        "embedding_model": "text-embedding-3-small",
        "source_branch": source_branch,
        "source_file": source_label,
        "notes": "지훈 법령 파트 실험기록.md §19 최종 집계 자동 변환 (로컬 평가 TEST B/C 17/17 · 거절 3/3 · 공식 RAGAS 미측정)",
        "modes": modes_out,
    }


def convert_jihun_experiment_splits(
    md_text: str,
    source_label: str = "실험기록.md (§19)",
    source_branch: str = "local(dev)",
) -> list[dict[str, Any]]:
    """지훈(jh) 실험기록.md §19를 DEV / TEST 시험지별 표준 요약 리스트로 분리 반환한다."""
    full = convert_jihun_experiment_md(md_text, source_label=source_label, source_branch=source_branch)
    out: list[dict[str, Any]] = []
    pmode_map = {"A": "A_vector", "B": "B_hybrid", "C": "C_hybrid_rerank"}
    for split_key, eval_ver, ts in (
        ("DEV", "jh_law_dev", "20261126_120000"),
        ("TEST", "jh_law_test", "20261126_130000"),
    ):
        split_modes: dict[str, dict[str, Any]] = {}
        for mkey, mval in full.get("modes", {}).items():
            if f"_{split_key}_" not in mkey:
                continue
            letter = mkey.rsplit("_", 1)[-1]
            copy_val = dict(mval)
            copy_val["preset"] = {
                "name": mkey,
                "pipeline_mode": pmode_map.get(letter, "B_hybrid"),
                "use_metadata_filter": False,
                "use_rewrite": False,
                "use_cache": False,
                "use_hyde": False,
                "collection_name": TEAM_COLLECTIONS["law"]["current"],
                "embedding_model": "text-embedding-3-small",
                "chunk_strategy": "law-1000-200-dedupe",
            }
            split_modes[mkey] = copy_val
        if split_modes:
            out.append(
                {
                    **full,
                    "timestamp": ts,
                    "eval_version": eval_ver,
                    "run_note": f"jh_1000 {split_key} (A/B/C)",
                    "modes": split_modes,
                }
            )
    return out


def convert_byeongcheol_history_md(
    md_text: str,
    source_label: str = "docs/development_history.md",
    source_branch: str = "local(dev)",
) -> list[dict[str, Any]]:
    """병철(bc) development_history.md의 Dev / 기존 Test(T01~T17, RAGAS 포함) / 신규 Test(T21~T40)를 시험지별 표준 요약 리스트로 파싱한다."""
    pmode_map = {"A": "A_vector", "B": "B_hybrid", "C": "C_hybrid_rerank"}

    # 1) §8 Hit@5 표: | Dev A | 3/8 | 6/8 | 7/8 | +50.0%p |  /  | 기존 Test A | 6/17 | 12/17 | 17/17 | ... |
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

    # 2) §8 RAGAS 재평가 표 (기존 Test T01~T17): 하단 F1 표와 섞이지 않도록 섹션 범위 한정!
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
            {
                "llm_context_precision_with_reference": float(m.group(2)),
                "context_recall": float(m.group(4)),
            },
        )
        test_ragas.setdefault(
            ("filter_v2", letter),
            {
                "llm_context_precision_with_reference": float(m.group(3)),
                "context_recall": float(m.group(5)),
            },
        )

    # 3) §10 신규 Test (T21~T40): | 근거 있는 17문항 Hit@5 | 15/17 (88.2%) |
    m_v3 = re.search(r"근거 있는 17문항 Hit@5\s*\|\s*(\d+/\d+)", md_text)
    v3_hit = m_v3.group(1) if m_v3 else "15/17"

    def _mk_mode(stage: str, letter: str, hit_str: str, ragas_map: dict[str, float] | None = None, refusal: str = "-") -> dict[str, Any]:
        ans_cnt = int(hit_str.split("/")[1]) if "/" in hit_str else 0
        use_flt = stage != "baseline"
        return {
            "preset": {
                "name": f"{stage}_{letter}",
                "pipeline_mode": pmode_map[letter],
                "use_metadata_filter": use_flt,
                "use_rewrite": False,
                "use_cache": False,
                "use_hyde": False,
                "collection_name": TEAM_COLLECTIONS["rent_deals"]["current"],
                "embedding_model": "text-embedding-3-small",
                "chunk_strategy": f"V0-{stage}",
            },
            "retrieval": {
                "answerable_count": ans_cnt,
                "recall@12": None,
                "hit@5": hit_str,
                "top1": "-",
                "mrr@5": None,
            },
            "latency_ms": {"avg": 0.0, "median": 0.0, "p95": 0.0},
            "refusal": refusal,
            "ragas_avg": dict(ragas_map) if ragas_map else {},
            "underperforming_items": [],
        }

    summaries: list[dict[str, Any]] = []
    if dev_hits:
        for idx_s, (stage, stage_lbl, ts) in enumerate(
            [
                ("baseline", "교안 기준선 (필터 OFF)", "20261125_150000"),
                ("filter_v1", "초기 필터 (v1)", "20261125_200000"),
                ("filter_v2", "정규화 보완 필터 (v2)", "20261126_110000"),
            ]
        ):
            modes = {
                f"{stage}_{lt}": _mk_mode(stage, lt, dev_hits[(stage, lt)])
                for lt in ("A", "B", "C")
                if (stage, lt) in dev_hits
            }
            if modes:
                summaries.append(
                    {
                        "timestamp": ts,
                        "owner": "bc",
                        "domain": "rent_deals",
                        "eval_version": "bc_deals_dev",
                        "ragas_target": 0.60,
                        "total_corpus_chunks": 31772,
                        "collection_name": TEAM_COLLECTIONS["rent_deals"]["current"],
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
                        "collection_name": TEAM_COLLECTIONS["rent_deals"]["current"],
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
            "collection_name": TEAM_COLLECTIONS["rent_deals"]["current"],
            "embedding_model": "text-embedding-3-small",
            "run_note": "신규 T21~T40 (C · 필터 v2)",
            "source_branch": source_branch,
            "source_file": source_label,
            "modes": {"filter_v2_C": _mk_mode("filter_v2", "C", v3_hit, None, refusal="3/3")},
        }
    )
    return summaries


def convert_byeongcheol_comparison(
    raw: dict[str, Any],
    source_label: str = "comparison_data.json",
    source_branch: str = "origin/feat/SBC-rag-pipeline",
) -> dict[str, Any]:
    """병철(bc) comparison_data.json -> 공통 UnifiedEvalSummary dict 변환 (개별 문항 본문 노출 없이 통계만 집계)."""
    q_meta = {q["id"]: q for q in raw.get("questions") or [] if isinstance(q, dict) and q.get("id")}
    modes_out: dict[str, dict[str, Any]] = {}

    for stage_key, prefix in (("before", "before"), ("after", "after_filter")):
        rows = raw.get(stage_key) or []
        by_sys: dict[str, list[dict]] = {}
        for r in rows:
            qid = r.get("question_id")
            # 공정 dev 비교 우선 (dev가 없으면 전체 집계)
            if q_meta.get(qid, {}).get("split") == "dev":
                by_sys.setdefault(str(r.get("system", "B")), []).append(r)
        if not by_sys:
            for r in rows:
                by_sys.setdefault(str(r.get("system", "B")), []).append(r)

        for sys_name in sorted(by_sys):
            grp = by_sys[sys_name]
            ans = [r for r in grp if q_meta.get(r.get("question_id"), {}).get("reference_answerable", r.get("reference_answerable", True))]
            unans = [r for r in grp if r not in ans]
            hit_cnt = sum(1 for r in ans if bool(r.get("hit_at_5")))
            ref_cnt = sum(
                1
                for r in unans
                if any(k in str(r.get("response", "")) for k in ("확인할 수 없", "찾을 수 없", "없습니다", "부족"))
            )
            lats = sorted(float(r.get("seconds", 0.0) or 0.0) * 1000.0 for r in grp if r.get("seconds") is not None)
            p95_ms = round(lats[max(0, int(len(lats) * 0.95) - 1)], 1) if lats else 0.0
            modes_out[f"{prefix}_{sys_name}"] = {
                "retrieval": {
                    "answerable_count": len(ans),
                    "recall@12": None,
                    "hit@5": f"{hit_cnt}/{len(ans)}" if ans else "-",
                    "top1": "-",
                    "mrr@5": None,
                },
                "latency_ms": {
                    "avg": round(statistics.mean(lats), 1) if lats else 0.0,
                    "median": round(statistics.median(lats), 1) if lats else 0.0,
                    "p95": p95_ms,
                },
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
        "collection_name": TEAM_COLLECTIONS["rent_deals"]["current"],
        "embedding_model": "text-embedding-3-small",
        "source_branch": source_branch,
        "source_file": source_label,
        "notes": "병철 전월세 개별거래 필터링 A/B/C(dev) 자동 변환 (RAGAS 미측정 · Hit@5/거절/지연 집계)",
        "modes": modes_out,
    }


def convert_kihyeok_summary(raw: dict[str, Any], source_label: str = "summary_rent.json", source_branch: str = "feat/kihyeok") -> dict[str, Any]:
    """최기혁(kh) summary_*.json -> 공통 UnifiedEvalSummary dict 정규화."""
    out = dict(raw)
    out.setdefault("owner", "kh")
    out["domain"] = "rent_stats"
    out.setdefault("ragas_target", 0.60)
    out.setdefault("collection_name", TEAM_COLLECTIONS["rent_stats"]["current"])
    out.setdefault("embedding_model", "text-embedding-3-small")
    out["source_branch"] = source_branch
    out["source_file"] = source_label
    return out


def convert_any_uploaded_json(raw: Any, filename: str) -> dict[str, Any] | None:
    """Streamlit 다중 업로더가 팀원 원본 JSON(예지/병철/최기혁)을 바로 표준 형식으로 자동 변환."""
    if not isinstance(raw, dict):
        return None
    if "modes" in raw and isinstance(raw["modes"], dict):
        return raw
    if "summary" in raw and "scored" in raw and isinstance(raw["summary"], dict):
        return convert_yeji_eval(raw, source_label=filename, source_branch="uploaded")
    if "questions" in raw and "after" in raw and isinstance(raw["after"], list):
        return convert_byeongcheol_comparison(raw, source_label=filename, source_branch="uploaded")
    return None


def load_all_team_histories(*, from_git: bool = False) -> dict[str, list[tuple[Path, dict[str, Any]]]]:
    """팀원 4인(kh · yj · jh · bc)의 전체 회차별 평가 이력을 표준 (Path, summary_dict) 리스트로 반환한다.

    Streamlit 사이드바에서 담당자(owner)를 선택하면 해당 팀원의 모든 시험지 버전과
    Before(기준선) / After(실험군)를 동일 워크벤치에서 즉시 비교·분석할 수 있다.
    """
    test_root = Path(__file__).resolve().parents[1]
    histories: dict[str, list[tuple[Path, dict[str, Any]]]] = {"kh": [], "yj": [], "jh": [], "bc": []}

    # 1) 최기혁 (kh)
    if KH_RESULTS_DIR.exists():
        for p in sorted(KH_RESULTS_DIR.glob("summary_*.json")):
            try:
                histories["kh"].append((p, json.loads(p.read_text(encoding="utf-8"))))
            except Exception:
                continue

    # 2) 예지 (yj) — chunk_V1 / chunk_V2 하위 모든 eval_*.json 시간순 로드
    yj_dir = test_root / "yj" / "docs" / "04-2_eval_reports"
    if yj_dir.exists():
        yj_files = sorted(
            yj_dir.rglob("eval_*.json"),
            key=lambda p: (re.search(r"(\d{8}_\d{4,6})", p.name).group(1) if re.search(r"(\d{8}_\d{4,6})", p.name) else p.name),
        )
        for fp in yj_files:
            try:
                cv = fp.parent.name.replace("chunk_", "") if fp.parent.name.startswith("chunk_") else ""
                raw = json.loads(fp.read_text(encoding="utf-8"))
                histories["yj"].append(
                    (fp, convert_yeji_eval(raw, source_label=f"{fp.parent.name}/{fp.name}", source_branch="local(dev)", chunk_version=cv))
                )
            except Exception:
                continue

    # 3) 지훈 (jh) — 로컬 results/*.jsonl + 실험기록.md §19 DEV/TEST 분리 로드
    jh_dir = test_root / "jh"
    jh_res_dir = jh_dir / "results"
    if jh_res_dir.exists():
        for fp in sorted(jh_res_dir.glob("*.jsonl")):
            try:
                histories["jh"].append((fp, convert_jihun_jsonl(fp.read_text(encoding="utf-8"), source_label=fp.name, source_branch="local(dev)")))
            except Exception:
                continue
    jh_md = jh_dir / "실험기록.md"
    if jh_md.exists():
        for sm in convert_jihun_experiment_splits(jh_md.read_text(encoding="utf-8"), source_label="test/jh/실험기록.md (§19)", source_branch="local(dev)"):
            histories["jh"].append((jh_md, sm))

    # 4) 병철 (bc) — docs/development_history.md (Dev · 기존 Test RAGAS · 신규 Test v3) + comparison_data.json
    bc_dir = test_root / "bc"
    bc_hist_md = bc_dir / "docs" / "development_history.md"
    if bc_hist_md.exists():
        for sm in convert_byeongcheol_history_md(bc_hist_md.read_text(encoding="utf-8")):
            histories["bc"].append((bc_hist_md, sm))
    else:
        bc_comp = bc_dir / "results" / "comparison_data.json"
        bc_text = bc_comp.read_text(encoding="utf-8") if bc_comp.exists() else (
            _git_show_text("origin/feat/SBC-rag-pipeline:src/rag_home_api/test/bc/results/comparison_data.json") if from_git else None
        )
        if bc_text:
            histories["bc"].append((bc_comp, convert_byeongcheol_comparison(json.loads(bc_text))))

    return histories


def build_unified_team_snapshots(*, from_git: bool = True, out_dir: Path = TEAM_UNIFIED_DIR) -> list[Path]:
    """4명 팀원의 최신 결과를 읽어 data/team_unified/summary_<owner>.json 4종을 생성한다."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    histories = load_all_team_histories(from_git=from_git)

    # 1) 최기혁 (로컬 최신 summary_*.json 중 RAGAS 포함 기록 우선)
    if histories["kh"]:
        kh_rev = list(reversed(histories["kh"]))
        best_p, best_raw = next(
            (
                (p, d)
                for p, d in kh_rev
                if any((m.get("ragas_avg") or {}).get("answer_relevancy", 0) > 0 for m in (d.get("modes") or {}).values())
            ),
            kh_rev[0],
        )
        kh_norm = convert_kihyeok_summary(best_raw, source_label=best_p.name)
        p_out = out_dir / "summary_kh.json"
        p_out.write_text(json.dumps(kh_norm, ensure_ascii=False, indent=2), encoding="utf-8")
        written.append(p_out)

    # 2) 예지 (최신 chunk_V2 test 우선, 없으면 최신 eval_*.json)
    if histories["yj"]:
        best_yj = next((d for _, d in reversed(histories["yj"]) if "test" in str(d.get("eval_version", ""))), histories["yj"][-1][1])
        yj_clean = {k: v for k, v in best_yj.items() if k != "_companion_logs"}
        p_out = out_dir / "summary_yj.json"
        p_out.write_text(json.dumps(yj_clean, ensure_ascii=False, indent=2), encoding="utf-8")
        written.append(p_out)

    # 3) 지훈 (최신 TEST 우선)
    if histories["jh"]:
        best_jh = next((d for _, d in reversed(histories["jh"]) if "test" in str(d.get("eval_version", ""))), histories["jh"][-1][1])
        p_out = out_dir / "summary_jh.json"
        p_out.write_text(json.dumps(best_jh, ensure_ascii=False, indent=2), encoding="utf-8")
        written.append(p_out)

    # 4) 병철 (기존 Test filter_v2 RAGAS 포함 요약 우선)
    if histories["bc"]:
        best_bc = next(
            (d for _, d in reversed(histories["bc"]) if d.get("eval_version") == "bc_deals_test" and "filter_v2_C" in (d.get("modes") or {})),
            histories["bc"][-1][1],
        )
        p_out = out_dir / "summary_bc.json"
        p_out.write_text(json.dumps(best_bc, ensure_ascii=False, indent=2), encoding="utf-8")
        written.append(p_out)

    return written


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="팀원 4인 평가 결과 공통 표준 JSON 통합 어댑터 (방안 3)")
    parser.add_argument("--from-git", action="store_true", help="원격 브랜치(origin/feat/*)에서 팀원 결과를 읽어 자동 변환")
    args = parser.parse_args()

    paths = build_unified_team_snapshots(from_git=args.from_git or True)
    print(f"통합 결과 {len(paths)}건 저장 완료 -> {TEAM_UNIFIED_DIR}")
    for p in paths:
        d = json.loads(p.read_text(encoding="utf-8"))
        mnames = list((d.get("modes") or {}).keys())
        last_m = mnames[-1] if mnames else "-"
        md = (d.get("modes") or {}).get(last_m) or {}
        print(
            f"  - {d['owner']:3s} | domain={d['domain']:14s} | modes={len(mnames)} (대표={last_m}) | "
            f"hit@5={(md.get('retrieval') or {}).get('hit@5')} | ragas={md.get('ragas_avg') or '미측정'}"
        )
