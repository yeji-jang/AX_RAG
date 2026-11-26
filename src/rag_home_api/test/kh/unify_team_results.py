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


def convert_yeji_eval(raw: dict[str, Any], source_label: str = "eval_dev.json", source_branch: str = "origin/feat/yeji") -> dict[str, Any]:
    """예지(yj) eval_dev_*.json -> 공통 UnifiedEvalSummary dict 변환."""
    summary_in = raw.get("summary") or {}
    scored_in = raw.get("scored") or {}
    modes_out: dict[str, dict[str, Any]] = {}

    for mode_name, m in summary_in.items():
        scored_rows = scored_in.get(mode_name) or []
        ans_rows = [r for r in scored_rows if r.get("query_type") != "unanswerable" and r.get("status") != "insufficient_evidence"]
        hit_str = _clean_ratio(m.get("hit_at_5"))
        ans_cnt = int(hit_str.split("/")[1]) if "/" in hit_str else len(ans_rows)

        avg_sec = float(m.get("avg_sec", 0.0) or 0.0)
        med_sec = float(m.get("median_sec", 0.0) or 0.0)
        lat_list = sorted(float(r.get("elapsed_sec", 0.0) or 0.0) * 1000.0 for r in scored_rows if r.get("elapsed_sec") is not None)
        p95_ms = round(lat_list[max(0, int(len(lat_list) * 0.95) - 1)], 1) if lat_list else round(avg_sec * 1000.0, 1)

        modes_out[mode_name] = {
            "retrieval": {
                "answerable_count": ans_cnt,
                "recall@12": None,
                "hit@5": hit_str,
                "top1": "-",
                "mrr@5": None,
            },
            "latency_ms": {
                "avg": round(avg_sec * 1000.0, 1),
                "median": round(med_sec * 1000.0, 1),
                "p95": p95_ms,
            },
            "refusal": _clean_ratio(m.get("unanswerable_refusal")),
            "ragas_avg": {
                "faithfulness": round(float(m.get("avg_faithfulness", 0.0) or 0.0), 4),
                "answer_relevancy": round(float(m.get("avg_answer_relevancy", 0.0) or 0.0), 4),
                "llm_context_precision_with_reference": round(float(m.get("avg_context_precision", 0.0) or 0.0), 4),
                "context_recall": round(float(m.get("avg_context_recall", 0.0) or 0.0), 4),
            },
            "underperforming_items": [],
        }

    m_ts = re.search(r"(\d{8}_\d{4,6})", source_label)
    ts = m_ts.group(1) if m_ts else datetime.now().strftime("%Y%m%d_%H%M%S")
    return {
        "timestamp": ts,
        "owner": "yj",
        "domain": "public_housing",
        "eval_version": f"yj_{raw.get('split', 'dev')}",
        "ragas_target": 0.70,
        "collection_name": str(raw.get("collection") or TEAM_COLLECTIONS["public_housing"]["current"]),
        "embedding_model": "text-embedding-3-small",
        "source_branch": source_branch,
        "source_file": source_label,
        "notes": "예지 공공임대 공고문 A/B/C RAGAS 평가 자동 변환",
        "modes": modes_out,
    }


def convert_jihun_jsonl(jsonl_text: str, source_label: str = "dev_원본로그.jsonl", source_branch: str = "origin/feat/cjh") -> dict[str, Any]:
    """지훈(jh) dev_원본로그.jsonl -> 공통 UnifiedEvalSummary dict 변환 (RAGAS 미측정 시 정직하게 비움)."""
    rows = [json.loads(line) for line in jsonl_text.splitlines() if line.strip()]
    metas_by_tag: dict[str, dict] = {}
    groups: dict[str, list[dict]] = {}
    latest_ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    for r in rows:
        if "meta" in r:
            tag = str(r["meta"].get("tag", "run"))
            metas_by_tag[tag] = r["meta"]
            if r["meta"].get("started"):
                latest_ts = str(r["meta"]["started"])
            continue
        run_id = str(r.get("run_id", ""))
        parts = run_id.split(":")
        tag = parts[0] if len(parts) >= 2 else "run"
        mode = str(r.get("mode", parts[1] if len(parts) >= 2 else "B"))
        key = f"{tag}_{mode}"
        groups.setdefault(key, []).append(r)

    modes_out: dict[str, dict[str, Any]] = {}
    for gkey, recs in groups.items():
        ans = [r for r in recs if r.get("expected_status") == "answered"]
        hit_cnt = sum(1 for r in ans if r.get("hit5") is True)
        top1_cnt = sum(1 for r in ans if r.get("first_hit_rank") == 1)
        mrr_vals = [(1.0 / int(r["first_hit_rank"])) if r.get("first_hit_rank") else 0.0 for r in ans]
        lats = sorted(float(r.get("latency_sec", 0.0) or 0.0) * 1000.0 for r in recs if r.get("latency_sec") is not None)
        p95_ms = round(lats[max(0, int(len(lats) * 0.95) - 1)], 1) if lats else 0.0
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
            "refusal": "검색전용",
            "ragas_avg": {},
            "underperforming_items": [],
        }

    return {
        "timestamp": latest_ts,
        "owner": "jh",
        "domain": "law",
        "eval_version": "jh_law_dev",
        "ragas_target": 0.70,
        "collection_name": TEAM_COLLECTIONS["law"]["current"],
        "embedding_model": "text-embedding-3-small",
        "source_branch": source_branch,
        "source_file": source_label,
        "notes": "지훈 법령 파트 검색 원본로그 자동 변환 (RAGAS 미측정 · 추후 RAGAS JSON 연동 시 자동 반영)",
        "modes": modes_out,
    }


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


def build_unified_team_snapshots(*, from_git: bool = True, out_dir: Path = TEAM_UNIFIED_DIR) -> list[Path]:
    """4명 팀원의 최신 결과를 읽어 data/team_unified/summary_<owner>.json 4종을 생성한다."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    # 1) 최기혁 (로컬 최신 summary_*.json 중 RAGAS 통과 기록 우선)
    kh_files = sorted(KH_RESULTS_DIR.glob("summary_*.json"), reverse=True) if KH_RESULTS_DIR.exists() else []
    if kh_files:
        # 가장 최신 RAGAS 포함 결과(예: 154004 등) 우선, 없으면 최신 파일
        best_kh = next(
            (
                p
                for p in kh_files
                if any((m.get("ragas_avg") or {}).get("answer_relevancy", 0) > 0 for m in json.loads(p.read_text(encoding="utf-8")).get("modes", {}).values())
            ),
            kh_files[0],
        )
        kh_raw = json.loads(best_kh.read_text(encoding="utf-8"))
        kh_norm = convert_kihyeok_summary(kh_raw, source_label=best_kh.name)
        p_out = out_dir / "summary_kh.json"
        p_out.write_text(json.dumps(kh_norm, ensure_ascii=False, indent=2), encoding="utf-8")
        written.append(p_out)

    # 2) 예지 (로컬 우선, 없으면 origin/feat/yeji)
    yj_local_dir = Path(__file__).resolve().parents[1] / "yj" / "docs" / "04-2_eval_reports"
    yj_text: str | None = None
    yj_name = "eval_dev_20261125_2155.json"
    if yj_local_dir.exists():
        yj_local = sorted(yj_local_dir.glob("eval_*.json"), reverse=True)
        if yj_local:
            yj_name = yj_local[0].name
            yj_text = yj_local[0].read_text(encoding="utf-8")
    if yj_text is None and from_git:
        yj_text = _git_show_text(f"origin/feat/yeji:src/rag_home_api/test/yj/docs/04-2_eval_reports/{yj_name}")
    if yj_text:
        yj_norm = convert_yeji_eval(json.loads(yj_text), source_label=yj_name)
        p_out = out_dir / "summary_yj.json"
        p_out.write_text(json.dumps(yj_norm, ensure_ascii=False, indent=2), encoding="utf-8")
        written.append(p_out)

    # 3) 지훈 (로컬 우선, 없으면 origin/feat/cjh)
    jh_local = Path(__file__).resolve().parents[1] / "jh" / "results" / "dev_원본로그.jsonl"
    jh_text = jh_local.read_text(encoding="utf-8") if jh_local.exists() else (
        _git_show_text("origin/feat/cjh:src/rag_home_api/test/jh/results/dev_원본로그.jsonl") if from_git else None
    )
    if jh_text:
        jh_norm = convert_jihun_jsonl(jh_text)
        p_out = out_dir / "summary_jh.json"
        p_out.write_text(json.dumps(jh_norm, ensure_ascii=False, indent=2), encoding="utf-8")
        written.append(p_out)

    # 4) 병철 (로컬 우선, 없으면 origin/feat/SBC-rag-pipeline)
    bc_local = Path(__file__).resolve().parents[1] / "bc" / "results" / "comparison_data.json"
    bc_text = bc_local.read_text(encoding="utf-8") if bc_local.exists() else (
        _git_show_text("origin/feat/SBC-rag-pipeline:src/rag_home_api/test/bc/results/comparison_data.json") if from_git else None
    )
    if bc_text:
        bc_norm = convert_byeongcheol_comparison(json.loads(bc_text))
        p_out = out_dir / "summary_bc.json"
        p_out.write_text(json.dumps(bc_norm, ensure_ascii=False, indent=2), encoding="utf-8")
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
