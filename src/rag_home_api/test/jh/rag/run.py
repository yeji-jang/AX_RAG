"""A/B 검색 평가 실행기: 평가셋 × 검색 방식 → 원본 로그.

예) uv run python src/rag_home_api/test/jh/rag/run.py --split dev --modes A B --preset tok --tag dev-t-tok
    --preset: r0(기본) / dedupe(개선 1) / route(개선 2) / tok(개선 3) — retrieve.PRESETS

결과는 지우지 않는다(실패·악화 포함). results/<split>_원본로그.jsonl 한 파일에 실행마다 이어 쓴다
(실행 하나 = {"meta": 설정} 한 줄 + 문항별 결과 줄들).
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

from common import MODELS, RESULTS, eval_sha, load_eval, require_dev_api_scope  # noqa: E402
from retrieve import CONFIG, PRESETS, Retriever, apply_preset, route  # noqa: E402
from score import evidence_rank  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["dev", "test"])
    ap.add_argument("--modes", nargs="+", default=["A", "B"])
    ap.add_argument("--collection", default="jh_1000")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--preset", default="tok", choices=list(PRESETS))
    args = ap.parse_args()

    if args.split == "dev":
        require_dev_api_scope()

    apply_preset(args.preset)
    ev = load_eval(args.split)
    ret = Retriever(args.collection)
    RESULTS.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = RESULTS / f"{args.split}_원본로그.jsonl"
    meta = {"tag": args.tag, "split": args.split, "eval_version": ev["version"], "eval_sha": eval_sha(),
            "collection": args.collection, "chunks": len(ret.chunks), "modes": args.modes, "models": MODELS,
            "code": "LangChain(BM25Retriever + EnsembleRetriever + EmbeddingsRedundantFilter + RunnableBranch)", "preset": args.preset,
            "retrieval_config": CONFIG, "started": stamp}
    print(json.dumps(meta, ensure_ascii=False))
    with open(out, "a", encoding="utf-8") as f:
        f.write(json.dumps({"meta": meta}, ensure_ascii=False) + "\n")
        for mode in args.modes:
            for q in ev["items"]:
                row = run_one(ret, mode, q, args)
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
                print(f"{mode} {q['question_id']:12} hit5={row.get('hit5')} {row['latency_sec']}s")
    print("저장:", out)


def run_one(ret: Retriever, mode: str, q: dict, args) -> dict:
    row = {"run_id": f"{args.tag}:{mode}:{q['question_id']}", "mode": mode, "question_id": q["question_id"],
           "eval_category": q["eval_category"], "expected_status": q["expected_status"], "question": q["question"],
           "query_used": q["question"], "status": "retrieval_only"}
    t0 = time.time()
    try:
        row["route"] = route(q["question"])  # law = 법령·가이드만, all = 전체
        hits = ret.search(mode, q["question"])  # 나누기는 검색기 안의 RunnableBranch가 함
        row["retrieved"] = hits
        row["evidence_unit"] = "chunk" if q.get("allowed_chunk_ids") else "row"
        if q["expected_status"] == "answered":
            rank = evidence_rank(hits, q)
            row["hit5"] = rank is not None
            row["first_hit_rank"] = rank
    except Exception as e:  # noqa: BLE001 — 검색 오류는 따로 기록
        row |= {"status": "error", "error": repr(e)[:300]}
    row["latency_sec"] = round(time.time() - t0, 2)
    return row


if __name__ == "__main__":
    main()
