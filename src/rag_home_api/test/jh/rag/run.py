"""A/B 검색 평가 실행기: 평가셋 × 검색 방식 → 원본 로그.

예) uv run python src/rag_home_api/test/jh/rag/run.py --split dev --modes A B --collection jh_1000 --tag dev-r0-1000

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

from common import MODELS, RESULTS, eval_sha, load_eval  # noqa: E402
from retrieve import CONFIG, Retriever, route  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["dev", "test"])
    ap.add_argument("--modes", nargs="+", default=["A", "B"])
    ap.add_argument("--collection", default="jh_1000")
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()

    ev = load_eval(args.split)
    ret = Retriever(args.collection)
    RESULTS.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = RESULTS / f"{args.split}_원본로그.jsonl"
    meta = {"tag": args.tag, "split": args.split, "eval_version": ev["version"], "eval_sha": eval_sha(),
            "collection": args.collection, "chunks": len(ret.chunks), "modes": args.modes, "models": MODELS,
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
        hits = ret.search(mode, q["question"], row["route"])
        row["retrieved"] = hits
        allowed = set(q.get("allowed_row_ids") or [])
        if allowed:
            ranks = [i + 1 for i, h in enumerate(hits) if h["row_id"] in allowed]
            row["hit5"] = bool(ranks)
            row["first_hit_rank"] = ranks[0] if ranks else None
    except Exception as e:  # noqa: BLE001 — 검색 오류는 따로 기록
        row |= {"status": "error", "error": repr(e)[:300]}
    row["latency_sec"] = round(time.time() - t0, 2)
    return row


if __name__ == "__main__":
    main()
