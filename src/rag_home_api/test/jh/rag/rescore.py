"""이미 돌린 검색 결과(retrieved)를 지금 평가셋의 허용 근거로 다시 채점한다(검색은 다시 하지 않음).

예) uv run python src/rag_home_api/test/jh/rag/rescore.py dev
평가셋을 고친 효과(채점 기준 변화)와 검색을 고친 효과를 나눠 보기 위함.
예전 평가셋 버전으로 채점하려면 그 버전이 커밋된 시점의 평가셋 파일을 꺼내 쓴다(git log 평가셋/).
"""

import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.stdout.reconfigure(encoding="utf-8")

from common import RESULTS, eval_sha, load_eval  # noqa: E402
from score import evidence_rank, load_groups  # noqa: E402

split = sys.argv[1] if len(sys.argv) > 1 else "dev"
questions = {q["question_id"]: q for q in load_eval(split)["items"]}
print(f"지금 평가셋({eval_sha()}) 기준")
for meta, rows in load_groups(RESULTS / f"{split}_원본로그.jsonl"):
    tally = defaultdict(lambda: [0, 0])
    for r in rows:
        if r["expected_status"] == "answered":
            tally[r["mode"]][0] += evidence_rank(r.get("retrieved", []), questions[r["question_id"]]) is not None
            tally[r["mode"]][1] += 1
    print(f"{meta['tag']:22} (실행 당시 평가셋 {meta['eval_sha']})", {m: f"{a}/{b}" for m, (a, b) in tally.items()})
    changed = sorted({r["question_id"] for r in rows
                      if r.get("question") != questions[r["question_id"]]["question"]})
    if changed:
        print("질문 문구가 달라 이전 질문 결과로만 해석할 문항:", ", ".join(changed))
