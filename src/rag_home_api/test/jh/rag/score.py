"""A/B 검색 원본로그 요약.

예) uv run python src/rag_home_api/test/jh/rag/score.py dev

- Hit@5: 답 있는 문항 중 상위 5개에 허용 근거 줄(allowed_row_ids)이 하나라도 있는 비율
- 호출 성공률(오류 아님), 응답 시간 평균·중앙값, 오류 수
"""

import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.stdout.reconfigure(encoding="utf-8")

from common import RESULTS  # noqa: E402


def load_groups(path: Path) -> list[tuple[dict, list[dict]]]:
    """원본로그 = 실행마다 {"meta"} 한 줄 + 결과 줄들. 실행 단위로 나눠 돌려준다."""
    groups: list[tuple[dict, list[dict]]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        x = json.loads(line)
        if "meta" in x:
            groups.append((x["meta"], []))
        else:
            groups[-1][1].append(x)
    return groups


def summarize(rows: list[dict]) -> dict:
    ans = [r for r in rows if r["expected_status"] == "answered"]
    lat = [r["latency_sec"] for r in rows if r["status"] != "error"]
    return {
        "n": len(rows),
        "hit5": f"{sum(bool(r.get('hit5')) for r in ans)}/{len(ans)}",
        "errors": sum(r["status"] == "error" for r in rows),
        "success": f"{sum(r['status'] != 'error' for r in rows)}/{len(rows)}",
        "lat_mean": round(statistics.mean(lat), 2) if lat else None,
        "lat_median": round(statistics.median(lat), 2) if lat else None,
    }


if __name__ == "__main__":
    split = sys.argv[1] if len(sys.argv) > 1 else "dev"
    for meta, rows in load_groups(RESULTS / f"{split}_원본로그.jsonl"):
        print(f"\n## {meta['tag']}  ({meta['collection']}, 평가셋={meta['eval_sha']})")
        for mode in dict.fromkeys(r["mode"] for r in rows):
            print(mode, json.dumps(summarize([r for r in rows if r["mode"] == mode]), ensure_ascii=False))
