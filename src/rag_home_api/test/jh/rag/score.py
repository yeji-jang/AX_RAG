"""A/B 검색 원본로그 요약.

예) uv run python src/rag_home_api/test/jh/rag/score.py dev

- Hit@5: 상위 5개에서 허용 청크 ID를 우선 대조하며, 매핑 전 문항은 원문 줄 ID로 잠정 채점
- 호출 성공률(오류 아님), 응답 시간 평균·중앙값, 오류 수
"""

import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from common import RESULTS  # noqa: E402


def evidence_rank(hits: list[dict], question: dict) -> int | None:
    """허용 청크가 있으면 청크로, 없으면 원문 줄로 상위 5개를 판정함."""
    chunk_ids = question.get("allowed_chunk_ids") or []
    field = "chunk_id" if chunk_ids else "row_id"
    allowed = set(chunk_ids or question.get("allowed_row_ids") or [])
    return next((rank for rank, hit in enumerate(hits[:5], 1) if hit.get(field) in allowed), None)


def load_groups(path: Path) -> list[tuple[dict, list[dict]]]:
    """원본로그 = 실행마다 {"meta"} 한 줄 + 결과 줄들. 실행 단위로 나눠 돌려준다."""
    groups: list[tuple[dict, list[dict]]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        x = json.loads(line)
        if "record_type" in x or x.get("meta", {}).get("experiment") == "C_vs_rewrite_C":
            continue  # 동일 로그에 이어 쓴 리라이트 비교는 별도로 집계함.
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
    sys.stdout.reconfigure(encoding="utf-8")
    split = sys.argv[1] if len(sys.argv) > 1 else "dev"
    for meta, rows in load_groups(RESULTS / f"{split}_원본로그.jsonl"):
        print(f"\n## {meta['tag']}  ({meta['collection']}, 평가셋={meta['eval_sha']})")
        for mode in dict.fromkeys(r["mode"] for r in rows):
            print(mode, json.dumps(summarize([r for r in rows if r["mode"] == mode]), ensure_ascii=False))
