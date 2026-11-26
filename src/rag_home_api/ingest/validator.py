"""공유 JSONL 데이터 정합성 검증 도구 (키 일관성 · 빈 값 · 중복 ID · JSON 파싱 오류 체크)."""

import json
from collections import Counter
from pathlib import Path

from rag_home_api.config import SHARED_DOCS_DIR


def validate_jsonl(jsonl_path: Path) -> None:
    if not jsonl_path.exists():
        print(f"[없음] {jsonl_path}")
        return

    key_sets: Counter[tuple[str, ...]] = Counter()
    empty_counts: Counter[str] = Counter()
    id_counts: Counter[str] = Counter()
    broken_lines: list[int] = []
    total_rows = 0

    with jsonl_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                broken_lines.append(line_no)
                continue

            total_rows += 1
            key_sets[tuple(sorted(row.keys()))] += 1
            id_counts[str(row.get("id", ""))] += 1
            for k, v in row.items():
                if v in ("", None, []):
                    empty_counts[k] += 1

    dup_ids = [k for k, c in id_counts.items() if c > 1]
    print(f"\n=== 검증: {jsonl_path.name} (총 {total_rows:,}줄) ===")
    print(f"  - 파싱 실패: {len(broken_lines)}건 | 중복 id: {len(dup_ids)}건 | 키 구성 종류: {len(key_sets)}종류")
    for keys, cnt in key_sets.items():
        print(f"    · {cnt:,}줄: {list(keys)}")
    non_zero_empty = {k: c for k, c in sorted(empty_counts.items()) if c > 0}
    print(f"  - 빈 값 있는 키: {non_zero_empty}")


if __name__ == "__main__":
    for name in ("일반전월세_법령가이드.jsonl", "공공임대_공고문.jsonl", "청년안심주택_공고문.jsonl"):
        validate_jsonl(SHARED_DOCS_DIR / name)