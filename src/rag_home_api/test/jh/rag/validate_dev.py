"""외부 호출 없이 활성 법령 DEV의 구성·원문·청크·기계적 중복을 확인함."""

import hashlib
import json
import re
from collections import Counter

from common import DOCS, EVAL, FILES, eval_sha, load_chunks, load_eval


def main():
    dev = load_eval("dev")["items"]
    test = load_eval("test")["items"]
    archive = json.loads((EVAL / "이전DEV_ce06a5d5ac03.json").read_text(encoding="utf-8"))
    assert len(dev) == 10 and len(test) == 20
    assert {q["question_id"] for q in dev} == {f"JH-DEV-{i:02d}" for i in range(1, 11)}
    assert Counter(q["expected_status"] for q in dev) == {"answered": 8, "insufficient_evidence": 2}
    test_hash = hashlib.sha256(json.dumps(sorted(test, key=lambda q: q["question_id"]),
                                        ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    assert test_hash == archive["test_items_sha256"], "기존 TEST 객체 변경"
    rows = {}
    for name in FILES:
        for line in (DOCS / f"{name}.jsonl").read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            rows[row["id"]] = row
    chunks = {d.metadata["chunk_id"]: d for collection in ("jh_500", "jh_1000")
              for d in load_chunks(collection)}
    quotes_checked = 0
    for q in dev:
        assert q["axis"] == "임대차 관련 법령" and q["application_conditions"]
        if q["expected_status"] == "insufficient_evidence":
            assert q["eval_category"] == "no_evidence" and q["absence_reason"]
            assert not any(q[key] for key in ("key_facts", "allowed_doc_ids", "allowed_row_ids",
                                             "allowed_chunk_ids", "evidence_quotes"))
            continue
        assert q["eval_category"] == "answerable" and q["key_facts"]
        assert q["source_location"] and q["source_url"]
        assert all(doc.startswith("LAW_") for doc in q["allowed_doc_ids"])
        assert {rows[rid]["doc_id"] for rid in q["allowed_row_ids"]} == set(q["allowed_doc_ids"])
        for evidence in q["evidence_quotes"]:
            assert evidence["row_id"] in q["allowed_row_ids"]
            assert evidence["quote"] in rows[evidence["row_id"]]["text"]
            quotes_checked += 1
        for size in (500, 1000):
            mapped = [cid for cid in q["allowed_chunk_ids"] if f"_s{size}_" in cid]
            assert mapped, "청킹 설정별 근거 청크 필요"
            for cid in mapped:
                doc = chunks[cid]
                assert doc.metadata["row_id"] in q["allowed_row_ids"]
                assert any(e["row_id"] == doc.metadata["row_id"] and e["quote"] in doc.page_content
                           for e in q["evidence_quotes"])
    normalize = lambda text: re.sub(r"\s+", "", text)
    assert len({normalize(q["question"]) for q in dev}) == 10
    assert not ({normalize(q["question"]) for q in dev} & {normalize(q["question"]) for q in test})
    dev_facts = {normalize(fact) for q in dev for fact in q["key_facts"]}
    test_facts = {normalize(fact) for q in test for fact in q["key_facts"]}
    assert not dev_facts.intersection(test_facts), "동일한 필수 정답 문구"
    for pattern in (r"벽지.{0,60}(?:제곱미터|㎡|단가)", r"(?:㎡|제곱미터).{0,60}배상",
                    r"지연기간", r"신고.{0,15}지연.{0,30}과태료"):
        assert not any(re.search(pattern, normalize(row["text"])) for row in rows.values())
    assert {row["article"] for row in rows.values() if row["doc_id"] == "LAW_REPORT_DECREE"} == {"제4조의3", "제20조"}
    print(json.dumps({"eval_sha": eval_sha(), "dev": 10, "answered": 8, "no_evidence": 2,
                      "test_unchanged": 20, "quotes_checked": quotes_checked,
                      "chunk_mappings": sum(len(q["allowed_chunk_ids"]) for q in dev),
                      "exact_question_overlap": 0, "exact_fact_overlap": 0,
                      "semantic_review": "평가셋.md의 200쌍 로컬 의미 대조 기록 참고"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
