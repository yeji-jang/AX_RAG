"""외부 호출 없이 개인 법령 DEV·TEST 30개의 사전검증을 수행함."""

import json
import re
from collections import Counter

from common import DOCS, EVAL, FILES, eval_sha, load_chunks, load_eval


def main():
    dev = load_eval("dev")["items"]
    test = load_eval("test")["items"]
    archive = json.loads((EVAL / "이전평가셋_c3d4c4ead6cb.json").read_text(encoding="utf-8"))
    assert len(dev) == 10 and len(test) == 20
    assert {q["question_id"] for q in dev} == {f"JH-DEV-{i:02d}" for i in range(1, 11)}
    assert {q["question_id"] for q in test} == {f"JH-TEST-{i:02d}" for i in range(1, 21)}
    assert Counter(q["expected_status"] for q in dev) == {"answered": 8, "insufficient_evidence": 2}
    assert Counter(q["expected_status"] for q in test) == {"answered": 17, "insufficient_evidence": 3}
    assert Counter(q["eval_category"] for q in test) == {"direct": 10, "paraphrase": 7, "no_evidence": 3}
    assert not archive["active"] and len(archive["items"]) == 30
    rows = {}
    for name in FILES:
        for line in (DOCS / f"{name}.jsonl").read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            rows[row["id"]] = row
    chunks = {d.metadata["chunk_id"]: d for collection in ("jh_500", "jh_1000")
              for d in load_chunks(collection)}
    quotes_checked = 0
    for q in dev + test:
        assert q["axis"] == "임대차 관련 법령" and q["application_conditions"]
        if q["expected_status"] == "insufficient_evidence":
            assert q["eval_category"] == "no_evidence" and q["absence_reason"]
            assert not any(q[key] for key in ("key_facts", "allowed_doc_ids", "allowed_row_ids",
                                             "allowed_chunk_ids", "evidence_quotes"))
            continue
        assert q["eval_category"] in {"answerable", "direct", "paraphrase"} and q["key_facts"]
        assert q["source_location"] and q["source_url"]
        assert q["source_url"] == rows[q["allowed_row_ids"][0]]["source_url"]
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
        assert {e["fact_index"] for e in q["fact_evidence"]} == set(range(1, len(q["key_facts"]) + 1))
        for evidence in q["fact_evidence"] + q.get("condition_evidence", []):
            assert evidence["row_id"] in q["allowed_row_ids"]
            assert evidence["quote"] in rows[evidence["row_id"]]["text"]
            assert evidence["chunk_ids"]
            for size in (500, 1000):
                assert any(f"_s{size}_" in cid for cid in evidence["chunk_ids"])
            for cid in evidence["chunk_ids"]:
                assert chunks[cid].metadata["row_id"] == evidence["row_id"]
                assert evidence["quote"] in chunks[cid].page_content
                if "fact_index" in evidence:
                    assert cid in q["allowed_chunk_ids"]
    normalize = lambda text: re.sub(r"\s+", "", text)
    assert len({normalize(q["question"]) for q in dev + test}) == 30
    assert len({q["issue"] for q in dev + test}) == 30
    assert not ({normalize(q["question"]) for q in dev} & {normalize(q["question"]) for q in test})
    dev_facts = {normalize(fact) for q in dev for fact in q["key_facts"]}
    test_facts = {normalize(fact) for q in test for fact in q["key_facts"]}
    assert not dev_facts.intersection(test_facts), "동일한 필수 정답 문구"
    primary = [q["allowed_row_ids"][0] for q in dev + test if q["expected_status"] == "answered"]
    assert len(set(primary)) == 25, "이번 버전의 답변 가능 쟁점은 서로 다른 25개 조문 행으로 분리함"
    for pattern in (r"벽지.{0,60}(?:제곱미터|㎡|단가)", r"(?:㎡|제곱미터).{0,60}배상",
                    r"지연기간", r"신고.{0,15}지연.{0,30}과태료"):
        assert not any(re.search(pattern, normalize(row["text"])) for row in rows.values())
    assert {row["article"] for row in rows.values() if row["doc_id"] == "LAW_REPORT_DECREE"} == {"제4조의3", "제20조"}
    assert not any(re.search(r"승소.{0,12}(?:확률|율|퍼센트|%)", normalize(row["text"])) for row in rows.values())
    assert not re.search(r"\d+일이내.{0,30}(?:결정|재판)|(?:결정|재판).{0,30}\d+일",
                         normalize(rows["LAW_HLPA_본칙_005"]["text"]))
    print(json.dumps({"eval_sha": eval_sha(), "dev": {"answered": 8, "no_evidence": 2},
                      "test": {"answered": 17, "no_evidence": 3}, "legacy_saved": 30,
                      "quotes_checked": quotes_checked,
                      "chunk_mappings": sum(len(q["allowed_chunk_ids"]) for q in dev + test),
                      "distinct_primary_rows": 25,
                      "exact_question_overlap": 0, "exact_fact_overlap": 0,
                      "semantic_review": "평가셋.md의 법적 쟁점·조건 로컬 대조 기록 참고",
                      "model_calls": "not_run"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
