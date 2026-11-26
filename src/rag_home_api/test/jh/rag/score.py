"""검색 지표만 집계한다. 생성 정답/충실도/적절 보류를 추정하지 않는다."""
from collections import Counter


def evidence_rank(hits, question, *, limit=None):
    """Document/dict 모두 같은 규칙. 전체 후보 순위와 Hit@5에 공동 사용한다."""
    chunk_ids = question.get("allowed_chunk_ids") or []
    field = "chunk_id" if chunk_ids else "row_id"
    allowed = set(chunk_ids or question.get("allowed_row_ids") or [])
    for rank, hit in enumerate(hits, 1):
        if limit is not None and rank > limit:
            break
        meta = hit.metadata if hasattr(hit, "metadata") else hit
        if meta.get(field) in allowed:
            return rank
    return None


def retrieval_metrics(candidates, top, question):
    if question["expected_status"] != "answered":
        return {"candidate_rank": None, "first_hit_rank": None, "hit5": None}
    rank = evidence_rank(top, question, limit=5)
    return {"candidate_rank": evidence_rank(candidates, question),
            "first_hit_rank": rank, "hit5": rank is not None}


def summarize(rows):
    eligible = [r for r in rows if r["expected_status"] == "answered"]
    denominator = len(eligible)
    hit = sum(r.get("hit5") is True for r in eligible)
    candidate_hits = sum(r.get("candidate_rank") is not None for r in eligible)
    return {"runs": len(rows), "answerable": denominator,
            "hit5": {"numerator": hit, "denominator": denominator,
                     "score": hit / denominator if denominator else None},
            "candidate_hits": {"numerator": candidate_hits, "denominator": denominator},
            "candidate_rank_counts": dict(Counter(str(r.get("candidate_rank")) for r in eligible)),
            "errors": sum(r["status"] == "error" for r in rows)}
