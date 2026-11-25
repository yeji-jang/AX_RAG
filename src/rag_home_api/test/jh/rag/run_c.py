"""Paired B/C dev experiment using the tutor's local Korean cross-encoder."""
import argparse
import json
import time
from collections import Counter
from datetime import datetime

from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

from common import MODELS, RESULTS, eval_sha, load_eval, require_dev_api_scope
from retrieve import CONFIG, Retriever, apply_preset, route

MODEL = "Dongjin-kr/ko-reranker"
COLLECTION = "jh_1000"


def evidence_rank(docs, question):
    chunk_ids = set(question.get("allowed_chunk_ids") or [])
    row_ids = set(question.get("allowed_row_ids") or [])
    field, refs = ("chunk_id", chunk_ids) if chunk_ids else ("row_id", row_ids)
    return next((i for i, d in enumerate(docs, 1) if d.metadata[field] in refs), None)


def run_pair(retriever, reranker, question):
    query = question["question"]
    top = CONFIG["k"]
    started = time.perf_counter()
    row = {"question_id": question["question_id"], "route": route(query),
           "evidence_unit": "chunk" if question.get("allowed_chunk_ids") else "row",
           "expected_status": question["expected_status"]}
    try:
        candidates = list(retriever.invoke(query))
    except Exception as exc:
        return row | {"outcome": "B_error", "error_type": type(exc).__name__}
    row["B_seconds"] = round(time.perf_counter() - started, 3)
    b_rank = evidence_rank(candidates[:top], question)
    candidate_rank = evidence_rank(candidates, question)
    row |= {"candidate_ids": [d.metadata["chunk_id"] for d in candidates],
            "B_rank_at_5": b_rank, "candidate_rank": candidate_rank}
    started = time.perf_counter()
    try:
        ranked = list(reranker.compress_documents(candidates, query)) if candidates else []
    except Exception as exc:
        return row | {"outcome": "C_error", "error_type": type(exc).__name__}
    row["rerank_seconds"] = round(time.perf_counter() - started, 3)
    c_ids = [d.metadata["chunk_id"] for d in ranked]
    if Counter(c_ids) - Counter(row["candidate_ids"]) or len(c_ids) != min(top, len(candidates)):
        return row | {"outcome": "C_error", "error_type": "InvalidRerankOutput"}
    c_rank = evidence_rank(ranked, question)
    if question["expected_status"] != "answered":
        outcome = "no_evidence_not_scored_in_retrieval"
    elif candidate_rank is None:
        outcome = "B_candidate_miss"
    elif b_rank and not c_rank:
        outcome = "C_regression"
    elif not b_rank and c_rank:
        outcome = "C_rescue"
    elif not c_rank:
        outcome = "C_rank_miss"
    else:
        outcome = "both_hit"
    return row | {"C_rank_at_5": c_rank, "C_top_ids": c_ids, "outcome": outcome}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = p.parse_args()
    require_dev_api_scope("ce06a5d5ac03")
    apply_preset("route")
    questions = load_eval("dev")["items"]
    if len(questions) != 10 or sum(q["expected_status"] == "answered" for q in questions) != 8:
        raise RuntimeError("Expected a dev set of 8 answerable and 2 no-evidence questions")
    ret = Retriever(COLLECTION)
    base = ret.retriever("B")  # Same B pipeline, before the final five-result slice.
    encoder = HuggingFaceCrossEncoder(model_name=MODEL, model_kwargs={"device": args.device})
    # Same compressor as the tutor's wrapper; invoke B just once for a paired comparison.
    reranker = CrossEncoderReranker(model=encoder, top_n=CONFIG["k"])
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"dev_BC_{datetime.now():%Y%m%d_%H%M%S}.jsonl"
    meta = {"split": "dev", "eval_sha": eval_sha(), "collection": COLLECTION, "preset": "route",
            "models": MODELS | {"reranker": MODEL}, "retrieval_config": CONFIG, "device": args.device,
            "score_scope": "retrieval only; page/row fallback if no exact chunk IDs"}
    rows = []
    with out.open("x", encoding="utf-8") as handle:
        print(json.dumps({"meta": meta}, ensure_ascii=False), file=handle, flush=True)
        for q in questions:
            row = run_pair(base, reranker, q)
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False), file=handle, flush=True)
            print(q["question_id"], row["outcome"], row.get("B_rank_at_5"), row.get("C_rank_at_5"), flush=True)
        eligible = [r for r in rows if r["expected_status"] == "answered"]
        summary = {"B_hits": sum(bool(r.get("B_rank_at_5")) for r in eligible),
                   "C_hits": sum(bool(r.get("C_rank_at_5")) for r in eligible),
                   "answerable": len(eligible), "outcomes": dict(Counter(r["outcome"] for r in rows))}
        print(json.dumps({"summary": summary}, ensure_ascii=False), file=handle)
    print(json.dumps(summary), out.name, flush=True)
    return int(any(r["outcome"].endswith("_error") for r in rows))


if __name__ == "__main__":
    raise SystemExit(main())
