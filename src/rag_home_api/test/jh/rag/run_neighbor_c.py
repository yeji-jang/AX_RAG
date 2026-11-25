"""저장 C의 DEV 상위 5개에 같은 문서 인접 청크 하나를 보강하여 비교함."""

import argparse
import hashlib
import json
from collections import defaultdict
from urllib.parse import urlsplit

import httpx
from openai import OpenAI

from common import DOCS, FILES, RESULTS, load_chunks, require_dev_api_scope
from rag_home_api.config import settings
from run_rewrite_c import ANSWER_PROMPT, PUBLIC_HOSTS, generate

OUTPUT = RESULTS / "dev_원본로그.jsonl"
BASE_RUN = "dev-rewrite-c-20261125"
RUN_ID = "dev-neighbor-c-20261125"
RULE = "상위5 순서대로 같은 doc_id의 바로 다음, 바로 이전 청크를 확인하여 기존5에 없는 첫 공개 청크 1개만 끝에 추가함"


def append(record):
    with OUTPUT.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def select_neighbor(base_ids, groups, positions, public_ids):
    """정답·질문·본문 내용을 사용하지 않고 순서와 문서 ID만으로 선택함."""
    for anchor in base_ids:
        doc_id, position = positions[anchor]
        for offset in (1, -1):
            adjacent = position + offset
            if 0 <= adjacent < len(groups[doc_id]):
                candidate = groups[doc_id][adjacent]
                if candidate not in base_ids and candidate in public_ids:
                    return {"anchor": anchor, "chunk_id": candidate, "offset": offset}
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["inspect", "compare"])
    args = parser.parse_args()
    require_dev_api_scope("ce06a5d5ac03")
    records = [json.loads(line) for line in OUTPUT.read_text(encoding="utf-8").splitlines()]
    baseline = [r for r in records if r.get("record_type") == "rewrite_result"
                and r.get("run_id") == BASE_RUN and r.get("mode") == "C"]
    if (len(baseline) != 10 or {r["question_id"] for r in baseline}
            != {f"JH-DEV-{number:02d}" for number in range(1, 11)}
            or any(r["split"] != "dev" or r["eval_sha"] != "ce06a5d5ac03" for r in baseline)):
        raise ValueError("ExpectedFixedDevSnapshot")
    snapshot = json.dumps(baseline, ensure_ascii=False, sort_keys=True).encode()
    baseline_hash = hashlib.sha256(snapshot).hexdigest()
    docs = load_chunks("jh_1000")
    by_id = {d.metadata["chunk_id"]: d for d in docs}
    groups = defaultdict(list)
    positions = {}
    for doc in docs:
        doc_id, chunk_id = doc.metadata["doc_id"], doc.metadata["chunk_id"]
        positions[chunk_id] = (doc_id, len(groups[doc_id]))
        groups[doc_id].append(chunk_id)
    sources = {row["id"]: row["source_url"] for name in FILES
               for line in (DOCS / f"{name}.jsonl").read_text(encoding="utf-8").splitlines()
               if (row := json.loads(line))}
    public_chunks = {cid: doc for cid, doc in by_id.items()
                     if urlsplit(sources[doc.metadata["row_id"]]).scheme == "https"
                     and urlsplit(sources[doc.metadata["row_id"]]).hostname in PUBLIC_HOSTS}
    for row in baseline:
        if list(row["refs"]) != [f"S{number}" for number in range(1, 6)]:
            raise ValueError("ExpectedFiveOrderedChunks")
        for ref in row["refs"].values():
            doc = public_chunks[ref["chunk_id"]]
            if (doc.page_content != ref["text"] or doc.metadata["row_id"] != ref["row_id"]
                    or sources[ref["row_id"]] != ref["source_url"]):
                raise ValueError("ChangedStoredContext")
    existing = [r for r in records if r.get("run_id") == RUN_ID]
    if args.stage == "inspect":
        if existing:
            raise ValueError("RunIdAlreadyExists")
        append({"record_type": "neighbor_meta", "run_id": RUN_ID, "rule": RULE,
                "baseline_sha256": baseline_hash, "base_run": BASE_RUN,
                "eval_sha": "ce06a5d5ac03", "prompt": ANSWER_PROMPT,
                "model": settings.llm_model, "max_completion_tokens": 900})
        for row in baseline:
            ids = [ref["chunk_id"] for ref in row["refs"].values()]
            selected = select_neighbor(ids, groups, positions, set(public_chunks))
            extra = None
            if selected:
                doc = by_id[selected["chunk_id"]]
                extra = {**selected, "doc_id": doc.metadata["doc_id"],
                         "row_id": doc.metadata["row_id"], "text": doc.page_content,
                         "source_url": sources[doc.metadata["row_id"]]}
            plan = {"record_type": "neighbor_plan", "run_id": RUN_ID, "split": "dev",
                    "question_id": row["question_id"], "baseline_ids": ids, "extra": extra}
            append(plan)
            print(json.dumps(plan, ensure_ascii=False), flush=True)
        return
    plans = {r["question_id"]: r for r in existing if r.get("record_type") == "neighbor_plan"}
    meta = next(r for r in existing if r.get("record_type") == "neighbor_meta")
    audits = [r for r in existing if r.get("record_type") == "neighbor_audit"]
    if len(plans) != 10 or meta["baseline_sha256"] != baseline_hash:
        raise ValueError("IncompleteOrChangedPlan")
    if not any(r.get("new_fact_indices") for r in audits):
        raise ValueError("NoEvidenceGainConfirmedLocally")
    if any(r.get("record_type") == "neighbor_result" for r in existing):
        raise ValueError("ComparisonAlreadyAttempted")
    with httpx.Client(trust_env=False, timeout=55) as client:
        api = OpenAI(api_key=settings.openai_api_key, base_url="https://api.openai.com/v1",
                     http_client=client, max_retries=1)
        for row in baseline:
            plan = plans[row["question_id"]]
            ids = plan["baseline_ids"]
            if ids != [ref["chunk_id"] for ref in row["refs"].values()]:
                raise ValueError("ChangedBaseOrder")
            selected = select_neighbor(ids, groups, positions, set(public_chunks))
            extra = plan["extra"]
            if (selected is None) != (extra is None):
                raise ValueError("ChangedNeighbor")
            if extra:
                if selected["chunk_id"] != extra["chunk_id"] or by_id[extra["chunk_id"]].page_content != extra["text"]:
                    raise ValueError("ChangedNeighbor")
                ids = ids + [extra["chunk_id"]]
            result = {"record_type": "neighbor_result", "run_id": RUN_ID, "split": "dev",
                      "question_id": row["question_id"], "question": row["question"],
                      "extra_id": extra["chunk_id"] if extra else None}
            try:
                result.update(generate(api, row["question"], [by_id[cid] for cid in ids], public_chunks, sources))
            except Exception as exc:
                result.update(status="error", error=type(exc).__name__)
            append(result)
            print(json.dumps({"question_id": row["question_id"], "status": result["status"],
                              "gen_sec": result.get("gen_sec")}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
