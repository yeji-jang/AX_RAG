"""저장된 C의 DEV 상위 5개로 주장별 근거 연결 규칙을 한 번 비교함."""

import hashlib
import json
import time
from urllib.parse import urlsplit

import httpx
from openai import OpenAI

from common import DOCS, FILES, RESULTS, load_chunks
from rag_home_api.config import settings
from run_rewrite_c import ANSWER_PROMPT, PUBLIC_HOSTS

BASE_RUN = "dev-rewrite-c-20261125"
RUN_ID = "dev-answer-grounding-20261125"
OUTPUT = RESULTS / "dev_원본로그.jsonl"
RULE = """답변을 작성하기 전에 질문에 직접 대응하는 근거 문장을 확인하고, 각 답변 주장에 그 근거를 연결하세요.
선택한 근거에서 그 주장과 직접 연결된 적용 대상·조건·예외·수치·단위를 함께 답하고, 근거가 없어 확인할 수 없는 부분은 missing_evidence에 명시하세요. 질문과 무관한 내용을 채우거나 빠진 사실을 추측하지 마세요.
claims는 text와 citations를 가진 객체 배열로 작성하세요. 각 객체는 답변의 사실 주장 하나와 그것을 지지하는 S 번호 배열을 포함해야 합니다. 최상위 citations는 claims에서 사용한 S 번호의 중복 없는 합집합이어야 합니다.
status가 insufficient_evidence이면 답변은 확인 불가 안내로 작성하고 citations와 claims를 모두 빈 배열로 반환하세요. 부족한 내용은 missing_evidence에 적으세요. 필요한 근거가 충분하면 missing_evidence는 빈 배열로 반환하세요."""


def append(record):
    with OUTPUT.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def validate_output(result, refs):
    errors = []
    if (not isinstance(result, dict)
            or result.get("status") not in ("answered", "insufficient_evidence")
            or not isinstance(result.get("answer"), str)
            or not isinstance(result.get("citations"), list)
            or not all(isinstance(slot, str) for slot in result["citations"])
            or not isinstance(result.get("claims"), list)
            or not isinstance(result.get("missing_evidence"), list)
            or not all(isinstance(text, str) for text in result["missing_evidence"])):
        return ["InvalidSchema"]
    linked_slots = set()
    for claim in result["claims"]:
        if (not isinstance(claim, dict) or not isinstance(claim.get("text"), str)
                or not isinstance(claim.get("citations"), list)
                or not claim["citations"]
                or not all(isinstance(slot, str) and slot in refs for slot in claim["citations"])):
            errors.append("InvalidClaimReference")
        else:
            linked_slots.update(claim["citations"])
    citations = set(result["citations"])
    if not citations.issubset(refs) or citations != linked_slots or len(citations) != len(result["citations"]):
        errors.append("CitationUnionMismatch")
    if result["status"] == "insufficient_evidence":
        if result["citations"] or result["claims"] or not result["missing_evidence"]:
            errors.append("InvalidInsufficientEvidenceOutput")
    elif not result["claims"]:
        errors.append("MissingClaimReferences")
    return errors


def generate(api, original, refs):
    context = "\n\n".join(f"[{slot}] {ref['text']}" for slot, ref in refs.items())
    response = api.chat.completions.create(
        model=settings.llm_model,
        messages=[{"role": "system", "content": ANSWER_PROMPT + "\n\n" + RULE},
                  {"role": "user", "content": original + "\n\n[참고 발췌]\n" + context}],
        response_format={"type": "json_object"}, reasoning_effort="low", max_completion_tokens=900,
    )
    result = json.loads(response.choices[0].message.content)
    return {"response": result, "output_errors": validate_output(result, refs),
            "usage": response.usage.model_dump(), "finish_reason": response.choices[0].finish_reason}


def main():
    records = [json.loads(line) for line in OUTPUT.read_text(encoding="utf-8").splitlines()]
    if any(r.get("run_id") == RUN_ID or r.get("meta", {}).get("tag") == RUN_ID for r in records):
        raise ValueError("RunIdAlreadyExists")
    baseline = [r for r in records if r.get("record_type") == "rewrite_result"
                and r.get("run_id") == BASE_RUN and r.get("mode") == "C"]
    if (len(baseline) != 10 or {r["question_id"] for r in baseline}
            != {f"JH-DEV-{number:02d}" for number in range(1, 11)}
            or any(r["split"] != "dev" or r["eval_sha"] != "ce06a5d5ac03" for r in baseline)):
        raise ValueError("ExpectedFixedDevSnapshot")
    canonical = {d.metadata["chunk_id"]: d for d in load_chunks("jh_1000")}
    sources = {row["id"]: row["source_url"] for name in FILES
               for line in (DOCS / f"{name}.jsonl").read_text(encoding="utf-8").splitlines()
               if (row := json.loads(line))}
    for row in baseline:
        if list(row["refs"]) != [f"S{number}" for number in range(1, 6)]:
            raise ValueError("ExpectedFiveOrderedChunks")
        for ref in row["refs"].values():
            doc = canonical[ref["chunk_id"]]
            url = urlsplit(ref["source_url"])
            if (doc.page_content != ref["text"] or doc.metadata["row_id"] != ref["row_id"]
                    or sources[ref["row_id"]] != ref["source_url"]
                    or url.scheme != "https" or url.hostname not in PUBLIC_HOSTS):
                raise ValueError("UnverifiedPublicContext")
    snapshot = json.dumps(baseline, ensure_ascii=False, sort_keys=True).encode()
    original_log = OUTPUT.read_bytes()
    append({"record_type": "answer_meta", "run_id": RUN_ID, "base_run": BASE_RUN,
            "eval_sha": "ce06a5d5ac03", "baseline_sha256": hashlib.sha256(snapshot).hexdigest(),
            "input_log_bytes": len(original_log),
            "input_log_sha256": hashlib.sha256(original_log).hexdigest(),
            "model": settings.llm_model, "rule": RULE, "base_prompt": ANSWER_PROMPT,
            "reasoning_effort": "low", "max_completion_tokens": 900,
            "context_source": "stored C refs, same S1-S5 order and full text"})
    with httpx.Client(trust_env=False, timeout=55) as client:
        api = OpenAI(api_key=settings.openai_api_key, base_url="https://api.openai.com/v1",
                     http_client=client, max_retries=1)
        for row in baseline:
            result = {"record_type": "answer_result", "run_id": RUN_ID, "split": "dev",
                      "question_id": row["question_id"], "question": row["question"],
                      "refs": row["refs"], "baseline_gen_sec": row["gen_sec"]}
            started = time.perf_counter()
            try:
                result.update(generate(api, row["question"], row["refs"]))
                result["status"] = "completed"
            except Exception as exc:
                result.update(status="error", error=type(exc).__name__)
            result["gen_sec"] = round(time.perf_counter() - started, 3)
            append(result)
            print(json.dumps({"question_id": row["question_id"], "status": result["status"],
                              "response_status": result.get("response", {}).get("status"),
                              "output_errors": result.get("output_errors"),
                              "gen_sec": result["gen_sec"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
