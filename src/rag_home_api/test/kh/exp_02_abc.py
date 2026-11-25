"""프리셋 기반 검색 A vs B(가중치 3종) 비교 & test/kh/results/ 자동 로그 저장."""

import json
import time
from datetime import datetime
from pathlib import Path

from rag_home_api.config import settings
from rag_home_api.retrieval.hybrid_rrf import search_hybrid_rrf
from rag_home_api.retrieval.vector_retriever import search_vector
from rag_home_api.test.kh.presets import NOW_AB_PRESETS, ExperimentPreset

KH_RESULTS_DIR = Path(__file__).resolve().parent / "results"

# (질문, 기대 정답 chunk_id 집합) — dev 미니 테스트셋 (214건 소량 DB 기준 검증 완료!)
MINI_DEV_CASES: list[dict] = [
    {
        "qid": "MINI-01",
        "question": "대항력이랑 확정일자 효력은 언제 생기나요?",
        "expected_chunks": {"LAW_HLPA_제3조_c00", "LAW_HLPA_제3조의2_c00"},
    },
    {
        "qid": "MINI-02",
        "question": "강동구 성지빌라 25B 1순위 청년 보증금과 월세는?",
        "expected_chunks": {"sh_306214_1_u00"},
    },
    {
        "qid": "MINI-03",
        "question": "행복주택 청년 계층 소득 기준은?",
        "expected_chunks": {"LH_2015122300021098_p009_c01", "LH_2015122300021098_p009_c00"},
    },
    {
        "qid": "MINI-04",
        "question": "원룸으로 이사 들어가서 동사무소에 주소 옮긴 당일에 집이 넘어가도 세입자 권리 주장할 수 있나요?",
        "expected_chunks": {"LAW_HLPA_제3조_c00"},
    },
]


def run_retrieval_preset(question: str, preset: ExperimentPreset) -> tuple[list[dict], float]:
    """프리셋 1개를 실행하고 (검색 결과, 소요시간 ms)를 반환한다."""
    t0 = time.perf_counter()

    if preset.pipeline_mode == "A_vector":
        hits = search_vector(question, top_k=preset.top_k)
    elif preset.pipeline_mode == "B_hybrid":
        hits = search_hybrid_rrf(
            keyword_query=question,
            semantic_query=question,
            top_k=preset.top_k,
            bm25_weight=preset.bm25_weight,
            vector_weight=preset.vector_weight,
            rrf_k=preset.rrf_k,
        )
    else:
        # 다음 단계에서 retrieval/reranker.py(C) + agent/pipeline.py 완성 시 여기 1줄 연결!
        raise NotImplementedError(f"아직 미구현 모드: {preset.pipeline_mode}")

    latency_ms = round((time.perf_counter() - t0) * 1000, 1)
    return hits, latency_ms


def save_scratch_log(records: list[dict]) -> Path:
    out_dir = KH_RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"exp_ab_kh_{ts}.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return out_path


def main(presets: list[ExperimentPreset] = NOW_AB_PRESETS) -> None:
    records: list[dict] = []
    score_board: dict[str, list[int]] = {p.name: [] for p in presets}

    for case in MINI_DEV_CASES:
        qid, q, expected = case["qid"], case["question"], case["expected_chunks"]
        print(f"\n==================== [{qid}] {q} ====================")

        for preset in presets:
            hits, latency_ms = run_retrieval_preset(q, preset)
            top_ids = [h["chunk_id"] for h in hits]
            hit_top_k = int(any(cid in expected for cid in top_ids))
            top1_ok = int(bool(top_ids) and top_ids[0] in expected)
            score_board[preset.name].append(hit_top_k)

            mark = "✅" if hit_top_k else "❌"
            print(f"  {mark} [{preset.name}] ({latency_ms}ms) | Hit@{preset.top_k}={hit_top_k} Top1={top1_ok}")
            for idx, h in enumerate(hits, start=1):
                m = h["document"].metadata
                star = "★" if h["chunk_id"] in expected else " "
                print(
                    f"    {star}#{idx} score={h['combined_score']} "
                    f"(bm25={h['bm25_rank']}/{h['bm25_score']}, vec={h['vector_rank']}/{h['vector_score']}) "
                    f"[{m['doc_type']}] {h['chunk_id']} ({m['location']})"
                )

            records.append(
                {
                    "qid": qid,
                    "question": q,
                    "index_version": settings.index_version,
                    "preset": preset.to_dict(),
                    "hit_at_k": hit_top_k,
                    "top1_ok": top1_ok,
                    "latency_ms": latency_ms,
                    "top_chunk_ids": top_ids,
                }
            )

    print("\n==================== 프리셋별 Hit@3 요약 ====================")
    for name, vals in score_board.items():
        print(f"  - {name:14s}: {sum(vals)}/{len(vals)}")

    saved = save_scratch_log(records)
    print(f"\n로그 저장 완료: {saved}")


if __name__ == "__main__":
    main()