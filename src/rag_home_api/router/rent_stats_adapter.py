"""[방안 3 · 4단계] 최기혁(test/kh) 서울 25구 전월세 시세 통계 RAG 무수정 어댑터.

설계 원칙:
  1. 최종 검증 파이프라인(KURE-v1 + FIX B 25:75 k=15 + Filter v2 + C Rerank + Versioned Cache)을 사용한다.
  2. 반환값은 팀 공통 RAGResponse DTO로 통일하며, 거절(insufficient_evidence) 시 빈 인용 규칙을 준수한다.
"""

from __future__ import annotations

import time
from typing import Any

from rag_home_api.config import settings
from rag_home_api.optimize.cache import VersionedRAGCache, build_cache_key
from rag_home_api.shared.core.vector_store import get_llm
from rag_home_api.shared.schemas.rag_contract import (
    CitationItem,
    RAGResponse,
    make_error_response,
    make_insufficient_response,
)

DEFAULT_RENT_STATS_PRESET = "C_rrf_25_75_k15_filter_v2_cache_kure_v1"


class RentStatsDomainAdapter:
    """최기혁 서울 25구 전월세 시세 통계(housing_rent_kh_kure_v1) -> 공통 RAGResponse 어댑터."""

    def __init__(self, preset_name: str = DEFAULT_RENT_STATS_PRESET) -> None:
        self.preset_name = preset_name
        self._preset: Any = None
        self._answer_chain: Any = None
        self._cache: VersionedRAGCache | None = None

    def _ensure_initialized(self) -> None:
        if self._preset is not None:
            return
        from rag_home_api.test.kh.exp_03_rent_eval import ANSWER_PROMPT
        from rag_home_api.test.kh.presets import PRESET_REGISTRY

        self._preset = PRESET_REGISTRY[self.preset_name]
        self._answer_chain = ANSWER_PROMPT | get_llm()
        self._cache = VersionedRAGCache()

    def __call__(self, question: str) -> RAGResponse:
        t0 = time.perf_counter()
        col_name = "housing_rent_kh_kure_v1"
        try:
            from rag_home_api.test.kh.exp_03_rent_eval import _run_full_retrieval_pipeline

            self._ensure_initialized()
            preset = self._preset
            col_name = preset.collection_name

            cache_key = build_cache_key(
                query=question,
                collection_name=preset.collection_name,
                index_version=preset.chunk_strategy,
                embedding_model=preset.embedding_model,
                llm_model=settings.llm_model,
                prompt_version=preset.prompt_version,
                preset_dict=preset.to_dict(),
                user_scope="router_public",
            )
            cached = self._cache.get(cache_key) if (preset.use_cache and self._cache) else None
            if cached is not None:
                top5_ids = list(cached.get("top5_ids") or [])
                contexts = list(cached.get("retrieved_contexts") or [])
                answer_text = str(cached.get("response") or "").strip()
                cache_hit = True
            else:
                cache_hit = False
                pipe = _run_full_retrieval_pipeline(question, preset)
                final_hits = pipe["final_hits"]
                if not final_hits:
                    return make_insufficient_response(
                        domain="rent_stats",
                        reason="서울 25개 구 최근 1년 실거래 요약 통계에서 일치하는 자료를 찾지 못했습니다.",
                        latency_ms=(time.perf_counter() - t0) * 1000,
                        collection_name=col_name,
                    )
                top5_ids = [h["chunk_id"] for h in final_hits]
                contexts = [h["document"].page_content for h in final_hits]
                ctx_joined = "\n\n---\n\n".join(contexts)
                answer_text = str(self._answer_chain.invoke({"context": ctx_joined, "question": question}).content).strip()
                if preset.use_cache and self._cache and answer_text:
                    self._cache.set(
                        cache_key,
                        {
                            "status": "ok",
                            "top5_ids": top5_ids,
                            "retrieved_contexts": contexts,
                            "response": answer_text,
                        },
                        ttl_seconds=preset.cache_ttl_seconds,
                    )

            is_refusal = "확인할 수 없음" in answer_text or "정보가 없" in answer_text
            status = "insufficient_evidence" if is_refusal else "answered"

            citations: list[CitationItem] = []
            if status == "answered":
                for cid in top5_ids[:3]:
                    parts = cid.split("_")
                    gu = parts[1] if len(parts) > 1 else ""
                    dong = parts[2] if len(parts) > 2 else ""
                    citations.append(
                        CitationItem(
                            chunk_id=cid,
                            title=f"{gu} {dong} 전월세 실거래 요약".strip(),
                            source=f"서울_전월세_최근1년_{gu}.csv" if gu else "서울_전월세_최근1년.csv",
                            location=cid,
                            url="",
                        )
                    )

            return RAGResponse(
                status=status,
                domain="rent_stats",
                answer=answer_text,
                citations=citations,
                retrieved_contexts=contexts,
                chunk_ids=top5_ids,
                latency_ms=round((time.perf_counter() - t0) * 1000, 1),
                collection_name=col_name,
                meta={
                    "preset": self.preset_name,
                    "cache_hit": cache_hit,
                    "embedding_model": preset.embedding_model,
                },
            )
        except Exception as exc:
            return make_error_response(
                domain="rent_stats",
                error_message=f"{type(exc).__name__}: {exc}",
                latency_ms=(time.perf_counter() - t0) * 1000,
                collection_name=col_name,
            )


def create_rent_stats_handler(preset_name: str = DEFAULT_RENT_STATS_PRESET) -> RentStatsDomainAdapter:
    """TeamDomainRouter.register('rent_stats', create_rent_stats_handler()) 용 팩토리."""
    return RentStatsDomainAdapter(preset_name=preset_name)
