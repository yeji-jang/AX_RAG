"""[방안 3 · 4단계] 병철(test/bc) 전월세 개별거래 조건 탐색 RAG(course_v0_small · CachedPipeline) 무수정 어댑터.

설계 원칙:
  1. src/rag_home_api/test/bc/ 하위 파일은 1줄도 수정하지 않는다.
  2. 병철 공식 진입점 CachedPipeline.answer(..., system='C')를 그대로 호출해 공통 RAGResponse로 변환한다.
  3. 확인 요청(needs_clarification) / 미지원(unsupported) / 빈 결과는 insufficient_evidence로 안전 매핑한다.
"""

from __future__ import annotations

import time
from typing import Any

from rag_home_api.shared.core.vector_store import TEAM_COLLECTIONS
from rag_home_api.shared.schemas.rag_contract import (
    CitationItem,
    RAGResponse,
    make_error_response,
)

DEFAULT_RENT_DEALS_COLLECTION = TEAM_COLLECTIONS["rent_deals"]["current"]  # "course_v0_small"


class RentDealsDomainAdapter:
    """병철 관악구 개별 거래 조건 필터 RAG (course_v0_small, system='C') -> 공통 RAGResponse 어댑터."""

    def __init__(
        self,
        collection_name: str = DEFAULT_RENT_DEALS_COLLECTION,
        system: str = "C",
        index_version: str = "course_v0_small_v2",
        use_cache: bool = True,
    ) -> None:
        self.collection_name = collection_name
        self.system = system
        self.index_version = index_version
        self.use_cache = use_cache
        self._pipeline: Any = None

    def _ensure_initialized(self) -> None:
        if self._pipeline is not None:
            return
        from rag_home_api.test.bc.cached_pipeline import CachedPipeline

        self._pipeline = CachedPipeline()

    def __call__(self, question: str) -> RAGResponse:
        t0 = time.perf_counter()
        try:
            self._ensure_initialized()
            raw = self._pipeline.answer(
                question,
                scope="router_public",
                index_version=self.index_version,
                system=self.system,
                use_cache=self.use_cache,
            )
            bc_status = str(raw.get("status") or "")
            answer_text = str(raw.get("response") or "").strip()
            sources = list(raw.get("sources") or [])

            is_answered = bc_status == "success" and len(sources) > 0
            status = "answered" if is_answered else "insufficient_evidence"

            citations: list[CitationItem] = []
            contexts: list[str] = []
            chunk_ids: list[str] = []
            for src in sources:
                cid = str(src.get("id") or "")
                meta = src.get("metadata") or {}
                chunk_ids.append(cid)
                contexts.append(str(src.get("page_content") or ""))
                if status == "answered":
                    citations.append(
                        CitationItem(
                            chunk_id=cid,
                            title=f"{meta.get('region_dong', '')} {meta.get('building_name', '')}".strip() or cid,
                            source="서울_전월세_최근1년_관악구.csv",
                            location=f"row {meta.get('source_row', '')}".strip(),
                            url="",
                        )
                    )

            return RAGResponse(
                status=status,
                domain="rent_deals",
                answer=answer_text,
                citations=citations,
                retrieved_contexts=contexts,
                chunk_ids=chunk_ids,
                latency_ms=round((time.perf_counter() - t0) * 1000, 1),
                collection_name=self.collection_name,
                meta={
                    "bc_status": bc_status,
                    "system": self.system,
                    "cache_hit": bool(raw.get("cache_hit", False)),
                },
            )
        except Exception as exc:
            return make_error_response(
                domain="rent_deals",
                error_message=f"{type(exc).__name__}: {exc}",
                latency_ms=(time.perf_counter() - t0) * 1000,
                collection_name=self.collection_name,
            )


def create_rent_deals_handler(
    collection_name: str = DEFAULT_RENT_DEALS_COLLECTION,
    system: str = "C",
) -> RentDealsDomainAdapter:
    """TeamDomainRouter.register('rent_deals', create_rent_deals_handler()) 용 팩토리."""
    return RentDealsDomainAdapter(collection_name=collection_name, system=system)

