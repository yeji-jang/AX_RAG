"""[방안 3 · 4단계] 예지(test/yj) 공공임대 공고문 RAG(chunkV2 · C_hybrid_rerank) 무수정 어댑터.

설계 원칙:
  1. src/rag_home_api/test/yj/ 하위 파일은 1줄도 수정하지 않는다.
  2. 단일 컬렉션 yj_public_rental 의 chunk_version='V2' (6,197청크) + C_hybrid_rerank(후보 15 -> Top-5)를 사용한다.
  3. 반환값은 팀 공통 RAGResponse DTO로 통일하며, 근거 부족·예외를 안전하게 격리한다.
"""

from __future__ import annotations

import time
from typing import Any

from rag_home_api.shared.core.vector_store import TEAM_COLLECTIONS
from rag_home_api.shared.schemas.rag_contract import (
    CitationItem,
    RAGResponse,
    make_error_response,
    make_insufficient_response,
)

DEFAULT_PUBLIC_HOUSING_COLLECTION = TEAM_COLLECTIONS["public_housing"]["current"]  # "yj_public_rental"


class PublicHousingDomainAdapter:
    """예지 공공임대 공고문 C_hybrid_rerank (yj_public_rental, V2) -> 공통 RAGResponse 어댑터."""

    def __init__(
        self,
        collection_name: str = DEFAULT_PUBLIC_HOUSING_COLLECTION,
        chunk_version: str = "V2",
        candidate_k: int = 15,
        top_k: int = 5,
    ) -> None:
        self.collection_name = collection_name
        self.chunk_version = chunk_version
        self.candidate_k = candidate_k
        self.top_k = top_k
        self._initialized = False
        self._bm25: Any = None

    def _ensure_initialized(self) -> None:
        if self._initialized:
            return
        from rag_home_api.test.yj.rag.B_hybrid_retriever import BM25Index

        self._bm25 = BM25Index(self.collection_name, chunk_version=self.chunk_version)
        self._initialized = True

    def __call__(self, question: str) -> RAGResponse:
        t0 = time.perf_counter()
        try:
            from rag_home_api.test.yj.rag.B_hybrid_retriever import search_hybrid
            from rag_home_api.test.yj.rag.C_reranker import rerank_chunks
            from rag_home_api.test.yj.rag.generator import generate_answer

            self._ensure_initialized()
            flt = {"chunk_version": self.chunk_version}
            candidates = search_hybrid(
                question,
                self._bm25,
                collection=self.collection_name,
                top_k=self.candidate_k,
                filters=flt,
            )
            if not candidates:
                return make_insufficient_response(
                    domain="public_housing",
                    reason="제공된 서울시 공공임대·청년안심주택 공고문에서 관련 정보를 찾지 못했습니다.",
                    latency_ms=(time.perf_counter() - t0) * 1000,
                    collection_name=self.collection_name,
                )

            reranked, trace = rerank_chunks(question, candidates, top_k=self.top_k)
            raw = generate_answer(question, reranked)
            answer_text = str(raw.get("answer") or "").strip()
            sources = list(raw.get("sources") or [])
            is_refusal = (
                not sources
                or "찾을 수 없습니다" in answer_text
                or "찾지 못했습니다" in answer_text
                or "확인할 수 없" in answer_text
            )
            status = "insufficient_evidence" if is_refusal else "answered"

            citations: list[CitationItem] = []
            if status == "answered":
                for src in sources:
                    citations.append(
                        CitationItem(
                            chunk_id=str(src.get("chunk_id") or ""),
                            title=str(src.get("title") or ""),
                            source="LH/SH 공공임대 공고문",
                            location=f"p.{src.get('page', '')}",
                            url=str(src.get("source_url") or ""),
                        )
                    )

            return RAGResponse(
                status=status,
                domain="public_housing",
                answer=answer_text,
                citations=citations,
                retrieved_contexts=list(raw.get("retrieved_contexts") or []),
                chunk_ids=list(raw.get("retrieved_chunk_ids") or []),
                latency_ms=round((time.perf_counter() - t0) * 1000, 1),
                collection_name=self.collection_name,
                meta={
                    "chunk_version": self.chunk_version,
                    "mode": "C_hybrid_rerank",
                    "cited_numbers": list(raw.get("cited_numbers") or []),
                    "rerank_trace_count": len(trace),
                },
            )
        except Exception as exc:
            return make_error_response(
                domain="public_housing",
                error_message=f"{type(exc).__name__}: {exc}",
                latency_ms=(time.perf_counter() - t0) * 1000,
                collection_name=self.collection_name,
            )


def create_public_housing_handler(
    collection_name: str = DEFAULT_PUBLIC_HOUSING_COLLECTION,
    chunk_version: str = "V2",
) -> PublicHousingDomainAdapter:
    """TeamDomainRouter.register('public_housing', create_public_housing_handler()) 용 팩토리."""
    return PublicHousingDomainAdapter(collection_name=collection_name, chunk_version=chunk_version)

