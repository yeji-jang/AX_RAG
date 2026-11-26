"""[방안 3 · 4단계] 지훈(test/jh) 법령 RAG 모듈을 무수정(Zero-Edit)으로 TeamDomainRouter에 연결하는 어댑터.

설계 원칙:
  1. src/rag_home_api/test/jh/ 하위 파일은 1줄도 수정하지 않는다.
  2. test/jh/rag/ 내부의 스크립트형 로컬 import(`from common import ...`)를 sys.path 격리 로더로 안전하게 감싼다.
  3. 서비스 실전 최적 모드인 B(Hybrid RRF · preset='route' · law_only=True · TEST 17/17 · ~1.7초)를 기본으로 사용하며,
     검증된 DB 저장 벡터(StoredDocumentEmbeddings)를 재사용하고 생성 단계는 원문 URL·메타데이터 일치 검증(generate_answer)을 통과한 청크만 주입한다.
  4. 반환값은 팀 공통 RAGResponse DTO로 통일하며, 예외 발생 시에도 라우터가 status='error'로 격리할 수 있게 안전 동작한다.
"""

from __future__ import annotations

import importlib
import sys
import time
from pathlib import Path
from typing import Any

from langchain_openai import OpenAIEmbeddings
from openai import OpenAI

from rag_home_api.config import settings
from rag_home_api.shared.core.vector_store import TEAM_COLLECTIONS
from rag_home_api.shared.schemas.rag_contract import (
    CitationItem,
    RAGResponse,
    make_error_response,
    make_insufficient_response,
)

JH_RAG_DIR = Path(__file__).resolve().parents[1] / "test" / "jh" / "rag"
DEFAULT_LAW_COLLECTION = TEAM_COLLECTIONS["law"]["current"]  # "jh_1000"


def _load_jh_modules() -> tuple[Any, Any, Any, Any]:
    """test/jh/rag/ 로컬 스크립트 모듈 4종(common, law_index, retrieve, run)을 안전 로드한다."""
    rag_dir_str = str(JH_RAG_DIR.resolve())
    if rag_dir_str not in sys.path:
        sys.path.insert(0, rag_dir_str)
    jh_common = importlib.import_module("common")
    jh_law_index = importlib.import_module("law_index")
    jh_retrieve = importlib.import_module("retrieve")
    jh_run = importlib.import_module("run")
    return jh_common, jh_law_index, jh_retrieve, jh_run


class LawDomainAdapter:
    """지훈 법령 Retriever(B 모드, jh_1000) + generate_answer() -> 공통 RAGResponse 어댑터."""

    def __init__(
        self,
        collection_name: str = DEFAULT_LAW_COLLECTION,
        preset: str = "route",
        mode: str = "B",
    ) -> None:
        self.collection_name = collection_name
        self.preset = preset
        self.mode = mode
        self._initialized = False
        self._retriever: Any = None
        self._openai_client: OpenAI | None = None
        self._jh_run: Any = None
        self._index_sha256: str = ""

    def _ensure_initialized(self) -> None:
        if self._initialized:
            return
        jh_common, jh_law_index, jh_retrieve, jh_run = _load_jh_modules()
        jh_retrieve.apply_preset(self.preset)

        dsn = settings.database_url.replace("postgresql://", "postgresql+psycopg://", 1)
        docs = jh_common.law_chunks(self.collection_name)
        snapshot = jh_law_index.read_index(dsn, self.collection_name, docs)
        self._index_sha256 = str(snapshot["index_sha256"])

        query_emb = OpenAIEmbeddings(
            model=settings.embedding_model,
            api_key=settings.openai_api_key,
            base_url=jh_common.API_BASE,
            max_retries=1,
            request_timeout=55,
            check_embedding_ctx_length=False,
        )
        stored_emb = jh_law_index.StoredDocumentEmbeddings(query_emb, snapshot["vectors_by_text"])
        self._retriever = jh_retrieve.Retriever(
            self.collection_name,
            law_only=True,
            law_documents=docs,
            embeddings=stored_emb,
            connection=dsn,
            law_documents=docs,
        )
        self._openai_client = OpenAI(
            api_key=settings.openai_api_key,
            base_url=jh_common.API_BASE,
            max_retries=1,
            timeout=55,
        )
        self._jh_run = jh_run
        self._initialized = True

    def __call__(self, question: str) -> RAGResponse:
        t0 = time.perf_counter()
        try:
            self._ensure_initialized()
            top_docs = self._retriever.documents(self.mode, question)
            if not top_docs:
                return make_insufficient_response(
                    domain="law",
                    reason="제공된 임대차 법령 자료에서 관련 조문을 찾지 못했습니다.",
                    latency_ms=(time.perf_counter() - t0) * 1000,
                    collection_name=self.collection_name,
                )

            raw_ans = self._jh_run.generate_answer(
                self._openai_client,
                settings.llm_model,
                {"question": question, "documents": top_docs},
                self._retriever.by_id,
            )
            status = raw_ans["status"]
            answer_text = str(raw_ans["answer"]).strip()
            cited_slots: list[str] = list(raw_ans.get("citations") or [])

            citations: list[CitationItem] = []
            if status == "answered":
                slot_to_doc = {f"S{i}": doc for i, doc in enumerate(top_docs, start=1)}
                for slot in cited_slots:
                    doc = slot_to_doc.get(slot)
                    if doc is None:
                        continue
                    meta = doc.metadata or {}
                    citations.append(
                        CitationItem(
                            chunk_id=str(meta.get("chunk_id") or ""),
                            title=str(meta.get("title") or ""),
                            source=str(meta.get("source") or "국가법령정보센터"),
                            location=str(meta.get("article") or ""),
                            url=str(meta.get("source_url") or ""),
                        )
                    )

            return RAGResponse(
                status=status,
                domain="law",
                answer=answer_text,
                citations=citations,
                retrieved_contexts=[d.page_content for d in top_docs],
                chunk_ids=[str((d.metadata or {}).get("chunk_id") or "") for d in top_docs],
                latency_ms=round((time.perf_counter() - t0) * 1000, 1),
                collection_name=self.collection_name,
                meta={
                    "law_mode": self.mode,
                    "law_preset": self.preset,
                    "index_sha256": self._index_sha256[:16],
                    "cited_slots": cited_slots,
                },
            )
        except Exception as exc:
            return make_error_response(
                domain="law",
                error_message=f"{type(exc).__name__}: {exc}",
                latency_ms=(time.perf_counter() - t0) * 1000,
                collection_name=self.collection_name,
            )


def create_law_handler(
    collection_name: str = DEFAULT_LAW_COLLECTION,
    preset: str = "route",
    mode: str = "B",
) -> LawDomainAdapter:
    """TeamDomainRouter.register('law', create_law_handler()) 용 팩토리."""
    return LawDomainAdapter(collection_name=collection_name, preset=preset, mode=mode)
