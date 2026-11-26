"""팀 4인(지훈·예지·최기혁·병철) 공통 RAG 응답 계약(DTO) 및 통합 평가 요약 스키마.

방안 3 (어댑터 선통합 + 멀티컬렉션·라우터 통합):
  - 각 팀원의 내부 검색·청킹·필터 구현은 보존하고,
  - 서비스 입구(domain_router)와 출구(RAGResponse), 평가 집계(UnifiedEvalSummary)만 공통 계약으로 통일한다.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

DomainType = Literal["law", "public_housing", "rent_stats", "rent_deals", "unsupported"]
ResponseStatus = Literal["answered", "insufficient_evidence", "error", "unsupported"]


class CitationItem(BaseModel):
    """출처 1건의 표준 구조."""

    chunk_id: str = Field(default="", description="청크 고유 ID")
    title: str = Field(default="", description="문서 제목 또는 법령명·조항")
    source: str = Field(default="", description="출처 기관 또는 파일명 (예: LH, SH, 서울_전월세_최근1년_강남구.csv)")
    location: str = Field(default="", description="세부 위치 (페이지, 조문번호, 동·유형·면적대 등)")
    url: str = Field(default="", description="공개 원본 URL (없으면 빈 문자열)")


class RAGResponse(BaseModel):
    """4개 도메인 핸들러가 공통으로 반환해야 하는 표준 응답 DTO."""

    status: ResponseStatus = Field(description="answered / insufficient_evidence / error / unsupported")
    domain: DomainType = Field(description="라우팅된 도메인")
    answer: str = Field(description="사용자에게 반환할 최종 답변 텍스트")
    citations: list[CitationItem] = Field(default_factory=list, description="검증된 출처 목록 (근거 부족/오류 시 빈 리스트)")
    retrieved_contexts: list[str] = Field(default_factory=list, description="RAGAS 평가 및 디버깅용 검색 원문")
    chunk_ids: list[str] = Field(default_factory=list, description="검색된 상위 chunk_id 목록")
    latency_ms: float = Field(default=0.0, ge=0.0, description="처리 시간(ms)")
    collection_name: str = Field(default="", description="실제 조회한 PGVector 컬렉션명")
    route_reason: str = Field(default="", description="도메인 라우팅 판단 사유")
    error_message: str = Field(default="", description="status='error'일 때의 내부 진단 메시지")
    meta: dict[str, Any] = Field(default_factory=dict, description="도메인별 확장 진단 메타데이터")

    def is_cacheable(self) -> bool:
        """오류 또는 빈 응답은 절대 캐시에 저장하지 않는다."""
        return self.status != "error" and bool(self.answer.strip())


class UnifiedEvalSummary(BaseModel):
    """Streamlit '5. 팀 통합 비교'가 4명 결과를 동일 표로 집계하기 위한 표준 요약 스키마."""

    timestamp: str
    owner: str
    domain: str
    eval_version: str
    ragas_target: float = 0.65
    collection_name: str = ""
    embedding_model: str = "text-embedding-3-small"
    source_branch: str = ""
    source_file: str = ""
    notes: str = ""
    modes: dict[str, dict[str, Any]] = Field(default_factory=dict)


def make_insufficient_response(
    *,
    domain: DomainType,
    reason: str,
    latency_ms: float = 0.0,
    collection_name: str = "",
    route_reason: str = "",
) -> RAGResponse:
    """근거 부족 표준 거절 응답 (임의 인용 금지)."""
    return RAGResponse(
        status="insufficient_evidence",
        domain=domain,
        answer=f"확인할 수 없음. {reason}",
        citations=[],
        retrieved_contexts=[],
        chunk_ids=[],
        latency_ms=round(latency_ms, 1),
        collection_name=collection_name,
        route_reason=route_reason,
    )


def make_unsupported_response(
    *,
    reason: str = "현재 서비스가 지원하는 서울 전월세·공공임대·주택임대차 법령 범위 밖의 질문입니다.",
    latency_ms: float = 0.0,
    route_reason: str = "out_of_scope",
) -> RAGResponse:
    """지원 범위 외 표준 응답."""
    return RAGResponse(
        status="unsupported",
        domain="unsupported",
        answer=f"확인할 수 없음. {reason}",
        citations=[],
        retrieved_contexts=[],
        chunk_ids=[],
        latency_ms=round(latency_ms, 1),
        route_reason=route_reason,
    )


def make_error_response(
    *,
    domain: DomainType,
    error_message: str,
    latency_ms: float = 0.0,
    collection_name: str = "",
    route_reason: str = "",
) -> RAGResponse:
    """검색·생성 예외 발생 시 서비스 중단을 막는 표준 오류 응답 (캐시 저장 자동 거부)."""
    return RAGResponse(
        status="error",
        domain=domain,
        answer="일시적인 검색 또는 답변 생성 오류로 응답을 완료하지 못했습니다. 잠시 후 다시 시도해 주세요.",
        citations=[],
        retrieved_contexts=[],
        chunk_ids=[],
        latency_ms=round(latency_ms, 1),
        collection_name=collection_name,
        route_reason=route_reason,
        error_message=error_message,
    )
