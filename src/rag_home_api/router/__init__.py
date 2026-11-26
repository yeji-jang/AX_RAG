"""서울 주거 RAG 4도메인 서비스 입구 라우터 패키지."""

from rag_home_api.router.domain_router import (
    RouteDecision,
    TeamDomainRouter,
    classify_domain,
)
from rag_home_api.router.law_adapter import LawDomainAdapter, create_law_handler

__all__ = [
    "LawDomainAdapter",
    "RouteDecision",
    "TeamDomainRouter",
    "classify_domain",
    "create_law_handler",
]
