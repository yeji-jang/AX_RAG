"""서울 주거 RAG 4도메인 서비스 입구 라우터 패키지."""

from rag_home_api.router.domain_router import (
    RouteDecision,
    TeamDomainRouter,
    classify_domain,
)
from rag_home_api.router.law_adapter import LawDomainAdapter, create_law_handler
from rag_home_api.router.public_housing_adapter import (
    PublicHousingDomainAdapter,
    create_public_housing_handler,
)
from rag_home_api.router.rent_deals_adapter import (
    RentDealsDomainAdapter,
    create_rent_deals_handler,
)
from rag_home_api.router.rent_stats_adapter import (
    RentStatsDomainAdapter,
    create_rent_stats_handler,
)


def build_default_team_router() -> TeamDomainRouter:
    """4명 팀원의 도메인 핸들러를 모두 등록한 통합 TeamDomainRouter 인스턴스를 반환한다."""
    router = TeamDomainRouter()
    router.register("law", create_law_handler())
    router.register("public_housing", create_public_housing_handler())
    router.register("rent_stats", create_rent_stats_handler())
    router.register("rent_deals", create_rent_deals_handler())
    return router


__all__ = [
    "LawDomainAdapter",
    "PublicHousingDomainAdapter",
    "RentDealsDomainAdapter",
    "RentStatsDomainAdapter",
    "RouteDecision",
    "TeamDomainRouter",
    "build_default_team_router",
    "classify_domain",
    "create_law_handler",
    "create_public_housing_handler",
    "create_rent_deals_handler",
    "create_rent_stats_handler",
]
