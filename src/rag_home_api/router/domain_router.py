"""[방안 3 · 서비스 입구 4도메인 라우터] 질문 의도를 분류해 각 팀원 전용 핸들러로 안전 연결한다.

설계 원칙:
  1. 지훈(feat/cjh) `law_router.py`의 임대차 법령 분류 규칙을 존중해 `law` 경계 케이스를 보존한다.
  2. 예지(공공임대 공고), 최기혁(서울 25구 전월세 시세 통계), 병철(전월세 개별 거래 조건 필터)을
     멀티 컬렉션 그대로 플러그인(register) 방식으로 연결한다.
  3. 지원 범위 외(서울 외 지역·매매/분양·개인 대출/전세사기 자문 등)는 `unsupported`로 정직 거절한다.
  4. 어떤 핸들러에서 예외가 발생해도 서비스 전체가 죽지 않고 `RAGResponse(status="error")`로 격리한다.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Callable

from rag_home_api.shared.core.vector_store import TEAM_COLLECTIONS
from rag_home_api.shared.schemas.rag_contract import (
    DomainType,
    RAGResponse,
    make_error_response,
    make_unsupported_response,
)

DomainHandler = Callable[[str], RAGResponse]

# ── 1) 지원 범위 외(명확한 제외 주제) ──
# 한글 단어 경계 가드: '무주택세대구성원'의 '대구' 오탐 등을 원천 차단
OUT_OF_SCOPE_REGION = (
    r"(?<![가-힣])(?:경기도|인천|수원|성남|용인|부천|고양|부산|대구|대전|광주|울산|세종|제주|지방)"
    r"(?:시|광역시|특별자치시|특별자치도|도|구|군|특례시|에|에서|의|은|는|이|가|\s|$)"
)
OUT_OF_SCOPE_TOPIC = r"매매가|매매시세|분양가|분양청약|전세사기.*안전|근저당.*안전|등기부등본.*판단|버팀목.*한도|청년전세대출.*금리"

# ── 2) 법령 (지훈 · jh_1000) — law_router.py 핵심 패턴 계승 ──
HOUSING_CTX = r"임대차|임대인|임차인|집주인|세입자|월세|전세|보증금|차임|임차|주택|원룸|빌라|오피스텔|아파트|전입신고|확정일자"
# 한글 단어 경계 가드: '방법으로'의 '법으로' 오탐 등을 원천 차단
LEGAL_INTENT = (
    r"(?<![가-힣])(?:법적|법정|법률|법령|법상|법으로|법대로|조문|조항|제\d+조|민법|주택임대차보호법|주민등록법)|"
    r"권리|의무|위법|불법|금지|청구|임차권등기|대항력|우선변제|월세상한|차임상한|임대료상한|계약갱신요구|묵시적갱신"
)
PROCEDURE_INTENT = r"(?:전입신고|확정일자|임대차신고).*(?:언제|기한|해야|조건|필요|가능|할수|받을|해도|돼|늦|효력)"
LEGAL_ACTION = r"올리|올려|올릴|인상|증액|해지|해제|갱신|취소|수리|고쳐|돌려|반환|상환|배상|부담|감액|깎|쫓아|내보내|나가야|퇴거|연체|못내|들어와|출입"
PERMISSION_INTENT = r"해야|줘야|내야|누가|가능|수있|하나요|되나|되나요|해도|돼|마음대로|안돌려|안해|상한|한도|몇퍼센트|몇%|기한|언제까지"

# ── 3) 공공임대·청년안심 공고 (예지 · yj_public_rental V2) ──
PUBLIC_HOUSING_KW = (
    r"공고|모집|행복주택|청년안심|국민임대|영구임대|매입임대|통합공공임대|공공임대|장기전세|전세임대|재개발임대|미리내집|희망하우징|"
    r"LH|SH|서울주택도시공사|청약플러스|입주자격|입주신청|예비입주자|청약신청|공급목표|소득기준|자산기준|정정공고"
)
PUBLIC_HOUSING_TYPE = r"\d{2,3}[A-Z]형"

# ── 4) 전월세 실거래: 개별 조건 필터(병철 · rent_deals) vs 시세 통계(최기혁 · rent_stats) ──
RENT_DOMAIN_KW = r"전세|월세|전월세|보증금|원룸|투룸|빌라|오피스텔|아파트|단독|다가구|시세|실거래|자취방"
SEOUL_AREA_HINT = r"서울|[가-힣]{1,4}구|[가-힣]{1,4}동|[가-힣]{1,4}가"

# 개별 거래 조건 검색(병철) 신호: 구체적 가격/층/연식 상하한 조건 + "집이 있나요 / 찾고 있어요"
DEAL_THRESHOLD_PAT = (
    r"(?:보증금|월세)\s*[0-9억천백만,\s]+(?:이하|이상|미만|초과)|"
    r"\d+\s*층\s*(?:이상|이하)|"
    r"20\d{2}\s*년\s*(?:이후|이상|부터)|"
    r"(?:집이\s*있나요|매물\s*(?:찾고|있나요)|자취방(?:을)?\s*찾고)"
)
# 시세 통계 요약(최기혁) 신호: 중앙값/평균/시세/비교/거래건수/대체로 몇 년식
STATS_INTENT_PAT = r"시세|보통\s*(?:얼마|보증금|월세)|중앙값|평균|비교|거래\s*(?:건수|얼마나|활발|자주)|대체로\s*몇\s*년식|어디가\s*더"


@dataclass(frozen=True)
class RouteDecision:
    domain: DomainType
    collection_name: str
    reason: str
    secondary_domain: DomainType | None = None


def classify_domain(question: str) -> RouteDecision:
    """사용자 질문을 4개 도메인(law / public_housing / rent_stats / rent_deals) 또는 unsupported로 분류한다."""
    if not isinstance(question, str) or not question.strip():
        return RouteDecision("unsupported", "", "빈 질문 입력")

    raw = question.strip()
    compact = re.sub(r"\s+", "", raw)

    # 1) 명백한 도메인 외 제외 주제 (단, 서울 언급이 같이 있으면 제외 완화)
    if re.search(OUT_OF_SCOPE_TOPIC, compact):
        return RouteDecision("unsupported", "", "지원 범위 외 주제 (매매/분양/대출/개별 법률자문)")
    if re.search(OUT_OF_SCOPE_REGION, raw) and "서울" not in compact:
        return RouteDecision("unsupported", "", "서울 외 지역 질문")

    # 2) 법령 의도 최우선 검사 (공고/단지 단어가 섞여도 법적 권리·의무 질문은 law 유지)
    has_housing = bool(re.search(HOUSING_CTX, compact))
    has_legal = bool(re.search(LEGAL_INTENT, raw) or re.search(PROCEDURE_INTENT, compact))
    has_action_perm = bool(re.search(LEGAL_ACTION, compact) and re.search(PERMISSION_INTENT, compact))
    data_only = bool(re.search(r"(?:법령|법률|조문|법적설명)(?:은|는|을|를|이|가)?(?:말고|빼고|필요없)", compact))
    is_stats_only = bool(re.search(STATS_INTENT_PAT, raw)) and not has_legal and not has_action_perm

    if not data_only and not is_stats_only and (has_legal or (has_housing and has_action_perm)):
        return RouteDecision(
            domain="law",
            collection_name=TEAM_COLLECTIONS["law"]["current"],
            reason="임대차 법령·권리·의무·절차 의도 감지",
        )

    # 3) 공공임대·청년안심주택 공고 (예지)
    if re.search(PUBLIC_HOUSING_KW, compact, flags=re.IGNORECASE) or re.search(PUBLIC_HOUSING_TYPE, compact):
        return RouteDecision(
            domain="public_housing",
            collection_name=TEAM_COLLECTIONS["public_housing"]["current"],
            reason="공공임대·청년안심 모집공고/입주자격 키워드 감지",
        )

    # 4) 전월세 실거래 (개별 조건 검색 vs 25구 시세 통계 요약)
    if re.search(RENT_DOMAIN_KW, compact) or re.search(SEOUL_AREA_HINT, compact):
        has_deal_cond = bool(re.search(DEAL_THRESHOLD_PAT, raw))
        has_stats_cue = bool(re.search(STATS_INTENT_PAT, raw))
        if has_deal_cond and not has_stats_cue:
            return RouteDecision(
                domain="rent_deals",
                collection_name=TEAM_COLLECTIONS["rent_deals"]["current"],
                reason="구체적 보증금·월세·층·연식 상하한 조건 탐색 감지",
                secondary_domain="rent_stats",
            )
        if re.search(RENT_DOMAIN_KW, compact):
            return RouteDecision(
                domain="rent_stats",
                collection_name=TEAM_COLLECTIONS["rent_stats"]["current"],
                reason="서울 전월세 실거래 시세·통계·비교 질문 감지",
                secondary_domain="rent_deals" if has_deal_cond else None,
            )

    return RouteDecision("unsupported", "", "서울 주거 4개 도메인 키워드 미탐지")


class TeamDomainRouter:
    """4명 팀원의 개별 RAG 핸들러를 공통 RAGResponse 계약으로 연결하는 플러그인 라우터."""

    def __init__(self, handlers: dict[DomainType, DomainHandler] | None = None) -> None:
        self._handlers: dict[DomainType, DomainHandler] = dict(handlers or {})

    def register(self, domain: DomainType, handler: DomainHandler) -> None:
        self._handlers[domain] = handler

    @property
    def registered_domains(self) -> list[DomainType]:
        return list(self._handlers.keys())

    def route(self, question: str) -> RouteDecision:
        return classify_domain(question)

    def invoke(self, question: str) -> RAGResponse:
        t0 = time.perf_counter()
        decision = self.route(question)
        if decision.domain == "unsupported":
            return make_unsupported_response(
                reason=decision.reason,
                latency_ms=(time.perf_counter() - t0) * 1000,
                route_reason=decision.reason,
            )

        handler = self._handlers.get(decision.domain)
        if handler is None and decision.secondary_domain:
            handler = self._handlers.get(decision.secondary_domain)
        if handler is None:
            return make_unsupported_response(
                reason=f"'{decision.domain}' 도메인 핸들러가 아직 등록되지 않았습니다.",
                latency_ms=(time.perf_counter() - t0) * 1000,
                route_reason=decision.reason,
            )

        try:
            resp = handler(question)
            if not isinstance(resp, RAGResponse):
                raise TypeError(f"핸들러는 RAGResponse를 반환해야 합니다 (받은 타입: {type(resp).__name__})")
            if resp.latency_ms <= 0:
                resp.latency_ms = round((time.perf_counter() - t0) * 1000, 1)
            if not resp.route_reason:
                resp.route_reason = decision.reason
            if not resp.collection_name:
                resp.collection_name = decision.collection_name
            return resp
        except Exception as exc:
            return make_error_response(
                domain=decision.domain,
                error_message=f"{type(exc).__name__}: {exc}",
                latency_ms=(time.perf_counter() - t0) * 1000,
                collection_name=decision.collection_name,
                route_reason=decision.reason,
            )
