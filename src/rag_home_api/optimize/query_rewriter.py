"""[STEP 5-1: Query Rewrite] 구어체·평수 정규화 + 2개 지역 복합 비교 서브쿼리 분해.

작성자: 최기혁 (kh)
원칙:
  - 단순 질문의 조건(자치구·법정동·유형·전월세·면적대)을 절대 왜곡·추가하지 않는다.
  - 구어체(빌라→연립다세대, 다달이→월세, 보증금만→전세) 및 평수(10평 이하→33㎡이하, 10~18평→33~60㎡, 18평 초과→60㎡초과)를 표준 검색어로 변환한다.
  - 두 지역/조건을 비교하는 복합 질문(예: DEV-05, DEV-06)은 sub_queries 2개로 분해해 각 지역별 누락(FN)을 방지한다.
  - 오류 시 원질문 그대로 안전 폴백한다.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from rag_home_api.shared.core.logger import get_logger
from rag_home_api.shared.core.vector_store import get_llm

logger = get_logger(__name__)

VALID_HOUSING = {"아파트", "오피스텔", "연립다세대", "단독다가구"}
VALID_LEASE = {"전세", "월세"}
VALID_AREA = {"33이하", "33초과_60이하", "60초과"}


class _SubQuerySchema(BaseModel):
    keyword_query: str = Field(description="BM25 키워드 검색용 표준 행정용어 질의 (구·동·주택유형·전월세·면적대 중심)")
    semantic_query: str = Field(description="Vector 의미 검색용 자연어 질의")
    district: str | None = Field(default=None, description="명확한 서울 자치구 1개 (예: 강남구), 불명확하면 null")
    dong: str | None = Field(default=None, description="명확한 법정동 1개 (예: 번동), 불명확하면 null")
    housing_type: str | None = Field(default=None, description="아파트/오피스텔/연립다세대/단독다가구 중 1개, 불명확하면 null")
    lease_type: str | None = Field(default=None, description="전세/월세 중 1개, 불명확하면 null")
    area_band: str | None = Field(default=None, description="33이하 / 33초과_60이하 / 60초과 중 1개, 불명확하면 null")


class _RewriteOutput(BaseModel):
    is_multi_hop: bool = Field(description="서로 다른 2개 지역이나 2개 조건을 비교하는 복합 질문이면 True")
    sub_queries: list[_SubQuerySchema] = Field(min_length=1, max_length=2, description="단순 질문은 1개, 2대상 비교 질문은 2개로 분해")


@dataclass
class SubQueryPlan:
    keyword_query: str
    semantic_query: str
    filters: dict[str, str] | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RewriteResult:
    original_query: str
    is_multi_hop: bool
    sub_queries: list[SubQueryPlan]
    fallback_used: bool = False

    def to_dict(self) -> dict:
        return {
            "original_query": self.original_query,
            "is_multi_hop": self.is_multi_hop,
            "sub_queries": [sq.to_dict() for sq in self.sub_queries],
            "fallback_used": self.fallback_used,
        }


REWRITE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "당신은 서울 청년 주거 RAG 검색용 질의 재작성(Query Rewrite) 전문가입니다.\n"
            "[변환 규칙]\n"
            "1. 일상 구어체를 DB 표준 행정용어로 정규화하세요:\n"
            "   - 빌라 / 다세대 / 연립 -> 연립다세대\n"
            "   - 단독주택 / 다가구 / 원룸건물 -> 단독다가구\n"
            "   - 보증금만 내는 / 월세 없이 -> 전세\n"
            "   - 다달이 내는 / 매달 임대료 / 월세 -> 월세\n"
            "   - 10평 이하 / 5~9평 / 원룸 / 작은 방 -> 33㎡이하 (area_band='33이하')\n"
            "   - 10~18평 / 12~17평 / 투룸 / 소형 -> 33~60㎡ (area_band='33초과_60이하')\n"
            "   - 18평 초과 / 20평대 이상 / 중대형 -> 60㎡초과 (area_band='60초과')\n"
            "   - 세입자/주소 옮김 -> 임차인/전입신고/대항력\n"
            "2. **절대 왜곡 금지**: 질문에 없는 자치구·법정동·유형·면적대를 임의로 지어내지 마세요! 모호하면 해당 필터 필드는 null로 두세요.\n"
            "3. **복합 비교 분해**: 서로 다른 2개 지역(또는 2개 주택/계약 조건)을 비교해 달라는 질문이면 `is_multi_hop=True`로 두고, 각 대상을 독립적으로 검색할 수 있도록 `sub_queries` 2개로 분해하세요.\n"
            "4. 단순 질문이면 `is_multi_hop=False`, `sub_queries` 1개만 반환하세요.",
        ),
        ("human", "원본 질문: {question}"),
    ]
)


def _sanitize_filters(sq: _SubQuerySchema) -> dict[str, str] | None:
    flt: dict[str, str] = {}
    if sq.district and sq.district.endswith("구"):
        flt["district"] = sq.district.strip()
    if sq.dong and any(sq.dong.strip().endswith(sfx) for sfx in ("동", "가", "로")):
        flt["dong"] = sq.dong.strip()
    if sq.housing_type in VALID_HOUSING:
        flt["housing_type"] = sq.housing_type
    if sq.lease_type in VALID_LEASE:
        flt["lease_type"] = sq.lease_type
    if sq.area_band in VALID_AREA:
        flt["area_band"] = sq.area_band
    return flt or None


def rewrite_query(question: str) -> RewriteResult:
    """사용자 질문을 검색 친화적 표준 질의(+ 필요 시 2개 서브쿼리)로 변환한다."""
    try:
        llm = get_llm()
        chain = REWRITE_PROMPT | llm.with_structured_output(_RewriteOutput)
        res: _RewriteOutput = chain.invoke({"question": question})
        plans = [
            SubQueryPlan(
                keyword_query=sq.keyword_query.strip() or question,
                semantic_query=sq.semantic_query.strip() or question,
                filters=_sanitize_filters(sq),
            )
            for sq in res.sub_queries
        ]
        if not plans:
            raise ValueError("empty sub_queries")
        return RewriteResult(
            original_query=question,
            is_multi_hop=bool(res.is_multi_hop and len(plans) > 1),
            sub_queries=plans,
            fallback_used=False,
        )
    except Exception as exc:
        logger.warning("Query Rewrite 실패 -> 원질문 폴백: %s", exc)
        return RewriteResult(
            original_query=question,
            is_multi_hop=False,
            sub_queries=[SubQueryPlan(keyword_query=question, semantic_query=question, filters=None)],
            fallback_used=True,
        )
