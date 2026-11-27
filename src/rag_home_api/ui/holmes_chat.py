"""찾아줘 홈즈 — 서울 주거 탐색 4도메인 통합 RAG 서비스 (Streamlit).

실행 (저장소 루트):
    .venv/bin/streamlit run src/rag_home_api/ui/holmes_chat.py
    # 또는
    uv run --no-sync --with streamlit streamlit run src/rag_home_api/ui/holmes_chat.py
"""

from __future__ import annotations

import base64
import html
import re
import time
from pathlib import Path
from typing import Any, Iterator, Literal

import streamlit as st
from pydantic import BaseModel, Field

ASSETS_DIR = Path(__file__).resolve().parent / "assets"
HERO_IMG_PATH = ASSETS_DIR / "holmes_hero.png"
AVATAR_IMG_PATH = ASSETS_DIR / "holmes_avatar.png"

PART_ORDER = ["auto", "public_housing", "law", "rent_deals", "rent_stats"]

PART_CONFIGS: dict[str, dict[str, Any]] = {
    "auto": {
        "label": "전체",
        "full_name": "전체 자동 탐색",
        "owner": "4인 통합 라우터",
        "collection": "4개 도메인 자동 분기",
        "desc": "질문 의도를 분석해 공공임대 · 임대차 법령 · 전월세 조건 · 25구 시세 중 가장 적합한 도메인으로 자동 연결합니다.",
        "placeholder": "서울 공공임대, 전월세 시세, 임대차 법령에 대해 질문해 보세요",
        "samples": [
            ("공공임대 · 대치르엘 35㎡ 청년 임대료", "2026년 1차 서울주택도시개발공사 행복주택 입주자 모집공고에서 강남구 대치르엘 35㎡ 청년(소득 있음)의 임대보증금과 월임대료는 얼마인가요?"),
            ("임대차 법령 · 월세 인상 법적 상한", "집주인이 월세를 올리려고 하는데 법적으로 몇 퍼센트까지 올릴 수 있나요?"),
            ("전월세 조건 · 신림동 보증금 500 / 월세 45 이하", "신림동에서 보증금 500만원 이하, 월세 45만원 이하인 집을 찾아줘"),
            ("25구 시세 · 신림동 원룸 월세 중앙값", "관악구 신림동 단독다가구 원룸(33㎡ 이하) 월세 시세는 보통 얼마인가요?"),
        ],
    },
    "public_housing": {
        "label": "공공임대",
        "full_name": "공공임대 · 청년안심 공고문",
        "owner": "장예지 (yj)",
        "collection": "yj_public_rental · chunkV2",
        "desc": "LH · SH 공고문 90건(2,482쪽)에서 표 구조 보존 청킹(V2)과 Hybrid RRF + LLM 리랭킹으로 입주자격 · 임대조건 · 일정을 찾습니다.",
        "placeholder": "공고명, 단지명, 주택형 또는 입주자격을 질문해 보세요",
        "samples": [
            ("근거 확인 · 대치르엘 35㎡ 청년 임대조건", "2026년 1차 서울주택도시개발공사 행복주택 입주자 모집공고에서 강남구 대치르엘 35㎡ 청년(소득 있음)의 임대보증금과 월임대료는 얼마인가요?"),
            ("근거 확인 · 관악봉천 행복주택 17C형 임대조건", "서울관악봉천 H-1·2·3BL 행복주택 예비입주자 모집공고의 17C형 기본 임대조건은 어떻게 되나요?"),
            ("근거 부족 · LH수서1 행복주택 예상 경쟁률·커트라인", "LH수서1 행복주택 36A형의 2026년 11월 기준 예상 당첨 경쟁률과 커트라인 점수는 몇 점인가요?"),
            ("근거 없음 · 부산 해운대구 국민임대 신청 기간", "부산광역시 해운대구 국민임대주택 예비입주자 모집 신청 기간은 언제인가요?"),
        ],
        "sample_groups": [
            {
                "type": "grounded",
                "badge": "유형 1 · 근거 확인 (이어묻기 가능)",
                "fg": "#0B7A6E",
                "bg": "#EDFAF7",
                "desc": "공고문 원문에서 임대조건·자격을 확인하고, 내 소득·자산 조건으로 이어서 상담할 수 있어요.",
                "hint": "답변 후 \"내 월소득이 350만원인데 나도 신청 가능한가?\"처럼 같은 공고로 이어서 질문해 보세요.",
                "items": [
                    (
                        "임대조건 → 자격 이어묻기",
                        "2026년 1차 서울주택도시개발공사 행복주택 입주자 모집공고에서 강남구 대치르엘 35㎡ 청년(소득 있음)의 임대보증금과 월임대료는 얼마인가요?",
                    ),
                    (
                        "임대조건 → 소득·서류 이어묻기",
                        "서울관악봉천 H-1·2·3BL 행복주택 예비입주자 모집공고의 17C형 기본 임대조건은 어떻게 되나요?",
                    ),
                ],
            },
            {
                "type": "partial_missing",
                "badge": "유형 2 · 근거 부족 문서 (공식 사이트 + 질문 유도)",
                "fg": "#A85507",
                "bg": "#FFF8EB",
                "desc": "공고문에 기재되지 않은 세부 결과(경쟁률·커트라인·대기기간)를 물으면 관련 공식 사이트와 대체 질문을 안내합니다.",
                "hint": "",
                "items": [
                    (
                        "경쟁률 · 커트라인 미기재",
                        "LH수서1 행복주택 36A형의 2026년 11월 기준 예상 당첨 경쟁률과 커트라인 점수는 몇 점인가요?",
                    ),
                    (
                        "실제 경쟁률 · 예비순번 대기기간",
                        "2026년 1차 행복주택 공고에서 강남구 대치르엘 35㎡ 청년 유형의 실제 청약 경쟁률과 예비순번 대기 기간은 얼마나 걸리나요?",
                    ),
                ],
            },
            {
                "type": "no_evidence",
                "badge": "유형 3 · 근거 없음 (환각 방지 + 대안 추천)",
                "fg": "#475467",
                "bg": "#F2F4F7",
                "desc": "수집 범위 밖(타 지역·미래 정책) 질문은 지어내지 않고 사실대로 모른다고 답한 뒤 가능한 질문을 추천합니다.",
                "hint": "",
                "items": [
                    (
                        "타 지역 공고 · 수집 범위 외",
                        "부산광역시 해운대구 국민임대주택 예비입주자 모집 신청 기간은 언제인가요?",
                    ),
                    (
                        "미래 정책 · 문서 없음",
                        "2028년 시행 예정인 서울시 공공임대주택 소득기준 개편안의 1인 가구 상한액은 얼마인가요?",
                    ),
                ],
            },
        ],
    },
    "law": {
        "label": "임대차 법령",
        "full_name": "주택임대차 법령 · 가이드",
        "owner": "최지훈 (jh)",
        "collection": "jh_1000 · 157조문",
        "desc": "주택임대차보호법 · 주민등록법 · 표준계약서 등 현행 법령 157조문에서 중복 제거 Hybrid RRF로 조문 근거와 원문을 안내합니다.",
        "placeholder": "계약갱신요구권, 전입신고, 확정일자, 증액 상한 등을 질문해 보세요",
        "samples": [
            ("차임상한 · 집주인 월세 인상 법적 상한선", "집주인이 월세를 올리려고 하는데 법적으로 몇 퍼센트까지 올릴 수 있나요?"),
            ("신고기한 · 이사 후 전입신고 법정 기한", "이사한 뒤 전입신고는 며칠 이내에 해야 하나요?"),
            ("계약갱신 · 계약갱신요구권 행사 기간과 조건", "임차인이 계약갱신요구권을 행사할 수 있는 기간과 조건이 어떻게 되나요?"),
            ("근거 부족 확인 · 법정 벽지 단가 기준", "임대차 계약 시 도배·장판 법정 벽지 단가 기준은 얼마인가요?"),
        ],
    },
    "rent_deals": {
        "label": "전월세 조건",
        "full_name": "전월세 개별 거래 조건 탐색",
        "owner": "선병철 (bc)",
        "collection": "course_v0_small · 31,772건",
        "desc": "관악구 실거래 31,772건에서 금액 · 면적 · 유형 조건을 추출해 Vector/BM25 공통 필터와 리랭킹으로 조건에 맞는 사례를 찾습니다.",
        "placeholder": "동네, 보증금/월세 상한, 면적 조건을 입력해 보세요",
        "samples": [
            ("월세 조건 · 신림동 보증금 500 / 월세 45 이하", "신림동에서 보증금 500만원 이하, 월세 45만원 이하인 집을 찾아줘"),
            ("전세 조건 · 남현동 빌라 2.5억 이하 / 8평 이상", "남현동 빌라 전세, 보증금 2억 5천만원 이하, 8평 이상 거래 사례가 있나요?"),
            ("원룸 조건 · 봉천동 보증금 1000 / 월세 55 이하", "봉천동에서 보증금 1000만원 이하, 월세 55만원 이하 원룸 거래를 알려줘"),
            ("범위 외 확인 · 부산 해운대구 아파트 매매가", "부산 해운대구 아파트 매매가는 얼마인가요?"),
        ],
    },
    "rent_stats": {
        "label": "25구 시세",
        "full_name": "서울 25구 전월세 시세 통계",
        "owner": "최기혁 (kh)",
        "collection": "housing_rent_kh_kure_v1 · 6,200블록",
        "desc": "서울 25개 자치구 최근 1년 실거래 562,624건을 집계해 KURE-v1 + 사전 필터 v2 + 버전 캐시로 지역별 시세 중앙값을 안내합니다.",
        "placeholder": "자치구, 동네, 평형대별 전월세 시세나 지역 비교를 질문해 보세요",
        "samples": [
            ("원룸 시세 · 관악구 신림동 33㎡ 이하 월세", "관악구 신림동 단독다가구 원룸(33㎡ 이하) 월세 시세는 보통 얼마인가요?"),
            ("지역 비교 · 강남 삼성동 전세 vs 노원 월계동 월세", "강남구 삼성동 빌라 전세(33㎡ 이하)와 노원구 월계동 단독다가구 월세(33~60㎡) 시세를 비교해줘"),
            ("전세 시세 · 용산구 한남동 33~60㎡ 전세 중앙값", "용산구 한남동 단독다가구 전세 33~60㎡ 보증금 중앙값은 얼마인가요?"),
            ("근거 부족 확인 · 2027년 강남구 원룸 월세 전망", "2027년 서울 강남구 원룸 월세 전망치는 얼마로 예상되나요?"),
        ],
    },
}

DOMAIN_LABELS = {
    "public_housing": "공공임대 공고",
    "law": "임대차 법령",
    "rent_deals": "전월세 조건",
    "rent_stats": "25구 시세",
    "unsupported": "지원 범위 외",
}

STATUS_STYLE = {
    "answered": ("근거 확인", "#0B7A6E", "#EDFAF7"),
    "insufficient_evidence": ("근거 부족", "#A85507", "#FFF8EB"),
    "unsupported": ("근거 없음", "#475467", "#F2F4F7"),
    "error": ("오류 격리", "#C92A2A", "#FFF2F2"),
}

EVIDENCE_STATUS_STYLE = {
    "grounded": ("근거 확인", "#0B7A6E", "#EDFAF7"),
    "partial_missing": ("근거 부족", "#A85507", "#FFF8EB"),
    "no_evidence": ("근거 없음", "#475467", "#F2F4F7"),
    "error": ("오류 격리", "#C92A2A", "#FFF2F2"),
}

RELATED_OFFICIAL_SITES: dict[str, list[tuple[str, str, str]]] = {
    "public_housing": [
        (
            "SH 서울주택도시공사 인터넷청약시스템",
            "https://www.i-sh.co.kr/app/index.do",
            "행복주택·청년안심주택 공고문, 청약 경쟁률·당첨자·예비순번 조회",
        ),
        (
            "LH 청약플러스",
            "https://apply.lh.or.kr",
            "LH 공공임대·매입임대·예비입주자 모집공고 및 청약 결과 확인",
        ),
        (
            "마이홈 포털 (국토교통부)",
            "https://www.myhome.go.kr",
            "공공임대 입주자격 자가진단 및 전국 임대주택 공고 통합 조회",
        ),
        (
            "서울주거포털",
            "https://housing.seoul.go.kr",
            "서울시 청년안심주택·미리내집·주거지원 정책 공식 안내",
        ),
    ],
    "law": [
        (
            "국가법령정보센터",
            "https://www.law.go.kr",
            "주택임대차보호법·주민등록법 현행 조문 원문 확인",
        ),
        (
            "찾기쉬운 생활법령정보",
            "https://easylaw.go.kr",
            "임대차 계약·보증금 보호·분쟁조정 해설 안내",
        ),
    ],
    "rent_deals": [
        (
            "국토교통부 실거래가 공개시스템",
            "https://rt.molit.go.kr",
            "전국 전월세·매매 공식 실거래가 조회",
        ),
        (
            "서울부동산정보광장",
            "https://land.seoul.go.kr",
            "서울시 자치구별 전월세 실거래 현황",
        ),
    ],
    "rent_stats": [
        (
            "서울부동산정보광장",
            "https://land.seoul.go.kr",
            "서울 25개 자치구 전월세 실거래 통계",
        ),
        (
            "국토교통부 실거래가 공개시스템",
            "https://rt.molit.go.kr",
            "전국 전월세·매매 공식 실거래가 조회",
        ),
    ],
    "unsupported": [
        (
            "마이홈 포털 (국토교통부)",
            "https://www.myhome.go.kr",
            "전국 공공임대 공고·입주자격 자가진단 통합 안내",
        ),
        (
            "서울주거포털",
            "https://housing.seoul.go.kr",
            "서울시 공공임대·청년 주거지원 공식 포털",
        ),
    ],
}

RECOMMENDED_GROUNDED_QUESTIONS: dict[str, list[str]] = {
    "public_housing": [
        "2026년 1차 서울주택도시개발공사 행복주택 입주자 모집공고에서 강남구 대치르엘 35㎡ 청년(소득 있음)의 임대보증금과 월임대료는 얼마인가요?",
        "2026년 1차 청년안심주택(공공임대) 모집공고에서 2순위 신청자의 총자산 기준은 얼마이며, 총자산가액은 어떻게 산출하나요?",
        "2025년 3차 서울 행복주택 공고에서 일반공급 1순위는 서울에 사는 사람만 해당하나요? 인접 지역도 포함되나요?",
    ],
    "law": [
        "집주인이 월세를 올리려고 하는데 법적으로 몇 퍼센트까지 올릴 수 있나요?",
        "이사한 뒤 전입신고는 며칠 이내에 해야 하나요?",
        "임차인이 계약갱신요구권을 행사할 수 있는 기간과 조건이 어떻게 되나요?",
    ],
    "rent_deals": [
        "신림동에서 보증금 500만원 이하, 월세 45만원 이하인 집을 찾아줘",
        "봉천동에서 보증금 1000만원 이하, 월세 55만원 이하 원룸 거래를 알려줘",
        "남현동 빌라 전세, 보증금 2억 5천만원 이하, 8평 이상 거래 사례가 있나요?",
    ],
    "rent_stats": [
        "관악구 신림동 단독다가구 원룸(33㎡ 이하) 월세 시세는 보통 얼마인가요?",
        "용산구 한남동 단독다가구 전세 33~60㎡ 보증금 중앙값은 얼마인가요?",
        "강남구 삼성동 빌라 전세(33㎡ 이하)와 노원구 월계동 단독다가구 월세(33~60㎡) 시세를 비교해줘",
    ],
    "unsupported": [
        "2026년 1차 서울주택도시개발공사 행복주택 입주자 모집공고에서 강남구 대치르엘 35㎡ 청년(소득 있음)의 임대보증금과 월임대료는 얼마인가요?",
        "2026년 1차 청년안심주택(공공임대) 모집공고에서 2순위 신청자의 총자산 기준은 얼마인가요?",
        "집주인이 월세를 올리려고 하는데 법적으로 몇 퍼센트까지 올릴 수 있나요?",
    ],
}

FALLBACK_FOLLOWUPS: dict[str, list[str]] = {
    "public_housing": [
        "내 월소득이 350만원이고 무주택 1인 가구인데 이 공고에 신청 가능한가요?",
        "이 공고의 청년 소득·자산 기준과 자동차 가액 기준을 알려줘",
        "청약 접수 일정과 제출해야 할 심사서류도 정리해줘",
    ],
    "law": [
        "이 규정에서 예외가 적용되는 경우도 있나요?",
        "임차인이 실제로 지켜야 할 기한이나 절차를 정리해줘",
        "전입신고와 확정일자 효력 차이도 쉽게 설명해줘",
    ],
    "rent_deals": [
        "비슷한 동네의 전월세 평균 시세와 비교하면 어떤가요?",
        "보증금을 조금 높이고 월세를 낮춘 조건으로도 찾아줘",
        "계약할 때 꼭 확인해야 할 법적 주의사항을 알려줘",
    ],
    "rent_stats": [
        "이 지역의 실제 조건별 거래 사례도 볼 수 있나요?",
        "같은 예산으로 들어갈 수 있는 공공임대 공고도 있나요?",
        "전월세 계약 시 월세 인상 상한선도 알려줘",
    ],
    "unsupported": [
        "강남구 대치르엘 35㎡ 청년 행복주택 보증금과 월세는 얼마인가요?",
        "집주인이 월세를 올릴 때 법적 상한선은 몇 퍼센트인가요?",
        "관악구 신림동 원룸(33㎡ 이하) 월세 시세는 보통 얼마인가요?",
    ],
}


class ContextualizedQuery(BaseModel):
    """이전 대화 맥락을 반영한 검색용 질의 재작성 스키마."""

    needs_retrieval: bool = Field(
        default=True,
        description="새 DB 검색이 필요하면 True, 직전 답변을 단순히 요약/쉽게 풀어달라는 요청이면 False",
    )
    resolved_query: str = Field(
        description="이전 대화의 공고명·단지·지역·대상을 반영해 단독으로도 의미가 완전한 한국어 질문"
    )
    reason: str = Field(default="", description="맥락 반영 사유 한 줄")


class ConversationalReply(BaseModel):
    """근거 기반 친절한 챗봇 응답 + 후속 추천 질문 스키마."""

    evidence_state: Literal["grounded", "partial_missing", "no_evidence"] = Field(
        default="grounded",
        description=(
            "근거 상태 분류: "
            "'grounded'(공고문 근거로 답변 완료), "
            "'partial_missing'(도메인 관련 문서이지만 경쟁률·커트라인·세부 결과 등 일부 근거 부족), "
            "'no_evidence'(타 지역·미래 정책·수집 범위 외 등 아예 문서 근거 없음)"
        ),
    )
    friendly_answer: str = Field(
        description="찾아온 근거와 숫자·[1],[2] 인용을 100% 유지하면서 홈즈 상담사처럼 친절하고 읽기 쉽게 풀어 쓴 답변"
    )
    follow_up_questions: list[str] = Field(
        default_factory=list,
        description="방금 답변과 자연스럽게 이어지거나(grounded) 대신 확인 가능한(partial_missing/no_evidence) 구체적 후속 질문 3개",
    )


_OUT_OF_SCOPE_PATTERNS = (
    "부산",
    "해운대",
    "대구",
    "대전",
    "광주광역시",
    "울산",
    "세종시",
    "제주",
    "강원도",
    "충북",
    "충남",
    "전북",
    "전남",
    "경북",
    "경남",
    "경기도",
    "2027년",
    "2028년",
    "2029년",
    "2030년",
    "개편안",
    "전망치",
    "매매가",
    "분양가",
)

_PARTIAL_MISSING_PATTERNS = (
    "경쟁률",
    "커트라인",
    "당첨자",
    "예비순번",
    "대기 기간",
    "대기기간",
    "발표 명단",
    "추첨 결과",
    "현금 지원",
    "이사비",
)


def _classify_evidence_state(
    question: str,
    resp: dict[str, Any],
) -> Literal["grounded", "partial_missing", "no_evidence", "error"]:
    """응답 상태와 질문 신호를 바탕으로 3가지 근거 유형(grounded / partial_missing / no_evidence)을 판별한다."""
    status = str(resp.get("status") or "error")
    if status == "error":
        return "error"
    if status == "answered":
        return "grounded"
    if status == "unsupported":
        return "no_evidence"

    q_text = question.strip()
    if any(tok in q_text for tok in _OUT_OF_SCOPE_PATTERNS):
        return "no_evidence"
    if any(tok in q_text for tok in _PARTIAL_MISSING_PATTERNS):
        return "partial_missing"

    contexts = list(resp.get("retrieved_contexts") or [])
    if not contexts:
        return "no_evidence"
    return "partial_missing"


@st.cache_resource(show_spinner=False)
def _get_chat_llm() -> Any | None:
    try:
        from langchain_openai import ChatOpenAI
        from pydantic import SecretStr

        from rag_home_api.config import settings

        return ChatOpenAI(
            model=settings.llm_model,
            api_key=SecretStr(settings.openai_api_key),
            reasoning_effort="low",
        )
    except Exception:
        return None


def _format_recent_history(history: list[dict[str, Any]], max_turns: int = 3) -> str:
    """최근 N턴의 대화를 프롬프트용 짧은 텍스트로 정리한다."""
    if not history:
        return ""
    recent = history[-(max_turns * 2) :]
    lines: list[str] = []
    for msg in recent:
        if msg["role"] == "user":
            lines.append(f"사용자: {msg['content']}")
        else:
            resp = msg.get("response") or {}
            ans = str(resp.get("raw_answer") or resp.get("answer") or "")[:260]
            dom = DOMAIN_LABELS.get(str(resp.get("domain") or ""), "")
            lines.append(f"홈즈({dom}): {ans}")
    return "\n".join(lines)


def contextualize_question(
    question: str,
    history: list[dict[str, Any]],
) -> tuple[str, bool, str]:
    """이전 대화가 있을 때 생략형 후속 질문을 완결된 검색 질의로 보완한다."""
    if not history:
        return question, True, ""

    llm = _get_chat_llm()
    if llm is None:
        return question, True, ""

    hist_text = _format_recent_history(history, max_turns=3)
    try:
        from langchain_core.messages import HumanMessage, SystemMessage

        structured = llm.with_structured_output(ContextualizedQuery)
        out: ContextualizedQuery = structured.invoke(
            [
                SystemMessage(
                    content=(
                        "당신은 서울 주거 상담 챗봇의 대화 맥락 분석기입니다.\n"
                        "직전 대화 기록과 사용자의 새 입력을 보고 다음을 판단하세요:\n"
                        "1. 새 입력이 '그럼 월세는?', '거기 2순위는?', '예비 입주자는 몇 명?', '내 월소득이 350만원인데 나도 신청 가능한가?'처럼 "
                        "이전 공고·단지를 이어 묻는 질문이면, 이전 대화의 공고명·단지명·지역·주택유형을 포함해 단독으로도 검색 가능한 완결 질문(resolved_query)으로 바꾸세요.\n"
                        "   - 특히 사용자가 본인 소득·자산·가구 조건을 말하며 신청 가능 여부를 물으면, 공고문의 공식 입주자격·소득기준·자산기준표를 잘 찾을 수 있도록 "
                        "해당 공고명 + 신청자 유형(예: 청년/신혼부부) + '입주자격 소득 기준 및 자산 기준' 키워드가 포함된 질의로 작성하세요.\n"
                        "2. 완전히 새로운 주제 질문이면 원질문을 그대로 resolved_query에 넣으세요.\n"
                        "3. '좀 더 쉽게 설명해줘', '표로 정리해줘', '핵심만 요약해줘'처럼 직전 답변만 다시 풀어달라는 요청이면 needs_retrieval=False로 설정하세요.\n"
                        "4. 원질문에 없는 숫자나 조건을 임의로 지어내지 마세요."
                    )
                ),
                HumanMessage(content=f"[최근 대화]\n{hist_text}\n\n[사용자 새 입력]\n{question}"),
            ]
        )
        resolved = out.resolved_query.strip() or question
        return resolved, bool(out.needs_retrieval), out.reason.strip()
    except Exception:
        return question, True, ""


def _build_followup_fallback(
    domain: str,
    selected_part: str,
    evidence_state: str = "grounded",
) -> list[str]:
    key = domain if domain in FALLBACK_FOLLOWUPS else selected_part
    if evidence_state in ("partial_missing", "no_evidence"):
        rec_key = domain if domain in RECOMMENDED_GROUNDED_QUESTIONS else selected_part
        return list(
            RECOMMENDED_GROUNDED_QUESTIONS.get(rec_key, RECOMMENDED_GROUNDED_QUESTIONS["unsupported"])
        )[:3]
    return list(FALLBACK_FOLLOWUPS.get(key, FALLBACK_FOLLOWUPS["unsupported"]))[:3]


def conversationalize_response(
    user_question: str,
    resolved_query: str,
    resp: dict[str, Any],
    history: list[dict[str, Any]],
) -> dict[str, Any]:
    """검색된 사실과 인용을 100% 보존하면서 3유형(근거 확인 / 근거 부족 / 근거 없음)에 맞춘 상담 답변과 후속 질문을 생성한다."""
    status = str(resp.get("status") or "error")
    domain = str(resp.get("domain") or "unsupported")
    selected_part = str(resp.get("selected_part") or "auto")
    raw_answer = str(resp.get("answer") or "").strip()
    resp["raw_answer"] = raw_answer

    base_state = _classify_evidence_state(user_question, resp)
    meta = dict(resp.get("meta") or {})
    meta["evidence_state"] = base_state
    resp["meta"] = meta

    # 오류 응답은 원본 오류 안내를 그대로 유지
    if status == "error":
        resp["follow_ups"] = _build_followup_fallback(domain, selected_part, "error")
        return resp

    llm = _get_chat_llm()
    if llm is None:
        resp["follow_ups"] = _build_followup_fallback(domain, selected_part, base_state)
        return resp

    contexts = list(resp.get("retrieved_contexts") or [])
    ctx_snippets = []
    for i, c in enumerate(contexts[:4], start=1):
        txt = str(c).strip()
        ctx_snippets.append(f"[{i}] {txt[:700]}")
    ctx_block = "\n\n".join(ctx_snippets) if ctx_snippets else "(검색된 근거 없음)"
    hist_text = _format_recent_history(history, max_turns=2)

    system_prompt = (
        "당신은 서울 상경 20~30대 청년을 돕는 친절하고 신뢰감 있는 주거 상담 AI '찾아줘 홈즈'입니다.\n"
        "검색 엔진이 찾아온 [원본 검색 답변]과 [검색 근거]만을 바탕으로, 사용자와 자연스럽게 대화하듯 이해하기 쉽게 답변을 다듬고 "
        "상황에 맞는 후속/추천 질문 3개를 제안하세요.\n\n"
        "절대 규칙:\n"
        "1. [원본 검색 답변]과 [검색 근거]에 없는 숫자·금액·기간·조건·공고명·경쟁률은 절대 추가하거나 추측하지 마세요.\n"
        "2. 원문의 숫자 단위(천원/원/만원/㎡/%)와 근거 인용 번호([1], [2] 등)는 정확히 유지하세요.\n"
        "3. [판정된 근거 유형]이 'grounded'일 때:\n"
        "   - 첫 문장은 사용자 질문에 바로 답하는 친절한 핵심 결론으로 시작하세요.\n"
        "   - 조건·금액·항목이 여러 개면 불릿 목록으로 한눈에 보기 쉽게 정리하세요.\n"
        "   - 사용자가 본인 조건(예: 월소득 350만원, 1인 가구 등)으로 신청 가능 여부를 물었다면, [검색 근거]에 나온 공식 기준과만 정확히 비교하고 "
        "     함께 확인해야 할 다른 요건(무주택·총자산·자동차 등 근거에 있는 항목)도 짚어 주세요.\n"
        "   - 마지막에 같은 공고의 자격·소득·일정 등도 이어서 물어보라는 한 줄 안내를 덧붙이세요.\n"
        "   - follow_up_questions에는 방금 공고/주제에서 사용자가 자연스럽게 이어서 물어볼 심화 질문 3개(예: 임대조건 답변 후 → '내 월소득이 350만원인데 이 공고에 신청 가능한가요?', '청년 소득·자산 기준 알려줘', '청약 일정과 제출 서류는?')를 작성하세요.\n"
        "4. [판정된 근거 유형]이 'partial_missing'일 때 (근거 부족 문서):\n"
        "   - 공고문 원문에 해당 세부 정보(예: 실제 경쟁률·커트라인·당첨자 발표 결과·대기 기간 등)가 기재되어 있지 않음을 솔직하고 부드럽게 안내하세요.\n"
        "   - 절대 수치를 추정하거나 지어내지 말고, 아래 공식 사이트(SH 청약시스템·LH 청약플러스·마이홈 포털 등)에서 직접 확인하실 수 있다고 안내하세요.\n"
        "   - 대신 현재 공고문에서 바로 확인할 수 있는 항목(임대조건·신청자격·소득/자산 기준·접수 일정 등)으로 자연스럽게 이어질 수 있도록 안내하세요.\n"
        "5. [판정된 근거 유형]이 'no_evidence'일 때 (근거 없음 · 환각 방지):\n"
        "   - 수집된 서울시 2025~2026년 공공임대 공고문 데이터에 해당 정보가 없어 사실대로 알 수 없다고 명확히 말씀드리세요.\n"
        "   - 추측성 답변이나 일반 상식 기반의 수치 제시는 절대 금지합니다.\n"
        "   - 왜 답할 수 없는지(타 지역 공고, 미래 미발표 정책, 수집 범위 외 등) 짧게 설명하고, 현재 홈즈가 정확히 도와드릴 수 있는 서울시 공공임대 공고 질문을 추천해 주세요."
    )

    user_msg = (
        f"[최근 대화]\n{hist_text or '(첫 질문)'}\n\n"
        f"[사용자 질문]\n{user_question}\n"
        f"[검색에 사용된 질의]\n{resolved_query}\n"
        f"[도메인 / 상태 / 판정된 근거 유형]\n{DOMAIN_LABELS.get(domain, domain)} / {status} / {base_state}\n\n"
        f"[원본 검색 답변]\n{raw_answer}\n\n"
        f"[검색 근거]\n{ctx_block}"
    )

    try:
        from langchain_core.messages import HumanMessage, SystemMessage

        structured = llm.with_structured_output(ConversationalReply)
        out: ConversationalReply = structured.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_msg),
            ]
        )
        friendly = out.friendly_answer.strip()
        if friendly:
            resp["answer"] = friendly

        # 확정 규칙(grounded / 명백한 범위 외·근거 부족) 우선 적용
        final_state = base_state
        if status == "insufficient_evidence" and base_state == "partial_missing":
            if out.evidence_state == "no_evidence" and not any(
                tok in user_question for tok in _PARTIAL_MISSING_PATTERNS
            ):
                final_state = "no_evidence"

        meta["evidence_state"] = final_state
        meta["conversational_mode"] = True
        resp["meta"] = meta

        if final_state in ("partial_missing", "no_evidence"):
            # 근거 부족/근거 없음 상태에서는 실제 DB에서 확실히 답변 가능한 추천 질문을 보장
            resp["follow_ups"] = _build_followup_fallback(domain, selected_part, final_state)
        else:
            follow_ups = [q.strip() for q in out.follow_up_questions if q and q.strip()][:3]
            resp["follow_ups"] = follow_ups or _build_followup_fallback(domain, selected_part, final_state)
    except Exception:
        resp["follow_ups"] = _build_followup_fallback(domain, selected_part, base_state)

    return resp


def format_inline_citations(text: str) -> str:
    """답변 본문의 [1], [2] 인용 번호를 마크다운 인라인 배지로 강조한다."""
    text = re.sub(r"\*\*\[(\d+)\]\*\*", r"`[\1]`", text)
    return re.sub(r"(?<!`)(?<!\])\[(\d+)\](?!\()(?!`)", r"`[\1]`", text)


def _stream_answer_chunks(text: str) -> Iterator[str]:
    """답변 텍스트를 자연스러운 챗봇 타이핑 속도로 스트리밍한다."""
    tokens = re.findall(r"\S+\s*|\n+", text)
    for tok in tokens:
        yield tok
        time.sleep(0.012)


@st.cache_data(show_spinner=False)
def _img_data_uri(path_str: str) -> str:
    p = Path(path_str)
    if not p.exists():
        return ""
    return f"data:image/png;base64,{base64.b64encode(p.read_bytes()).decode('ascii')}"


@st.cache_resource(show_spinner=False)
def load_team_router() -> tuple[Any | None, str]:
    try:
        from rag_home_api.router import build_default_team_router

        return build_default_team_router(), ""
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def dispatch_question(
    question: str,
    selected_part: str,
    history: list[dict[str, Any]] | None = None,
    conversational_mode: bool = True,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    hist = list(history or [])
    router, init_err = load_team_router()

    if router is None:
        return {
            "status": "error",
            "domain": selected_part if selected_part != "auto" else "unsupported",
            "selected_part": selected_part,
            "resolved_query": question,
            "answer": "환경변수(`.env`의 `OPENAI_API_KEY`, `DATABASE_URL`)가 설정되지 않아 검색 엔진을 초기화할 수 없습니다.",
            "citations": [],
            "retrieved_contexts": [],
            "chunk_ids": [],
            "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
            "collection_name": "",
            "route_reason": "라우터 초기화 실패",
            "error_message": init_err,
            "meta": {},
            "follow_ups": _build_followup_fallback("unsupported", selected_part),
        }

    # 1) 멀티턴 대화 맥락 반영 (직전 대화 기반 후속 질문 보완)
    resolved_query = question
    needs_retrieval = True
    context_reason = ""
    if conversational_mode and hist:
        resolved_query, needs_retrieval, context_reason = contextualize_question(question, hist)

    # 2) 직전 답변을 단순히 쉽게 풀어달라/요약해달라는 후속 요청이면 직전 근거 재활용
    last_resp = hist[-1].get("response") if hist and hist[-1].get("role") == "assistant" else None
    if (
        conversational_mode
        and not needs_retrieval
        and last_resp is not None
        and last_resp.get("status") == "answered"
    ):
        reused = {
            "status": "answered",
            "domain": last_resp.get("domain", "unsupported"),
            "selected_part": selected_part,
            "resolved_query": resolved_query,
            "answer": str(last_resp.get("raw_answer") or last_resp.get("answer") or ""),
            "citations": list(last_resp.get("citations") or []),
            "retrieved_contexts": list(last_resp.get("retrieved_contexts") or []),
            "chunk_ids": list(last_resp.get("chunk_ids") or []),
            "latency_ms": 0.0,
            "collection_name": str(last_resp.get("collection_name") or ""),
            "route_reason": f"직전 검색 근거 재활용 ({context_reason or '대화형 재설명'})",
            "error_message": "",
            "meta": {"reused_context": True},
        }
        reused = conversationalize_response(question, resolved_query, reused, hist)
        reused["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        return reused

    # 3) 4도메인 라우터 또는 선택 도메인 핸들러 실행
    if selected_part == "auto":
        resp = router.invoke(resolved_query)
        data = resp.model_dump()
        data["selected_part"] = "auto"
    else:
        handler = router._handlers.get(selected_part)
        if handler is None:
            data = {
                "status": "unsupported",
                "domain": selected_part,
                "selected_part": selected_part,
                "answer": f"확인할 수 없음. '{PART_CONFIGS[selected_part]['full_name']}' 핸들러가 등록되어 있지 않습니다.",
                "citations": [],
                "retrieved_contexts": [],
                "chunk_ids": [],
                "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
                "collection_name": "",
                "route_reason": f"도메인 직접 선택 ({selected_part})",
                "error_message": "",
                "meta": {},
            }
        else:
            try:
                resp = handler(resolved_query)
                data = resp.model_dump()
                if not data.get("route_reason"):
                    data["route_reason"] = f"사용자 도메인 선택 → {PART_CONFIGS[selected_part]['full_name']}"
                data["selected_part"] = selected_part
            except Exception as exc:
                data = {
                    "status": "error",
                    "domain": selected_part,
                    "selected_part": selected_part,
                    "answer": "검색 또는 답변 생성 중 오류가 발생했습니다. DB 연결 및 환경 설정을 확인해 주세요.",
                    "citations": [],
                    "retrieved_contexts": [],
                    "chunk_ids": [],
                    "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
                    "collection_name": "",
                    "route_reason": f"도메인 직접 선택 ({selected_part})",
                    "error_message": f"{type(exc).__name__}: {exc}",
                    "meta": {},
                }

    data["resolved_query"] = resolved_query
    if resolved_query != question and context_reason:
        data["route_reason"] = f"{data.get('route_reason') or ''} · 맥락 보완({context_reason})".strip(" ·")

    # 4) 대화형 상담 모드: 검색된 근거 기반 자연스러운 답변 + 후속 추천 질문 3개 생성
    if conversational_mode:
        data = conversationalize_response(question, resolved_query, data, hist)
    else:
        ev_state = _classify_evidence_state(question, data)
        meta = dict(data.get("meta") or {})
        meta["evidence_state"] = ev_state
        data["meta"] = meta
        data["raw_answer"] = str(data.get("answer") or "")
        data["follow_ups"] = _build_followup_fallback(
            str(data.get("domain") or "unsupported"), selected_part, ev_state
        )

    data["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    return data


def inject_production_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/variable/pretendardvariable-dynamic-subset.min.css');

        :root {
            --bg: #FFFFFF;
            --sidebar: #F8F9FB;
            --surface: #F4F6F9;
            --border: #E6EAF0;
            --border-strong: #D3DAE5;
            --ink: #191F28;
            --body: #333D4B;
            --muted: #6B7684;
            --subtle: #8B95A1;
            --accent: #1B64DA;
            --accent-soft: #EEF4FF;
            --accent-border: #C9DDFF;
        }

        html, body, [class*="css"] {
            font-family: "Pretendard Variable", "Pretendard", "Apple SD Gothic Neo", "Malgun Gothic", sans-serif !important;
        }

        /* Clean white canvas & hide noisy Streamlit chrome */
        .stApp {
            background: var(--bg) !important;
            color: var(--ink) !important;
        }
        div[data-testid="stDecoration"] { display: none !important; }
        #MainMenu { visibility: hidden; }
        footer { visibility: hidden; }
        header[data-testid="stHeader"] {
            background: rgba(255, 255, 255, 0.85) !important;
            backdrop-filter: blur(8px);
        }

        /* Centered clean editorial width (Perplexity / Toss style) */
        .block-container {
            max-width: 820px !important;
            padding-top: 2.2rem !important;
            padding-bottom: 6rem !important;
        }

        /* Minimalist light-gray sidebar */
        section[data-testid="stSidebar"] {
            background: var(--sidebar) !important;
            border-right: 1px solid var(--border) !important;
        }
        section[data-testid="stSidebar"] .block-container {
            padding-top: 1.4rem !important;
        }

        .sb-brand {
            display: flex;
            align-items: center;
            gap: 11px;
            padding-bottom: 16px;
            border-bottom: 1px solid var(--border);
            margin-bottom: 16px;
        }
        .sb-brand img {
            width: 36px;
            height: 36px;
            border-radius: 50%;
            object-fit: cover;
            border: 1px solid var(--border-strong);
        }
        .sb-brand-name {
            font-size: 16px;
            font-weight: 800;
            color: var(--ink);
            letter-spacing: -0.025em;
            line-height: 1.2;
        }
        .sb-brand-sub {
            font-size: 11.5px;
            color: var(--subtle);
            font-weight: 500;
        }

        .sb-label {
            font-size: 11px;
            font-weight: 700;
            color: var(--subtle);
            letter-spacing: 0.05em;
            text-transform: uppercase;
            margin: 18px 0 8px 2px;
        }

        .sb-domain-row {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 7px 8px;
            border-radius: 8px;
            font-size: 13px;
            color: var(--body);
        }
        .sb-domain-row.active {
            background: var(--accent-soft);
            color: var(--accent);
            font-weight: 700;
        }
        .sb-dot {
            display: inline-block;
            width: 6px;
            height: 6px;
            border-radius: 50%;
            margin-right: 8px;
            background: #10B981;
            vertical-align: middle;
        }
        .sb-dot.off { background: #CBD5E1; }
        .sb-owner {
            font-size: 11.5px;
            color: var(--subtle);
            font-weight: 500;
        }

        .sb-history-item {
            font-size: 12.5px;
            color: var(--body);
            padding: 6px 8px;
            border-radius: 7px;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            border-bottom: 1px solid rgba(230, 234, 240, 0.5);
        }

        section[data-testid="stSidebar"] .stButton > button[kind="primary"] {
            background: #FFFFFF !important;
            border: 1px solid var(--border-strong) !important;
            color: var(--ink) !important;
            border-radius: 10px !important;
            font-size: 13px !important;
            font-weight: 700 !important;
            height: 38px !important;
            box-shadow: 0 1px 2px rgba(25, 31, 40, 0.04) !important;
        }
        section[data-testid="stSidebar"] .stButton > button[kind="primary"]:hover {
            border-color: var(--ink) !important;
            background: #FFFFFF !important;
        }
        section[data-testid="stSidebar"] .stButton > button[kind="secondary"] {
            background: transparent !important;
            border: 1px solid transparent !important;
            color: var(--body) !important;
            border-radius: 8px !important;
            font-size: 12.5px !important;
            font-weight: 500 !important;
            min-height: 32px !important;
            padding: 5px 8px !important;
            text-align: left !important;
            justify-content: flex-start !important;
            box-shadow: none !important;
        }
        section[data-testid="stSidebar"] .stButton > button[kind="secondary"]:hover {
            background: #EEF2F7 !important;
            color: var(--ink) !important;
        }
        section[data-testid="stSidebar"] .stButton > button[kind="secondary"] p {
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            margin: 0 !important;
        }

        /* Top domain bar */
        .top-meta-bar {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 12px;
            margin-bottom: 8px;
        }
        .top-meta-title {
            font-size: 12.5px;
            font-weight: 700;
            color: var(--muted);
        }
        .top-meta-col {
            font-size: 11.5px;
            color: var(--subtle);
            font-family: "SF Mono", ui-monospace, Menlo, monospace;
        }

        /* Sleek segmented control styling */
        div[data-testid="stSegmentedControl"] {
            margin-bottom: 12px;
        }

        /* Welcome Hero (disappears once chat begins!) */
        .welcome-box {
            display: flex;
            flex-direction: column;
            align-items: center;
            text-align: center;
            padding: 26px 0 22px;
        }
        .mascot-frame {
            width: 138px;
            height: 138px;
            border-radius: 32px;
            overflow: hidden;
            background: #FAF5E8;
            border: 1px solid #ECE4D5;
            box-shadow: 0 14px 32px rgba(25, 31, 40, 0.08), 0 2px 6px rgba(25, 31, 40, 0.04);
            margin-bottom: 20px;
        }
        .mascot-frame img {
            width: 100%;
            height: 100%;
            object-fit: cover;
            display: block;
        }
        .welcome-kicker {
            font-size: 12px;
            font-weight: 700;
            color: var(--accent);
            letter-spacing: 0.02em;
            margin-bottom: 6px;
        }
        .welcome-title {
            font-size: 27px;
            font-weight: 800;
            letter-spacing: -0.03em;
            color: var(--ink);
            margin: 0 0 8px 0;
            line-height: 1.25;
        }
        .welcome-desc {
            font-size: 14px;
            color: var(--muted);
            line-height: 1.6;
            max-width: 540px;
            margin: 0;
        }

        /* Sample cards (clean 2x2 grid buttons) */
        div[data-testid="stHorizontalBlock"] .stButton > button {
            background: #FFFFFF !important;
            border: 1px solid var(--border) !important;
            border-radius: 13px !important;
            padding: 13px 15px !important;
            text-align: left !important;
            justify-content: flex-start !important;
            color: var(--ink) !important;
            font-size: 13px !important;
            font-weight: 600 !important;
            line-height: 1.45 !important;
            min-height: 58px !important;
            box-shadow: 0 1px 3px rgba(25, 31, 40, 0.03) !important;
            transition: all 0.15s ease !important;
        }
        div[data-testid="stHorizontalBlock"] .stButton > button:hover {
            border-color: var(--accent-border) !important;
            background: #FAFCFF !important;
            color: var(--accent) !important;
            box-shadow: 0 6px 16px rgba(27, 100, 218, 0.08) !important;
        }

        /* Clean borderless editorial chat messages */
        div[data-testid="stChatMessage"] {
            background: transparent !important;
            border: none !important;
            padding: 12px 0 !important;
            border-bottom: 1px solid #F1F3F7 !important;
            border-radius: 0 !important;
        }

        /* User bubble pill look */
        .user-bubble-wrap {
            display: flex;
            justify-content: flex-end;
            margin: 6px 0 2px;
        }
        .user-bubble {
            max-width: 78%;
            background: var(--surface);
            color: var(--ink);
            padding: 11px 16px;
            border-radius: 16px 16px 4px 16px;
            font-size: 14.5px;
            line-height: 1.6;
            font-weight: 500;
            word-break: keep-all;
        }

        /* Assistant meta strip */
        .meta-bar {
            display: flex;
            align-items: center;
            flex-wrap: wrap;
            gap: 8px;
            margin-bottom: 10px;
            font-size: 12px;
        }
        .st-pill {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            font-weight: 700;
            padding: 3px 9px;
            border-radius: 6px;
        }
        .st-dot {
            width: 6px;
            height: 6px;
            border-radius: 50%;
            display: inline-block;
        }
        .dm-name { font-weight: 700; color: var(--ink); }
        .meta-sep { color: var(--border-strong); }
        .lat-txt {
            color: var(--subtle);
            font-family: "SF Mono", ui-monospace, Menlo, monospace;
            font-size: 11.5px;
        }
        .cache-pill {
            font-size: 11px;
            font-weight: 700;
            color: var(--accent);
            background: var(--accent-soft);
            padding: 2px 7px;
            border-radius: 5px;
        }
        .mode-pill {
            font-size: 11px;
            font-weight: 600;
            color: #4E5968;
            background: #F2F4F7;
            padding: 2px 7px;
            border-radius: 5px;
        }

        /* Inline citation badges inside chat answers */
        div[data-testid="stChatMessage"] p code,
        div[data-testid="stChatMessage"] li code {
            background: var(--accent-soft) !important;
            color: var(--accent) !important;
            border: 1px solid var(--accent-border) !important;
            border-radius: 5px !important;
            padding: 1px 5px !important;
            font-size: 12px !important;
            font-weight: 700 !important;
        }

        /* Follow-up question chips ("이어서 물어보기") */
        .followup-heading {
            font-size: 12px;
            font-weight: 700;
            color: var(--muted);
            margin: 14px 0 6px;
        }
        div[data-testid="stChatMessage"] .stButton > button {
            background: #F8FAFC !important;
            border: 1px solid var(--border) !important;
            border-radius: 10px !important;
            padding: 7px 13px !important;
            min-height: 36px !important;
            text-align: left !important;
            justify-content: flex-start !important;
            color: var(--body) !important;
            font-size: 13px !important;
            font-weight: 500 !important;
            line-height: 1.4 !important;
            box-shadow: none !important;
            transition: all 0.14s ease !important;
        }
        div[data-testid="stChatMessage"] .stButton > button:hover {
            background: var(--accent-soft) !important;
            border-color: var(--accent-border) !important;
            color: var(--accent) !important;
        }
        div[data-testid="stChatMessage"] .stButton > button p {
            color: inherit !important;
            margin: 0 !important;
        }

        /* 3-Type Question Group UI (public_housing) */
        .qtype-head {
            margin: 14px 0 8px;
            padding-top: 10px;
            border-top: 1px solid #EEF1F6;
        }
        .qtype-head.first {
            margin-top: 6px;
            padding-top: 0;
            border-top: none;
        }
        .qtype-row {
            display: flex;
            align-items: center;
            flex-wrap: wrap;
            gap: 8px;
            margin-bottom: 4px;
        }
        .qtype-badge {
            display: inline-flex;
            align-items: center;
            gap: 5px;
            font-size: 11.5px;
            font-weight: 800;
            padding: 3px 9px;
            border-radius: 999px;
        }
        .qtype-desc {
            font-size: 12.5px;
            color: var(--muted);
            line-height: 1.5;
        }
        .qtype-hint {
            font-size: 12px;
            color: #0B7A6E;
            background: #F3FBF9;
            border: 1px solid #CBEFE6;
            border-radius: 8px;
            padding: 6px 10px;
            margin: 6px 0 4px;
            line-height: 1.45;
        }

        /* Evidence-state guidance box (근거 부족 / 근거 없음) */
        .guide-box {
            border-radius: 12px;
            padding: 12px 14px;
            margin: 12px 0 10px;
            border: 1px solid var(--border);
        }
        .guide-box.partial {
            background: #FFFBF2;
            border-color: #F4D7A3;
        }
        .guide-box.none {
            background: #F7F8FA;
            border-color: #DCE2EC;
        }
        .guide-title {
            font-size: 13px;
            font-weight: 800;
            color: var(--ink);
            margin-bottom: 3px;
        }
        .guide-sub {
            font-size: 12.5px;
            color: var(--muted);
            line-height: 1.5;
        }

        /* Official reference sites grid (근거 부족 문서용) */
        .site-section {
            margin: 10px 0 8px;
        }
        .site-heading {
            font-size: 12px;
            font-weight: 700;
            color: var(--muted);
            margin-bottom: 7px;
        }
        .site-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
            gap: 8px;
        }
        a.site-card {
            display: flex;
            flex-direction: column;
            gap: 3px;
            padding: 10px 12px;
            border-radius: 10px;
            border: 1px solid #E2E8F0;
            background: #FFFFFF;
            text-decoration: none !important;
            color: inherit !important;
            transition: all 0.14s ease;
        }
        a.site-card:hover {
            border-color: var(--accent-border);
            background: var(--accent-soft);
        }
        .site-name {
            font-size: 12.5px;
            font-weight: 700;
            color: var(--accent);
        }
        .site-desc {
            font-size: 11.5px;
            color: var(--muted);
            line-height: 1.4;
        }

        /* Perplexity-style horizontal citations grid */
        .cite-section { margin-top: 14px; margin-bottom: 8px; }
        .cite-heading {
            font-size: 12px;
            font-weight: 700;
            color: var(--muted);
            margin-bottom: 8px;
        }
        .cite-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(210px, 1fr));
            gap: 8px;
        }
        .cite-card {
            display: flex;
            flex-direction: column;
            gap: 4px;
            padding: 10px 12px;
            border-radius: 10px;
            border: 1px solid var(--border);
            background: var(--sidebar);
            text-decoration: none !important;
            color: inherit !important;
            transition: all 0.14s ease;
        }
        a.cite-card:hover {
            border-color: var(--accent-border);
            background: var(--accent-soft);
        }
        .cite-top {
            display: flex;
            align-items: center;
            gap: 6px;
        }
        .cite-num {
            width: 18px;
            height: 18px;
            border-radius: 5px;
            background: #FFFFFF;
            border: 1px solid var(--border-strong);
            color: var(--ink);
            font-size: 10.5px;
            font-weight: 800;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            flex-shrink: 0;
        }
        .cite-title {
            font-size: 12.5px;
            font-weight: 700;
            color: var(--ink);
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .cite-sub {
            font-size: 11.5px;
            color: var(--muted);
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }

        /* Subtle clean expander */
        details[data-testid="stExpander"] {
            border: 1px solid var(--border) !important;
            border-radius: 10px !important;
            background: #FFFFFF !important;
            margin-top: 8px !important;
        }
        details[data-testid="stExpander"] summary {
            font-size: 12.5px !important;
            color: var(--muted) !important;
            font-weight: 600 !important;
        }

        /* Clean bottom chat input (force light contrast even on OS dark mode) */
        div[data-testid="stBottom"],
        div[data-testid="stBottom"] > div {
            background: #FFFFFF !important;
        }
        div[data-testid="stChatInput"] > div {
            background: #F4F6F9 !important;
            border: 1.5px solid #D3DAE5 !important;
            border-radius: 14px !important;
        }
        div[data-testid="stChatInput"] > div:focus-within {
            background: #FFFFFF !important;
            border-color: #1B64DA !important;
            box-shadow: 0 0 0 3px rgba(27, 100, 218, 0.12) !important;
        }
        div[data-testid="stChatInput"] textarea {
            font-size: 14.5px !important;
            color: #191F28 !important;
            caret-color: #1B64DA !important;
            -webkit-text-fill-color: #191F28 !important;
        }
        div[data-testid="stChatInput"] textarea::placeholder {
            color: #8B95A1 !important;
            -webkit-text-fill-color: #8B95A1 !important;
        }
        div[data-testid="stChatInput"] button {
            color: #1B64DA !important;
        }

        /* Ensure markdown, sidebar, segmented control, and expander text stay crisp dark */
        .stApp p, .stApp li, .stApp span, .stApp label,
        section[data-testid="stSidebar"] p,
        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] span {
            color: var(--ink);
        }
        div[data-testid="stSegmentedControl"] button {
            background: #F4F6F9 !important;
            color: #333D4B !important;
            border-color: #E6EAF0 !important;
        }
        div[data-testid="stSegmentedControl"] button[aria-checked="true"],
        div[data-testid="stSegmentedControl"] button[kind="segmented_controlActive"] {
            background: #1B64DA !important;
            color: #FFFFFF !important;
            border-color: #1B64DA !important;
        }
        div[data-testid="stSegmentedControl"] button[aria-checked="true"] p,
        div[data-testid="stSegmentedControl"] button[kind="segmented_controlActive"] p {
            color: #FFFFFF !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar(router: Any | None, init_err: str, selected_part: str) -> tuple[bool, bool]:
    avatar_uri = _img_data_uri(str(AVATAR_IMG_PATH))
    registered = set(router.registered_domains) if router is not None else set()
    history = st.session_state.get("history", [])

    with st.sidebar:
        img_tag = f'<img src="{avatar_uri}" alt="찾아줘 홈즈" />' if avatar_uri else ""
        st.markdown(
            f"""
            <div class="sb-brand">
                {img_tag}
                <div>
                    <div class="sb-brand-name">찾아줘 홈즈</div>
                    <div class="sb-brand-sub">Seoul Housing RAG</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.button("+ 새 대화", type="primary", use_container_width=True):
            st.session_state.history = []
            st.rerun()

        st.markdown('<div class="sb-label">연결된 도메인</div>', unsafe_allow_html=True)
        for key in ("public_housing", "law", "rent_deals", "rent_stats"):
            cfg = PART_CONFIGS[key]
            owner_short = cfg["owner"].split(" ")[0]
            status_mark = "●" if key in registered else "○"
            active_prefix = "✓ " if selected_part == key else ""
            btn_label = f"{status_mark} {active_prefix}{cfg['label']} · {owner_short}"
            if st.button(
                btn_label,
                key=f"sb_domain_{key}",
                type="secondary",
                use_container_width=True,
                help=f"{cfg['full_name']} 탭으로 전환 (다시 클릭하면 전체 자동)",
            ):
                st.session_state["pending_part"] = "auto" if selected_part == key else key
                st.rerun()

        st.markdown('<div class="sb-label">최근 질문</div>', unsafe_allow_html=True)
        user_qs = [m["content"] for m in history if m["role"] == "user"]
        if user_qs:
            recent_unique: list[str] = []
            for q in reversed(user_qs):
                if q not in recent_unique:
                    recent_unique.append(q)
                if len(recent_unique) >= 6:
                    break
            for r_idx, q in enumerate(recent_unique):
                short_q = q if len(q) <= 22 else q[:21] + "…"
                if st.button(
                    f"↺ {short_q}",
                    key=f"sb_hist_{r_idx}",
                    type="secondary",
                    use_container_width=True,
                    help=q,
                ):
                    st.session_state.pending_question = q
                    st.rerun()
        else:
            st.caption("질문을 입력하면 이곳에 대화 목록이 표시됩니다.")

        st.divider()
        conversational_mode = st.toggle(
            "대화형 상담 모드",
            value=True,
            help="검색된 근거를 바탕으로 멀티턴 문맥을 이해하고 친절한 설명과 이어서 물어보기 질문을 제공합니다.",
        )
        filter_by_part = st.toggle(
            "선택 도메인 대화만 보기",
            value=False,
            disabled=(selected_part == "auto"),
        )

        if init_err:
            st.caption("상태: .env 설정 필요 (UI 미리보기)")
        else:
            st.caption(f"상태: {len(registered)}/4 도메인 라우터 연결됨")

    return filter_by_part, conversational_mode


def render_assistant_response(
    resp: dict[str, Any],
    *,
    user_question: str = "",
    stream: bool = False,
    show_followups: bool = False,
    turn_idx: int = 0,
) -> None:
    status = str(resp.get("status") or "error")
    domain = str(resp.get("domain") or "unsupported")
    latency_ms = float(resp.get("latency_ms") or 0.0)
    meta = resp.get("meta") or {}
    evidence_state = str(meta.get("evidence_state") or _classify_evidence_state(user_question, resp))

    st_label, st_fg, st_bg = EVIDENCE_STATUS_STYLE.get(
        evidence_state,
        STATUS_STYLE.get(status, (status, "#6B7684", "#F2F4F7")),
    )
    dm_label = DOMAIN_LABELS.get(domain, domain)
    lat_txt = f"{latency_ms:.0f}ms" if latency_ms < 1000 else f"{latency_ms / 1000:.2f}s"

    badges: list[str] = []
    if meta.get("cache_hit"):
        badges.append('<span class="cache-pill">캐시 적중</span>')
    if meta.get("reused_context"):
        badges.append('<span class="cache-pill">직전 근거 활용</span>')
    if meta.get("conversational_mode"):
        badges.append('<span class="mode-pill">홈즈 상담</span>')
    badges_html = "".join(badges)

    st.markdown(
        f"""
        <div class="meta-bar">
            <span class="st-pill" style="color:{st_fg}; background:{st_bg};"><span class="st-dot" style="background:{st_fg};"></span>{st_label}</span>
            <span class="dm-name">{dm_label}</span>
            <span class="meta-sep">·</span>
            <span class="lat-txt">{lat_txt}</span>
            {badges_html}
        </div>
        """,
        unsafe_allow_html=True,
    )

    formatted_answer = format_inline_citations(str(resp.get("answer") or ""))
    if stream:
        st.write_stream(_stream_answer_chunks(formatted_answer))
    else:
        st.markdown(formatted_answer)

    citations = list(resp.get("citations") or [])
    if citations:
        cards = []
        for idx, c in enumerate(citations, start=1):
            title = html.escape(str(c.get("title") or c.get("chunk_id") or "출처"))
            sub = html.escape(" · ".join(p for p in (str(c.get("source") or ""), str(c.get("location") or "")) if p))
            url = str(c.get("url") or "").strip()
            inner = (
                f'<div class="cite-top"><span class="cite-num">{idx}</span>'
                f'<span class="cite-title">{title}</span></div>'
                f'<div class="cite-sub">{sub or "원문 참조"}</div>'
            )
            if url:
                cards.append(f'<a class="cite-card" href="{html.escape(url)}" target="_blank">{inner}</a>')
            else:
                cards.append(f'<div class="cite-card">{inner}</div>')
        st.markdown(
            f'<div class="cite-section"><div class="cite-heading">참고 출처 ({len(citations)})</div>'
            f'<div class="cite-grid">{"".join(cards)}</div></div>',
            unsafe_allow_html=True,
        )

    # 유형 2 (근거 부족 문서): 안내 배너 + 관련 공식 사이트 링크 카드
    if evidence_state == "partial_missing":
        st.markdown(
            """
            <div class="guide-box partial">
                <div class="guide-title">공고문 원문에 해당 세부 정보가 부족합니다</div>
                <div class="guide-sub">추측해서 안내하지 않고, 최신 현황을 직접 확인할 수 있는 공식 사이트와 대신 물어볼 수 있는 질문을 준비했습니다.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        sites = RELATED_OFFICIAL_SITES.get(domain, RELATED_OFFICIAL_SITES["public_housing"])
        site_cards = "".join(
            f'<a class="site-card" href="{html.escape(url)}" target="_blank" rel="noopener noreferrer">'
            f'<div class="site-name">{html.escape(name)} ↗</div>'
            f'<div class="site-desc">{html.escape(desc)}</div></a>'
            for name, url, desc in sites
        )
        st.markdown(
            f'<div class="site-section"><div class="site-heading">관련 공식 사이트에서 직접 확인하기</div>'
            f'<div class="site-grid">{site_cards}</div></div>',
            unsafe_allow_html=True,
        )
    # 유형 3 (근거 없음): 환각 방지 사실 안내 배너
    elif evidence_state == "no_evidence":
        st.markdown(
            """
            <div class="guide-box none">
                <div class="guide-title">수집된 문서에 근거가 없어 사실대로 모른다고 안내드립니다</div>
                <div class="guide-sub">찾아줘 홈즈는 확인되지 않은 내용을 지어내지 않습니다(환각 방지). 아래에서 현재 서울시 데이터로 정확히 확인 가능한 질문을 선택해 보세요.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    follow_ups = [str(q).strip() for q in (resp.get("follow_ups") or []) if str(q).strip()]
    if show_followups and follow_ups:
        if evidence_state == "partial_missing":
            fu_title = "대신 이런 질문은 바로 확인해 드릴 수 있어요"
        elif evidence_state == "no_evidence":
            fu_title = "홈즈가 정확히 답할 수 있는 다른 질문 추천"
        else:
            fu_title = "이어서 물어보기"
        st.markdown(f'<div class="followup-heading">{fu_title}</div>', unsafe_allow_html=True)
        for f_idx, fq in enumerate(follow_ups[:3]):
            if st.button(
                f"↳  {fq}",
                key=f"followup_{turn_idx}_{f_idx}",
                use_container_width=True,
            ):
                st.session_state.pending_question = fq
                st.rerun()

    with st.expander("검색 근거 및 라우팅 상세"):
        c1, c2 = st.columns(2)
        c1.markdown(f"**라우팅 사유**  \n{resp.get('route_reason') or '-'}")
        c2.markdown(f"**컬렉션**  \n`{resp.get('collection_name') or '-'}`")

        resolved_q = str(resp.get("resolved_query") or "").strip()
        if resolved_q and user_question and resolved_q != user_question.strip():
            st.markdown(f"**맥락 반영 검색 질의**  \n`{resolved_q}`")

        raw_answer = str(resp.get("raw_answer") or "").strip()
        if raw_answer and raw_answer != str(resp.get("answer") or "").strip():
            st.markdown("**원본 RAG 검색 답변**")
            with st.container(border=True):
                st.markdown(raw_answer)

        err_msg = str(resp.get("error_message") or "").strip()
        if err_msg:
            st.markdown("**오류 상세**")
            st.code(err_msg, language=None)

        chunk_ids = list(resp.get("chunk_ids") or [])
        if chunk_ids:
            st.markdown("**검색된 청크 ID**")
            st.code("\n".join(chunk_ids), language=None)

        contexts = list(resp.get("retrieved_contexts") or [])
        for i, ctx in enumerate(contexts[:4], start=1):
            st.markdown(f"**검색 근거 [{i}]**")
            txt = str(ctx).strip()
            with st.container(border=True):
                st.markdown(txt[:1500] + ("\n\n…" if len(txt) > 1500 else ""))


def render_theme_samples(selected_part: str, cfg: dict[str, Any], *, key_prefix: str) -> None:
    """현재 선택한 테마(도메인)의 대표 질문을 렌더링한다. sample_groups가 있으면 유형별 카드 UI로 표시한다."""
    sample_groups = cfg.get("sample_groups")
    if sample_groups:
        for g_idx, grp in enumerate(sample_groups):
            head_cls = "qtype-head first" if g_idx == 0 else "qtype-head"
            badge = html.escape(str(grp["badge"]))
            desc = html.escape(str(grp["desc"]))
            fg = grp["fg"]
            bg = grp["bg"]
            st.markdown(
                f"""
                <div class="{head_cls}">
                    <div class="qtype-row">
                        <span class="qtype-badge" style="color:{fg}; background:{bg};">{badge}</span>
                    </div>
                    <div class="qtype-desc">{desc}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            items = list(grp.get("items") or [])
            cols = st.columns(len(items), gap="small") if items else []
            for i_idx, (label, sample_q) in enumerate(items):
                preview = sample_q if len(sample_q) <= 46 else sample_q[:45] + "…"
                btn_text = f"**{label}**  \n{preview}"
                if cols[i_idx].button(
                    btn_text,
                    key=f"{key_prefix}_{selected_part}_{grp['type']}_{i_idx}",
                    use_container_width=True,
                ):
                    st.session_state.pending_question = sample_q
                    st.rerun()
            hint = str(grp.get("hint") or "").strip()
            if hint:
                st.markdown(f'<div class="qtype-hint">💡 {html.escape(hint)}</div>', unsafe_allow_html=True)
        return

    samples = cfg["samples"]
    for row_start in range(0, len(samples), 2):
        row_cols = st.columns(2, gap="small")
        for offset in range(2):
            idx = row_start + offset
            if idx < len(samples):
                label, sample_q = samples[idx]
                preview = sample_q if len(sample_q) <= 44 else sample_q[:43] + "…"
                btn_text = f"**{label}**  \n{preview}"
                if row_cols[offset].button(
                    btn_text,
                    key=f"{key_prefix}_{selected_part}_{idx}",
                    use_container_width=True,
                ):
                    st.session_state.pending_question = sample_q
                    st.rerun()


def main() -> None:
    st.set_page_config(
        page_title="찾아줘 홈즈 — 서울 주거 탐색 AI",
        page_icon=str(AVATAR_IMG_PATH) if AVATAR_IMG_PATH.exists() else None,
        layout="centered",
        initial_sidebar_state="expanded",
    )
    inject_production_css()
    st.session_state.setdefault("history", [])
    st.session_state.setdefault("selected_part", "auto")
    st.session_state.setdefault("part_tabs", st.session_state["selected_part"])

    # Apply deferred sidebar domain switch BEFORE segmented_control instantiation
    if "pending_part" in st.session_state:
        next_part = st.session_state.pop("pending_part")
        if next_part in PART_CONFIGS:
            st.session_state["selected_part"] = next_part
            st.session_state["part_tabs"] = next_part

    router, init_err = load_team_router()

    # Clean Top Segmented Part Bar
    picked = st.segmented_control(
        "도메인 선택",
        options=PART_ORDER,
        format_func=lambda k: PART_CONFIGS[k]["label"],
        selection_mode="single",
        label_visibility="collapsed",
        key="part_tabs",
    )
    selected_part = picked if picked in PART_CONFIGS else "auto"
    st.session_state["selected_part"] = selected_part

    cfg = PART_CONFIGS[selected_part]
    st.markdown(
        f"""
        <div class="top-meta-bar">
            <span class="top-meta-title">{cfg['full_name']} · {cfg['owner']}</span>
            <span class="top-meta-col">{cfg['collection']}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    filter_by_part, conversational_mode = render_sidebar(router, init_err, selected_part)

    # Chat input (pinned at bottom)
    typed = st.chat_input(cfg["placeholder"])
    pending = st.session_state.pop("pending_question", None)
    question = typed or pending

    has_history = bool(st.session_state.history) or bool(question)

    # Empty Welcome State (disappears cleanly once conversation starts)
    if not has_history:
        hero_uri = _img_data_uri(str(HERO_IMG_PATH))
        mascot_html = (
            f'<div class="mascot-frame"><img src="{hero_uri}" alt="찾아줘 홈즈" /></div>'
            if hero_uri
            else ""
        )
        st.markdown(
            f"""
            <div class="welcome-box">
                {mascot_html}
                <div class="welcome-kicker">{cfg['full_name']} · {cfg['owner']}</div>
                <h1 class="welcome-title">서울 주거 탐색, 무엇이 궁금하신가요?</h1>
                <p class="welcome-desc">{cfg['desc']}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        render_theme_samples(selected_part, cfg, key_prefix="welcome_sample")
    else:
        # Keep per-theme clickable sample questions tucked away cleanly during active chat
        with st.expander(f"{cfg['label']} 추천 질문 바로 해보기", expanded=False):
            render_theme_samples(selected_part, cfg, key_prefix="active_sample")

    # Render existing conversation
    avatar_arg: str | None = str(AVATAR_IMG_PATH) if AVATAR_IMG_PATH.exists() else None
    history = st.session_state.history
    total_turns = len(history) // 2
    idx = 0
    while idx < len(history):
        turn_idx = idx // 2
        u_msg = history[idx]
        a_msg = history[idx + 1] if idx + 1 < len(history) else None
        idx += 2

        if filter_by_part and selected_part != "auto" and a_msg is not None:
            r_dom = a_msg["response"].get("domain")
            r_part = a_msg["response"].get("selected_part")
            if r_dom != selected_part and r_part != selected_part:
                continue

        st.markdown(
            f'<div class="user-bubble-wrap"><div class="user-bubble">{html.escape(u_msg["content"])}</div></div>',
            unsafe_allow_html=True,
        )
        if a_msg is not None:
            is_last_turn = (turn_idx == total_turns - 1) and not question
            with st.chat_message("assistant", avatar=avatar_arg):
                render_assistant_response(
                    a_msg["response"],
                    user_question=str(u_msg.get("content") or ""),
                    stream=False,
                    show_followups=is_last_turn,
                    turn_idx=turn_idx,
                )

    # Run newly submitted question
    if question:
        new_turn_idx = len(st.session_state.history) // 2
        st.markdown(
            f'<div class="user-bubble-wrap"><div class="user-bubble">{html.escape(question)}</div></div>',
            unsafe_allow_html=True,
        )
        with st.chat_message("assistant", avatar=avatar_arg):
            spinner_txt = (
                f"{cfg['label']} 문서를 검색하고 맞춤 상담 답변을 정리하는 중입니다..."
                if conversational_mode
                else f"{cfg['label']} 문서를 검색하고 근거를 확인하는 중입니다..."
            )
            with st.spinner(spinner_txt):
                resp_dict = dispatch_question(
                    question,
                    selected_part,
                    history=st.session_state.history,
                    conversational_mode=conversational_mode,
                )
            render_assistant_response(
                resp_dict,
                user_question=question,
                stream=conversational_mode,
                show_followups=True,
                turn_idx=new_turn_idx,
            )

        st.session_state.history.append({"role": "user", "content": question})
        st.session_state.history.append({"role": "assistant", "response": resp_dict})


if __name__ == "__main__":
    main()
