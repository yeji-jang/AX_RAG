"""찾아줘 홈즈 Next.js UI 전용 FastAPI 백엔드 브리지 서버.

실행 (rag-home-api 루트에서):
    uv run uvicorn rag_home_api.nextjs_ui.api_server:app --reload --port 8000

설계 원칙:
  1. 팀원 도메인 코드(test/{yj,jh,bc,kh}) 및 router/* 는 무수정으로 연결한다.
  2. Streamlit UI(holmes_chat.py)의 모든 핵심 기능(4도메인 라우팅, 멀티턴 맥락 보완,
     직전 근거 재활용, 근거 100% 보존 대화형 상담, 3가지 근거 유형 분류, 공식 사이트/추천 질문)을
     그대로 REST API로 제공한다.
"""

from __future__ import annotations

import time
from functools import lru_cache
from typing import Any, Literal

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

DomainKey = Literal["auto", "public_housing", "law", "rent_deals", "rent_stats"]
EvidenceState = Literal["grounded", "partial_missing", "no_evidence", "error"]

DOMAIN_LABELS: dict[str, str] = {
    "public_housing": "공공임대 공고",
    "law": "임대차 법령",
    "rent_deals": "전월세 조건",
    "rent_stats": "25구 시세",
    "unsupported": "지원 범위 외",
}

DOMAIN_CHARACTER_MAP: dict[str, str] = {
    "auto": "conan",
    "public_housing": "conan",
    "law": "haibara",
    "rent_deals": "ran",
    "rent_stats": "agasa",
    "unsupported": "conan",
}

RELATED_OFFICIAL_SITES: dict[str, list[dict[str, str]]] = {
    "public_housing": [
        {
            "name": "SH 서울주택도시공사 인터넷청약시스템",
            "url": "https://www.i-sh.co.kr/app/index.do",
            "desc": "행복주택·청년안심주택 공고문, 청약 경쟁률·당첨자·예비순번 조회",
        },
        {
            "name": "LH 청약플러스",
            "url": "https://apply.lh.or.kr",
            "desc": "LH 공공임대·매입임대·예비입주자 모집공고 및 청약 결과 확인",
        },
        {
            "name": "마이홈 포털 (국토교통부)",
            "url": "https://www.myhome.go.kr",
            "desc": "공공임대 입주자격 자가진단 및 전국 임대주택 공고 통합 조회",
        },
        {
            "name": "서울주거포털",
            "url": "https://housing.seoul.go.kr",
            "desc": "서울시 청년안심주택·미리내집·주거지원 정책 공식 안내",
        },
    ],
    "law": [
        {
            "name": "국가법령정보센터",
            "url": "https://www.law.go.kr",
            "desc": "주택임대차보호법·주민등록법 현행 조문 원문 확인",
        },
        {
            "name": "찾기쉬운 생활법령정보",
            "url": "https://easylaw.go.kr",
            "desc": "임대차 계약·보증금 보호·분쟁조정 해설 안내",
        },
    ],
    "rent_deals": [
        {
            "name": "국토교통부 실거래가 공개시스템",
            "url": "https://rt.molit.go.kr",
            "desc": "전국 전월세·매매 공식 실거래가 조회",
        },
        {
            "name": "서울부동산정보광장",
            "url": "https://land.seoul.go.kr",
            "desc": "서울시 자치구별 전월세 실거래 현황",
        },
    ],
    "rent_stats": [
        {
            "name": "서울부동산정보광장",
            "url": "https://land.seoul.go.kr",
            "desc": "서울 25개 자치구 전월세 실거래 통계",
        },
        {
            "name": "국토교통부 실거래가 공개시스템",
            "url": "https://rt.molit.go.kr",
            "desc": "전국 전월세·매매 공식 실거래가 조회",
        },
    ],
    "unsupported": [
        {
            "name": "마이홈 포털 (국토교통부)",
            "url": "https://www.myhome.go.kr",
            "desc": "전국 공공임대 공고·입주자격 자가진단 통합 안내",
        },
        {
            "name": "서울주거포털",
            "url": "https://housing.seoul.go.kr",
            "desc": "서울시 공공임대·청년 주거지원 공식 포털",
        },
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

CHARACTER_PERSONA_GUIDES: dict[str, str] = {
    "conan": "명탐정 코난 스타일: 똑똑하고 친근한 추리 탐정 말투(예: '좋아, 공고문 근거를 딱 찾아냈어!' 또는 친절한 해요체 상담 톤 유지). 과장하지 말고 정확하고 믿음직하게 설명하세요.",
    "haibara": "홍장미(하이바라) 스타일: 차분하고 명석한 법령 분석가 톤. 조문과 법적 기한·예외를 논리적이고 깔끔하게 짚어주는 친절한 해요체로 설명하세요.",
    "ran": "유미란 스타일: 따뜻하고 꼼꼼하게 실제 집을 같이 찾아주는 든든한 상담 파트너 톤. 보증금·월세·면적 조건을 다정하고 명쾌한 해요체로 설명하세요.",
    "agasa": "브라운 박사 스타일: 자상하고 유쾌한 데이터·시세 분석 박사 톤. 서울 25구 실거래 중앙값과 시세 비교를 이해하기 쉬운 해요체로 풀어 설명하세요.",
}


class ContextualizedQuery(BaseModel):
    needs_retrieval: bool = Field(
        default=True,
        description="새 DB 검색이 필요하면 True, 직전 답변을 단순히 요약/쉽게 풀어달라는 요청이면 False",
    )
    resolved_query: str = Field(
        description="이전 대화의 공고명·단지·지역·대상을 반영해 단독으로도 의미가 완전한 한국어 질문"
    )
    reason: str = Field(default="", description="맥락 반영 사유 한 줄")


class ConversationalReply(BaseModel):
    evidence_state: Literal["grounded", "partial_missing", "no_evidence"] = Field(
        default="grounded",
        description="근거 상태 분류: grounded / partial_missing / no_evidence",
    )
    friendly_answer: str = Field(
        description="찾아온 근거와 숫자·[1],[2] 인용을 100% 유지하면서 캐릭터 상담사처럼 친절하고 읽기 쉽게 풀어 쓴 답변"
    )
    follow_up_questions: list[str] = Field(
        default_factory=list,
        description="방금 답변과 자연스럽게 이어지는 구체적 후속 질문 3개",
    )


class HistoryMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = ""
    response: dict[str, Any] | None = None


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    selected_part: DomainKey = "auto"
    history: list[HistoryMessage] = Field(default_factory=list)
    conversational_mode: bool = True


@lru_cache(maxsize=1)
def get_team_router() -> tuple[Any | None, str]:
    try:
        from rag_home_api.router import build_default_team_router

        return build_default_team_router(), ""
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


@lru_cache(maxsize=1)
def get_chat_llm() -> Any | None:
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


def classify_evidence_state(question: str, resp: dict[str, Any]) -> EvidenceState:
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
    return "no_evidence" if not contexts else "partial_missing"


def build_followup_fallback(
    domain: str,
    selected_part: str,
    evidence_state: str = "grounded",
) -> list[str]:
    if evidence_state in ("partial_missing", "no_evidence"):
        rec_key = domain if domain in RECOMMENDED_GROUNDED_QUESTIONS else selected_part
        return list(
            RECOMMENDED_GROUNDED_QUESTIONS.get(rec_key, RECOMMENDED_GROUNDED_QUESTIONS["unsupported"])
        )[:3]
    key = domain if domain in FALLBACK_FOLLOWUPS else selected_part
    return list(FALLBACK_FOLLOWUPS.get(key, FALLBACK_FOLLOWUPS["unsupported"]))[:3]


def _format_recent_history(history: list[dict[str, Any]], max_turns: int = 3) -> str:
    if not history:
        return ""
    recent = history[-(max_turns * 2) :]
    lines: list[str] = []
    for msg in recent:
        if msg.get("role") == "user":
            lines.append(f"사용자: {msg.get('content', '')}")
        else:
            resp = msg.get("response") or {}
            ans = str(resp.get("raw_answer") or resp.get("answer") or "")[:260]
            dom = DOMAIN_LABELS.get(str(resp.get("domain") or ""), "")
            lines.append(f"홈즈({dom}): {ans}")
    return "\n".join(lines)


def contextualize_question(question: str, history: list[dict[str, Any]]) -> tuple[str, bool, str]:
    if not history:
        return question, True, ""

    llm = get_chat_llm()
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


def conversationalize_response(
    user_question: str,
    resolved_query: str,
    resp: dict[str, Any],
    history: list[dict[str, Any]],
) -> dict[str, Any]:
    status = str(resp.get("status") or "error")
    domain = str(resp.get("domain") or "unsupported")
    selected_part = str(resp.get("selected_part") or "auto")
    raw_answer = str(resp.get("answer") or "").strip()
    resp["raw_answer"] = raw_answer

    base_state = classify_evidence_state(user_question, resp)
    character_id = DOMAIN_CHARACTER_MAP.get(domain, DOMAIN_CHARACTER_MAP.get(selected_part, "conan"))
    resp["character_id"] = character_id

    meta = dict(resp.get("meta") or {})
    meta["evidence_state"] = base_state
    resp["meta"] = meta

    if status == "error":
        resp["follow_ups"] = build_followup_fallback(domain, selected_part, "error")
        resp["official_sites"] = []
        return resp

    llm = get_chat_llm()
    if llm is None:
        resp["follow_ups"] = build_followup_fallback(domain, selected_part, base_state)
        resp["official_sites"] = (
            RELATED_OFFICIAL_SITES.get(domain, RELATED_OFFICIAL_SITES["public_housing"])
            if base_state == "partial_missing"
            else []
        )
        return resp

    contexts = list(resp.get("retrieved_contexts") or [])
    ctx_snippets = [f"[{i}] {str(c).strip()[:700]}" for i, c in enumerate(contexts[:4], start=1)]
    ctx_block = "\n\n".join(ctx_snippets) if ctx_snippets else "(검색된 근거 없음)"
    hist_text = _format_recent_history(history, max_turns=2)
    persona = CHARACTER_PERSONA_GUIDES.get(character_id, CHARACTER_PERSONA_GUIDES["conan"])

    system_prompt = (
        f"당신은 서울 상경 20~30대 청년과 1:1로 대화하는 '구해줘 홈즈' 팀의 캐릭터입니다.\n"
        f"[캐릭터 톤]: {persona}\n"
        "AI 보고서처럼 딱딱하거나 기계적인 문체('~에 대해 안내해 드리겠습니다', '도움이 되셨길 바랍니다', 불필요한 '### 1. 개요')를 절대 쓰지 말고, "
        "실제 사람이 옆에서 공고문을 짚어 주듯 자연스럽고 담백한 한국어로 답하세요.\n\n"
        "절대 규칙:\n"
        "1. [원본 검색 답변]과 [검색 근거]에 없는 숫자·금액·기간·조건·공고명·경쟁률은 절대 추가하거나 추측하지 마세요.\n"
        "2. 원문의 숫자 단위(천원/원/만원/㎡/%)와 근거 인용 번호([1], [2] 등)는 정확히 유지하세요.\n"
        "3. 자연스러운 가독성 규칙:\n"
        "   - 첫 줄은 핵심 답을 담은 간결한 한 문장(`> ` 인용구)으로 바로 시작하세요.\n"
        "   - 금액·자격·일정 항목은 군더더기 없는 짧은 불릿(`- 보증금: **43,860천원** [1]`)으로 정리하세요.\n"
        "   - `###` 소제목은 항목이 2개 이상 묶음으로 나뉠 때만 최소한으로 쓰세요.\n"
        "   - 마지막엔 실제 대화하듯 자연스러운 한 줄 코멘트만 덧붙이세요.\n"
        "4. [판정된 근거 유형]이 'grounded'일 때:\n"
        "   - 사용자가 본인 조건(예: 월소득 350만원, 1인 가구 등)으로 신청 가능 여부를 물었다면, [검색 근거]에 나온 공식 기준과만 정확히 비교하고 "
        "     함께 체크해야 할 나머지 요건(무주택·총자산·자동차 등 근거에 있는 항목)도 간단히 짚어 주세요.\n"
        "   - follow_up_questions에는 방금 공고에서 자연스럽게 이어서 물어볼 실용적인 후속 질문 3개를 작성하세요.\n"
        "5. [판정된 근거 유형]이 'partial_missing'일 때 (근거 부족 문서):\n"
        "   - 공고문 원문에 해당 세부 정보(경쟁률·커트라인·당첨 결과 등)가 없다는 점을 솔직하게 말하고, 아래 공식 사이트에서 확인하시라고 담백하게 안내하세요.\n"
        "6. [판정된 근거 유형]이 'no_evidence'일 때 (근거 없음 · 환각 방지):\n"
        "   - 수집된 서울시 2025~2026년 주거·공공임대 데이터에 해당 정보가 없어 사실대로 알 수 없다고 분명히 말하세요.\n"
        "   - 추측이나 일반 상식 기반 수치는 절대 말하지 말고, 대신 확인 가능한 서울시 공고 주제를 짧게 제안하세요."
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

        final_state: EvidenceState = base_state
        if status == "insufficient_evidence" and base_state == "partial_missing":
            if out.evidence_state == "no_evidence" and not any(
                tok in user_question for tok in _PARTIAL_MISSING_PATTERNS
            ):
                final_state = "no_evidence"

        meta["evidence_state"] = final_state
        meta["conversational_mode"] = True
        resp["meta"] = meta

        if final_state in ("partial_missing", "no_evidence"):
            resp["follow_ups"] = build_followup_fallback(domain, selected_part, final_state)
        else:
            follow_ups = [q.strip() for q in out.follow_up_questions if q and q.strip()][:3]
            resp["follow_ups"] = follow_ups or build_followup_fallback(domain, selected_part, final_state)

        resp["official_sites"] = (
            RELATED_OFFICIAL_SITES.get(domain, RELATED_OFFICIAL_SITES["public_housing"])
            if final_state == "partial_missing"
            else []
        )
    except Exception:
        resp["follow_ups"] = build_followup_fallback(domain, selected_part, base_state)
        resp["official_sites"] = (
            RELATED_OFFICIAL_SITES.get(domain, RELATED_OFFICIAL_SITES["public_housing"])
            if base_state == "partial_missing"
            else []
        )

    return resp


def dispatch_question(
    question: str,
    selected_part: DomainKey,
    history: list[dict[str, Any]] | None = None,
    conversational_mode: bool = True,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    hist = list(history or [])
    router, init_err = get_team_router()

    if router is None:
        return {
            "status": "error",
            "domain": selected_part if selected_part != "auto" else "unsupported",
            "selected_part": selected_part,
            "character_id": DOMAIN_CHARACTER_MAP.get(selected_part, "conan"),
            "resolved_query": question,
            "answer": "환경변수(`.env`의 `OPENAI_API_KEY`, `DATABASE_URL`)가 설정되지 않아 검색 엔진을 초기화할 수 없습니다.",
            "raw_answer": "",
            "citations": [],
            "retrieved_contexts": [],
            "chunk_ids": [],
            "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
            "collection_name": "",
            "route_reason": "라우터 초기화 실패",
            "error_message": init_err,
            "meta": {"evidence_state": "error"},
            "follow_ups": build_followup_fallback("unsupported", selected_part, "error"),
            "official_sites": [],
        }

    resolved_query = question
    needs_retrieval = True
    context_reason = ""
    if conversational_mode and hist:
        resolved_query, needs_retrieval, context_reason = contextualize_question(question, hist)

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
                "answer": f"확인할 수 없음. '{DOMAIN_LABELS.get(selected_part, selected_part)}' 핸들러가 등록되어 있지 않습니다.",
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
                    data["route_reason"] = f"사용자 도메인 선택 → {DOMAIN_LABELS.get(selected_part, selected_part)}"
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

    if conversational_mode:
        data = conversationalize_response(question, resolved_query, data, hist)
    else:
        ev_state = classify_evidence_state(question, data)
        meta = dict(data.get("meta") or {})
        meta["evidence_state"] = ev_state
        data["meta"] = meta
        data["character_id"] = DOMAIN_CHARACTER_MAP.get(
            str(data.get("domain") or ""),
            DOMAIN_CHARACTER_MAP.get(selected_part, "conan"),
        )
        data["raw_answer"] = str(data.get("answer") or "")
        data["follow_ups"] = build_followup_fallback(
            str(data.get("domain") or "unsupported"), selected_part, ev_state
        )
        data["official_sites"] = (
            RELATED_OFFICIAL_SITES.get(
                str(data.get("domain") or "public_housing"),
                RELATED_OFFICIAL_SITES["public_housing"],
            )
            if ev_state == "partial_missing"
            else []
        )

    data["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    return data


app = FastAPI(title="구해줘 홈즈 RAG API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3001",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict[str, Any]:
    router, init_err = get_team_router()
    registered = sorted(list(router.registered_domains)) if router is not None else []
    return {
        "ok": router is not None,
        "registered_domains": registered,
        "total_domains": 4,
        "error": init_err,
    }


@app.post("/api/chat")
def chat(req: ChatRequest) -> dict[str, Any]:
    hist_dicts = [m.model_dump() for m in req.history]
    return dispatch_question(
        question=req.question.strip(),
        selected_part=req.selected_part,
        history=hist_dicts,
        conversational_mode=req.conversational_mode,
    )
