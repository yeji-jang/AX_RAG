"""찾아줘 홈즈 — 서울 주거 탐색 4도메인 통합 RAG 서비스 (Streamlit).

실행 (저장소 루트):
    .venv/bin/streamlit run src/rag_home_api/ui/holmes_chat.py
    # 또는
    uv run --no-sync --with streamlit streamlit run src/rag_home_api/ui/holmes_chat.py
"""

from __future__ import annotations

import base64
import html
import time
from pathlib import Path
from typing import Any

import streamlit as st

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
            ("임대조건 · 대치르엘 35㎡ 청년 보증금/월세", "2026년 1차 서울주택도시개발공사 행복주택 입주자 모집공고에서 강남구 대치르엘 35㎡ 청년(소득 있음)의 임대보증금과 월임대료는 얼마인가요?"),
            ("자산기준 · 청년안심주택 2순위 총자산 기준", "2026년 1차 청년안심주택(공공임대) 모집공고에서 2순위 신청자의 총자산 기준은 얼마이며, 총자산가액은 어떻게 산출하나요?"),
            ("자격판단 · 미리내집 배우자 실업급여 맞벌이", "2025년 제2-1차 미리내집 신혼신생아 매입임대주택2 공고에서 배우자가 실업급여를 받고 있으면 맞벌이로 인정되나요?"),
            ("근거 부족 확인 · 행복주택 이사비 현금 지원", "2026년 서울시 행복주택 입주자에게 지급되는 이사비 현금 지원 금액은 얼마인가요?"),
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
    "unsupported": ("범위 외", "#6B7684", "#F2F4F7"),
    "error": ("오류 격리", "#C92A2A", "#FFF2F2"),
}


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


def dispatch_question(question: str, selected_part: str) -> dict[str, Any]:
    t0 = time.perf_counter()
    router, init_err = load_team_router()

    if router is None:
        return {
            "status": "error",
            "domain": selected_part if selected_part != "auto" else "unsupported",
            "selected_part": selected_part,
            "answer": "환경변수(`.env`의 `OPENAI_API_KEY`, `DATABASE_URL`)가 설정되지 않아 검색 엔진을 초기화할 수 없습니다.",
            "citations": [],
            "retrieved_contexts": [],
            "chunk_ids": [],
            "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
            "collection_name": "",
            "route_reason": "라우터 초기화 실패",
            "error_message": init_err,
            "meta": {},
        }

    if selected_part == "auto":
        resp = router.invoke(question)
        data = resp.model_dump()
        data["selected_part"] = "auto"
        return data

    handler = router._handlers.get(selected_part)
    if handler is None:
        return {
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

    try:
        resp = handler(question)
        data = resp.model_dump()
        if not data.get("route_reason"):
            data["route_reason"] = f"사용자 도메인 선택 → {PART_CONFIGS[selected_part]['full_name']}"
        data["selected_part"] = selected_part
        return data
    except Exception as exc:
        return {
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

        section[data-testid="stSidebar"] .stButton > button {
            background: #FFFFFF !important;
            border: 1px solid var(--border-strong) !important;
            color: var(--ink) !important;
            border-radius: 10px !important;
            font-size: 13px !important;
            font-weight: 700 !important;
            height: 38px !important;
            box-shadow: 0 1px 2px rgba(25, 31, 40, 0.04) !important;
        }
        section[data-testid="stSidebar"] .stButton > button:hover {
            border-color: var(--ink) !important;
            background: #FFFFFF !important;
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


def render_sidebar(router: Any | None, init_err: str, selected_part: str) -> bool:
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

        if st.button("+ 새 대화", use_container_width=True):
            st.session_state.history = []
            st.rerun()

        st.markdown('<div class="sb-label">연결된 도메인</div>', unsafe_allow_html=True)
        rows = []
        for key in ("public_housing", "law", "rent_deals", "rent_stats"):
            cfg = PART_CONFIGS[key]
            dot_cls = "sb-dot" if key in registered else "sb-dot off"
            row_cls = "sb-domain-row active" if selected_part == key else "sb-domain-row"
            owner_short = cfg["owner"].split(" ")[0]
            rows.append(
                f'<div class="{row_cls}"><span><span class="{dot_cls}"></span>{cfg["label"]}</span>'
                f'<span class="sb-owner">{owner_short}</span></div>'
            )
        st.markdown("".join(rows), unsafe_allow_html=True)

        st.markdown('<div class="sb-label">최근 질문</div>', unsafe_allow_html=True)
        user_qs = [m["content"] for m in history if m["role"] == "user"]
        if user_qs:
            q_html = "".join(
                f'<div class="sb-history-item" title="{html.escape(q)}">{html.escape(q)}</div>'
                for q in reversed(user_qs[-8:])
            )
            st.markdown(q_html, unsafe_allow_html=True)
        else:
            st.caption("질문을 입력하면 이곳에 대화 목록이 표시됩니다.")

        st.divider()
        filter_by_part = st.toggle(
            "선택 도메인 대화만 보기",
            value=False,
            disabled=(selected_part == "auto"),
        )

        if init_err:
            st.caption("상태: .env 설정 필요 (UI 미리보기)")
        else:
            st.caption(f"상태: {len(registered)}/4 도메인 라우터 연결됨")

    return filter_by_part


def render_assistant_response(resp: dict[str, Any]) -> None:
    status = str(resp.get("status") or "error")
    domain = str(resp.get("domain") or "unsupported")
    latency_ms = float(resp.get("latency_ms") or 0.0)
    meta = resp.get("meta") or {}

    st_label, st_fg, st_bg = STATUS_STYLE.get(status, (status, "#6B7684", "#F2F4F7"))
    dm_label = DOMAIN_LABELS.get(domain, domain)
    lat_txt = f"{latency_ms:.0f}ms" if latency_ms < 1000 else f"{latency_ms / 1000:.2f}s"
    cache_html = '<span class="cache-pill">캐시 적중</span>' if meta.get("cache_hit") else ""

    st.markdown(
        f"""
        <div class="meta-bar">
            <span class="st-pill" style="color:{st_fg}; background:{st_bg};"><span class="st-dot" style="background:{st_fg};"></span>{st_label}</span>
            <span class="dm-name">{dm_label}</span>
            <span class="meta-sep">·</span>
            <span class="lat-txt">{lat_txt}</span>
            {cache_html}
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(str(resp.get("answer") or ""))

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

    with st.expander("검색 근거 및 라우팅 상세"):
        c1, c2 = st.columns(2)
        c1.markdown(f"**라우팅 사유**  \n{resp.get('route_reason') or '-'}")
        c2.markdown(f"**컬렉션**  \n`{resp.get('collection_name') or '-'}`")

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
            txt = str(ctx)
            st.text(txt[:1200] + (" …" if len(txt) > 1200 else ""))


def render_theme_samples(selected_part: str, cfg: dict[str, Any], *, key_prefix: str) -> None:
    """현재 선택한 테마(도메인)의 대표 질문 4개를 2x2 카드로 렌더링한다."""
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

    router, init_err = load_team_router()

    # Clean Top Segmented Part Bar
    picked = st.segmented_control(
        "도메인 선택",
        options=PART_ORDER,
        format_func=lambda k: PART_CONFIGS[k]["label"],
        default=st.session_state["selected_part"],
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

    filter_by_part = render_sidebar(router, init_err, selected_part)

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
        # Keep per-theme clickable sample questions accessible during active chat for smooth demos
        with st.expander(f"{cfg['label']} 추천 질문 바로 해보기", expanded=True):
            render_theme_samples(selected_part, cfg, key_prefix="active_sample")

    # Render existing conversation
    avatar_arg: str | None = str(AVATAR_IMG_PATH) if AVATAR_IMG_PATH.exists() else None
    history = st.session_state.history
    idx = 0
    while idx < len(history):
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
            with st.chat_message("assistant", avatar=avatar_arg):
                render_assistant_response(a_msg["response"])

    # Run newly submitted question
    if question:
        st.markdown(
            f'<div class="user-bubble-wrap"><div class="user-bubble">{html.escape(question)}</div></div>',
            unsafe_allow_html=True,
        )
        with st.chat_message("assistant", avatar=avatar_arg):
            with st.spinner(f"{cfg['label']} 문서를 검색하고 근거를 확인하는 중입니다..."):
                resp_dict = dispatch_question(question, selected_part)
            render_assistant_response(resp_dict)

        st.session_state.history.append({"role": "user", "content": question})
        st.session_state.history.append({"role": "assistant", "response": resp_dict})


if __name__ == "__main__":
    main()
