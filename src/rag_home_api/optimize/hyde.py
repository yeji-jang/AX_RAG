"""[STEP 5-3: HyDE — Hypothetical Document Embeddings] 추상·변형 질문 Vector 회수 보강 + 절대 출처 격리.

작성자: 최기혁 (kh)
절대 규칙 (출처 격리):
  1. HyDE 가상 문서는 오직 Vector 검색(semantic_query 임베딩)에만 사용한다.
  2. BM25(keyword_query)에는 원질문(또는 Query Rewrite keyword_query)을 그대로 유지한다.
  3. 가상 문서는 Vector 검색 직후 즉시 폐기하며, Re-ranker(C) · Generator · 인용(Citation)에 1글자도 유입시키지 않는다.
"""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate

from rag_home_api.shared.core.logger import get_logger
from rag_home_api.shared.core.vector_store import get_llm

logger = get_logger(__name__)

HYDE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "당신은 서울 주거 전월세 실거래 요약 문서를 작성하는 검색 보조기입니다.\n"
            "사용자 질문에 답이 들어있을 법한 **가상의 실거래 요약 문서 형태(3~4줄)**를 작성하세요.\n"
            "[규칙]\n"
            "- 질문에 언급된 지역(자치구/법정동), 주택유형(아파트/오피스텔/연립다세대/단독다가구), 계약유형(전세/월세), 면적대(33㎡이하/33~60㎡/60㎡초과)를 표준 용어로 자연스럽게 포함하세요.\n"
            "- 이 문서는 오직 벡터 검색 임베딩에만 쓰이고 사용자에게 절대 노출되지 않습니다.\n"
            "- CSV 파일명이나 가짜 출처 URL은 쓰지 마세요.",
        ),
        ("human", "사용자 질문: {question}"),
    ]
)


def generate_hyde_document(question: str) -> tuple[str, bool]:
    """Vector 검색 임베딩 전용 가상 문서를 생성한다. 오류 시 원질문으로 폴백한다."""
    try:
        llm = get_llm()
        chain = HYDE_PROMPT | llm
        text = str(chain.invoke({"question": question}).content).strip()
        if not text:
            return question, True
        return text, False
    except Exception as exc:
        logger.warning("HyDE 생성 실패 -> 원질문 폴백: %s", exc)
        return question, True


def verify_hyde_isolation(
    *,
    response_text: str,
    retrieved_contexts: list[str],
    hypothetical_doc: str | None,
) -> dict[str, bool]:
    """가상 문서가 최종 컨텍스트나 응답에 섞이지 않았는지(출처 격리) 검증한다."""
    if not hypothetical_doc:
        return {"hyde_used": False, "context_isolated": True, "response_isolated": True}

    marker = hypothetical_doc.strip()[:80]
    in_ctx = any(marker and marker in c for c in retrieved_contexts)
    in_resp = bool(marker and marker in response_text)
    return {
        "hyde_used": True,
        "context_isolated": not in_ctx,
        "response_isolated": not in_resp,
    }
