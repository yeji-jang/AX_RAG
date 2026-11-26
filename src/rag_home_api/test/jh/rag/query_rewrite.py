"""선택형 검색 질의 리라이트. 생성에는 원질문을 사용하며 기본 C는 변경하지 않는다."""
import re
import time

from langchain_core.prompts import PromptTemplate

# 기존 실험의 조건 보존 규칙. 검색문에 원질문 앵커를 남긴다.
def rewrite_query(question, chain):
    started = time.perf_counter()
    fallback = None
    raw = ""
    try:
        raw = chain.invoke({"question": question}).content.strip()
        protected = re.findall(r"\d+(?:[.,]\d+)*|[A-Za-z]+\d*", question)
        if not raw or any(v not in raw for v in protected):
            raise ValueError("MissingProtectedLiteral")
        if set(re.findall(r"\d+(?:[.,]\d+)*", raw)) - set(re.findall(r"\d+(?:[.,]\d+)*", question)):
            raise ValueError("AddedNumber")
        if re.search(r"제\s*\d+조", raw) and not re.search(r"제\s*\d+조", question):
            raise ValueError("AddedArticle")
        # 원질문의 조건/당사자/부정·예외/고유명사를 변환문에 통째로 유지한다.
        query = question if raw == question else question + "\n검색 핵심: " + raw
    except Exception as exc:
        query, fallback = question, type(exc).__name__
    return {"query": query, "raw_rewrite": raw, "fallback": fallback,
            "transform_sec": round(time.perf_counter() - started, 4), "original_anchor_preserved": question in query}




def build_query_chain(model, template):
    # 튜터 12_rag/chapter05/04_04.ipynb와 같은 연결 방식.
    return PromptTemplate.from_template(template) | model
