"""서비스 입구용 의도 분류와 법령 응답 어댑터. 개인 DEV 검색과 분리함."""

import re
from urllib.parse import urlsplit

from langchain_core.runnables import RunnableBranch, RunnableLambda

HOUSING = r"임대차|임대인|임차인|집주인|세입자|월세|전세|보증금|차임|임차|주택|원룸|빌린집|사는집|전입신고|확정일자"
LEGAL_INTENT = r"법적|법정|법률|법령|법상|법으로|법대로|조문|조항|제\d+조|권리|의무|위법|불법|금지|청구|임차권등기|월세상한|차임상한|임대료상한"
PROCEDURE_INTENT = r"(?:전입신고|확정일자|임대차신고).*(?:언제|기한|해야|조건|필요|가능|할수|받을|해도|돼|늦)"
ACTION = r"올리|올려|올릴|인상|증액|해지|해제|갱신|취소|수리|고쳐|돌려|반환|상환|배상|부담|감액|깎|쫓아|내보내|나가야|퇴거|연체|못내|들어와|출입"
PERMISSION_INTENT = r"해야|줘야|내야|누가|가능|수있|하나요|되나|되나요|해도|돼|마음대로|안돌려|안해"
ACTION_INTENT = PERMISSION_INTENT + r"|상한|한도|얼마까지|몇퍼센트|조건|기준|기한|언제"
NON_LAW = r"단지|모집|공고|행복주택|청년안심|입주자격|입주신청|인프라|마트|병원|학교|지하철|역세권|정류장|평면|옵션|시세|실거래|평균|매물|주소|위치|거리|도보|보증금얼마|월세얼마|관리비얼마"


def classify_question(question: str) -> str:
    """확인 가능한 임대차 법적 의도는 law, 명확한 정보 탐색은 non_law로 분류함."""
    if not isinstance(question, str) or not question.strip():
        return "unsupported"
    text = re.sub(r"\s+", "", question)
    housing = re.search(HOUSING, text)
    data_only = re.search(r"(?:법령|법률|조문|법적설명)(?:은|는|을|를|이|가)?(?:말고|빼고|필요없)", text)
    action_intent = re.search(ACTION, text) and re.search(ACTION_INTENT, text)
    if data_only and re.search(NON_LAW, text) and not action_intent:
        return "non_law"
    if housing and (re.search(LEGAL_INTENT, text) or re.search(PROCEDURE_INTENT, text)):
        return "law"
    if re.search(r"시세|실거래|평균|추세", text) and not re.search(PERMISSION_INTENT, text):
        return "non_law"
    if housing and action_intent:
        return "law"
    return "non_law" if re.search(NON_LAW, text) else "unsupported"


def unsupported_response(question):
    return {"status": "unsupported", "answer": "현재 연결된 경로에서 지원하는 질문인지 확인할 수 없습니다.",
            "citations": []}


def build_question_router(law_handler, non_law_handler=None):
    """비법령 핸들러가 제공되면 입력과 반환값을 그대로 전달함."""
    fallback = RunnableLambda(unsupported_response)
    return RunnableBranch(
        (lambda question: classify_question(question) == "law", law_handler),
        (lambda question: classify_question(question) == "non_law", non_law_handler or fallback),
        fallback,
    )


def is_statute(doc):
    meta = doc.metadata
    if (not doc.page_content.strip() or meta.get("category") != "일반 전월세 법령"
            or not all(isinstance(meta.get(key), str) and meta[key].strip()
                       for key in ("chunk_id", "doc_id", "row_id", "title", "article", "source_url"))):
        return False
    try:
        source = urlsplit(meta["source_url"])
        return meta["doc_id"].startswith("LAW_") and source.scheme == "https" and source.hostname == "www.law.go.kr"
    except ValueError:
        return False


def law_response(documents, generated):
    """검색된 법령 청크의 실제 메타데이터로 인용을 구성함. 의미 채점은 별도임."""
    insufficient = {"status": "insufficient_evidence",
                    "answer": "수집된 법령 원문에서 답변 근거를 확인할 수 없습니다.", "citations": []}
    if not documents:
        return insufficient
    if not isinstance(generated, dict):
        return {"status": "error", "answer": "법령 응답 형식을 확인할 수 없습니다.", "citations": []}
    if generated.get("status") == "insufficient_evidence":
        return insufficient
    ids = generated.get("cited_chunk_ids")
    if (generated.get("status") != "answered" or not isinstance(generated.get("answer"), str)
            or not generated["answer"].strip() or not isinstance(ids, list) or not ids
            or not all(isinstance(cid, str) for cid in ids)):
        return {"status": "error", "answer": "법령 응답의 근거를 확인할 수 없습니다.", "citations": []}
    by_id = {doc.metadata.get("chunk_id"): doc for doc in documents}
    citations = []
    for cid in dict.fromkeys(ids):
        doc = by_id.get(cid)
        if doc is None:
            return {"status": "error", "answer": "검색 결과에 없는 조문이 인용되었습니다.", "citations": []}
        meta = doc.metadata
        if not is_statute(doc):
            return {"status": "error", "answer": "법령 원문 출처를 확인할 수 없습니다.", "citations": []}
        citations.append({"chunk_id": cid, "row_id": meta["row_id"], "law_name": meta["title"],
                          "article": meta["article"], "source_url": meta["source_url"], "quote": doc.page_content})
    return {"status": "answered", "answer": generated["answer"], "citations": citations}


def build_law_handler(retriever, generator):
    """Runnable 검색기와 구조화 응답 생성기를 주입함. 이 모듈은 외부 모델을 만들지 않음."""
    def answer(question):
        try:
            documents = retriever.invoke(question)
            if not documents:
                return law_response([], None)
            if any(not is_statute(doc) for doc in documents):
                return {"status": "error", "answer": "법령 외 문서가 검색되었습니다.", "citations": []}
            generated = generator.invoke({"question": question, "documents": documents})
            return law_response(documents, generated)
        except Exception:
            return {"status": "error", "answer": "법령 검색 또는 답변 처리 중 오류가 발생했습니다.", "citations": []}
    return RunnableLambda(answer)
