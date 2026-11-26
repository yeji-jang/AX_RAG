"""
# STEP 5 최적화 : 쿼리 리라이트 (Query Rewriter)

(순서)
    1. 리라이트 시스템 프롬프트 설정
    2. 숫자·영문코드 보호 안전성 검사 (check_literal_safety)
    3. 검색용 질의 변환 및 오류·왜곡 시 원질문 자동 폴백 (rewrite_query)
    4. 단독 실행 테스트
"""

import re
import time

from langchain_core.messages import HumanMessage, SystemMessage

from rag_home_api.test.yj.core.logger import get_logger
from rag_home_api.test.yj.core.schemas.query_rewrite_output import QueryRewriteOutput
from rag_home_api.test.yj.core.vector_store import get_llm

logger = get_logger(__name__)


#===================================================================
# 1. 리라이트 시스템 프롬프트 설정
#===================================================================
REWRITE_SYSTEM_PROMPT = """
당신은 서울시 공공임대주택 모집공고 검색 정확도를 높이는 질의 변환 전문가입니다.

사용자의 [원질문]을 공고문 원문에서 잘 검색되도록 행정 용어를 보강한 짧은 검색용 질의로 다듬으세요.

규칙 :
1. 원질문에 있는 공고명·단지명·지역명·숫자·면적·주택형(예: 45㎡, 17C, 2순위, 2026년)은 절대 빠뜨리거나 바꾸지 마세요.
2. 원질문에 없는 새로운 숫자·금액·날짜·조문 번호를 절대 지어내지 마세요.
3. 일상 표현이 있으면 공고문 행정 용어를 함께 보강하세요.
   - 예: "집 없는 사람" -> "무주택세대구성원"
   - 예: "돈 벌이 기준" -> "월평균소득 기준"
   - 예: "당첨 발표 언제" -> "당첨자 발표일"
4. 답변을 생성하지 말고, 검색용 질의 한 문장만 반환하세요.
""".strip()

# 숫자(소수점·쉼표 포함)와 주택형 코드(예: 17C, 36A) 추출 패턴
LITERAL_RE = re.compile(r"\d+(?:[.,]\d+)*|[A-Za-z]+\d+|\d+[A-Za-z]+")
NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)*")


#===================================================================
# 2. 숫자·영문코드 보호 안전성 검사
#===================================================================
def check_literal_safety(original_query : str, rewritten_query : str) -> str | None :
    """
    변환 질의가 원질문의 숫자·코드를 빠뜨리거나 없는 숫자를 지어냈는지 검사
    - 안전하면 None 반환
    - 문제가 있으면 폴백 사유 문자열 반환
    """
    if not rewritten_query or not rewritten_query.strip() :
        return "EmptyRewrite"

    # 1) 원질문에 있던 숫자·영문코드가 하나라도 빠졌는지 확인
    protected = LITERAL_RE.findall(original_query)
    for token in protected :
        if token not in rewritten_query :
            return f"MissingProtectedLiteral({token})"

    # 2) 원질문에 없던 새로운 숫자가 임의로 추가됐는지 확인
    orig_nums = set(NUMBER_RE.findall(original_query))
    new_nums = set(NUMBER_RE.findall(rewritten_query))
    added_nums = new_nums - orig_nums
    if added_nums :
        return f"AddedNumber({sorted(added_nums)})"

    return None


#===================================================================
# 3. 검색용 질의 변환 + 안전 폴백 함수
#===================================================================
def rewrite_query(question : str) -> dict :
    """
    원질문을 검색용 질의로 변환합니다.
    - 검색·리랭킹에는 search_query(원질문 앵커 + 변환 핵심어)를 사용
    - 답변 생성(generator)에는 반드시 original_query를 사용
    - API 오류나 숫자 왜곡이 감지되면 자동으로 original_query로 폴백
    """
    start_time = time.perf_counter()
    raw_rewrite = ""
    reason = ""
    fallback : str | None = None

    try :
        llm = get_llm().with_structured_output(QueryRewriteOutput)
        output : QueryRewriteOutput = llm.invoke([
            SystemMessage(content=REWRITE_SYSTEM_PROMPT),
            HumanMessage(content=f"[원질문]\n{question}"),
        ])
        raw_rewrite = output.rewritten_query.strip()
        reason = output.reason.strip()

        # 1) 숫자·영문코드 누락 또는 임의 추가 여부 검사
        safety_issue = check_literal_safety(question, raw_rewrite)
        if safety_issue is not None :
            fallback = safety_issue
            search_query = question
            logger.warning(f"쿼리 리라이트 폴백({safety_issue}) -> 원질문 유지")
        elif raw_rewrite == question :
            search_query = question
        else :
            # 2) 원질문 앵커를 유지하면서 변환 질의를 함께 붙여 검색 범위 보강
            search_query = f"{question}\n검색 핵심어: {raw_rewrite}"

    except Exception as err :
        fallback = type(err).__name__
        search_query = question
        logger.warning(f"쿼리 리라이트 오류({fallback}) -> 원질문 유지 : {err}")

    elapsed = round(time.perf_counter() - start_time, 2)
    return {
        "original_query": question,
        "rewritten_query": raw_rewrite if fallback is None else "",
        "search_query": search_query,
        "reason": reason,
        "fallback": fallback,
        "elapsed_sec": elapsed,
    }


#===================================================================
# 4. 단독 실행 테스트
#===================================================================
if __name__ == "__main__" :
    sample_questions = [
        "2026년 영구임대주택 예비입주자모집 공고에서 배우자와 주민등록상 주소가 따로 되어 있는 경우, 등본은 어떻게 내야 하나요?",
        "2026년 재개발임대주택 일반모집 공고에서 부양가족으로 인정되는 무주택세대구성원은 누구인가요?",
    ]

    for q in sample_questions :
        res = rewrite_query(q)
        print(f"\n[원질문] {res['original_query']}")
        print(f"[변환됨] {res['rewritten_query']} (fallback={res['fallback']}, {res['elapsed_sec']}s)")
        print(f"[사유]   {res['reason']}")
