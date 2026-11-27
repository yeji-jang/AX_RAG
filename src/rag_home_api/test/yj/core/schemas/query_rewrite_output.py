"""
D 구성(쿼리 리라이트)의 LLM 구조화 출력 스키마
"""

from pydantic import BaseModel, Field


class QueryRewriteOutput(BaseModel) :
    rewritten_query : str = Field(
        description="원질문의 공고명·단지·숫자·조건을 유지하면서 공고문 행정 용어를 보강한 검색용 질의"
    )
    reason : str = Field(
        description="어떤 표현을 어떤 공고문 용어로 보강했는지 한 줄 설명"
    )
