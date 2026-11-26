"""
RAG 답변 생성기에서 사용하는 LLM 구조화 출력 스키마
"""

from pydantic import BaseModel, Field

#===================================================================
# Generator 답변 출력 스키마
#===================================================================
class CitedAnswerOutput(BaseModel) :
    answer : str = Field(
        description="참고 문맥만을 근거로 작성된 한국어 답변 (문장 끝에 [1], [2] 인용 표기)"
    )

    cited_numbers : list[int] = Field(
        default_factory=list,
        description="답변에서 실제로 인용한 문맥 번호 목록"
    )