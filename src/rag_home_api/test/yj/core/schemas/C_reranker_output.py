"""
RAG 리랭커에서 사용하는 LLM 구조화 출력 스키마
"""

from pydantic import BaseModel, Field


#===================================================================
# Re-ranker 출력 스키마 (C 구성)
#===================================================================
class RankedItem(BaseModel) :
    candidate_no : int = Field(
        description="후보 청크 번호 (1부터 시작)")

    relevance_score : float = Field(
        ge=0.0,
        le=10.0,
        description="질문과의 관련도 점수 (0~10점)")
    
    reason : str = Field(
        description="이 순위로 판단한 한 줄 이유")

class RerankOutput(BaseModel) :
    ranked_items : list[RankedItem] = Field(
        description="관련도가 높은 순서대로 정렬된 상위 후보 목록")
