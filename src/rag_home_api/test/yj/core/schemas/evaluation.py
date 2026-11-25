from typing import Literal
from pydantic import BaseModel, Field


class EvalQAItem(BaseModel):
    """
    공공임대 공고문 Golden QA 평가셋 단일 문항 스키마 
    (지침서 STEP 2 + RAGAS 호환)
    """

    # 1. 문항 식별 및 분류 (개발 10 / 최종 평가 20 분리)
    question_id: str = Field(description="평가 문항 고유 ID (예: DEV-001, TEST-001)")
    split: Literal["dev", "test"] = Field(description="개발용(dev 10문항) / 최종 평가용(test 20문항)")
    query_type: Literal["direct", "paraphrase_or_similar", "unanswerable"] = Field(
        description="직접 검색(direct) / 표현 변형·유사 조건 구분(paraphrase_or_similar) / 근거 없음(unanswerable)"
    )

    # 2. 질문 & 모범 정답 (RAGAS 입력용)
    question: str = Field(description="평가용 사용자 질문")
    ground_truth: str = Field(
        description="공고문 근거 기반 모범 정답 (근거 없음 문항은 '정보 부족 안내'로 기록)"
    )
    required_facts: list[str] = Field(
        default_factory=list,
        description="정답 충족(/17) 채점을 위한 필수 핵심 사실 목록",
    )

    # 3. 정답 근거 매핑 (Hit@5 및 근거 적합 채점용)
    allowed_doc_ids: list[str] = Field(
        default_factory=list, description="허용 근거 문서 doc_id 목록 (근거 없음은 [])"
    )
    allowed_chunk_ids: list[str] = Field(
        default_factory=list, description="허용 근거 청크 chunk_id 목록 (Hit@5 채점 기준, 근거 없음은 [])"
    )
    source_location: str = Field(
        default="", description="원문 위치 확인용 (예: LH_2015122300021098 p.3)"
    )
    unanswerable_reason: str | None = Field(
        default=None, description="근거 없음 문항일 때 부재 판정 이유"
    )


class SampleEvaluationResult(BaseModel):
    """단일 문항 평가 결과 (지침서 필수 지표 + RAGAS 4대 지표 + 자동 진단)"""
    question_id: str
    query_type: str
    pipeline_mode: Literal["A_vector", "B_hybrid", "C_hybrid_rerank", "C_plus_opt"]
    question: str
    answer: str
    status: Literal["answered", "insufficient_evidence", "error"]
    hit_at_5: int | None = Field(default=None, description="답변 가능 문항 Hit@5 (0/1), 근거 없음은 None")
    elapsed_sec: float = Field(description="검색+재정렬+생성 전체 응답 시간(초)")
    context_precision: float | None = None
    context_recall: float | None = None
    faithfulness: float | None = None
    answer_relevancy: float | None = None
    prescription: str = Field(description="자동 진단된 개선 처방")