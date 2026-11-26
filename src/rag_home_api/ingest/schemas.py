from typing import Literal
from pydantic import BaseModel, Field

# 청킹할 문서 유형
DocType = Literal["rental_law", "public_notice", "youth_housing", "housing_unit", "rent_summary"]

class ChunkMetadata(BaseModel):
    """청크 1개가 반드시 들고 다녀야 하는 최소 꼬리포(메타데이타)"""

    chunk_id: str = Field(description="안정적 고유 ID (예: LAW_HLPA__본칙_제3조__c00)")
    doc_id: str = Field(description="원본 문서 ID (예: LAW_HLPA, LH_2015122300021098)")
    doc_type: DocType
    source: str = Field(description="발행 기관 (법제처, LH, SH, 청년안심주택)")
    category: str = Field(description="세부 분류")
    title: str = Field(description="문서·공고·법령 제목")
    location: str = Field(description="원문 위치 (예: 본칙 제3조, p.1)")
    source_url: str = Field(default="",description="원문 확인 URL")
    index_version: str = Field(description="색인/청킹 버전 태그")
    content_hash: str = Field(description="SHA-256 해시 (5단계 중복 적재 방지용)")
    district: str | None = Field(default=None, description="서울 자치구 (공급호실 등 해당 시)")
    min_deposit_krw: int | None = Field(default=None, description="최소 보증금(원, 미확인 시 None)")
    min_monthly_rent_krw: int | None = Field(default=None, description="최소 월세(원, 미확인 시 None)")
    exclusive_area_m2: float | None = Field(default=None, description="전용면적(㎡)")