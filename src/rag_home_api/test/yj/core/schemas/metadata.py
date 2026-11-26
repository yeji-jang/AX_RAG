"""
전처리된 문서의 메타데이터를 정의하는 스키마
"""

from pydantic import BaseModel, Field


class PublicRentalNoticeMetadata(BaseModel):
    """
    - 원본 문서 : [공공임대_공고문.jsonl]
    - PGVector JSONB 컬럼에 저장할 metadata 스키마
    - 원본 JSONL 키 알파벳 순 
    - text 제외(page_content)
    """
    category : str = Field(
        description="문서 분류 (예: 공공임대 모집공고)")
    
    doc_id : str = Field(
        description="원본 문서 고유 ID (예: SH_308971)")
    
    # has_table : bool = Field(
    #     description="표 포함 여부")
    
    house_type : str = Field(
        description="주택 유형 (예: 매입임대, 장기전세주택)")
    
    id : str = Field(
        description="원본 행/페이지 고유 ID (예: SH_308944_p024)")
    
    notice_date : str = Field(
        description="모집 공고일 YYYYMMDD (예: 20261119)")
    
    page : int = Field(
        description="원본 문서 페이지 번호")
    
    same_text_ids : list[str] = Field(
        default_factory=list, 
        description="동일 본문을 가진 다른 페이지 ID 목록")
    
    source : str = Field(
        description="공급/출처 기관 (예: LH, SH)")

    source_url : str = Field(
        description="원본 공고/문서 URL")
    
    # spacing_restored : bool = Field(
    #     description="띄어쓰기 복원 여부")

    title : str = Field(
        description="공고문/문서 제목")


class PublicRentalChunkMetadata(PublicRentalNoticeMetadata) :
    """
    - 청킹 후 PGVector에 저장할 metadata 스키마
    - PublicRentalNoticeMetadata의 10개 필드 + 청크 정보
    """
    chunk_id : str = Field(
        description="청크 고유 ID, {페이지ID}_{청크버전}_c{순번 2자리}")

    chunk_index : int = Field(
        description="문서 내 청크 순번")
    
    content_hash : str = Field(
        description="증분 인덱싱 판별을 위한 SHA-256 해시값")