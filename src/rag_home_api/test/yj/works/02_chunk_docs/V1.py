"""
# 2단계: 전처리된 문서 청킹
JSONL 읽기 → Document 변환 → 청킹

(순서)
    1. 경로 & 청킹 설정값 지정
    2. 전처리된 JSONL → LangChain Document 리스트로 불러오기
    3. 청킹 + Prefix 붙이기
    4. 청킹된 결과 저장 & 샘플 확인
"""

import json
import hashlib
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rag_home_api.test.yj.core.schemas.metadata import PublicRentalChunkMetadata, PublicRentalNoticeMetadata

#===================================================================
# 1. 입력/출력 경로와 청킹 설정값 지정
#===================================================================
DOC_TITLE = "공공임대_공고문" # 문서 제목
DOC_PREPROCESS_VERSION = "V2"   # 읽어올 전처리 문서 버전

CHUNK_VERSION = "V1"        # 이번 청킹 코드 버전
CHUNK_SIZE = 800 # 청킹 단위 (문자 수)
CHUNK_OVERLAP = 100 # 청킹 단위 겹침 (문자 수)


# 1) 1단계에서 만든 전처리 파일 읽어오기
IN_JSONL_PATH = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "01_preprocessed"
    / f"{DOC_TITLE}_{DOC_PREPROCESS_VERSION}.jsonl"
)

# 2) 청킹된 결과 저장할 경로
OUT_JSONL_PATH = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "02_chunked"
    / f"{DOC_TITLE}_pre{DOC_PREPROCESS_VERSION}_chunk{CHUNK_VERSION}_cs{CHUNK_SIZE}_ov{CHUNK_OVERLAP}.jsonl"
)

#===================================================================
# 2. 전처리된 JSONL -> LangChain Document 리스트로 불러오기
#===================================================================

def load_documents_from_jsonl(jsonl_path : Path = IN_JSONL_PATH) -> list[Document] :
    docs : list[Document] = []

    with jsonl_path.open("r", encoding="utf-8") as file : 
        for line in file : 
            line = line.strip()

            if not line : 
                continue

            data = json.loads(line)
            text = data.pop("text", "") # text만 꺼내고 나머지 키들은 전부 metadata로 사용

            docs.append(Document(
                page_content=text,
                metadata=data
            ))

    return docs

#===================================================================
# 3. Document 리스트 청킹 + Contextual Prefix 붙이기
#===================================================================
def compute_sha256(text: str) -> str:
    """청크 본문의 고유 지문(SHA-256 해시)을 생성합니다."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def split_and_enrich(
    raw_docs: list[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    add_prefix: bool = True,
) -> list[Document]:
    # 1) 가위 세팅
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n■", "\n•", "\n※", "\n", " ", ""],
    )
    enriched: list[Document] = []

    # 2) 페이지(Document)마다 따로 청킹 → chunk_index가 페이지 안에서 0, 1, 2 …
    for doc in raw_docs:
        pieces = splitter.split_text(doc.page_content)

        # 3) 각 조각에 Prefix + PublicRentalChunkMetadata 검증 적용
        for idx, piece in enumerate(pieces):
            text = piece.strip()
            if not text:
                continue

            if add_prefix:
                prefix = (
                    f"[{doc.metadata['source']} | {doc.metadata['house_type']} | "
                    f"공고일: {doc.metadata['notice_date']}] "
                    f"{doc.metadata['title']} (p.{doc.metadata['page']})\n\n"
                )
                content = prefix + text
            else:
                content = text

            meta = PublicRentalChunkMetadata(
                **doc.metadata,
                chunk_index=idx,
                content_hash=compute_sha256(content),  # prefix 포함 본문 기준
            )
            enriched.append(Document(page_content=content, metadata=meta.model_dump()))

    return enriched

# ===================================================================
# 4. 청킹된 결과 JSONL 저장 & 샘플 확인
# ===================================================================
def save_chunked_jsonl(chunks: list[Document], out_path: Path = OUT_JSONL_PATH) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            record = {"text": chunk.page_content, **chunk.metadata}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    lengths = [len(c.page_content) for c in chunks]
    print(f"총 청크 수: {len(chunks):,}개")
    print(f"길이 -> 최소: {min(lengths)}, 평균: {sum(lengths) // len(lengths)}, 최대: {max(lengths)}")
    print(f"저장 경로: {out_path}")


# ===================================================================
# 5. 실행
# ===================================================================
if __name__ == "__main__":
    raw_docs = load_documents_from_jsonl()
    chunks = split_and_enrich(raw_docs, add_prefix=True)
    save_chunked_jsonl(chunks)

    print("\n--- [샘플 청크 0번] ---")
    print(chunks[0].page_content[:400])
