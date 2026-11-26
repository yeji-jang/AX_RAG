"""
# 2-2단계: 전처리된 문서 청킹 (V2 : 표 인식 청킹)
JSONL 읽기 → Document 변환 → 청킹

- V1 대비 변경 : 표는 행 단위로 자르고, 청크마다 표 헤더를 반복하며, 비어 있는 앞쪽 칸(병합 셀)을 위 행 값으로 채움
- 본문은 V1과 동일 (RecursiveCharacterTextSplitter, 800/100)

(순서)
    1. 경로 & 청킹 설정값 지정 (V2)
    2. 전처리된 JSONL -> LangChain Document 리스트로 불러오기
    3. 페이지 본문을 [일반 문단]과 [표] 블록으로 분리
    4. 표 블록 처리 (2단 병합 헤더 감지 · 앞쪽 병합 셀 채우기 · 헤더 복제 행 분할)
    5. 짧은 자투리 조각 병합 & 전체 청킹 + Contextual Prefix 붙이기
    6. 청킹된 결과 JSONL 저장 & 샘플 확인
    7. 실행
"""

import hashlib
import json
import re
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag_home_api.test.yj.core.schemas.metadata import PublicRentalChunkMetadata


#===================================================================
# 1. 입력/출력 경로와 청킹 설정값 지정
#===================================================================
DOC_TITLE = "공공임대_공고문"
DOC_PREPROCESS_VERSION = "V2"   # 읽어올 전처리 문서 버전

CHUNK_VERSION = "V2"            # 이번 청킹 코드 버전 (표 보존 청킹)
CHUNK_SIZE = 800                # 기본 청킹 단위 (문자 수)
CHUNK_OVERLAP = 100             # 일반 문단 청킹 겹침 (문자 수)
MIN_CHUNK_CHARS = 200           # 이 길이보다 짧은 조각은 같은 페이지 앞/뒤 조각과 병합 시도
MAX_MERGE_CHARS = 950           # 병합했을 때 허용하는 최대 글자 수

IN_JSONL_PATH = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "01_preprocessed"
    / f"{DOC_TITLE}_{DOC_PREPROCESS_VERSION}.jsonl"
)

OUT_JSONL_PATH = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "02_chunked"
    / f"{DOC_TITLE}_pre{DOC_PREPROCESS_VERSION}_chunk{CHUNK_VERSION}_cs{CHUNK_SIZE}_ov{CHUNK_OVERLAP}.jsonl"
)

# 마크다운 표 구분선 패턴 (예: |---|---|---| 또는 |:---|---:|)
SEPARATOR_RE = re.compile(r"^\|[\s:\-|]+\|$")
SUMMARY_ROW_LABELS = ("계", "합계", "소계", "총계")


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
            text = data.pop("text", "")

            docs.append(Document(
                page_content=text,
                metadata=data,
            ))

    return docs


#===================================================================
# 3. 페이지 본문을 [일반 문단]과 [표] 블록으로 분리
#===================================================================
def is_table_line(line : str) -> bool :
    """마크다운 표 행인지 판별 ('|'로 시작하고 끝나며 구분자가 2개 이상)"""
    stripped = line.strip()
    return stripped.startswith("|") and stripped.endswith("|") and stripped.count("|") >= 2


def is_separator_line(line : str) -> bool :
    """마크다운 표 헤더 구분선(|---|---|)인지 판별"""
    stripped = line.strip()
    return bool(SEPARATOR_RE.match(stripped)) and "-" in stripped


def parse_blocks(text : str) -> list[tuple[str, list[str]]] :
    """
    페이지 텍스트를 순서대로 ('text', [...]) / ('table', [...]) 블록 리스트로 분리
    """
    blocks : list[tuple[str, list[str]]] = []
    current_type : str | None = None
    current_lines : list[str] = []

    for line in text.splitlines() :
        line_type = "table" if is_table_line(line) else "text"

        if line_type != current_type :
            if current_lines and current_type is not None :
                blocks.append((current_type, current_lines))
            current_type = line_type
            current_lines = [line]
        else :
            current_lines.append(line)

    if current_lines and current_type is not None :
        blocks.append((current_type, current_lines))

    return blocks


#===================================================================
# 4. 표 블록 처리 (헤더 추출 · 앞쪽 병합 셀 채우기 · 헤더 복제 행 분할)
#===================================================================
def split_table_cells(line : str) -> list[str] :
    """'| A | B | C |' 한 줄을 ['A', 'B', 'C'] 셀 리스트로 분해"""
    stripped = line.strip()
    inner = stripped[1:-1]
    return [cell.strip() for cell in inner.split("|")]


def join_table_cells(cells : list[str]) -> str :
    """셀 리스트를 다시 마크다운 표 한 줄로 조립"""
    return "| " + " | ".join(cells) + " |"


def fill_leading_merged_cells(body_lines : list[str]) -> list[str] :
    """
    세로 병합으로 비어 있는 앞쪽 칸(예: 단지명, 주택형)을 바로 위 행의 값으로 채움
    - 맨 왼쪽부터 처음 값이 나오는 칸 직전까지만 채우므로, 오른쪽 빈칸(비고 등)은 건드리지 않음
    """
    filled_lines : list[str] = []
    prev_cells : list[str] | None = None

    for line in body_lines :
        cells = split_table_cells(line)

        # 1) 첫 번째 비어있지 않은 칸 위치 찾기
        first_non_empty = -1
        for idx, cell in enumerate(cells) :
            if cell :
                first_non_empty = idx
                break

        # 전부 빈 행이면 그대로 두고 넘어감
        if first_non_empty == -1 :
            filled_lines.append(line)
            continue

        # 2) 위 행과 열 개수가 같고 앞쪽에 빈 칸이 있으면 위 행 값으로 채움
        if prev_cells is not None and len(cells) == len(prev_cells) and first_non_empty > 0 :
            for idx in range(first_non_empty) :
                if not cells[idx] and prev_cells[idx] :
                    cells[idx] = prev_cells[idx]

        filled_lines.append(join_table_cells(cells))

        # 3) 합계/소계 행은 다음 행의 단지명으로 복사되지 않도록 기준 행에서 제외
        first_label = cells[0] if cells else ""
        if first_label not in SUMMARY_ROW_LABELS :
            prev_cells = cells

    return filled_lines


def split_table_block(table_lines : list[str], max_chars : int = CHUNK_SIZE) -> list[str] :
    """
    표 블록을 행 단위로 나누되, 모든 청크 상단에 표 헤더(+2단 병합 소제목)를 자동 복제
    """
    # 1) 표 구분선(|---|) 위치 찾기
    sep_idx = -1
    for idx, line in enumerate(table_lines) :
        if is_separator_line(line) :
            sep_idx = idx
            break

    # 구분선이 없는 비표준 표는 일반 줄 묶음처럼 길이 기준으로만 행 단위 분할
    if sep_idx == -1 :
        header_lines : list[str] = []
        body_lines = table_lines
    else :
        header_end = sep_idx + 1

        # 2) 2단 병합 소제목 감지 : 구분선 바로 다음 줄이 빈 첫 칸('|  | ...')으로 시작하면 헤더에 포함
        if header_end < len(table_lines) :
            next_cells = split_table_cells(table_lines[header_end])
            if next_cells and next_cells[0] == "" :
                header_end += 1

        header_lines = table_lines[:header_end]
        body_lines = fill_leading_merged_cells(table_lines[header_end:])

    # 본문 행이 없으면 표 전체를 그대로 1개 조각으로 반환
    if not body_lines :
        return ["\n".join(table_lines)]

    header_text = "\n".join(header_lines)
    pieces : list[str] = []
    current_rows : list[str] = []
    current_len = len(header_text)

    # 3) 행 단위로 묶다가 max_chars를 넘으면 청크 확정 후 새 청크에 헤더를 다시 붙임
    for row in body_lines :
        added_len = len(row) + 1  # 줄바꿈 포함 길이

        if current_rows and (current_len + added_len > max_chars) :
            if header_text :
                pieces.append(header_text + "\n" + "\n".join(current_rows))
            else :
                pieces.append("\n".join(current_rows))
            current_rows = [row]
            current_len = len(header_text) + added_len
        else :
            current_rows.append(row)
            current_len += added_len

    if current_rows :
        if header_text :
            pieces.append(header_text + "\n" + "\n".join(current_rows))
        else :
            pieces.append("\n".join(current_rows))

    return pieces


#===================================================================
# 5. 짧은 조각 병합 & 전체 청킹 + Contextual Prefix 붙이기
#===================================================================
def merge_short_pieces(
    pieces : list[str],
    min_chars : int = MIN_CHUNK_CHARS,
    max_merge_chars : int = MAX_MERGE_CHARS,
) -> list[str] :
    """같은 페이지 안에서 너무 짧은 자투리 조각을 앞/뒤 조각과 병합"""
    if len(pieces) <= 1 :
        return pieces

    merged : list[str] = []

    for piece in pieces :
        text = piece.strip()
        if not text :
            continue

        # 1) 이전 조각이 짧거나 현재 조각이 짧고, 합쳐도 max_merge_chars 이하라면 병합
        if merged :
            prev = merged[-1]
            combined_len = len(prev) + 2 + len(text)
            if (len(prev) < min_chars or len(text) < min_chars) and combined_len <= max_merge_chars :
                merged[-1] = prev + "\n\n" + text
                continue

        merged.append(text)

    return merged


def compute_sha256(text : str) -> str :
    """청크 본문의 고유 지문(SHA-256 해시)을 생성합니다."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def split_page_table_aware(
    page_text : str,
    text_splitter : RecursiveCharacterTextSplitter,
    chunk_size : int = CHUNK_SIZE,
) -> list[str] :
    """한 페이지 안에서 일반 문단과 표를 구분해 각각 최적 방식으로 분할"""
    blocks = parse_blocks(page_text)
    raw_pieces : list[str] = []

    for block_type, lines in blocks :
        if block_type == "table" :
            table_pieces = split_table_block(lines, max_chars=chunk_size)
            raw_pieces.extend(table_pieces)
        else :
            paragraph_text = "\n".join(lines).strip()
            if not paragraph_text :
                continue
            raw_pieces.extend(text_splitter.split_text(paragraph_text))

    return merge_short_pieces(raw_pieces)


def split_and_enrich(
    raw_docs : list[Document],
    chunk_size : int = CHUNK_SIZE,
    chunk_overlap : int = CHUNK_OVERLAP,
    add_prefix : bool = True,
) -> list[Document] :
    # 1) 일반 문단용 가위 세팅 (V1과 동일)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n■", "\n•", "\n※", "\n", " ", ""],
    )
    enriched : list[Document] = []

    # 2) 페이지마다 표 인식 분할 수행 -> chunk_index가 페이지 안에서 0, 1, 2 ...
    for doc in raw_docs :
        pieces = split_page_table_aware(doc.page_content, text_splitter, chunk_size=chunk_size)

        # 3) 각 조각에 Contextual Prefix + PublicRentalChunkMetadata 검증 적용
        for idx, piece in enumerate(pieces) :
            text = piece.strip()
            if not text :
                continue

            if add_prefix :
                prefix = (
                    f"[{doc.metadata['source']} | {doc.metadata['house_type']} | "
                    f"공고일: {doc.metadata['notice_date']}] "
                    f"{doc.metadata['title']} (p.{doc.metadata['page']})\n\n"
                )
                content = prefix + text
            else :
                content = text

            chunk_id = f"{doc.metadata['id']}_ck{CHUNK_VERSION}_c{idx:02d}"

            meta = PublicRentalChunkMetadata(
                **doc.metadata,
                chunk_id=chunk_id,
                chunk_index=idx,
                content_hash=compute_sha256(content),
            )
            enriched.append(Document(
                id=chunk_id,
                page_content=content,
                metadata=meta.model_dump(),
            ))

    return enriched


#===================================================================
# 6. 청킹된 결과 JSONL 저장 & 샘플 확인
#===================================================================
def save_chunked_jsonl(chunks : list[Document], out_path : Path = OUT_JSONL_PATH) -> None :
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f :
        for chunk in chunks :
            record = {"text": chunk.page_content, **chunk.metadata}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    lengths = [len(c.page_content) for c in chunks]
    print(f"총 청크 수: {len(chunks):,}개")
    print(f"길이 -> 최소: {min(lengths)}, 평균: {sum(lengths) // len(lengths)}, 최대: {max(lengths)}")
    print(f"저장 경로: {out_path}")


#===================================================================
# 7. 실행
#===================================================================
if __name__ == "__main__" :
    raw_docs = load_documents_from_jsonl()
    chunks = split_and_enrich(raw_docs, add_prefix=True)
    save_chunked_jsonl(chunks)

    print("\n--- [샘플 청크 0번] ---")
    print(chunks[0].page_content[:400])


import hashlib
import json
import re
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag_home_api.test.yj.core.schemas.metadata import PublicRentalChunkMetadata


#===================================================================
# 1. 입력/출력 경로와 청킹 설정값 지정
#===================================================================
DOC_TITLE = "공공임대_공고문"
DOC_PREPROCESS_VERSION = "V2"   # 읽어올 전처리 문서 버전

CHUNK_VERSION = "V2"            # 이번 청킹 코드 버전 (표 보존 청킹)
CHUNK_SIZE = 800                # 기본 청킹 단위 (문자 수)
CHUNK_OVERLAP = 100             # 일반 문단 청킹 겹침 (문자 수)
MIN_CHUNK_CHARS = 200           # 이 길이보다 짧은 조각은 같은 페이지 앞/뒤 조각과 병합 시도
MAX_MERGE_CHARS = 950           # 병합했을 때 허용하는 최대 글자 수

IN_JSONL_PATH = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "01_preprocessed"
    / f"{DOC_TITLE}_{DOC_PREPROCESS_VERSION}.jsonl"
)

OUT_JSONL_PATH = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "02_chunked"
    / f"{DOC_TITLE}_pre{DOC_PREPROCESS_VERSION}_chunk{CHUNK_VERSION}_cs{CHUNK_SIZE}_ov{CHUNK_OVERLAP}.jsonl"
)

# 마크다운 표 구분선 패턴 (예: |---|---|---| 또는 |:---|---:|)
SEPARATOR_RE = re.compile(r"^\|[\s:\-|]+\|$")
SUMMARY_ROW_LABELS = ("계", "합계", "소계", "총계")


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
            text = data.pop("text", "")

            docs.append(Document(
                page_content=text,
                metadata=data,
            ))

    return docs


#===================================================================
# 3. 페이지 본문을 [일반 문단]과 [표] 블록으로 분리
#===================================================================
def is_table_line(line : str) -> bool :
    """마크다운 표 행인지 판별 ('|'로 시작하고 끝나며 구분자가 2개 이상)"""
    stripped = line.strip()
    return stripped.startswith("|") and stripped.endswith("|") and stripped.count("|") >= 2


def is_separator_line(line : str) -> bool :
    """마크다운 표 헤더 구분선(|---|---|)인지 판별"""
    stripped = line.strip()
    return bool(SEPARATOR_RE.match(stripped)) and "-" in stripped


def parse_blocks(text : str) -> list[tuple[str, list[str]]] :
    """
    페이지 텍스트를 순서대로 ('text', [...]) / ('table', [...]) 블록 리스트로 분리
    """
    blocks : list[tuple[str, list[str]]] = []
    current_type : str | None = None
    current_lines : list[str] = []

    for line in text.splitlines() :
        line_type = "table" if is_table_line(line) else "text"

        if line_type != current_type :
            if current_lines and current_type is not None :
                blocks.append((current_type, current_lines))
            current_type = line_type
            current_lines = [line]
        else :
            current_lines.append(line)

    if current_lines and current_type is not None :
        blocks.append((current_type, current_lines))

    return blocks


#===================================================================
# 4. 표 블록 처리 (헤더 추출 · 앞쪽 병합 셀 채우기 · 헤더 복제 행 분할)
#===================================================================
def split_table_cells(line : str) -> list[str] :
    """'| A | B | C |' 한 줄을 ['A', 'B', 'C'] 셀 리스트로 분해"""
    stripped = line.strip()
    inner = stripped[1:-1]
    return [cell.strip() for cell in inner.split("|")]


def join_table_cells(cells : list[str]) -> str :
    """셀 리스트를 다시 마크다운 표 한 줄로 조립"""
    return "| " + " | ".join(cells) + " |"


def fill_leading_merged_cells(body_lines : list[str]) -> list[str] :
    """
    세로 병합으로 비어 있는 앞쪽 칸(예: 단지명, 주택형)을 바로 위 행의 값으로 채움
    - 맨 왼쪽부터 처음 값이 나오는 칸 직전까지만 채우므로, 오른쪽 빈칸(비고 등)은 건드리지 않음
    """
    filled_lines : list[str] = []
    prev_cells : list[str] | None = None

    for line in body_lines :
        cells = split_table_cells(line)

        # 1) 첫 번째 비어있지 않은 칸 위치 찾기
        first_non_empty = -1
        for idx, cell in enumerate(cells) :
            if cell :
                first_non_empty = idx
                break

        # 전부 빈 행이면 그대로 두고 넘어감
        if first_non_empty == -1 :
            filled_lines.append(line)
            continue

        # 2) 위 행과 열 개수가 같고 앞쪽에 빈 칸이 있으면 위 행 값으로 채움
        if prev_cells is not None and len(cells) == len(prev_cells) and first_non_empty > 0 :
            for idx in range(first_non_empty) :
                if not cells[idx] and prev_cells[idx] :
                    cells[idx] = prev_cells[idx]

        filled_lines.append(join_table_cells(cells))

        # 3) 합계/소계 행은 다음 행의 단지명으로 복사되지 않도록 기준 행에서 제외
        first_label = cells[0] if cells else ""
        if first_label not in SUMMARY_ROW_LABELS :
            prev_cells = cells

    return filled_lines


def split_table_block(table_lines : list[str], max_chars : int = CHUNK_SIZE) -> list[str] :
    """
    표 블록을 행 단위로 나누되, 모든 청크 상단에 표 헤더(+2단 병합 소제목)를 자동 복제
    """
    # 1) 표 구분선(|---|) 위치 찾기
    sep_idx = -1
    for idx, line in enumerate(table_lines) :
        if is_separator_line(line) :
            sep_idx = idx
            break

    # 구분선이 없는 비표준 표는 일반 줄 묶음처럼 길이 기준으로만 행 단위 분할
    if sep_idx == -1 :
        header_lines : list[str] = []
        body_lines = table_lines
    else :
        header_end = sep_idx + 1

        # 2) 2단 병합 소제목 감지 : 구분선 바로 다음 줄이 빈 첫 칸('|  | ...')으로 시작하면 헤더에 포함
        if header_end < len(table_lines) :
            next_cells = split_table_cells(table_lines[header_end])
            if next_cells and next_cells[0] == "" :
                header_end += 1

        header_lines = table_lines[:header_end]
        body_lines = fill_leading_merged_cells(table_lines[header_end:])

    # 본문 행이 없으면 표 전체를 그대로 1개 조각으로 반환
    if not body_lines :
        return ["\n".join(table_lines)]

    header_text = "\n".join(header_lines)
    pieces : list[str] = []
    current_rows : list[str] = []
    current_len = len(header_text)

    # 3) 행 단위로 묶다가 max_chars를 넘으면 청크 확정 후 새 청크에 헤더를 다시 붙임
    for row in body_lines :
        added_len = len(row) + 1  # 줄바꿈 포함 길이

        if current_rows and (current_len + added_len > max_chars) :
            if header_text :
                pieces.append(header_text + "\n" + "\n".join(current_rows))
            else :
                pieces.append("\n".join(current_rows))
            current_rows = [row]
            current_len = len(header_text) + added_len
        else :
            current_rows.append(row)
            current_len += added_len

    if current_rows :
        if header_text :
            pieces.append(header_text + "\n" + "\n".join(current_rows))
        else :
            pieces.append("\n".join(current_rows))

    return pieces


#===================================================================
# 5. 짧은 조각 병합 & 전체 청킹 + Contextual Prefix 붙이기
#===================================================================
def merge_short_pieces(
    pieces : list[str],
    min_chars : int = MIN_CHUNK_CHARS,
    max_merge_chars : int = MAX_MERGE_CHARS,
) -> list[str] :
    """같은 페이지 안에서 너무 짧은 자투리 조각을 앞/뒤 조각과 병합"""
    if len(pieces) <= 1 :
        return pieces

    merged : list[str] = []

    for piece in pieces :
        text = piece.strip()
        if not text :
            continue

        # 1) 이전 조각이 짧거나 현재 조각이 짧고, 합쳐도 max_merge_chars 이하라면 병합
        if merged :
            prev = merged[-1]
            combined_len = len(prev) + 2 + len(text)
            if (len(prev) < min_chars or len(text) < min_chars) and combined_len <= max_merge_chars :
                merged[-1] = prev + "\n\n" + text
                continue

        merged.append(text)

    return merged


def compute_sha256(text : str) -> str :
    """청크 본문의 고유 지문(SHA-256 해시)을 생성합니다."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def split_page_table_aware(
    page_text : str,
    text_splitter : RecursiveCharacterTextSplitter,
    chunk_size : int = CHUNK_SIZE,
) -> list[str] :
    """한 페이지 안에서 일반 문단과 표를 구분해 각각 최적 방식으로 분할"""
    blocks = parse_blocks(page_text)
    raw_pieces : list[str] = []

    for block_type, lines in blocks :
        if block_type == "table" :
            table_pieces = split_table_block(lines, max_chars=chunk_size)
            raw_pieces.extend(table_pieces)
        else :
            paragraph_text = "\n".join(lines).strip()
            if not paragraph_text :
                continue
            raw_pieces.extend(text_splitter.split_text(paragraph_text))

    return merge_short_pieces(raw_pieces)


def split_and_enrich(
    raw_docs : list[Document],
    chunk_size : int = CHUNK_SIZE,
    chunk_overlap : int = CHUNK_OVERLAP,
    add_prefix : bool = True,
) -> list[Document] :
    # 1) 일반 문단용 가위 세팅 (V1과 동일)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n■", "\n•", "\n※", "\n", " ", ""],
    )
    enriched : list[Document] = []

    # 2) 페이지마다 표 인식 분할 수행 -> chunk_index가 페이지 안에서 0, 1, 2 ...
    for doc in raw_docs :
        pieces = split_page_table_aware(doc.page_content, text_splitter, chunk_size=chunk_size)

        # 3) 각 조각에 Contextual Prefix + PublicRentalChunkMetadata 검증 적용
        for idx, piece in enumerate(pieces) :
            text = piece.strip()
            if not text :
                continue

            if add_prefix :
                prefix = (
                    f"[{doc.metadata['source']} | {doc.metadata['house_type']} | "
                    f"공고일: {doc.metadata['notice_date']}] "
                    f"{doc.metadata['title']} (p.{doc.metadata['page']})\n\n"
                )
                content = prefix + text
            else :
                content = text

            chunk_id = f"{doc.metadata['id']}_ck{CHUNK_VERSION}_c{idx:02d}"

            meta = PublicRentalChunkMetadata(
                **doc.metadata,
                chunk_id=chunk_id,
                chunk_index=idx,
                content_hash=compute_sha256(content),
            )
            enriched.append(Document(
                id=chunk_id,
                page_content=content,
                metadata=meta.model_dump(),
            ))

    return enriched


#===================================================================
# 6. 청킹된 결과 JSONL 저장 & 샘플 확인
#===================================================================
def save_chunked_jsonl(chunks : list[Document], out_path : Path = OUT_JSONL_PATH) -> None :
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f :
        for chunk in chunks :
            record = {"text": chunk.page_content, **chunk.metadata}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    lengths = [len(c.page_content) for c in chunks]
    print(f"총 청크 수: {len(chunks):,}개")
    print(f"길이 -> 최소: {min(lengths)}, 평균: {sum(lengths) // len(lengths)}, 최대: {max(lengths)}")
    print(f"저장 경로: {out_path}")


#===================================================================
# 7. 실행
#===================================================================
if __name__ == "__main__" :
    raw_docs = load_documents_from_jsonl()
    chunks = split_and_enrich(raw_docs, add_prefix=True)
    save_chunked_jsonl(chunks)

    print("\n--- [샘플 청크 0번] ---")
    print(chunks[0].page_content[:400])