import re

import csv
from collections import defaultdict

import hashlib
import json
from pathlib import Path
import pandas as pd

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag_home_api.config import settings
from rag_home_api.ingest.schemas import ChunkMetadata, DocType

from rag_home_api.shared.core.logger import get_logger

# 기존 (20줄)
# SEPARATORS = ["\n\n","\n",". "," ",""]
# 수정: 문단(\n\n) -> 공고문 항목 기호(\n■, \n•, \n※) -> 줄바꿈(\n) -> 문장(. ) -> 공백( ) -> 글자("")
SEPARATORS = ["\n\n", "\n■", "\n•", "\n※", "\n", ". ", " ", ""]

def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

_TABLE_SEP_RE = re.compile(r"^\|\s*:?-{2,}")


def _is_table_line(line: str) -> bool:
    s = line.strip()
    return s.startswith("|") and s.endswith("|") and s.count("|") >= 2


def _split_into_blocks(text: str) -> list[tuple[str, str]]:
    """[1층: 구조적 분리] 한 쪽 텍스트를 ('text', 일반글)과 ('table', 마크다운 표) 블록으로 나눈다."""
    lines = text.splitlines()
    blocks: list[tuple[str, str]] = []
    buf: list[str] = []
    i = 0

    while i < len(lines):
        # 현재 줄이 '|...|'이고 다음 줄이 '|---|' 구분선이면 마크다운 표 시작으로 인식
        if (
            i + 1 < len(lines)
            and _is_table_line(lines[i])
            and _TABLE_SEP_RE.match(lines[i + 1].strip())
        ):
            if buf:
                t = "\n".join(buf).strip()
                if t:
                    blocks.append(("text", t))
                buf = []
            tlines = [lines[i].strip(), lines[i + 1].strip()]
            i += 2
            while i < len(lines) and _is_table_line(lines[i]):
                tlines.append(lines[i].strip())
                i += 1
            blocks.append(("table", "\n".join(tlines)))
        else:
            buf.append(lines[i])
            i += 1

    if buf:
        t = "\n".join(buf).strip()
        if t:
            blocks.append(("text", t))
    return blocks


def _split_table_preserving_header(table_text: str, chunk_size: int) -> list[str]:
    """[2층: 표 전용 분할] 표를 행 단위로 자르고, 모든 조각 맨 위에 헤더 2줄(|컬럼|, |---|)을 재부착한다."""
    tlines = table_text.splitlines()
    if len(tlines) <= 2 or len(table_text) <= chunk_size:
        return [table_text]

    header = f"{tlines[0]}\n{tlines[1]}"
    rows = tlines[2:]
    pieces: list[str] = []
    cur_rows: list[str] = []
    cur_len = len(header)

    for row in rows:
        add_len = 1 + len(row)
        # 최소 1행은 반드시 포함해 한 행이 중간에서 뚝 잘리지 않도록 보호
        if cur_rows and cur_len + add_len > chunk_size:
            pieces.append(header + "\n" + "\n".join(cur_rows))
            cur_rows = [row]
            cur_len = len(header) + add_len
        else:
            cur_rows.append(row)
            cur_len += add_len

    if cur_rows:
        pieces.append(header + "\n" + "\n".join(cur_rows))
    return pieces


def split_text_structured(
    raw_text: str,
    splitter: RecursiveCharacterTextSplitter,
    chunk_size: int,
    min_heading_chars: int = 120,
) -> list[str]:
    """구조적(표·소제목) + 문단·재귀적 하이브리드 청킹."""
    if len(raw_text) <= chunk_size:
        return [raw_text]

    # 1층·2층: 블록 성격별 분할
    pieces: list[str] = []
    for kind, content in _split_into_blocks(raw_text):
        if kind == "table":
            pieces.extend(_split_table_preserving_header(content, chunk_size))
        elif len(content) <= chunk_size:
            pieces.append(content)
        else:
            pieces.extend(splitter.split_text(content))

    # 3층: 표 위 소제목('2. 임대조건' 등 120자 미만 단독 조각)을 다음 조각 앞에 병합
    merged: list[str] = []
    pending = ""
    for p in pieces:
        p = p.strip()
        if not p:
            continue
        if pending:
            p = f"{pending}\n\n{p}"
            pending = ""
        if len(p) < min_heading_chars and "|---" not in p:
            pending = p
        else:
            merged.append(p)
    if pending:
        if merged:
            merged[-1] = f"{merged[-1]}\n\n{pending}"
        else:
            merged.append(pending)

    return merged

def load_jsonl_chunks(
    jsonl_path: Path,
    doc_type: DocType,
    chunk_size: int = 600,
    chunk_overlap: int = 100,
    max_lines: int | None = None,
) -> list[Document]:
    """
    JSONL 파일 (1줄 = 1조문 또는 1쪽)을 읽어 청킹된 Langchain Document 목록으로 반환한다.

    Note (pandas.read_json(..., lines=True) 대신 open() + json.loads()를 쓰는 이유):
    1. 줄마다 키 구조가 달라 NaN 함정이 생김:
    예)
    일반전월세_법령가이드.jsonl은 법령 줄에 section/article만 있고 page가 없으며,
    가이드북 줄에는 page만 있고 article이 없음. 
    pandas 표로 읽으면 빈 칸이 float('nan')이 되어 
    .strip()에서 에러가 나고, 정수 page도 116.0 같은 실수로 바뀜.

                    doc_id section article   page
    0             LAW_HLPA      본칙     제1조    NaN
    1             LAW_HLPA      본칙     제2조    NaN
    425  GUIDE_SEOUL_CASES     NaN     NaN  117.0
    426  GUIDE_SEOUL_CASES     NaN     NaN  118.0

    article 값: nan | 타입: <class 'float'> | (val or empty): nan

    2. 문자열 숫자 코드의 자동 형변환 위험:
    notice_date("20261119")나 숫자 형태의 ID를 pandas가 임의로 int/float로 변환할 수 있음.

    3. 표 집계가 아닌 단순 한 줄씩 순회(스트리밍) 작업:
    open()으로 한 줄씩 읽으면 메모리가 가볍고, max_lines 테스트 시 앞 N줄만 일고 바로 종료됨.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=SEPARATORS,
    )
    docs: list[Document] = []

    # 파일을 통째로 DataFrame에 올리지 말고, 한 줄씩 스트리밍으로 파싱
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f,1):
            if max_lines is not None and line_no > max_lines:
                break
            if not line.strip():
                continue

            # 1. JSONL 한 줄을 순수 dict로 파싱
            # 없는 키는 NaN이 아니라 안전하게 None/"" 처리 가능
            row = json.loads(line)
            raw_text = (row.get("text") or "").strip()
            if not raw_text:
                continue

            # 2. 원문 위치 통일 
            # 법령 : section+article 
            # 공고문·가이드북: page -> 모두 location 하나로 통합
            doc_id = str(row["doc_id"])
            section = (row.get("section") or "").strip()
            article = (row.get("article") or "").strip()
            if article:
                loc_key = f"{section}_{article}" if section else article
                loc_display = f"{section} {article}".strip() if section else article
            else:
                page_num = int(row.get("page") or 1)
                loc_key = f"p{page_num:03d}"
                loc_display = f"p.{page_num}"

            # 3. 청킹 
            # chunk_size 이하면 굳이 자르지 않고 1조각 유지
            # 초과하면 자연스러운 경계로 분할

            # 기존: pieces = [raw_text] if len(raw_text) <= chunk_size else splitter.split_text(raw_text)
            pieces = split_text_structured(raw_text, splitter, chunk_size)

            # 4. 풍부한 문맥 Prefix ([기관 | 주택유형 | 공고일] 제목 (위치)) + 메타데이터 포장
            title = row.get("title", "")
            prefix_parts = [
                p
                for p in (
                    (row.get("source") or "").strip(),
                    (row.get("house_type") or "").strip(),
                    f"공고일: {row['notice_date']}" if row.get("notice_date") else "",
                )
                if p
            ]
            prefix_tag = f"[{' | '.join(prefix_parts)}] " if prefix_parts else ""

            for c_idx, piece in enumerate(pieces):
                chunk_id = f"{doc_id}_{loc_key}_c{c_idx:02d}"
                body = f"{prefix_tag}{title} ({loc_display})\n\n{piece.strip()}"

                meta = ChunkMetadata(
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    doc_type=doc_type,
                    source=row.get("source",""),
                    category=row.get("category",""),
                    title=title,
                    location=loc_display,
                    source_url=row.get("source_url",""),
                    index_version=settings.index_version,
                    content_hash=_sha256(f"{chunk_id}|{body}"),
                )
                docs.append(Document(page_content=body, metadata=meta.model_dump()))

    return docs

def load_housing_unit_chunks(
    units_csv: Path,
    scenarios_csv: Path,
    max_lines: int | None = None,
) -> list[Document]:
    """
    공급호실(units) + 자격별 가격 시나리오(scenarios) CSV를 조인해 1호실 = 1청크로 만든다.

    Note:
0        - 긴 글이 아니라 호실 1개 = 완결된 짧은 카드(300~450자)이므로
           글자 수 기반 splitter(chunk_size/overlap)를 쓰지 않는다. 
         - pandas.read_csv() 대신 open() + csv.DictReader() 쓰면 빈 칸이 아니라 NaN이 아니라 ""로
           들어오고, 호수("0201") 같은 앞자리 0이 숫자로 증발하지 않아 텍스트 조립에 안전하다.
    """

    # 1. 가격 시나리오 CSV를 먼저 읽어 {unit_record_id: [시나리오행들]로 묶어두기 (1:N 조인 준비)}
    scenario_by_unit: dict[str, list[dict[str,str]]] = defaultdict(list)
    with scenarios_csv.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            scenario_by_unit[row["unit_record_id"]].append(row)

    docs: list[Document] = []

    # 2. 호실 CSV를 한 행씩 읽으며 텍스트 카드 1장 + 메타데이터로 조힙
    with units_csv.open("r", encoding="utf-8-sig", newline="") as f:
        for line_no, u in enumerate(csv.DictReader(f), start=1):
            if max_lines is not None and line_no > max_lines:
                break

            uid = u["unit_record_id"]

            # 도메인 전처리: price_status가 "published"이고 보증금이 실제로 있는 조건만 유효 가격으로 취급
            pub = [
                s
                for s in scenario_by_unit.get(uid, [])
                if s.get("price_status") == "published" and s.get("deposit_krw")
            ]

            deposits = [int(float(s["deposit_krw"])) for s in pub]
            rents = [int(float(s["monthly_rent_krw"])) for s in pub if s.get("monthly_rent_krw")]

            price_lines = [
                f"- {s['eligibility_label']}({s['conversion_mode']}): "
                f"보증금 {int(float(s['deposit_krw'])):,}원 / "
                f"월세 {int(float(s.get('monthly_rent_krw') or 0)):,}원"
                for s in pub
            ]

            price_text = "\n".join(price_lines) if price_lines else "- 원문 가격 미확인"

            area = float(u["exclusive_area_m2"]) if u.get("exclusive_area_m2") else None

            area_display = f"{area}㎡" if area is not None else "미확인"
            room_display = u.get("room_layout") or u.get("housing_type_raw") or "미확인"

            # 3. 원문 위치 통일 & 본문 카드 조립
            loc_display = f"p.{u.get('source_page') or '-'} row.{u.get('source_row') or "-"}"
            chunk_id = f"{uid}_u00"
            title = f"{u['district']} {u.get('building_name','')} {u.get('unit_type','')}".strip()

            body = (
                f"[공급호실 {uid} | 공고 {u['notice_id']} | {u['provider']}]\n"
                f"자치구: {u['district']} | 주소: {u.get('address', '')} | "
                f"건물명: {u.get('building_name', '')} {u.get('ho', '')}호\n"
                f"주택형: {u.get('unit_type') or '-'} | 방구조: {room_display} | "
                f"전용면적: {area_display} | 공고일: {u.get('notice_date', '')}\n"
                f"임대조건:\n{price_text}\n"
                f"참고: 공고 시점 공급 기록이며 가전옵션·관리비·실시간 공실은 원문 기준 미확인."
            )

            # 4. 메타데이터 + Document 포장
            meta = ChunkMetadata(
                chunk_id=chunk_id,
                doc_id=u["notice_id"],
                doc_type="housing_unit",
                source=u["provider"],
                category="공공임대 공급호실",
                title=title,
                location=loc_display,
                source_url=u.get("notice_url") or u.get("source_url") or "",
                index_version=settings.index_version,
                content_hash=_sha256(f"{chunk_id}|{body}"),
                district=u.get("district") or None,
                min_deposit_krw=min(deposits) if deposits else None,
                min_monthly_rent_krw=min(rents) if rents else None,
                exclusive_area_m2=area,
            )
            docs.append(Document(page_content=body, metadata=meta.model_dump()))
    return docs
