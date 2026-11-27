"""수집한 PDF·법령을 RAG용 JSONL로 변환해 팀 공유 폴더(shared/docs)에 저장한다.

- 공고문·가이드 PDF: 한 줄 = 한 쪽. 글자는 PyMuPDF, 표는 pdfplumber로 찾아 마크다운 표로 바꾸고
  쪽 안의 위아래 순서대로 합친다. 글자가 거의 없는 쪽(도면·사진)은 빼고 문서 목록에 쪽수만 남긴다.
- 법령: 한 줄 = 한 조문 (본칙만).
- 문서목록.csv: 모든 문서의 출처·공고일·쪽수 (엑셀용 UTF-8 BOM).

정제(v1, clean_text·dedupe_boilerplate):
- 보이지 않는 제어 글자(NUL 등) 제거
- 쪽 번호(- 3 -)·꼬리말(www.i-sh.co.kr | 61)·다운로드 워터마크 줄 제거
- PDF 줄바꿈 때문에 끊긴 문장 잇기. 이어 붙인 자리의 띄어쓰기는 kiwipiepy가 앞뒤 글자를 보고 정한다.
  글머리표(■ ● ※ 1. 가. ① …)로 시작하는 줄과 문장 끝(…다.)은 줄을 그대로 둔다.
- 흩어진 표 조각(칸 값이 한 줄씩 흩어져 읽을 수 없는 부분) 제거
- 마크다운 표의 빈 행·빈 열 제거
- 두 단 편집 쪽은 왼쪽 단을 다 읽고 오른쪽 단을 읽는 순서로 정렬(reading_order)
- 공고문: 여러 공고에 똑같이 나오는 공통 안내문은 처음 한 번만 남김
- 가이드: 같은 책 안에서 쪽마다 반복되는 머리말 제거
- 정제 뒤 글자가 거의 없는 쪽은 제외
- 출력 칸: 처리 기록용 칸(has_table·spacing_restored·section)은 빼고, 주택 유형 이름을 통일

실행: uv run python -m rag_home_api.collect.export_jsonl
"""

import csv
import json
import re
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pdfplumber
import pymupdf

from rag_home_api.config import DATA_DIR, ROOT

RAW = DATA_DIR / "raw"
OUT = ROOT / "src" / "rag_home_api" / "shared" / "docs"
MIN_CHARS = 30  # 이보다 글자가 적은 쪽은 그림만 있는 쪽으로 보고 뺀다


def table_to_markdown(rows: list[list]) -> str:
    def cell(c) -> str:
        return (c or "").replace("\n", " ").replace("|", "/").strip()

    rows = [[cell(c) for c in r] for r in rows if any(c for c in r)]
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    lines = ["| " + " | ".join(rows[0]) + " |", "|" + "---|" * width]
    lines += ["| " + " | ".join(r) + " |" for r in rows[1:]]
    return "\n".join(lines)


def inside(box, area) -> bool:
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    return area[0] <= cx <= area[2] and area[1] <= cy <= area[3]


def block_text(block: dict) -> str:
    """PyMuPDF rawdict 블록 하나의 글을 글자 위치로 다시 짠다.
    일부 PDF(LH 공고 등)는 문장부호를 다른 글꼴 조각으로 따로 저장해서, 조각 순서대로 이으면
    '바랍니다아울러 신청⏎.' 처럼 밀린다. 같은 높이의 글자를 한 줄로 묶고 x 순서로 정렬해 제자리에 놓는다.
    글자 조각(span)이 바뀌는 자리에서 글자 사이가 글자 폭의 35% 넘게 벌어졌는데 공백이 없으면 띄어쓰기를 넣는다
    (예: 본인이입주자 → 본인이 입주자). 한 조각 안에서는 원래 공백을 따른다(자간이 넓은 글자가 쪼개지지 않게)."""
    spans = [sp for ln in block.get("lines", []) for sp in ln["spans"]]
    chars = [(c["c"], c["bbox"], si) for si, sp in enumerate(spans) for c in sp["chars"]]
    lines: list[dict] = []
    for ch, (x0, y0, x1, y1), si in sorted(chars, key=lambda c: (c[1][1] + c[1][3]) / 2):
        yc, h = (y0 + y1) / 2, max(y1 - y0, 1)
        for L in reversed(lines[-3:]):  # 최근 줄 몇 개만 비교(같은 높이면 같은 줄)
            if abs(L["yc"] - yc) <= max(L["h"], h) * 0.5:
                L["chars"].append((x0, x1, ch, si))
                break
        else:
            lines.append({"yc": yc, "h": h, "chars": [(x0, x1, ch, si)]})
    out = []
    for L in sorted(lines, key=lambda L: L["yc"]):
        s, prev = "", None
        for x0, x1, ch, si in sorted(L["chars"], key=lambda c: c[0]):
            if (prev is not None and si != prev[1] and ch != " " and not s.endswith(" ")
                    and x0 - prev[0] > (x1 - x0) * 0.35):
                s += " "
            s += ch
            prev = (x1, si)
        s = re.sub(r"[  -​　\t]", " ", s)  # 양쪽 정렬용 특수 공백 → 일반 공백
        s = re.sub(r" {2,}", " ", s).strip()
        if s:
            out.append(s)
    return "\n".join(out)


def reading_order(items: list[tuple], width: float) -> list[tuple]:
    """블록(x0, y0, x1, y1, 내용)을 읽는 순서로 정렬한다.
    두 단 편집 쪽(왼쪽에만 있는 블록·오른쪽에만 있는 블록이 각각 5개 이상, 양쪽에 걸친 블록이 20% 이하)은
    양쪽에 걸친 블록 사이 구간마다 왼쪽 단을 위에서 아래로 다 읽고 오른쪽 단을 읽는다. 그 밖의 쪽은 위에서 아래로."""
    def side(it):
        if it[2] <= width * 0.52:
            return 0
        if it[0] >= width * 0.48:
            return 1
        return -1
    by_y = sorted(items, key=lambda it: (it[1], it[0]))
    sides = [side(it) for it in by_y]

    def wide_and_full(s):  # 그 단의 블록이 넓고(중앙값 너비 ≥ 쪽의 20%) 글이 충분(200자 이상)한가
        col = [it for it, x in zip(by_y, sides) if x == s]
        widths = sorted(it[2] - it[0] for it in col)
        return (bool(col) and widths[len(widths) // 2] >= width * 0.2
                and sum(len(it[4]) for it in col) >= 200)

    # "항목명 | 내용"처럼 한쪽이 좁은 이름표인 배치는 두 단이 아니다 → 양쪽 단이 모두 넓고 글이 있을 때만 두 단
    two_col = (sides.count(0) >= 5 and sides.count(1) >= 5 and sides.count(-1) <= 0.2 * len(by_y)
               and wide_and_full(0) and wide_and_full(1))
    if not two_col:
        return by_y
    out, seg = [], []
    for it, s in zip(by_y + [None], sides + [-1]):
        if s == -1:  # 양쪽에 걸친 블록(제목 등)이 나오면 지금까지 모은 구간을 왼쪽 → 오른쪽 순으로 내보낸다
            out += [x for x, xs in seg if xs == 0] + [x for x, xs in seg if xs == 1]
            seg = []
            if it is not None:
                out.append(it)
        else:
            seg.append((it, s))
    return out


def pdf_pages(path: Path) -> tuple[list[dict], int]:
    """PDF를 쪽 단위 [{page, text, tables}]로 바꾼다. (글자 없는 쪽 수도 함께 돌려줌)"""
    pages, empty = [], 0
    try:
        pl = pdfplumber.open(path)
        pl_pages = pl.pages
    except Exception:  # pdfplumber가 못 여는 PDF(일부 청년안심주택 공고)는 표 찾기도 PyMuPDF로 대신한다
        pl, pl_pages = None, None
    with pymupdf.open(path) as mu:
        for i, mp in enumerate(mu, start=1):
            try:
                if pl_pages is not None:
                    found = [t for t in pl_pages[i - 1].find_tables()
                             if len(t.rows) >= 2 and len(t.rows[0].cells) >= 2]
                else:
                    found = [t for t in mp.find_tables().tables if t.row_count >= 2 and t.col_count >= 2]
            except Exception:
                found = []
            areas = [t.bbox for t in found]
            parts = []  # (x0, y0, x1, y1, 내용)
            for bl in mp.get_text("rawdict")["blocks"]:
                if bl.get("type") != 0 or any(inside(bl["bbox"], a) for a in areas):
                    continue
                text = block_text(bl)
                if text.strip():
                    parts.append((*bl["bbox"], text.strip()))
            for t in found:
                try:
                    parts.append((*t.bbox, table_to_markdown(t.extract())))
                except Exception:
                    pass
            text = "\n\n".join(p[4] for p in reading_order(parts, mp.rect.width))
            if len(re.sub(r"\s", "", text)) < MIN_CHARS:
                empty += 1
                continue
            pages.append({"page": i, "text": text, "tables": len(found)})
    if pl is not None:
        pl.close()
    return pages, empty


# 호실별 주택목록 표(주소·호·면적·보증금 목록)는 OCR로 다시 넣지 않는다(팀 결정: 주택목록은 별도 표 데이터)
HOUSE_LIST = re.compile(r"주택목록\s*:|공급대상주택\s*상세현황|\|\s*연번\s*\|[^\n]*(주소|소재지)[^\n]*\|\s*호\s*\|")
# 원본 글자층에 없는 숫자가 있지만 사람이 원본 그림과 대조해 맞다고 확인한 쪽 (문서ID, 쪽): 확인 내용
OCR_MANUAL_OK = {("SH_301759", 1): "표지 '공고일: 2026.02.27.(금)' — 제목이 그림이라 글자층에 없음, 2026-11-24 원본 그림 대조"}


def _numbers(text: str) -> set[str]:
    """금액·날짜·면적 대조용 숫자 값(쉼표 제거, 두 자리 이상). 줄 맨 앞 '01 공급일정' 같은 장 번호는 뺀다."""
    text = re.sub(r"(?m)^\s*0\d(?=\s)", "", text.replace(",", ""))
    return {v for v in re.findall(r"\d+", text) if len(v) >= 2}


def add_ocr(doc_id: str, pages: list[dict], empty: int) -> tuple[list[dict], int, str]:
    """collect/ocr_scans.py가 판독해 둔 data/raw/ocr/<문서ID>.json이 있으면 그 쪽 본문을 판독 결과로 쓴다.
    - scan : 글자가 없던 쪽을 채운다
    - table: 선 없는 표가 흩어진 쪽을 판독 결과(마크다운 표)로 바꾼다.
             단, 판독 결과의 숫자(금액·날짜·면적)가 하나라도 원본 글자층에 없으면 모델이 잘못 읽었을 수 있으므로
             그 쪽은 판독 결과를 쓰지 않고 원래 글자층을 그대로 둔다(예: 소액임차인 기준 날짜를 지어낸 경우).
    - 호실별 주택목록 표는 쓰지 않는다."""
    import html

    path = RAW / "ocr" / f"{doc_id}.json"
    if not path.exists():
        return pages, empty, ""
    ocr = json.loads(path.read_text(encoding="utf-8"))
    by_page = {p["page"]: p for p in pages}
    counts = Counter()
    skipped_pages: list[str] = []
    for o in ocr["pages"]:
        if len(re.sub(r"\s", "", o["text"])) < MIN_CHARS:
            continue
        if HOUSE_LIST.search(o["text"]):
            counts["skipped_house_list"] += 1
            continue
        kind = o.get("kind", "scan")
        if kind == "table" and o["page"] in by_page and (doc_id, o["page"]) not in OCR_MANUAL_OK:
            unverified = _numbers(o["text"]) - _numbers(by_page[o["page"]]["text"])
            if unverified:
                counts["skipped_unverified"] += 1
                orig = re.sub(r"\s", "", by_page[o["page"]]["text"])
                n = max(len(orig), 1)
                hangul = len(re.findall(r"[가-힣]", orig))
                marks = len(re.findall(r"[\[\]().,“”‘’※*:|]", orig))
                if hangul < 0.15 * n and marks > 0.3 * n:
                    # 원래 글자층도 글꼴 정보가 없어 한글이 빠진 깨진 글자뿐이면 읽을 수도 확인할 수도 없으므로 넣지 않는다
                    del by_page[o["page"]]
                    empty += 1
                    skipped_pages.append(f"p{o['page']}(원래 글자층도 깨져 쪽 제외)")
                else:
                    skipped_pages.append(f"p{o['page']}(원래 글자층 유지)")
                continue
        if o["page"] not in by_page:
            empty -= 1
        by_page[o["page"]] = {"page": o["page"], "text": html.unescape(o["text"]), "tables": o["text"].count("\n|---"),
                              "ocr": True}
        counts[kind] += 1
    if not counts:
        return pages, empty, ""
    pages = list(by_page.values())
    parts = []
    if counts["scan"]:
        parts.append(f"스캔 PDF {counts['scan']}쪽")
    if counts["table"]:
        parts.append(f"선 없는 표 {counts['table']}쪽")
    note = (f"{'·'.join(parts)} OCR({ocr['model']}, {ocr['date']}) 판독 — 모델 판독이라 원문과 다를 수 있음"
            if parts else "")
    if counts["skipped_house_list"]:
        note += f"{' / ' if note else ''}주택목록 표 {counts['skipped_house_list']}쪽은 OCR 제외"
    if counts["skipped_unverified"]:
        note += f"{' / ' if note else ''}OCR 숫자가 원본 글자층과 달라 OCR 미사용: {', '.join(skipped_pages)}"
    return sorted(by_page.values(), key=lambda p: p["page"]), empty, note


def convert_pdf(job: tuple[dict, str]) -> tuple[dict, list[dict]]:
    meta, path = job
    try:
        pages, empty = pdf_pages(Path(path))
        pages, empty, note = add_ocr(meta["doc_id"], pages, empty)
    except Exception as e:  # 깨진 PDF 등
        pages, empty, note = [], 0, f"변환 실패: {e}"[:200]
    rows = [{"id": f"{meta['doc_id']}_p{p['page']:03d}", **meta, "page": p["page"],
             "has_table": p["tables"] > 0, "text": p["text"], "_ocr": p.get("ocr", False)} for p in pages]
    return meta | {"pages_with_text": len(pages), "pages_without_text": empty, "note": note}, rows


# 법 전체가 아니라 일부 조문만 쓰는 법령 (v1 보강, 2026-11-24). 관련 없는 조문은 검색 방해꾼이 되므로 주제별로 고른다.
# 값: (첫 조, 끝 조) 범위 또는 조문 번호 묶음
LAW_ARTICLE_RANGE = {
    "LAW_CIVIL": (618, 654),  # 원상회복·중도 해지·보증금: 민법 제3편 제2장 제7절 임대차
    "LAW_REPORT_DECREE": {"제4조의3", "제20조"},  # 임대차 신고: 신고 대상 금액·지역, 과태료
    "LAW_RESIDENT": {"제10조", "제11조", "제16조", "제16조의2", "제18조", "제29조의2", "제40조"},  # 전입신고 + 전입세대확인서·과태료
}


def _keep_article(doc_id: str, article: str) -> bool:
    rule = LAW_ARTICLE_RANGE.get(doc_id)
    if rule is None:
        return True
    if isinstance(rule, tuple):
        return rule[0] <= int(re.match(r"제(\d+)조", article).group(1)) <= rule[1]
    return article in rule


def _range_note(doc_id: str) -> str:
    rule = LAW_ARTICLE_RANGE.get(doc_id)
    if rule is None:
        return ""
    if isinstance(rule, tuple):
        return f" — 제{rule[0]}조~제{rule[1]}조만 사용"
    return " — " + "·".join(sorted(rule, key=lambda a: [int(x) for x in re.findall(r"\d+", a)])) + "만 사용"


def law_rows(meta: dict, html_path: Path) -> list[dict]:
    """법령 본문 HTML을 조문(div.pgroup) 단위로 나눈다. 항·호는 줄바꿈으로 구분하고,
    링크 때문에 끊긴 단어(「민법」, 대통령령 등)는 이어 붙인다. 부칙 머리글 뒤의 조문은 '부칙'으로 표시."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html_path.read_text(encoding="utf-8"), "html.parser")
    headers = [t.sourceline for t in soup.find_all(string=re.compile(r"^\s*부칙\s*<")) if t.parent]
    first_appendix = min((h for h in headers if h), default=None)
    out, counts = [], {"본칙": 0, "부칙": 0}
    for g in soup.find_all("div", class_="pgroup"):
        paras = [re.sub(r"\s+", " ", p.get_text("", strip=False)).strip() for p in g.find_all("p")]
        text = "\n".join(p for p in paras if p) or re.sub(r"\s+", " ", g.get_text("", strip=False)).strip()
        m = re.match(r"(제\d+조(?:의\d+)?)", text)
        if not m:
            continue
        section = "부칙" if first_appendix and g.sourceline and g.sourceline > first_appendix else "본칙"
        counts[section] += 1  # 번호는 삭제 조문까지 세어서 매김 → 조문 id가 법 전체 순서와 계속 맞음
        if re.match(r"제\d+조(?:의\d+)?\s*삭제", text):  # 삭제된 조문은 내용이 없으므로 제외
            continue
        out.append({"id": f"{meta['doc_id']}_{section}_{counts[section]:03d}", **meta, "section": section,
                    "article": m.group(1), "text": text})
    if meta["doc_id"] in LAW_ARTICLE_RANGE:  # 고른 조문만 남김(id 번호는 법 전체 순서 그대로 유지 → 원문 추적 가능)
        out = [r for r in out if r["section"] == "본칙" and _keep_article(meta["doc_id"], r["article"])]
    return out


def collapse_same_text(rows: list[dict]) -> list[dict]:
    """본문(text)이 완전히 같은 줄은 처음 것만 남기고, 나머지 줄의 id를 same_text_ids에 모은다.
    (공고마다 반복되는 서식·유의사항 쪽 때문에 검색 상위가 같은 내용으로 채워지는 것을 막되, 출처 정보는 잃지 않음)"""
    # 대표는 공고일이 가장 최근인 줄(정정공고가 있으면 정정공고). 같은 PDF를 다시 첨부한 게시글은 main에서 파일 비교로 따로 거른다.
    order = lambda r: (r.get("notice_date") or "00000000", r["id"])
    best: dict[str, dict] = {}
    for r in rows:
        if r["text"] not in best or order(r) > order(best[r["text"]]):
            best[r["text"]] = r
    out, keep_of = [], {}
    for r in rows:
        if best[r["text"]] is r:
            r = r | {"same_text_ids": []}
            keep_of[r["text"]] = r
            out.append(r)
    for r in rows:
        if best[r["text"]] is not r:
            keep_of[r["text"]]["same_text_ids"].append(r["id"])
    return out


_KIWI = None
# 띄어쓰기 복원 때 한 단어로 다룰 분야 용어 (kiwipiepy가 "청년 안심 주택", "세 대"처럼 잘못 띄우는 것 방지)
DOMAIN_WORDS = ["청년안심주택", "공공임대주택", "민간임대주택", "공공지원민간임대주택", "공공지원민간임대",
                "공공임대", "민간임대",
                "임대사업자", "임대보증금", "임대주택", "보증보험", "세대", "세대원", "세대주", "세대구성원",
                "확정일자", "전입신고", "행복주택", "매입임대", "전세임대", "장기전세주택", "무주택",
                "예비입주자", "입주자", "임차인", "임대인", "임대차계약", "월임대료", "관리비"]


def _kiwi():
    global _KIWI
    if _KIWI is None:
        from kiwipiepy import Kiwi
        _KIWI = Kiwi()
        for w in DOMAIN_WORDS:
            _KIWI.add_user_word(w, "NNP" if "주택" in w else "NNG")
    return _KIWI


def _space_ratio(s: str):
    h = len(re.findall(r"[가-힣]", s))
    return (s.count(" ") / h) if h >= 20 else None


def restore_spacing(rows: list[dict]) -> list[dict]:
    """원본 PDF부터 띄어쓰기가 빠진 줄·표 칸(한글 20자 이상, 한글 대비 공백 비율 0.08 미만)만 kiwipiepy로
    띄어쓰기를 복원한다. (v1: 쪽 전체가 아니라 줄 단위로 판단 — 일부 문단만 붙어 있는 쪽도 고친다)"""
    out = []
    for r in rows:
        if not any((_space_ratio(l) or 1) < 0.08 for l in re.split(r"[\n|]", r["text"])):
            out.append(r | {"spacing_restored": False})
            continue

        def fix(s: str) -> str:
            rr = _space_ratio(s)
            return _kiwi().space(s, reset_whitespace=False) if rr is not None and rr < 0.08 else s

        lines = []
        for line in r["text"].split("\n"):
            if line.startswith("|") and not re.match(r"^\|(-+\|)+$", line):
                cells = line.strip("|").split("|")
                line = "| " + " | ".join(fix(c.strip()) for c in cells) + " |"
            elif not line.startswith("|"):
                line = fix(line)
            lines.append(line)
        out.append(r | {"text": "\n".join(lines), "spacing_restored": True})
    return out


# ---------- 정제(v1) ----------

CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
PAGE_NO = re.compile(r"^[-–—]?\s*\d{1,3}\s*(/\s*\d{1,3}\s*)?[-–—]?$")           # "- 3 -", "12", "- 3 / 20 -"
FOOTER = re.compile(r"^(\d{1,3}\s*[|IⅠΙ]\s*.{1,40}|.{1,40}\s*[|IⅠΙ]\s*\d{1,3})$")  # "60 | 서울주택도시개발공사", "114 Ι 115"
# 다운로드 워터마크: 경고 문구가 들어 있는 줄은 통째로 지운다
# (글자 위치로 줄을 다시 짜면 IP·해시·시각·경고 문구가 한 줄로 합쳐지므로 줄 전체가 아니라 '들어 있는지'로 찾음)
WATERMARK = re.compile(r"공사\s*승인\s*없이\s*무단\s*유출\s*시\s*법적\s*책임")
WATERMARK_TAIL = re.compile(r"^([0-9A-F]{16,}|[\d.\-: ]+)$")                    # 워터마크 뒤 IP·해시·시각
BULLET_ONLY = re.compile(r"^[•●○◯◦■□▪▶▷►※*\-–·ㆍŸ]+$")     # 글머리표만 있는 줄
CLOSE_ONLY = re.compile(r"^[)\]}”’」』.,:;·]+$")                                 # 닫는 괄호·문장부호만 있는 줄
OPEN_ONLY = re.compile(r"^[(\[{“‘「『]+$")                                        # 여는 괄호만 있는 줄
LIST_START = re.compile(
    r"^([•●○◯◦■□▪▶▷►※*\-–·ㆍŸ<\[]|[①-⑳㉠-㉭]|\(?\d{1,2}[.)]|\(?[가-하][.)]|[ⅰ-ⅹⅠ-Ⅹ][.)\s]"
    r"|제\d+[조항호장절]|\[?별[표첨지]|붙임|Q\d|A\d|[A-Z]\d?[.)])")
SENTENCE_END = re.compile(r"([다요음함됨임것]\.|[.!?:])$")
POLITE_END = re.compile(r"(니다|시오|세요)$")                                   # 마침표 없이 끝난 문장
POLITE_GLUED = re.compile(r"(니다|시오|세요)(?=[가-힣])(?!만)")                 # "합니다신혼부부형의" → 띄움
# 글자마다 띄운 제목: 4자 이상은 어디서든("신 청 자 격"), 3자는 줄·표 칸 전체가 그것뿐일 때만("| 주 택 형 |")
# ("이 중 한 곳"처럼 한 글자 단어가 이어지는 정상 문장은 건드리지 않기 위해)
LETTER_SPACED = re.compile(r"(?<![가-힣0-9])(?:[가-힣] ){3,}[가-힣](?![가-힣])")
LETTER_SPACED3 = re.compile(r"(?m)(^|\| |\d\. )((?:[가-힣] ){2}[가-힣])(?=[ \t]*(?:\||$))")
# 혼자 쓰이는 한 글자 단어가 끼어 있으면 정상 문장으로 보고 붙이지 않는다 ("제53조 각 호 및", "몇 번 쓸 수", "두 명 중 한")
SINGLE_WORDS = set("및각호조제중한두세수더번몇그외등것때곳개명쓸살할또는이저그새옛안못잘좀다꼭늘곧")


def _join_letters(s: str) -> str:
    chars = s.split(" ")
    return s if any(c in SINGLE_WORDS for c in chars) else "".join(chars)
WIDE_GAP = re.compile(r"(?<![가-힣])([가-힣]) {2,}(?=[가-힣])")                   # "무  주택" → 무주택(양쪽 정렬 빈칸)
FRAG_RUN = 4          # 흩어진 조각 덩어리가 이만큼 이어지면 표 조각으로 보고 뺀다
FRAG_LINES = 8        # 또는 이어진 조각 덩어리들의 원래 줄 수가 이만큼이면 뺀다(숫자만 여러 줄인 덩어리)
_HANGUL = re.compile(r"[가-힣]")


def _is_fragment(block: str) -> bool:
    """흩어진 표 칸 하나로 보이는 덩어리: 6자 이하로 아주 짧거나, 40자 이하인데 숫자·기호가 절반 이상인 것.
    (한글 7자 이상인 짧은 문장·제목은 조각으로 보지 않는다. 예: 상담사례집 질문 제목)"""
    s = re.sub(r"\s", "", block)
    if not s or block.startswith("|"):
        return False
    if LIST_START.match(block.strip()) and len(_HANGUL.findall(block)) >= 2:  # "2. 수원지부" 같은 번호 목록은 조각 아님
        return False
    digits = len(re.findall(r"[\d,.()+\-~%㎡/]", s))
    hangul = len(_HANGUL.findall(s))
    # 숫자 위주라도 한글이 4자 이상이면 내용(주소·전화번호 등)으로 본다
    return len(s) <= 6 or (len(s) <= 40 and digits / len(s) >= 0.5 and hangul <= 3)


def _space_at_joins(pairs: list[tuple[str, str]]) -> list[bool]:
    """줄 끝과 다음 줄 시작을 붙였을 때 그 자리에 띄어쓰기가 필요한지 kiwipiepy로 판단한다.
    앞뒤 12글자만 넣어 보고, 붙인 자리에 공백이 생기면 띄어 쓴다(다른 자리는 건드리지 않음)."""
    if not pairs:
        return []
    windows = [a[-12:] + b[:12] for a, b in pairs]
    out = []
    for (a, _), spaced in zip(pairs, _kiwi().space(windows, reset_whitespace=False)):
        k = len(a[-12:])
        i = seen = 0
        while i < len(spaced) and seen < k:  # 원래 글자 k개를 지난 위치 찾기(공백만 끼어들 수 있음)
            if spaced[i] != " " or a[-12:][seen] == " ":
                seen += 1
            i += 1
        out.append(i < len(spaced) and spaced[i] == " ")
    return out


def _join_lines(lines: list[str], stats: dict) -> str:
    """한 덩어리(문단) 안의 줄을 정리해 끊긴 문장을 잇는다."""
    # 1) 글머리표·괄호만 있는 줄을 앞뒤 줄에 붙인다
    merged: list[str] = []
    carry = ""
    for line in lines:
        if BULLET_ONLY.match(line) or OPEN_ONLY.match(line):
            carry += line + ("" if OPEN_ONLY.match(line) else " ")
            continue
        if CLOSE_ONLY.match(line) and merged:
            merged[-1] += line
            continue
        merged.append(carry + line)
        carry = ""
    if carry.strip() and merged:           # 덩어리 끝에 남은 글머리표는 버린다
        stats["bullet_dropped"] += 1
    # 2) 줄 사이를 이을지 정한다
    decide, ask = [], []
    for a, b in zip(merged, merged[1:]):
        if LIST_START.match(b) or (SENTENCE_END.search(a) and len(a) > 4):
            decide.append("\n")
        elif POLITE_END.search(a):
            decide.append(" ")
        elif _HANGUL.match(a[-1]) and _HANGUL.match(b[0]):
            decide.append(None)
            ask.append((a, b))
        elif a[-1] in "([{“‘「『" or b[0] in ")]}”’」』.,:;%":
            decide.append("")
        else:
            decide.append(" ")
    answers = iter(_space_at_joins(ask))
    out = merged[0] if merged else ""
    for sep, b in zip(decide, merged[1:]):
        if sep is None:
            sep = " " if next(answers) else ""
        if sep != "\n":
            stats["lines_joined"] += 1
        out += sep + b
    return out


def _clean_table(block: str, stats: dict) -> str:
    """마크다운 표에서 빈 행·빈 열을 지운다. 한 열만 남으면 일반 줄로 바꾼다."""
    rows = [[re.sub(r" {2,}", " ", c.strip()) for c in l.strip().strip("|").split("|")] for l in block.split("\n")
            if l.strip() and not re.match(r"^\|(\s*-+\s*\|)+$", l.strip())]
    kept = [r for r in rows if any(r)]
    stats["table_rows_dropped"] += len(rows) - len(kept)
    if not kept:
        return ""
    width = max(len(r) for r in kept)
    kept = [r + [""] * (width - len(r)) for r in kept]
    cols = [j for j in range(width) if any(r[j] for r in kept)]
    stats["table_cols_dropped"] += width - len(cols)
    kept = [[r[j] for j in cols] for r in kept]
    # 표 칸 전체가 글자마다 띄운 머리글이면 붙인다 ("전 화 번 호" → 전화번호). 칸 안이라 문장과 섞일 걱정이 없다.
    kept = [[re.sub(r" ", "", c) if re.fullmatch(r"(?:[가-힣] )+[가-힣]", c) else c for c in r] for r in kept]
    if len(cols) == 1:
        return "\n".join(r[0] for r in kept)
    lines = ["| " + " | ".join(kept[0]) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(r) + " |" for r in kept[1:]]
    return "\n".join(lines)


def clean_text(text: str, stats: dict, keep_fragments: bool = False) -> str:
    """쪽 하나의 본문을 정제한다(제어 글자·쪽 번호·꼬리말·워터마크 제거, 끊긴 줄 잇기, 표 정리, 흩어진 조각 제거).
    keep_fragments=True(OCR 판독 쪽): 흐름도·일정표의 짧은 칸(날짜 등)이 정상 내용이므로 조각 제거를 하지 않는다."""
    text = CONTROL.sub(" ", text)
    blocks = [b for b in text.split("\n\n") if b.strip()]
    out: list[tuple[str, bool, bool, int]] = []  # (내용, 표인가, 흩어진 조각인가, 원래 줄 수)
    for bi, block in enumerate(blocks):
        if block.startswith("|"):
            t = _clean_table(block, stats)
            if t:
                out.append((t, t.startswith("|"), False, 0))
            continue
        lines = [l.strip() for l in block.split("\n")]
        kept, skip_tail = [], False
        for li, line in enumerate(lines):
            if not line:
                continue
            edge = (bi == 0 and li == 0) or (bi == len(blocks) - 1 and li == len(lines) - 1)
            if WATERMARK.search(line):
                stats["watermark_lines"] += 1
                skip_tail = True
                continue
            if skip_tail and WATERMARK_TAIL.match(line):
                stats["watermark_lines"] += 1
                continue
            skip_tail = False
            if PAGE_NO.match(line) and (edge or not line.isdigit()):
                stats["page_no_lines"] += 1
                continue
            if edge and FOOTER.match(line):
                stats["footer_lines"] += 1
                continue
            kept.append(line)
        if kept:
            joined = _join_lines(kept, stats)
            # 조각인지는 줄을 잇기 전 모양으로 판단: 줄 대부분이 6자 이하이거나, 이은 뒤에도 짧은·숫자 위주 덩어리
            short = sum(_is_fragment(l) for l in kept)
            longest_normal = max((len(l) for l in kept if not _is_fragment(l)), default=0)
            frag = not keep_fragments and ((len(kept) >= 3 and short / len(kept) >= 0.6 and longest_normal <= 20)
                                           or _is_fragment(joined))
            out.append((joined, False, frag, len(kept)))
    # 흩어진 표 조각: 조각 덩어리가 FRAG_RUN개 이상 이어지거나, 조각 덩어리들의 원래 줄이 FRAG_LINES줄 이상이면 뺀다
    final: list[tuple[str, bool]] = []
    run: list[tuple[str, bool]] = []
    run_lines = 0
    for item in out + [None]:
        if item is not None and item[2]:
            run.append(item[:2])
            run_lines += item[3]
            continue
        if len(run) >= FRAG_RUN or run_lines >= FRAG_LINES:
            stats["fragment_blocks"] += len(run)
            stats["fragment_runs"] += 1
        else:
            final += run
        run, run_lines = [], 0
        if item is not None:
            final.append(item[:2])
    # 빈 줄(덩어리 경계)을 넘어 끊긴 문장 잇기: 앞 덩어리가 길고 문장 끝이 아니며, 다음 덩어리가 한글로 이어질 때
    merged: list[tuple[str, bool]] = []
    ask = []
    for text_, is_table in final:
        text_ = text_.strip()
        if not text_:
            continue
        prev = merged[-1] if merged else None
        if (prev and not prev[1] and not is_table and len(prev[0]) >= 25
                and (_HANGUL.match(prev[0][-1]) or prev[0][-1] == ",") and not SENTENCE_END.search(prev[0])
                and _HANGUL.match(text_[0]) and not LIST_START.match(text_)):
            if _HANGUL.match(prev[0][-1]):
                ask.append((prev[0].split("\n")[-1].replace("\x00", " "), text_.split("\n")[0]))
            merged[-1] = (prev[0] + "\x00" + text_, False)  # 자리 표시, 아래에서 띄어쓰기를 정해 바꿈
            continue
        merged.append((text_, is_table))
    answers = iter(_space_at_joins(ask))
    result = []
    for text_, _ in merged:
        parts = text_.split("\x00")
        s = parts[0]
        for p in parts[1:]:
            stats["blocks_joined"] += 1
            if _HANGUL.match(s[-1]):
                s += (" " if next(answers) else "") + p
            else:
                s += " " + p
        result.append(s)
    out_text = "\n\n".join(b for b in result if b.strip())
    stats["polite_end_spaced"] += len(POLITE_GLUED.findall(out_text))
    out_text = POLITE_GLUED.sub(r"\1 ", out_text)
    stats["letter_spaced_joined"] += len(LETTER_SPACED.findall(out_text)) + len(WIDE_GAP.findall(out_text))
    out_text = LETTER_SPACED.sub(lambda m: _join_letters(m.group(0)), out_text)
    out_text = LETTER_SPACED3.sub(lambda m: m.group(1) + _join_letters(m.group(2)), out_text)
    return WIDE_GAP.sub(r"\1", out_text)


def dedupe_boilerplate(rows: list[dict], stats: dict, min_len: int = 40, min_docs: int = 3) -> list[dict]:
    """여러 공고(min_docs개 이상)에 똑같이 나오는 문단은 처음 나온 곳에만 남긴다. 한 공고 안의 반복은 건드리지 않는다."""
    docs_of: dict[str, set] = {}
    for r in rows:
        for p in r["text"].split("\n\n"):
            if len(p) >= min_len:
                docs_of.setdefault(p, set()).add(r["doc_id"])
    common = {p for p, d in docs_of.items() if len(d) >= min_docs}
    first_doc: dict[str, str] = {}
    out = []
    for r in rows:
        paras = []
        for p in r["text"].split("\n\n"):
            if p in common:
                owner = first_doc.setdefault(p, r["doc_id"])
                if owner != r["doc_id"]:
                    stats["boilerplate_removed"] += 1
                    continue
            paras.append(p)
        out.append(r | {"text": "\n\n".join(paras)})
    stats["boilerplate_kinds"] += len(common)
    return out


def drop_running_headers(rows: list[dict], stats: dict, max_len: int = 120, min_len: int = 10,
                         min_pages: int = 5) -> list[dict]:
    """가이드 PDF: 같은 문서 안에서 min_pages쪽 이상 똑같이 나오는 문단(쪽마다 찍힌 책 제목 머리말 등)을 지운다.
    "분쟁내용" 같은 짧은 소제목(min_len자 미만)은 건드리지 않는다."""
    key = lambda p: re.sub(r"\d+", "#", p.split("\n")[0])  # 쪽 번호가 달라도 같은 머리말로 본다("54 만화로 보는 …")
    pages_of: dict[tuple, set] = {}
    for r in rows:
        if "page" not in r:
            continue
        for p in r["text"].split("\n\n"):
            if min_len <= len(p.split("\n")[0]) <= max_len:
                pages_of.setdefault((r["doc_id"], key(p)), set()).add(r["page"])
    heads = {k for k, v in pages_of.items() if len(v) >= min_pages}
    out = []
    for r in rows:
        if "page" in r:
            paras = []
            for p in r["text"].split("\n\n"):
                if (r["doc_id"], key(p)) in heads:  # 머리말 줄만 빼고, 같은 덩어리에 붙은 본문 줄은 남긴다
                    rest = "\n".join(p.split("\n")[1:]).strip()
                    if rest:
                        paras.append(rest)
                else:
                    paras.append(p)
            stats["running_headers"] += r["text"].count("\n\n") + 1 - len(paras)
            r = r | {"text": "\n\n".join(paras)}
        out.append(r)
    return out


HOUSE_TYPE = {"매입임대주택": "매입임대", "영구임대주택": "영구임대"}
OUT_FIELDS = ["id", "doc_id", "source", "category", "title", "house_type", "notice_date", "source_url",
              "page", "article", "text", "same_text_ids"]


def to_output(r: dict) -> dict:
    """출력 칸만 남긴다(page·article은 해당하는 줄에만)."""
    r = r | {"house_type": HOUSE_TYPE.get(r.get("house_type", ""), r.get("house_type", ""))}
    return {k: r[k] for k in OUT_FIELDS if k in r}


def read_manifest(path: Path) -> list[dict]:
    return list(csv.DictReader(open(path, encoding="utf-8-sig")))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    groups = {"공공임대_공고문": [], "청년안심주택_공고문": [], "일반전월세_법령가이드": []}
    jobs = []  # (그룹, (메타, 경로))

    for r in read_manifest(RAW / "notices" / "manifest.csv"):
        meta = {"doc_id": r["doc_id"], "source": r["source"], "category": "공공임대 모집공고",
                "title": r["title"], "house_type": r["house_type"], "notice_date": r["notice_date"],
                "source_url": r["page_url"]}
        path = RAW / "notices" / r["source"].lower() / f"{r['doc_id']}.pdf"
        if path.exists():
            jobs.append(("공공임대_공고문", (meta, str(path))))
        else:
            groups["공공임대_공고문"].append((meta | {"pages_with_text": 0, "pages_without_text": 0,
                                                    "note": r.get("note", "")}, []))

    for r in read_manifest(RAW / "youth" / "manifest.csv"):
        meta = {"doc_id": r["doc_id"], "source": "청년안심주택", "category": f"청년안심주택 {r['kind']}",
                "title": r["title"], "house_type": r["kind"], "notice_date": r["notice_date"],
                "source_url": r["page_url"]}
        jobs.append(("청년안심주택_공고문", (meta, str(RAW / "youth" / f"{r['doc_id']}.pdf"))))

    for r in read_manifest(RAW / "rental" / "manifest.csv"):
        meta = {"doc_id": r["doc_id"], "source": r["publisher"], "category": f"일반 전월세 {r['category']}",
                "title": r["title"], "house_type": "", "notice_date": "", "source_url": r["page_url"]}
        if r["category"] == "법령":
            rows = law_rows(meta, (RAW / "rental" / r["file"]).with_suffix(".html"))
            groups["일반전월세_법령가이드"].append((meta | {"pages_with_text": len(rows), "pages_without_text": 0,
                                                       "note": f"본칙 조문 {len(rows)}개(쪽 대신 조문 단위, 부칙 제외)"
                                                       + _range_note(r["doc_id"])},
                                                rows))
        elif r["file"]:
            jobs.append(("일반전월세_법령가이드", (meta, str(RAW / "rental" / r["file"]))))
        else:
            groups["일반전월세_법령가이드"].append((meta | {"pages_with_text": 0, "pages_without_text": 0,
                                                       "note": r["note"]}, []))

    # 파일 내용이 똑같은 PDF(같은 공고문을 "접수 결과 안내" 게시글 등에 다시 첨부)는 공고일이 가장 이른 공고 하나만 변환한다.
    # 나머지는 줄 없이 문서목록에 "같은 PDF: <대표 ID>"로 남겨 추적할 수 있게 한다. (OCR 결과가 실행마다 조금 달라
    # 같은 본문 합치기로는 못 잡는 이중 입력을 막기 위함)
    import hashlib

    first_of: dict[str, dict] = {}
    for group, (meta, path) in sorted(jobs, key=lambda j: (j[1][0]["notice_date"] or "99999999", j[1][0]["doc_id"])):
        first_of.setdefault(hashlib.md5(Path(path).read_bytes()).hexdigest(), meta)
    kept_jobs = []
    for group, (meta, path) in jobs:
        keeper = first_of[hashlib.md5(Path(path).read_bytes()).hexdigest()]
        if keeper is meta:
            kept_jobs.append((group, (meta, path)))
        else:
            groups[group].append((meta | {"pages_with_text": 0, "pages_without_text": 0,
                                          "note": f"같은 PDF: {keeper['doc_id']}에 포함(그 문서로 인용)"}, []))
    jobs = kept_jobs

    with ProcessPoolExecutor(max_workers=4) as pool:
        for n, ((group, _), result) in enumerate(zip(jobs, pool.map(convert_pdf, [j for _, j in jobs])), start=1):
            groups[group].append(result)
            if n % 25 == 0 or n == len(jobs):
                print(f"PDF {n}/{len(jobs)} 변환")

    manifest = []
    for group, items in groups.items():
        all_rows = []
        for meta, rows in items:
            manifest.append({"group": group, **meta})
            all_rows += rows
        stats = Counter()
        # 법령 조문은 HTML에서 깔끔하게 뽑았으므로 제어 글자만 지우고, PDF에서 온 쪽만 정제한다
        cleaned = [r | {"text": CONTROL.sub(" ", r["text"]).strip()} if "article" in r
                   else r | {"text": clean_text(r["text"], stats, keep_fragments=r.get("_ocr", False))}
                   for r in all_rows]
        kept = collapse_same_text(restore_spacing(cleaned))
        # 정제 자체로 글자가 거의 없어진 쪽(표지·MEMO·흩어진 표만 있던 쪽)은 뺀다
        nchar = lambda r: len(re.sub(r"\s", "", r["text"]))
        short = [r for r in kept if nchar(r) < MIN_CHARS]
        kept = [r for r in kept if nchar(r) >= MIN_CHARS]
        if group != "일반전월세_법령가이드":
            kept = dedupe_boilerplate(kept, stats)
        else:  # 가이드 책의 쪽마다 찍힌 머리말 (공고문은 반복되는 공고 제목이 문맥이라 그대로 둠)
            kept = drop_running_headers(kept, stats)
        # 공통 안내문·머리말을 지워 짧아진 쪽은 남은 내용(예: "공급대상 주택 : 총 119호")이 그 공고만의 정보라 남긴다
        emptied = [r for r in kept if nchar(r) < 8]
        stats["pages_emptied_by_dedupe"] += len(emptied)
        kept = [r for r in kept if nchar(r) >= 8]
        short += emptied
        with open(OUT / f"{group}.jsonl", "w", encoding="utf-8") as f:
            for row in kept:
                f.write(json.dumps(to_output(row), ensure_ascii=False) + "\n")
        rows_per_doc = Counter(r["doc_id"] for r in kept)
        for m in manifest:
            if m["group"] == group:
                m["rows_in_jsonl"] = rows_per_doc.get(m["doc_id"], 0)
        size = (OUT / f"{group}.jsonl").stat().st_size
        print(f"{group}.jsonl → 문서 {len(items)}개, 줄 {len(kept):,}개, {size / 1e6:.1f}MB")
        print(f"  같은 본문 합침 {len(all_rows) - len(kept) - len(short):,}쪽, 정제 뒤 글자가 거의 없어 뺀 쪽 {len(short):,}")
        print(f"  정제: {dict(sorted(stats.items()))}")

    fields = ["group", "doc_id", "source", "category", "title", "house_type", "notice_date",
              "pages_with_text", "pages_without_text", "rows_in_jsonl", "note", "source_url"]
    with open(OUT / "문서목록.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(manifest)
    print("문서목록.csv →", len(manifest), "개 문서")


if __name__ == "__main__":
    main()
