"""수집한 PDF·법령을 RAG용 JSONL로 변환해 팀 공유 폴더(shared/docs)에 저장한다.

- 공고문·가이드 PDF: 한 줄 = 한 쪽. 글자는 PyMuPDF, 표는 pdfplumber로 찾아 마크다운 표로 바꾸고
  쪽 안의 위아래 순서대로 합친다. 글자가 거의 없는 쪽(도면·사진)은 빼고 문서 목록에 쪽수만 남긴다.
- 법령: 한 줄 = 한 조문 (본칙/부칙 구분).
- 문서목록.csv: 모든 문서의 출처·공고일·쪽수 (엑셀용 UTF-8 BOM).

실행: uv run python -m rag_home_api.collect.export_jsonl
"""

import csv
import json
import re
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
            parts = []  # (위치 y, 내용)
            for x0, y0, x1, y1, text, *_rest, btype in mp.get_text("blocks"):
                if btype == 0 and text.strip() and not any(inside((x0, y0, x1, y1), a) for a in areas):
                    parts.append((y0, text.strip()))
            for t in found:
                try:
                    parts.append((t.bbox[1], table_to_markdown(t.extract())))
                except Exception:
                    pass
            text = "\n\n".join(p for _, p in sorted(parts, key=lambda p: p[0]))
            if len(re.sub(r"\s", "", text)) < MIN_CHARS:
                empty += 1
                continue
            pages.append({"page": i, "text": text, "tables": len(found)})
    if pl is not None:
        pl.close()
    return pages, empty


def convert_pdf(job: tuple[dict, str]) -> tuple[dict, list[dict]]:
    meta, path = job
    try:
        pages, empty = pdf_pages(Path(path))
        note = ""
    except Exception as e:  # 깨진 PDF 등
        pages, empty, note = [], 0, f"변환 실패: {e}"[:200]
    rows = [{"id": f"{meta['doc_id']}_p{p['page']:03d}", **meta, "page": p["page"],
             "has_table": p["tables"] > 0, "text": p["text"]} for p in pages]
    return meta | {"pages_with_text": len(pages), "pages_without_text": empty, "note": note}, rows


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
        counts[section] += 1
        out.append({"id": f"{meta['doc_id']}_{section}_{counts[section]:03d}", **meta, "section": section,
                    "article": m.group(1), "text": text})
    return out


def collapse_same_text(rows: list[dict]) -> list[dict]:
    """본문(text)이 완전히 같은 줄은 처음 것만 남기고, 나머지 줄의 id를 same_text_ids에 모은다.
    (공고마다 반복되는 서식·유의사항 쪽 때문에 검색 상위가 같은 내용으로 채워지는 것을 막되, 출처 정보는 잃지 않음)"""
    first: dict[str, dict] = {}
    out = []
    for r in rows:
        keep = first.get(r["text"])
        if keep is None:
            r = r | {"same_text_ids": []}
            first[r["text"]] = r
            out.append(r)
        else:
            keep["same_text_ids"].append(r["id"])
    return out


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
                                                       "note": f"본칙 조문 {len(rows)}개(쪽 대신 조문 단위, 부칙 제외)"}, rows))
        elif r["file"]:
            jobs.append(("일반전월세_법령가이드", (meta, str(RAW / "rental" / r["file"]))))
        else:
            groups["일반전월세_법령가이드"].append((meta | {"pages_with_text": 0, "pages_without_text": 0,
                                                       "note": r["note"]}, []))

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
        kept = collapse_same_text(all_rows)
        with open(OUT / f"{group}.jsonl", "w", encoding="utf-8") as f:
            for row in kept:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        size = (OUT / f"{group}.jsonl").stat().st_size
        print(f"{group}.jsonl → 문서 {len(items)}개, 줄 {len(kept):,}개(같은 본문 {len(all_rows) - len(kept):,}줄 합침), {size / 1e6:.1f}MB")

    fields = ["group", "doc_id", "source", "category", "title", "house_type", "notice_date",
              "pages_with_text", "pages_without_text", "note", "source_url"]
    with open(OUT / "문서목록.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(manifest)
    print("문서목록.csv →", len(manifest), "개 문서")


if __name__ == "__main__":
    main()
