"""PDF에서 글자로 뽑을 수 없는 쪽을 OpenAI 모델로 판독(OCR)해 저장한다. 두 종류:

1) scan  : 스캔 PDF — 모든 쪽에 뽑을 글자가 없고 그림이 있거나 글자를 선으로 그린 문서
2) table : 선 없는 표 — 글자는 있지만 표 칸이 흩어져 정제(clean_text)에서 조각이 TABLE_FRAG개 이상 빠지는 쪽
           (예: LH 행복주택 "임대대상 및 임대조건" 표의 보증금·임대료·면적)
- 쪽마다 200dpi 그림으로 바꿔 모델에 넣고, 그림의 글자를 그대로 옮겨 적게 한다(추측·요약 금지, 못 읽는 글자는 [?]).
- 결과: data/raw/ocr/<문서ID>.json (깃 제외). 이미 판독한 쪽은 다시 부르지 않는다.
- export_jsonl이 이 결과로 그 쪽의 본문을 바꾼다. 모델 판독이라 틀릴 수 있으므로 문서목록·데이터목록에 OCR로 표시한다.

실행: uv run python -m rag_home_api.collect.ocr_scans [--doc-ids SH_313268 ...] [--no-tables] [--dry-run]
"""

import argparse
import base64
import html
import json
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import date

import pymupdf

from rag_home_api.config import DATA_DIR, settings

RAW = DATA_DIR / "raw"
OCR_DIR = RAW / "ocr"
MIN_CHARS = 30
TABLE_FRAG = 8
DPI = 200
PROMPT_VERSION = "ocr-v2"
PROMPT = ("이 그림은 한국 공공임대주택 모집공고문(또는 주택임대차 안내서)의 한 쪽입니다. 그림에 적힌 글자를 빠짐없이 그대로 옮겨 적으세요. "
          "추측하거나 고치거나 요약하지 말고, 읽을 수 없는 글자는 [?]로 적으세요. "
          "표는 마크다운 표(| 칸 | 칸 |)로, 그 밖의 글은 원래 줄 순서대로 적으세요. "
          "표의 머리글이 여러 줄이면 위 머리글과 아래 머리글을 '-'로 이어 한 줄로 적으세요(예: 임대보증금(천원)-계약금). "
          "여러 행에 걸친(병합된) 칸은 해당하는 행마다 같은 값을 적으세요. "
          "장식용 선(━, ─), 쪽 번호, 다운로드 워터마크(IP 주소·다운로드 시각·'무단유출' 경고 문구), 설명 문장은 적지 마세요.")


def pdf_paths():
    """수집한 PDF 전체 (문서ID, 경로)."""
    for p in sorted((RAW / "notices").glob("*/*.pdf")):
        yield p.stem, p
    for p in sorted((RAW / "youth").glob("*.pdf")):
        yield p.stem, p
    for p in sorted((RAW / "rental" / "guides").glob("*.pdf")):
        yield p.stem, p


def is_scanned(path) -> bool:
    """모든 쪽에 뽑을 수 있는 글자가 없고, 그림이 있거나 글자를 선으로 그린(도형이 많은) 문서."""
    try:
        with pymupdf.open(path) as d:
            return len(d) > 0 and all(len(re.sub(r"\s", "", p.get_text())) < MIN_CHARS
                                      and (p.get_images() or len(p.get_drawings()) >= 100) for p in d)
    except Exception:
        return False


def tidy(text: str) -> str:
    """모델 출력 정리: 장식선 줄 제거, 표 칸 안 <br>은 띄어쓰기로, &lt; 같은 HTML 기호는 원래 글자로."""
    lines = [html.unescape(l.replace("<br>", " ")).rstrip() for l in text.split("\n")]
    drop = lambda l: (re.fullmatch(r"[\s━─═—_\-]{3,}", l) or "무단유출" in l
                      or re.fullmatch(r"\s*\d{4}-\d{2}-\d{2} \d{2}:\d{2}(:\d{2})?\s*", l))
    return "\n".join(l for l in lines if not drop(l)).strip()


def table_pages(path) -> list[int]:
    """선 없는 표 때문에 정제에서 흩어진 조각이 TABLE_FRAG개 이상 빠지는 쪽 번호."""
    from collections import Counter

    from rag_home_api.collect.export_jsonl import clean_text, pdf_pages
    try:
        pages, _ = pdf_pages(path)
    except Exception:
        return []
    out = []
    for p in pages:
        st = Counter()
        clean_text(p["text"], st)
        if st["fragment_blocks"] >= TABLE_FRAG:
            out.append(p["page"])
    return out


def _find(job):
    doc_id, path, want_tables = job
    if is_scanned(path):
        return doc_id, path, "scan", list(range(1, pymupdf.open(path).page_count + 1))
    if want_tables:
        pages = table_pages(path)
        if pages:
            return doc_id, path, "table", pages
    return None


def ocr_page(client, path, page_no: int) -> dict:
    with pymupdf.open(path) as d:
        png = d[page_no - 1].get_pixmap(dpi=DPI).tobytes("png")
    resp = client.responses.create(
        model=settings.llm_model,
        input=[{"role": "user", "content": [
            {"type": "input_text", "text": PROMPT},
            {"type": "input_image", "image_url": "data:image/png;base64," + base64.b64encode(png).decode()},
        ]}],
    )
    return {"page": page_no, "text": tidy(resp.output_text),
            "tokens": [resp.usage.input_tokens, resp.usage.output_tokens]}


def main():
    from concurrent.futures import ProcessPoolExecutor

    ap = argparse.ArgumentParser()
    ap.add_argument("--doc-ids", nargs="*", help="이 문서만 (안 주면 전체에서 자동으로 찾음)")
    ap.add_argument("--no-tables", action="store_true", help="스캔 PDF만, 선 없는 표 쪽은 찾지 않음")
    ap.add_argument("--dry-run", action="store_true", help="대상만 보여 주고 판독하지 않음")
    a = ap.parse_args()

    jobs = [(d, p, not a.no_tables) for d, p in pdf_paths() if not a.doc_ids or d in a.doc_ids]
    with ProcessPoolExecutor(max_workers=4) as pool:
        targets = [t for t in pool.map(_find, jobs) if t]
    for kind in ["scan", "table"]:
        sel = [(d, len(pg)) for d, _, k, pg in targets if k == kind]
        print(f"[{kind}] 문서 {len(sel)}건, 쪽 {sum(n for _, n in sel)}:", sel[:20])
    if a.dry_run or not targets:
        return

    from openai import OpenAI
    client = OpenAI(api_key=settings.openai_api_key)
    OCR_DIR.mkdir(parents=True, exist_ok=True)
    # 모든 문서의 판독할 쪽을 한꺼번에 병렬로 부른다
    saved_of, todo = {}, []
    for doc_id, path, kind, want in targets:
        out = OCR_DIR / f"{doc_id}.json"
        saved_of[doc_id] = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {"pages": []}
        done = {p["page"] for p in saved_of[doc_id]["pages"]}
        todo += [(doc_id, path, kind, n) for n in want if n not in done]
    print(f"새로 판독할 쪽 {len(todo)}")

    def run(job):
        doc_id, path, kind, n = job
        try:
            return doc_id, ocr_page(client, path, n) | {"kind": kind, "prompt_version": PROMPT_VERSION}
        except Exception as e:  # 실패한 쪽은 저장하지 않음 → 다음 실행 때 다시 시도
            print(f"  실패 {doc_id} p{n}: {e}"[:200])
            return doc_id, None

    new_of: dict[str, list] = {}
    with ThreadPoolExecutor(max_workers=8) as pool:
        for i, (doc_id, r) in enumerate(pool.map(run, todo), start=1):
            if r:
                new_of.setdefault(doc_id, []).append(r)
            if i % 20 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)}")
    total = [0, 0]
    for doc_id, path, kind, want in targets:
        out = OCR_DIR / f"{doc_id}.json"
        saved, new = saved_of[doc_id], new_of.get(doc_id, [])
        if not new:
            continue
        pages = sorted(saved["pages"] + new, key=lambda p: p["page"])
        out.write_text(json.dumps({
            "doc_id": doc_id, "source_pdf": str(path.relative_to(RAW)), "model": settings.llm_model,
            "dpi": DPI, "date": str(date.today()), "pages": pages,
        }, ensure_ascii=False, indent=1), encoding="utf-8")
        for p in new:
            total[0] += p["tokens"][0]; total[1] += p["tokens"][1]
        print(f"{doc_id} [{kind}]: {len(pages)}쪽 보관 (새로 {len(new)}쪽)")
    print(f"새로 판독한 토큰: 입력 {total[0]:,} / 출력 {total[1]:,}")


if __name__ == "__main__":
    main()
