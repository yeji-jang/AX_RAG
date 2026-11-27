"""서울 공공임대 모집공고문 PDF 수집 (정적 크롤링).

- LH: 공공데이터포털 LH 공고 API 목록 → 공고 상세(DTL_URL) → 첨부 공고문 PDF
- SH: 서울주거포털 SH 공공임대 목록 → SH 누리집 상세 → 첨부 공고문 PDF

결과: data/raw/notices/{lh,sh}/<문서ID>.pdf, 문서 목록 data/raw/notices/manifest.csv
실행: uv run python -m rag_home_api.collect.notices
"""

import csv
import json
import re
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import requests
from bs4 import BeautifulSoup

from rag_home_api.collect.public_api import fetch_lh
from rag_home_api.config import DATA_DIR

OUT = DATA_DIR / "raw" / "notices"
HEADERS = {"User-Agent": "Mozilla/5.0 (AX_RAG education project)"}
MAX_BYTES = 60 * 1024 * 1024  # 이보다 큰 첨부(도면·팸플릿 위주)는 건너뜀
DELAY = 1.0  # 요청 간격(초)

LH_FILE_URL = "https://apply.lh.or.kr/lhapply/lhFile.do?fileid={}"
SH_LIST_URL = "https://housing.seoul.go.kr/site/main/sh/publicLease/list?cp={}&supplyType=publicLease"
SH_FILE_URL = "https://www.i-sh.co.kr/main/com/file/innoFD.do?brdId={brdId}&seq={seq}&fileTp={fileTp}&fileSeq={fileSeq}"

session = requests.Session()
session.headers.update(HEADERS)


def get(url: str, **kw) -> requests.Response:
    time.sleep(DELAY)
    resp = session.get(url, timeout=60, **kw)
    resp.raise_for_status()
    return resp


def pick_notice_pdf(names: list[str]) -> int | None:
    """첨부 이름 목록에서 '공고문' PDF 하나를 고른다.
    요약본·팸플릿·도면은 건너뛰고, 당첨자·예비자 명단(개인정보)은 절대 받지 않는다."""
    pdfs = [(i, n) for i, n in enumerate(names) if n.lower().endswith(".pdf")]
    for keyword in ("공고문", "모집공고", "모집", "공고"):
        hits = [i for i, n in pdfs
                if keyword in n and not re.search(r"요약|팸플릿|팜플렛|평면|조감|명단|명부|당첨|커트라인", n)]
        if hits:
            return hits[0]
    return None


def no_pdf_reason(title: str, names: list[str]) -> str:
    """공고문 PDF를 못 고른 이유. 문서 목록(manifest)의 note 칸에 남긴다."""
    if not names:
        return "첨부 없음(안내 게시물)"
    if re.search(r"당첨자|예비자", title):
        return "당첨자 발표 게시물(명단·계약안내만 첨부, 개인정보라 받지 않음)"
    return "공고문 PDF 없음(보조 자료만 첨부)"


def download(url: str, path: Path, referer: str) -> int:
    if path.exists() and path.stat().st_size > 0:  # 다시 실행할 때 이미 받은 파일은 건너뜀
        return path.stat().st_size
    resp = get(url, headers={"Referer": referer}, stream=True)
    size = int(resp.headers.get("Content-Length") or 0)
    if size > MAX_BYTES:
        return -size
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        for chunk in resp.iter_content(1 << 16):
            f.write(chunk)
    return path.stat().st_size


def base_title(title: str) -> str:
    """'[정정공고]' 접두어를 떼서 원공고와 정정공고를 같은 공고로 묶는다."""
    return re.sub(r"^(\s*\[정정공고\])+\s*", "", title).strip()


# ---------- LH ----------

def collect_lh(start: str, end: str) -> list[dict]:
    rows = []
    for notice_type in ("06", "13"):  # 임대주택 + 주거복지(청년 매입임대·전세임대)
        raw = fetch_lh(start, end, notice_type=notice_type)
        rows += next((d["dsList"] for d in raw if "dsList" in d), [])

    # 같은 공고의 원본·정정본이 있으면 가장 최근 것만 남긴다 (중복 문서로 검색이 왜곡되지 않게)
    latest: dict[str, dict] = {}
    for r in rows:
        key = base_title(r["PAN_NM"])
        if key not in latest or r["PAN_DT"] > latest[key]["PAN_DT"]:
            latest[key] = r

    records = []
    for r in latest.values():
        page = get(r["DTL_URL"]).text
        soup = BeautifulSoup(page, "lxml")
        links = [(re.search(r"fileDownLoad\('(\d+)'\)", a["href"]).group(1), a.get_text(strip=True))
                 for a in soup.select("a[href^=\"javascript:fileDownLoad('\"]")]
        idx = pick_notice_pdf([name for _, name in links])
        rec = {
            "doc_id": f"LH_{r['PAN_ID']}", "source": "LH", "title": r["PAN_NM"],
            "house_type": r["AIS_TP_CD_NM"], "notice_date": r["PAN_DT"], "status": r["PAN_SS"],
            "page_url": r["DTL_URL"], "file_name": "", "file_url": "", "bytes": 0, "note": "",
        }
        if idx is not None:
            file_id, name = links[idx]
            rec |= {"file_name": name, "file_url": LH_FILE_URL.format(file_id)}
            rec["bytes"] = download(rec["file_url"], OUT / "lh" / f"{rec['doc_id']}.pdf", r["DTL_URL"])
        else:
            rec["note"] = no_pdf_reason(rec["title"], [name for _, name in links])
        records.append(rec)
        print(f"[LH] {rec['notice_date']} {rec['title'][:40]} → {rec['file_name'] or '공고문 PDF 없음'} ({rec['bytes']:,}B)")
    return records


# ---------- SH ----------

def sh_list(since: date) -> list[dict]:
    """서울주거포털 SH 공공임대 목록에서 since 이후 공고만 모은다."""
    items = []
    for cp in range(1, 30):
        soup = BeautifulSoup(get(SH_LIST_URL.format(cp)).text, "lxml")
        rows = soup.select("table tbody tr")
        if not rows:
            break
        for tr in rows:
            tds = [td.get_text(" ", strip=True) for td in tr.find_all("td")]
            link = tr.find("a", href=re.compile(r"i-sh\.co\.kr"))
            if len(tds) < 7 or not link:
                continue
            notice_date = datetime.strptime(tds[3], "%Y-%m-%d").date()
            if notice_date < since:
                return items
            items.append({"no": tds[0], "house_type": tds[1], "title": tds[2], "notice_date": tds[3],
                          "status": tds[5], "page_url": link["href"]})
    return items


def collect_sh(since: date) -> list[dict]:
    records = []
    for it in sh_list(since):
        page = get(it["page_url"]).text
        m = re.search(r"initParam\.downList\s*=\s*(\[.*?\]);", page)
        files = json.loads(m.group(1)) if m else []
        idx = pick_notice_pdf([f["oriFileNm"] for f in files])
        seq = re.search(r"seq=(\d+)", it["page_url"]).group(1)
        rec = {
            "doc_id": f"SH_{seq}", "source": "SH", "title": it["title"], "house_type": it["house_type"],
            "notice_date": it["notice_date"].replace("-", ""), "status": it["status"],
            "page_url": it["page_url"], "file_name": "", "file_url": "", "bytes": 0, "note": "",
        }
        if idx is not None:
            f = files[idx]
            rec |= {"file_name": f["oriFileNm"], "file_url": SH_FILE_URL.format(**f)}
            rec["bytes"] = download(rec["file_url"], OUT / "sh" / f"{rec['doc_id']}.pdf", it["page_url"])
        else:
            rec["note"] = no_pdf_reason(rec["title"], [f["oriFileNm"] for f in files])
        records.append(rec)
        print(f"[SH] {rec['notice_date']} {rec['title'][:40]} → {rec['file_name'] or '공고문 PDF 없음'} ({rec['bytes']:,}B)")
    return records


def save_manifest(records: list[dict]) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "manifest.csv"
    collected_at = datetime.now().isoformat(timespec="seconds")
    fields = list(records[0]) + ["collected_at", "license"]
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in records:
            w.writerow(r | {"collected_at": collected_at,
                            "license": "공공기관 모집공고(공개 게시물), 목록은 공공데이터포털 이용허락범위 제한 없음"})
    return path


if __name__ == "__main__":
    today = date.today()
    since = today - timedelta(days=365)
    records = collect_lh(f"{since:%Y%m%d}", f"{today:%Y%m%d}") + collect_sh(since)
    print("문서 목록 →", save_manifest(records))
    ok = [r for r in records if r["bytes"] > 0]
    print(f"공고 {len(records)}건 중 PDF {len(ok)}건 저장")
