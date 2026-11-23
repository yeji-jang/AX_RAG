"""청년안심주택 모집공고 수집 (서울시 청년안심주택 누리집 soco.seoul.go.kr, 정적 요청).

- 목록: /youth/pgm/home/yohome/bbsListJson.json (POST, 게시판 BMSR00015)
- 상세: /youth/bbs/BMSR00015/view.do?boardId=… → 첨부 fileDown.do 링크
결과: data/raw/youth/<문서ID>.pdf, 문서 목록 data/raw/youth/manifest.csv
실행: uv run python -m rag_home_api.collect.youth_housing            (목록 통계만)
      uv run python -m rag_home_api.collect.youth_housing --download (공고문 PDF까지)
"""

import csv
import re
import sys
import time
from datetime import date, datetime, timedelta

import requests
from bs4 import BeautifulSoup

from rag_home_api.collect.notices import pick_notice_pdf
from rag_home_api.config import DATA_DIR

BASE = "https://soco.seoul.go.kr"
LIST_URL = BASE + "/youth/pgm/home/yohome/bbsListJson.json"
VIEW_URL = BASE + "/youth/bbs/BMSR00015/view.do?boardId={}&menuNo=400008"
OUT = DATA_DIR / "raw" / "youth"

session = requests.Session()
session.headers.update({"User-Agent": "Mozilla/5.0 (AX_RAG education project)",
                        "Referer": BASE + "/youth/bbs/BMSR00015/list.do?menuNo=400008"})


def get(url: str, method: str = "GET", **kw) -> requests.Response:
    time.sleep(1.0)
    r = session.request(method, url, timeout=60, **kw)
    r.raise_for_status()
    return r


def fetch_list(since: date) -> list[dict]:
    """최신순 목록을 since 이전 공고가 나올 때까지 넘겨 가며 모은다."""
    items, page = [], 1
    while True:
        data = get(LIST_URL, "POST", data={"bbsId": "BMSR00015", "pageIndex": page,
                                           "searchCondition": "", "searchKeyword": ""}).json()
        rows = data["resultList"]
        if not rows:
            return items
        for r in rows:
            notice_date = (r.get("optn1") or "")[:10]
            if notice_date and datetime.strptime(notice_date, "%Y-%m-%d").date() < since:
                return items
            m = re.match(r"\[(.+?)\]", r["nttSj"])
            items.append({"board_id": r["boardId"], "title": r["nttSj"], "notice_date": notice_date,
                          "kind": m.group(1) if m else "", "operator": r.get("optn3") or ""})
        if page >= data["pagingInfo"]["totPage"]:
            return items
        page += 1


def download_notice(it: dict) -> dict:
    page_url = VIEW_URL.format(it["board_id"])
    soup = BeautifulSoup(get(page_url).text, "lxml")
    files = [(a["href"], a.get_text(strip=True)) for a in soup.select('a[href*="fileDown.do"]')]
    rec = {"doc_id": f"YOUTH_{it['board_id']}", "source": "청년안심주택", "title": it["title"],
           "kind": it["kind"], "operator": it["operator"], "notice_date": it["notice_date"].replace("-", ""),
           "page_url": page_url, "file_name": "", "file_url": "", "bytes": 0, "note": ""}
    idx = pick_notice_pdf([name for _, name in files])
    if idx is None:
        rec["note"] = "공고문 PDF 없음" if files else "첨부 없음"
        return rec
    href, name = files[idx]
    rec |= {"file_name": name, "file_url": BASE + href}
    path = OUT / f"{rec['doc_id']}.pdf"
    if not (path.exists() and path.stat().st_size > 0):
        OUT.mkdir(parents=True, exist_ok=True)
        path.write_bytes(get(rec["file_url"], headers={"Referer": page_url}).content)
    rec["bytes"] = path.stat().st_size
    return rec


if __name__ == "__main__":
    since = date.today() - timedelta(days=365)
    items = fetch_list(since)
    kinds: dict[str, int] = {}
    for it in items:
        kinds[it["kind"]] = kinds.get(it["kind"], 0) + 1
    print(f"최근 1년 공고 {len(items)}건 | 종류별 {kinds}")

    if "--download" in sys.argv:
        records = []
        for it in items:
            rec = download_notice(it)
            records.append(rec)
            print(f"[청년] {rec['notice_date']} {rec['title'][:40]} → {rec['file_name'] or rec['note']} ({rec['bytes']:,}B)")
        collected_at = datetime.now().isoformat(timespec="seconds")
        with open(OUT / "manifest.csv", "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(records[0]) + ["collected_at"])
            w.writeheader()
            w.writerows(r | {"collected_at": collected_at} for r in records)
        print(f"공고 {len(records)}건 중 PDF {sum(r['bytes'] > 0 for r in records)}건 저장 → {OUT / 'manifest.csv'}")
