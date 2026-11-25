"""일반 전월세 근거 문서 수집 (정적 크롤링): 법령 본문 + 정부·서울시 가이드 PDF.

결과: data/raw/rental/{laws,guides}/, 문서 목록 data/raw/rental/manifest.csv
실행: uv run python -m rag_home_api.collect.rental_docs
"""

import csv
import re
import time
from datetime import datetime
from urllib.parse import quote

import requests
from bs4 import BeautifulSoup

from rag_home_api.config import DATA_DIR

OUT = DATA_DIR / "raw" / "rental"
session = requests.Session()
session.headers.update({"User-Agent": "Mozilla/5.0 (AX_RAG education project)"})

# 법령: 국가법령정보센터 현행 법령 (원문 위치 = 조·항·호)
LAWS = [
    ("LAW_HLPA", "주택임대차보호법"),
    ("LAW_HLPA_DECREE", "주택임대차보호법 시행령"),
    ("LAW_REPORT", "부동산 거래신고 등에 관한 법률"),  # 주택 임대차 계약 신고(전월세 신고제) 조항
    # v1 보강(2026-11-24): 사회초년생 이사 흐름에 필요한 것만
    ("LAW_REPORT_DECREE", "부동산 거래신고 등에 관한 법률 시행령"),  # 임대차 신고 대상 금액·지역
    ("LAW_CIVIL", "민법"),  # 임대차 절(제618조~제654조)만 사용 → export_jsonl.LAW_ARTICLE_RANGE
    ("LAW_RESIDENT", "주민등록법"),  # 전입신고(거주지 이동 신고) 기한
]

# 가이드 PDF: (문서ID, 제목, 발행처, 파일 URL, Referer)
GUIDES = [
    ("GUIDE_STD_CONTRACT", "주택임대차 표준계약서(2023.10.6. 개정, 원본게시용)", "법무부",
     "https://www.moj.go.kr/bbs/moj/118/459550/download.do",
     "https://www.moj.go.kr/bbs/moj/118/575868/artclView.do"),
    ("GUIDE_HLPA_COMMENTARY", "주택임대차보호법 해설집", "국토교통부·법무부",
     "https://www.molit.go.kr/portal/common/download/DownloadMltm2.jsp?FilePath=portal/DextUpload/202008/20200828_165507_655.pdf"
     "&FileName=" + quote("주택임대차보호법 해설집(국토교통부-법무부).pdf"),
     "https://www.molit.go.kr/USR/policyData/m_34681/dtl.jsp?id=4484"),
    ("GUIDE_SEOUL_HLPA_2020", "2020.7.31. 개정 주택임대차보호법 가이드북", "서울특별시",
     "https://news.seoul.go.kr/citybuild/files/2020/12/60011a27be8dd0.61694881.pdf",
     "https://news.seoul.go.kr/citybuild/archives/511154"),
    ("GUIDE_SEOUL_CASES", "실제 사례와 Q&A로 알아보는 주택임대차 상담사례집", "서울특별시",
     "https://housing.seoul.go.kr/site/main/file/download/uu/80ccc952c14c4e20a544fcfb1a5840ea",
     "https://housing.seoul.go.kr/"),
]

# 정적으로 못 받는 문서 (Selenium 단계에서 처리)
DYNAMIC = [
    ("GUIDE_SEOUL_YOUTH_ATOZ", "전세 계약, 두렵지 않아요 – 전세사기 예방 AtoZ (2025.10)", "서울특별시",
     "https://housing.seoul.go.kr/site/main/content/sh05_070200", "전자책 뷰어(ebook.seoul.go.kr)라 정적 수집 불가 → Selenium 대상"),
]


def get(url: str, **kw) -> requests.Response:
    time.sleep(1.0)
    r = session.get(url, timeout=60, **kw)
    r.raise_for_status()
    return r


def collect_law(doc_id: str, name: str) -> dict:
    """법령 페이지에서 현행 법령 번호(lsiSeq)·시행일을 찾고, 본문 HTML과 정리된 텍스트를 저장한다."""
    page_url = f"https://www.law.go.kr/법령/{quote(name)}"
    html = get(page_url).text
    lsi = re.search(r"lsiSeq\s*:\s*'(\d+)", html).group(1)
    ef = re.search(r"efYd\s*[:=]\s*'(\d+)", html).group(1)
    body_url = f"https://www.law.go.kr/LSW/lsInfoR.do?lsiSeq={lsi}&efYd={ef}&chrClsCd=010202&ancYnChk=0"
    body = get(body_url, headers={"Referer": "https://www.law.go.kr/"}).text

    d = OUT / "laws"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{doc_id}.html").write_text(body, encoding="utf-8")
    text = BeautifulSoup(body, "lxml").get_text("\n", strip=True)
    (d / f"{doc_id}.txt").write_text(text, encoding="utf-8")
    articles = len(re.findall(r"^제\d+조(의\d+)?\(", text, re.M))
    print(f"[법령] {name} (시행 {ef}) → 조문 {articles}개, {len(text):,}자")
    return {"doc_id": doc_id, "category": "법령", "title": name, "publisher": "법제처 국가법령정보센터",
            "version": f"현행(시행일 {ef}, lsiSeq {lsi})", "page_url": page_url, "file_url": body_url,
            "file": f"laws/{doc_id}.txt", "bytes": len(text.encode()),
            "license": "법령은 저작권 보호 대상 아님(저작권법 제7조)", "note": f"조문 {articles}개"}


def collect_guide(doc_id: str, title: str, publisher: str, url: str, referer: str) -> dict:
    r = get(url, headers={"Referer": referer})
    d = OUT / "guides"
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{doc_id}.pdf"
    path.write_bytes(r.content)
    ok = r.content[:5] == b"%PDF-"
    print(f"[가이드] {title} → {len(r.content):,}B {'PDF' if ok else '⚠ PDF 아님'}")
    return {"doc_id": doc_id, "category": "가이드", "title": title, "publisher": publisher,
            "version": "", "page_url": referer, "file_url": url, "file": f"guides/{doc_id}.pdf",
            "bytes": len(r.content), "license": "공공기관 공개 자료(공공누리 유형은 원문 페이지 확인)",
            "note": "" if ok else "PDF 형식 아님 — 확인 필요"}


def main():
    records = [collect_law(*law) for law in LAWS] + [collect_guide(*g) for g in GUIDES]
    for doc_id, title, publisher, page, note in DYNAMIC:
        records.append({"doc_id": doc_id, "category": "가이드", "title": title, "publisher": publisher,
                        "version": "", "page_url": page, "file_url": "", "file": "", "bytes": 0,
                        "license": "", "note": note})

    collected_at = datetime.now().isoformat(timespec="seconds")
    with open(OUT / "manifest.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0]) + ["collected_at"])
        w.writeheader()
        w.writerows(r | {"collected_at": collected_at} for r in records)
    print("문서 목록 →", OUT / "manifest.csv")


if __name__ == "__main__":
    main()
