"""공공데이터포털 API로 서울 공공임대 모집공고 목록을 받아 data/raw/에 원본 JSON으로 저장한다.

실행: uv run python -m rag_home_api.collect.public_api
"""

import json
from datetime import datetime

import requests

from rag_home_api.config import DATA_DIR, settings

SEOUL = "11"  # 광역시도 코드: 서울 (마이홈 첨부 코드표로 재확인)

# 국토교통부 마이홈포털 공공주택 모집공고 조회 (data.go.kr 15108420)
MYHOME_URL = "https://apis.data.go.kr/1613000/HWSPR02/rsdtRcritNtcList"
# 한국토지주택공사 분양임대공고문 조회 (data.go.kr 15058530)
LH_URL = "https://apis.data.go.kr/B552555/lhLeaseNoticeInfo1/lhLeaseNoticeInfo1"


def call(url: str, params: dict) -> dict | list:
    """API를 한 번 호출해 JSON을 돌려준다. 키 오류 등은 응답 본문을 그대로 보여 주고 멈춘다."""
    resp = requests.get(url, params={"serviceKey": settings.data_go_kr_service_key, **params}, timeout=20)
    try:
        return resp.json()
    except ValueError:
        # 인증키 오류 등은 XML/텍스트로 오는 경우가 있다
        raise RuntimeError(f"JSON이 아닌 응답 (HTTP {resp.status_code}): {resp.text[:300]}")


def fetch_myhome(year_month_begin: str, year_month_end: str, rows: int = 100) -> dict:
    """마이홈: 서울 공공임대 모집공고 (공급기관명에 LH·SH가 함께 나오는지 확인용)."""
    return call(MYHOME_URL, {
        "brtcCode": SEOUL,
        "yearMtBegin": year_month_begin,  # YYYYMM
        "yearMtEnd": year_month_end,
        "numOfRows": rows,
        "pageNo": 1,
    })


def fetch_lh(start: str, end: str, rows: int = 100, notice_type: str = "06") -> dict | list:
    """LH: 서울 공고 목록. 기간을 안 주면 최근 2개월만 온다(2026-11-23 확인).
    notice_type: 06=임대주택(행복·국민·영구 등), 13=주거복지(매입임대·전세임대)"""
    return call(LH_URL, {
        "PG_SZ": rows,
        "PAGE": 1,
        "UPP_AIS_TP_CD": notice_type,
        "CNP_CD": SEOUL,
        "PAN_ST_DT": start,  # 공고 게시일 시작 (YYYYMMDD)
        "PAN_ED_DT": end,
    })


def save_raw(name: str, data) -> str:
    out = DATA_DIR / "raw" / "api"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}_{datetime.now():%Y%m%d_%H%M%S}.json"
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(path)


if __name__ == "__main__":
    if not settings.data_go_kr_service_key:
        raise SystemExit(".env의 DATA_GO_KR_SERVICE_KEY가 비어 있습니다.")

    today = datetime.now()
    begin = f"{today.year - 1}{today.month:02d}"  # 최근 1년
    end = f"{today:%Y%m}"

    lh_start = f"{today.year - 1}{today:%m%d}"
    lh_end = f"{today:%Y%m%d}"

    for name, fetch in [("myhome", lambda: fetch_myhome(begin, end)), ("lh", lambda: fetch_lh(lh_start, lh_end))]:
        try:
            print(name, "→", save_raw(name, fetch()))
        except Exception as e:
            print(name, "실패:", e)
