"""서울시 부동산 전월세가(서울 열린데이터 OA-21276)를 '최근 1년 · RAG에 필요한 칸만' CSV로 정제한다.

입력
- --json-dir : 2026 접수 JSON(자치구별 {"DESCRIPTION","DATA"}) 폴더
- --zip2025  : 2025 접수 원본 zip(서울특별시_전월세가_2025.csv, CP949) — 2025년 11~12월 계약 보충용
출력
- shared/docs/서울_전월세_최근1년_<자치구>.csv (UTF-8 BOM, CRLF)
  source: 2026접수 = 2026 접수 JSON, 2025접수 = 2025 접수 원본(11~12월 계약 보충),
          국토부 = 팀원이 받은 국토부 실거래(yj 구로구 CSV, bc 신림동 CSV) 중 서울시 자료에 없는 거래
  id: SR_ + 거래 내용 해시 12자리

처리
1) 계약일이 기간(기본 2025-11-24 ~ 2026-11-23) 안인 행만
2) 공통 칸으로 변환, 완전히 같은 거래는 1건만(2026 접수 우선)
3) yj·bc 국토부 거래를 같은 칸으로 바꿔 합침. 서울시 자료에 이미 있는 거래는 서울시 쪽만 남김
   (구·동·계약일·보증금·월세 같음 + 지번·층은 양쪽에 있을 때만 같아야 함, 한 행당 한 행만)
   → 구로구·신림동도 자치구 파일 하나에 모두 들어감

실행: uv run python -m rag_home_api.collect.refine_rent --json-dir <폴더> --zip2025 <zip> --yj-dir <폴더> --bc-dir <폴더>
"""

import argparse
import csv
import glob
import hashlib
import io
import json
import zipfile
from collections import defaultdict
from pathlib import Path

from rag_home_api.config import ROOT

DOCS = ROOT / "src" / "rag_home_api" / "shared" / "docs"
OUT = DOCS
# id 해시에 쓰는 칸 (구간 칸을 더해도 id가 바뀌지 않도록 고정)
KEY_FIELDS = ["region_gu", "region_dong", "jibun", "building_name", "housing_type", "lease_type",
              "area_m2", "floor", "build_year", "contract_date", "contract_period", "deposit_manwon",
              "monthly_rent_manwon", "new_or_renewal", "renewal_right_used", "prev_deposit_manwon",
              "prev_rent_manwon"]
FIELDS = ["id", *KEY_FIELDS[:13], "deposit_range", "rent_range", *KEY_FIELDS[13:], "source"]


def area2(a) -> str:
    """면적(㎡)은 소수점 둘째 자리까지 (반올림, 끝의 0은 뺌: 97.924→97.92, 42.10→42.1)"""
    try:
        return f"{round(float(a), 2):.2f}".rstrip("0").rstrip(".")
    except (TypeError, ValueError):
        return ""


def price_range(manwon) -> str:
    """만원 단위 금액을 1만·10만·100만·1000만·1억 경계로 나눈 구간 이름 (필터·검색용)"""
    v = num(manwon)
    if v is None:
        return ""
    if v <= 0:
        return "없음"
    for limit, label in ((10, "1만~10만"), (100, "10만~100만"), (1000, "100만~1000만"), (10000, "1000만~1억")):
        if v < limit:
            return label
    return "1억 이상"


def num(x):
    try:
        return int(float(str(x).replace(",", "")))
    except (TypeError, ValueError):
        return None


def jibun(main, sub) -> str:
    m, s = str(main or "").lstrip("0"), str(sub or "").lstrip("0")
    return m + (f"-{s}" if s else "") if m else ""


def period(p) -> str:
    """'26.02~28.02'(서울시) 또는 '202602~202802'(국토부) → '2026-02~2028-02'"""
    try:
        a, b = str(p).strip().split("~")
        if "." in a:
            f = lambda x: f"20{x.split('.')[0]}-{x.split('.')[1]}"
        else:
            f = lambda x: f"{x[:4]}-{x[4:6]}"
        return f"{f(a)}~{f(b)}"
    except (ValueError, IndexError):
        return ""


def row(gu, dong, main, sub, bldg, usg, rent_se, area, flr, arch, day, prd, grfe, rtfe, new, used, bgrfe, brtfe, source):
    r = {"region_gu": gu, "region_dong": dong, "jibun": jibun(main, sub), "building_name": (bldg or "").strip(),
         "housing_type": usg, "lease_type": rent_se, "area_m2": f"{float(area):g}" if area not in (None, "") else "",
         "floor": num(flr) if num(flr) is not None else "", "build_year": num(arch) if num(arch) else "",
         "contract_date": f"{day[:4]}-{day[4:6]}-{day[6:8]}", "contract_period": period(prd),
         "deposit_manwon": num(grfe), "monthly_rent_manwon": num(rtfe), "new_or_renewal": new or "",
         "renewal_right_used": "예" if str(used or "").strip() in ("○", "O", "Y", "사용") else "",
         "prev_deposit_manwon": num(bgrfe) if num(bgrfe) is not None else "",
         "prev_rent_manwon": num(brtfe) if num(brtfe) is not None else "", "source": source}
    key = "|".join(str(r[k]) for k in KEY_FIELDS)
    r["id"] = "SR_" + hashlib.sha1(key.encode()).hexdigest()[:12]  # id는 원래 면적 값으로 계산(바뀌지 않게)
    r["area_m2"] = area2(r["area_m2"])
    r["deposit_range"] = price_range(r["deposit_manwon"])
    r["rent_range"] = price_range(r["monthly_rent_manwon"])
    return key, r


def load(json_dir: Path, zip2025: Path, start: str, end: str) -> list[dict]:
    seen, out = set(), []
    for f in sorted(json_dir.glob("*.json")):
        for d in json.load(open(f, encoding="utf-8"))["DATA"]:
            if start <= d["ctrt_day"] <= end:
                k, r = row(d["cgg_nm"], d["stdg_nm"], d["mno"], d["sno"], d["bldg_nm"], d["bldg_usg"], d["rent_se"],
                           d["rent_area"], d["flr"], d["arch_yr"], d["ctrt_day"], d["ctrt_prd"], d["grfe"], d["rtfe"],
                           d["new_updt_yn"], d["ctrt_updt_use_yn"], d["bfr_grfe"], d["bfr_rtfe"], "2026접수")
                if k not in seen:
                    seen.add(k)
                    out.append(r)
    with zipfile.ZipFile(zip2025) as z:
        text = z.read(z.namelist()[0]).decode("cp949")
    for d in csv.DictReader(io.StringIO(text)):
        if start <= d["계약일"] <= end:
            k, r = row(d["자치구명"], d["법정동명"], d["본번"], d["부번"], d["건물명"], d["건물용도"], d["전월세구분"],
                       d["임대면적"], d["층"], d["건축년도"], d["계약일"], d["계약기간"], d["보증금(만원)"], d["임대료(만원)"],
                       d["신규계약구분"], d["갱신청구권사용"], d["종전보증금"], d["종전임대료"], "2025접수")
            if k not in seen:  # 원본에 두 번씩 든 행, 2026 접수와 같은 거래는 한 번만
                seen.add(k)
                out.append(r)
    return out


def clean_jibun(j) -> str:
    """국토부 번지 '685-222'는 그대로, 단독다가구의 가려진 번지('7**')는 빈칸"""
    j = str(j or "").strip()
    return "" if (not j or "*" in j) else j


def load_team(yj_dir: Path, bc_dir: Path, start: str, end: str) -> list[dict]:
    """yj(구로구 국토부 실거래 CSV)·bc(신림동 국토부 API 정제 CSV)를 공통 칸으로 바꾼다."""
    out = []
    for f in sorted(yj_dir.glob("*실거래가*.csv")):
        t = open(f, encoding="utf-8-sig", newline="").read()
        default_type = "오피스텔" if "오피스텔" in f.name else ""
        for d in csv.DictReader(io.StringIO("\r\n".join(t.split("\r\n")[15:]))):
            day = d["계약년월"] + d["계약일"].zfill(2)
            if not (start <= day <= end):
                continue
            p = d["시군구"].split()
            _, r = row(p[1], p[2], clean_jibun(d.get("번지")), "", d.get("단지명") or d.get("건물명"),
                       d.get("주택유형") or default_type, d["전월세구분"],
                       d.get("전용면적(㎡)") or d.get("계약면적(㎡)"), d.get("층"), d["건축년도"], day, d["계약기간"],
                       d["보증금(만원)"], d["월세금(만원)"], d["계약구분"], d["갱신요구권 사용"],
                       d["종전계약 보증금(만원)"], d["종전계약 월세(만원)"], "국토부")
            out.append(r)
    type_map = {"아파트": "아파트", "오피스텔": "오피스텔", "빌라": "연립다세대", "단독/다가구": "단독다가구"}
    for f in sorted(bc_dir.glob("*.csv")):
        for d in csv.DictReader(open(f, encoding="utf-8-sig", newline="")):
            day = d["contract_date"].replace("-", "")
            if not (start <= day <= end):
                continue
            s = json.loads(d["source_raw"])
            rent = num(d["monthly_rent_manwon"])
            _, r = row(d["region_gu"], d["region_dong"], clean_jibun(s.get("jibun")), "",
                       s.get("aptNm") or s.get("mhouseNm") or s.get("offiNm") or d.get("building_name"),
                       type_map.get(d["property_category"], d["property_category"]), "전세" if rent == 0 else "월세",
                       d["area_m2"], d["floor"], d["built_year"], day, s.get("contractTerm"), d["deposit_manwon"],
                       d["monthly_rent_manwon"], s.get("contractType"), s.get("useRRRight"),
                       s.get("preDeposit"), s.get("preMonthlyRent"), "국토부")
            out.append(r)
    return out


def merge_team(rows: list[dict], team: list[dict]) -> tuple[list[dict], int]:
    """서울시 자료(rows)에 없는 국토부 거래(team)만 더한다. 같은 거래면 서울시 쪽을 남긴다."""
    idx = defaultdict(list)  # 서울시 거래: (구,동,날짜,보증금,월세) → [[지번, 층], ...]
    for r in rows:
        idx[(r["region_gu"], r["region_dong"], r["contract_date"], r["deposit_manwon"], r["monthly_rent_manwon"])].append(
            [r["jibun"] or None, r["floor"] if r["floor"] != "" else None])
    added = []
    for r in team:
        key = (r["region_gu"], r["region_dong"], r["contract_date"], r["deposit_manwon"], r["monthly_rent_manwon"])
        j, fl = r["jibun"] or None, (r["floor"] if r["floor"] != "" else None)
        for i, (jj, ff) in enumerate(idx.get(key, [])):
            if (j is None or jj is None or j == jj) and (fl is None or ff is None or fl == ff):
                idx[key].pop(i)
                break
        else:
            added.append(r)
    return rows + added, len(added)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json-dir", required=True, type=Path)
    ap.add_argument("--zip2025", required=True, type=Path)
    ap.add_argument("--yj-dir", type=Path, default=DOCS / "yj")
    ap.add_argument("--bc-dir", type=Path, default=DOCS / "bc")
    ap.add_argument("--start", default="20251124")
    ap.add_argument("--end", default="20261123")
    a = ap.parse_args()

    rows = load(a.json_dir, a.zip2025, a.start, a.end)
    print(f"기간 {a.start}~{a.end} 서울시 고유 거래 {len(rows):,}건")
    team = load_team(a.yj_dir, a.bc_dir, a.start, a.end)
    rows, added = merge_team(rows, team)
    print(f"국토부(yj·bc) 기간 내 {len(team):,}건 중 서울시 자료에 없는 {added:,}건 추가 → {len(rows):,}건")
    # 칸이 모두 같은 국토부 거래(같은 건물의 다른 호실일 수 있어 남김)는 id 뒤에 -2, -3 …을 붙여 구분
    seen = defaultdict(int)
    for r in rows:
        seen[r["id"]] += 1
        if seen[r["id"]] > 1:
            r["id"] = f"{r['id']}-{seen[r['id']]}"

    OUT.mkdir(parents=True, exist_ok=True)
    by_gu = defaultdict(list)
    for r in rows:
        by_gu[r["region_gu"]].append(r)
    for gu, rs in sorted(by_gu.items()):
        rs.sort(key=lambda r: (r["contract_date"], r["region_dong"], r["jibun"], r["id"]))
        with open(OUT / f"서울_전월세_최근1년_{gu}.csv", "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS, lineterminator="\r\n")
            w.writeheader()
            w.writerows(rs)
    print(f"자치구 {len(by_gu)}개 파일 → {OUT}")


if __name__ == "__main__":
    main()
