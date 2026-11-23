"""서울시 부동산 전월세가(서울 열린데이터 OA-21276)를 '최근 1년 · RAG에 필요한 칸만' CSV로 정제한다.

입력
- --json-dir : 2026 접수 JSON(자치구별 {"DESCRIPTION","DATA"}) 폴더
- --zip2025  : 2025 접수 원본 zip(서울특별시_전월세가_2025.csv, CP949) — 2025년 11~12월 계약 보충용
출력
- shared/docs/kh/서울_전월세_최근1년/서울_전월세_최근1년_<자치구>.csv (UTF-8 BOM, CRLF)
  source: 2026접수 = 2026 접수 JSON, 2025접수 = 2025 접수 원본(11~12월 계약 보충). id: SR_ + 거래 내용 해시 12자리

처리
1) 계약일이 기간(기본 2025-11-24 ~ 2026-11-23) 안인 행만
2) 공통 칸으로 변환, 완전히 같은 거래는 1건만(2026 접수 우선)
3) yj(구로구 국토부 실거래)·bc(신림동) CSV에 이미 있는 거래는 그쪽에 두고 여기서 뺌
   (구·동·계약일·보증금·월세 같음 + 지번·층은 양쪽에 있을 때만 같아야 함, 한 행당 한 행만)

실행: uv run python -m rag_home_api.collect.refine_rent --json-dir <폴더> --zip2025 <zip>
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
OUT = DOCS / "kh" / "서울_전월세_최근1년"
FIELDS = ["id", "region_gu", "region_dong", "jibun", "building_name", "housing_type", "lease_type",
          "area_m2", "floor", "build_year", "contract_date", "contract_period", "deposit_manwon",
          "monthly_rent_manwon", "new_or_renewal", "renewal_right_used", "prev_deposit_manwon",
          "prev_rent_manwon", "source"]


def num(x):
    try:
        return int(float(str(x).replace(",", "")))
    except (TypeError, ValueError):
        return None


def jibun(main, sub) -> str:
    m, s = str(main or "").lstrip("0"), str(sub or "").lstrip("0")
    return m + (f"-{s}" if s else "") if m else ""


def period(p) -> str:
    """'26.02~28.02' → '2026-02~2028-02'"""
    try:
        a, b = str(p).split("~")
        f = lambda x: f"20{x.split('.')[0]}-{x.split('.')[1]}"
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
    key = "|".join(str(r[k]) for k in FIELDS[1:-1])
    r["id"] = "SR_" + hashlib.sha1(key.encode()).hexdigest()[:12]
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


def remove_team_overlap(rows: list[dict]) -> tuple[list[dict], int]:
    idx = defaultdict(list)  # yj·bc 거래: (구,동,날짜,보증금,월세) → [[지번, 층], ...]
    for f in glob.glob(str(DOCS / "yj" / "*실거래가*.csv")):
        t = open(f, encoding="utf-8-sig", newline="").read()
        for d in csv.DictReader(io.StringIO("\r\n".join(t.split("\r\n")[15:]))):
            p = d["시군구"].split()
            day = d["계약년월"] + d["계약일"].zfill(2)
            idx[(p[1], p[2], f"{day[:4]}-{day[4:6]}-{day[6:8]}", num(d["보증금(만원)"]), num(d["월세금(만원)"]))].append(
                [d.get("번지") or None, num(d.get("층"))])
    for f in glob.glob(str(DOCS / "bc" / "*.csv")):
        for d in csv.DictReader(open(f, encoding="utf-8-sig", newline="")):
            a = d["address"].split()
            jb = a[3] if len(a) > 3 and a[3][:1].isdigit() else None
            idx[(d["region_gu"], d["region_dong"], d["contract_date"], num(d["deposit_manwon"]),
                 num(d["monthly_rent_manwon"]))].append([jb, num(d["floor"])])
    kept, removed = [], 0
    for r in rows:
        key = (r["region_gu"], r["region_dong"], r["contract_date"], r["deposit_manwon"], r["monthly_rent_manwon"])
        j, fl = r["jibun"] or None, (r["floor"] if r["floor"] != "" else None)
        for i, (jj, ff) in enumerate(idx.get(key, [])):
            if (j is None or jj is None or j == jj) and (fl is None or ff is None or fl == ff):
                idx[key].pop(i)
                removed += 1
                break
        else:
            kept.append(r)
    return kept, removed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json-dir", required=True, type=Path)
    ap.add_argument("--zip2025", required=True, type=Path)
    ap.add_argument("--start", default="20251124")
    ap.add_argument("--end", default="20261123")
    a = ap.parse_args()

    rows = load(a.json_dir, a.zip2025, a.start, a.end)
    print(f"기간 {a.start}~{a.end} 고유 거래 {len(rows):,}건")
    rows, removed = remove_team_overlap(rows)
    print(f"yj·bc와 겹치는 {removed:,}건 제외 → {len(rows):,}건")

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
