# KH 공유 자료 (`src/rag_home_api/shared/docs/kh`)

서울 전월세 시세는 **최근 1년(계약일 2025-11-24 ~ 2026-11-23), RAG에 필요한 칸만** 남긴 CSV로 정제해 **서울 25개 자치구별**로 나눠 저장했습니다. 모든 개별 파일은 **9MB 미만**입니다.

## 폴더 구조

```text
kh/
├── README.md
├── 서울주택도시개발공사_주택관리현황.json            # SH 주택관리현황 (0.18 MB)
├── 서울_전월세_최근1년/                              # 서울 전월세 거래 CSV 자치구별 25개 (총 527,145건)
│   └── 서울_전월세_최근1년_<자치구>.csv
└── mvp/                                              # 서울주거 RAG MVP 핵심 자료
    ├── config/                                       # mvp.json, seoul_districts.json
    ├── docs/                                         # MVP_README.md, MVP_REPORT.md, MVP_SOURCES.md
    ├── reports/                                      # mvp_validation.json
    ├── geo/                                          # 단지 좌표, 목적지 직선거리, 지하철역 좌표, 청년안심주택 지도 JSON
    └── data/
        ├── public_housing_units.csv                  # SH·LH 공급 호실 정제본 (1,269건)
        ├── public_rent_scenarios.csv                 # 자격·전환 조건별 보증금/월세 정제본 (5,076건)
        └── notices.json / files.json / summary.json / package_manifest.json
```

## 서울_전월세_최근1년 (CSV, UTF-8 BOM)

- 출처: 서울시 부동산 전월세가 정보([OA-21276](https://data.seoul.go.kr/dataList/OA-21276/S/1/datasetView.do)) 2026 접수 JSON + 2025 접수 원본(2025년 11~12월 계약 보충)
- 완전히 같은 거래는 1건만 남겼습니다(2025 접수 원본은 같은 행이 두 번씩 들어 있었음).
- 구로구(yj 국토부 실거래)·신림동(bc) CSV에 이미 있는 거래 33,328건은 그쪽에 두고 여기서 뺐습니다. **지역 시세를 볼 때는 kh + yj + bc를 합쳐서** 쓰세요.
- 다시 만들기: `uv run python -m rag_home_api.collect.refine_rent --json-dir <2026 JSON 폴더> --zip2025 <seoul_2025.zip>`

| 칸 | 내용 |
|---|---|
| `id` | 거래 고유 번호 (`SR_` + 거래 내용 해시) |
| `region_gu`, `region_dong`, `jibun`, `building_name` | 자치구·법정동·지번(본번-부번)·건물명. 단독다가구는 원본에서 지번·건물명·층이 비공개라 빈칸 |
| `housing_type`, `lease_type` | 아파트·오피스텔·연립다세대·단독다가구 / 전세·월세 |
| `area_m2`, `floor`, `build_year` | 임대면적(㎡)·층·건축년도 |
| `contract_date`, `contract_period` | 계약일, 계약기간(예: 2026-02~2028-02) |
| `deposit_manwon`, `monthly_rent_manwon` | 보증금·월세(만원) |
| `new_or_renewal`, `renewal_right_used` | 신규/갱신, 갱신청구권 사용(예/빈칸) |
| `prev_deposit_manwon`, `prev_rent_manwon` | 갱신 계약의 종전 보증금·월세(만원) |
| `source` | `2026접수` / `2025접수` |

```python
import pandas as pd
from pathlib import Path

df = pd.concat(
    (pd.read_csv(p, encoding="utf-8-sig") for p in sorted(Path("서울_전월세_최근1년").glob("*.csv"))),
    ignore_index=True,
)
```
