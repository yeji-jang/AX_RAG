# KH 공유 자료 (`src/rag_home_api/shared/docs/kh`)

GitHub 단일 파일 용량 제한(100MB 차단 / 50MB 경고)을 준수하도록 대용량 전월세 원본은 **서울 25개 자치구별**로 무손실 분할하여 저장했습니다. 모든 개별 파일은 **17MB 미만**입니다.

## 폴더 구조

```text
kh/
├── README.md
├── 서울주택도시개발공사_주택관리현황.json            # SH 주택관리현황 (0.18 MB)
├── 서울시_부동산_전월세가_정보/                      # 서울시 부동산 전월세가 정보 JSON 자치구별 25개 (총 481,589건 · yj·bc와 겹치는 30,744건 제외, 최대 16.84 MB)
│   └── 서울시_부동산_전월세가_정보_<자치구>.json
└── mvp/                                              # 서울주거 RAG MVP 핵심 자료
    ├── config/                                       # mvp.json, seoul_districts.json
    ├── docs/                                         # MVP_README.md, MVP_REPORT.md, MVP_SOURCES.md
    ├── reports/                                      # mvp_validation.json
    ├── geo/                                          # 단지 좌표, 목적지 직선거리, 지하철역 좌표, 청년안심주택 지도 JSON
    └── data/
        ├── public_housing_units.csv                  # SH·LH 공급 호실 정제본 (1,269건)
        ├── public_rent_scenarios.csv                 # 자격·전환 조건별 보증금/월세 정제본 (5,076건)
        ├── notices.json / files.json / summary.json / package_manifest.json
        └── rent_contracts/                           # 2025년 서울 전월세 계약 CSV 자치구별 25개 (총 580,277건, 최대 7.23 MB · 원본 중복 439,318건·yj·bc와 겹치는 3,156건 제거)
            └── rent_contracts_<자치구>.csv
```

## 분할 파일 로드 예시 (Python)

### 1. 서울시 부동산 전월세가 정보 (JSON)
각 파일은 원본과 동일한 `{"DESCRIPTION": {...}, "DATA": [...]}` 구조를 유지합니다.

```python
import json
from pathlib import Path

records = []
for path in sorted(Path("서울시_부동산_전월세가_정보").glob("*.json")):
    with path.open(encoding="utf-8") as f:
        records.extend(json.load(f)["DATA"])
```

### 2. 2025년 전월세 계약 정제본 (CSV)
각 파일은 동일한 CSV 헤더를 포함한 독립 파일입니다.

```python
import pandas as pd
from pathlib import Path

df = pd.concat(
    (pd.read_csv(p) for p in sorted(Path("mvp/data/rent_contracts").glob("*.csv"))),
    ignore_index=True,
)
```
