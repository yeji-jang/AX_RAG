# 서울시 공공임대 공고문 RAG 파이프라인 최종 결과 및 실패 분석 보고서

- **담당 파트**: 공공임대·청년안심주택 모집공고문 (`src/rag_home_api/test/yj/`)
- **대상 데이터**: 서울시 LH·SH 공공임대 공고문 90건 (전처리 후 2,482페이지)
- **사용 모델**: 임베딩 `text-embedding-3-small` (1,536차원) / LLM `gpt-6-luna` (`reasoning_effort="low"`)
- **최종 채택 구성**: **`chunkV2` (표 구조 보존 청킹) + `C_hybrid_rerank` (Kiwi BM25 + Vector RRF + LLM 리랭킹) + 주어 명시 프롬프트**

---

## 1. 파이프라인 구성 요약

```text
01_preprocess (V2) → 02_chunk (V2: 표 헤더·병합 셀 보존) → 03_build_index (PGVector 단일 컬렉션 yj_public_rental)
  → rag/ (A: Vector / B: Hybrid RRF / C: Hybrid + LLM Rerank / D: Query Rewrite)
  → rag/generator.py (근거 번호 [1],[2] 인용 · 출처 매핑 · 거절/오류 분리 · 인젝션 방어)
  → 04_eval/run_eval.py (Hit@5 · 거절률 · RAGAS 4대 지표 · 자동 실패 진단)
```

- **단일 컬렉션 + 버전 필터 구조**: `yj_public_rental` 컬렉션 1개에 `V1`(6,346개)과 `V2`(6,197개)를 공존시키되, 전역 기본키(`e.id`)를 `{collection}:{chunk_id}`로 분리해 충돌을 방지하고 검색 시 `filters={"chunk_version": "V2"}`로 버전을 구분합니다.
- **고정 평가셋 (30문항)**:
  - `docs/04-1_eval_set/dev.json` (10문항): `direct` 5 · `paraphrase_or_similar` 3 · `unanswerable` 2 (개발·튜닝 전용)
  - `docs/04-1_eval_set/test.json` (20문항): `direct` 10 · `paraphrase_or_similar` 7 · `unanswerable` 3 (사람 검수 후 고정, 최종 평가 전용)

---

## 2. 청킹 고도화: `chunkV1` vs `chunkV2`

공고문 2,482페이지 중 **66.3%(1,646페이지)가 마크다운 표**이며, 그중 **743페이지가 2단 병합 헤더**(`공급호수 → 계 / 일반 / 우선`)를 포함합니다.

| 구분 | `chunkV1` (기본 800/100 분할) | `chunkV2` (표 구조 보존 청킹) | 개선 효과 |
|---|---:|---:|---|
| 분할 방식 | 페이지 전체 글자 수 기준 일괄 분할 | 일반 문단 / 마크다운 표 블록 분리 후 맞춤 분할 | 표 중간 잘림 제거 |
| 표 헤더 처리 | 분할 시 뒷 청크에서 헤더 유실 | 행 단위 분할 + **2단 병합 헤더 상단 자동 복제** | **헤더 유실 `1,192 → 0건` (100% 해결)** |
| 세로 병합 셀 | 빈 앞 칸(`\| \| \| 49 \|`) 그대로 유지 | 위 행 단지명·구분 값 자동 채움(Forward-fill) | 단일 청크 내 단지·평형 정보 완결 |
| 짧은 조각 | 250자 미만 자투리 `884개` | 200자 미만 조각 인접 병합 (`MAX_MERGE_CHARS=950`) | **자투리 `884 → 147건` (83.4% 감소)** |
| 총 청크 수 | 6,346개 (평균 600자) | 6,197개 (평균 681자) | 중복 ID `0`, `id 불일치 0` |

---

## 3. 개발셋(`dev.json` 10문항) 단계별 튜닝 결과

*(단일 컬렉션 `yj_public_rental` / 동일 `dev.json` / 전 실행 오류 0건 · 결측 0건)*

| 단계 | 리포트 | 모드 | Hit@5 (`/8`) | 거절 (`/2`) | CP | CR | Faithfulness | Answer Relevancy | 평균(초) | 미달 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ① V1 기준선 | `chunk_V1/_1437` | `C_hybrid_rerank` | 7/8 (87.5%) | 2/2 | 0.8747 | 1.0000 | 0.9722 | 0.6285 | 6.96 | 6/10 |
| ② V2 표 청킹 | `chunk_V2/_1445` | `C_hybrid_rerank` | 7/8 (87.5%) | 2/2 | **0.9688** | 0.9375 | **1.0000** | 0.6490 | 7.03 | 5/10 |
| ③ V2 + 프롬프트 | `chunk_V2/_1559` | `C_hybrid_rerank` | **7/8 (87.5%)** | **2/2** | **0.9484** | **0.9375** | **0.9844** | **0.9027** | **7.02** | **1/10** |
| ④ V2 + 리라이트 | `chunk_V2/_1559` | `C_plus_opt` | 6/8 (75.0%) | 2/2 | 0.8688 | 0.9375 | 0.9583 | 0.8711 | 8.22 | 2/10 |

- **② 표 보존(`V2`) 효과**: `C_hybrid_rerank`의 `CP`가 `+9.4%p`(0.87 → 0.97) 상승하고 `Faithfulness`가 `1.00` 만점을 기록했으며, `V1`에서 놓쳤던 `DEV-006`도 정확히 검색됐습니다.
- **③ 프롬프트 1줄 보강 효과**: `generator.py`에 *"답변 첫 문장에 질문의 공고명·단지·주택형·대상 계층을 주어로 명시"* 규칙을 추가하자 `Answer Relevancy`가 **`0.6490 → 0.9027` (+25.4%p)** 급상승하며 10문항 중 **9문항 PASS**를 달성했습니다.

---

## 4. STEP 6 최종 고정 평가 (`test.json` 20문항 × 4모드 = 80회)

- **원본 리포트**: `docs/04-2_eval_reports/chunk_V2/eval_test_20261126_1630.{md,json}`
- **실행 조건**: `yj_public_rental` / `chunk_version=V2` / Top-K=5 / 리랭크 후보 15개 / 캐시 미사용
- **호출 성공률**: **80/80 (100%, 오류 0건 · RAGAS 결측 0건)**

| 모드 | Hit@5 (`/17`) | direct (`/10`) | paraphrase (`/7`) | 거절 (`/3`) | CP | CR | Faithfulness | Answer Relevancy | 인용표시율 | 평균(초) | 중앙값(초) | PASS |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `A_vector` | 8/17 (47.1%) | 5/10 | 3/7 | 3/3 | 0.5120 | 0.6176 | 0.9628 | 0.6372 | 11/11 (100%) | 2.30 | 2.03 | 9/20 |
| `B_hybrid` | 10/17 (58.8%) | 5/10 | 5/7 | 3/3 | 0.6023 | 0.7941 | 0.8922 | 0.7762 | 15/15 (100%) | 2.01 | 1.92 | 9/20 |
| **`C_hybrid_rerank`** | **13/17 (76.5%)** | **8/10** | **5/7** | **3/3** | **0.8578** | **0.8824** | **0.9706** | **0.8939** | **16/16 (100%)** | **6.78** | **6.72** | **15/20** |
| `C_plus_opt` *(C+리라이트)* | 12/17 (70.6%) | 7/10 | 5/7 | 3/3 | 0.8569 | 0.8529 | 0.9706 | 0.8479 | 16/16 (100%) | 8.70 | 8.86 | 14/20 |

### 핵심 결론
1. **A → B → C 계단식 상승 입증**: 미관측 `test.json`에서도 `A → B → C` 순으로 **Hit@5 `47.1% → 58.8% → 76.5%` (+29.4%p)**, **CP `0.51 → 0.60 → 0.86`**, **CR `0.62 → 0.79 → 0.88`**, **AR `0.64 → 0.78 → 0.89`**까지 뚜렷하게 개선되었습니다.
2. **목표 지표 초과 달성**: 최종 채택 모델(`chunkV2 + C_hybrid_rerank`)은 RAGAS 4대 지표 모두 **`0.86 ~ 0.97`**로 팀 목표(`≥ 0.7`)를 크게 넘겼고, 근거없음 거절(`3/3`)과 출처 인용 표시율(`16/16`) 모두 100%를 달성했습니다.
3. **STEP 5 쿼리 리라이트(`C_plus_opt`) 최종 판정**: 숫자·주택형 보호 가드로 20문항 모두 폴백 `0건`·오류 `0건`으로 안전하게 동작했으나, `dev`와 `test` 모두에서 `direct` 1문항 하락 및 평균 응답 시간 `+1.92초` 증가가 관찰되어 **기본 파이프라인에서는 미채택(기각)**하고 `chunkV2 + C_hybrid_rerank`를 최종 구성으로 확정했습니다.

---

## 5. 실패 사례 상세 분석 (`C_hybrid_rerank` 미달 5건 / 20건)

| 문항 | 유형 | 진단 | 주요 지표 | 원인 분석 | 개선 방향 |
|---|---|---|---|---|---|
| `TEST-014` | paraphrase | `RETRIEVAL_FAILURE` | Hit=0, CP=0.83, CR=1.0, F=1.0, AR=0.83 | 동일 공고의 다른 안내 청크를 가져와 **실제 정답은 맞혔으나** `allowed_chunk_ids` 외 청크 인용 | 동일 사실이 반복 기재된 페이지의 허용 근거 범위 보완 |
| `TEST-015` | paraphrase | `RETRIEVAL_FAILURE` | Hit=0, CP=1.0, CR=1.0, F=1.0, AR=0.80 | `TEST-014`와 동일 — 중복 신청 무효 조항이 다른 공고/청크에도 존재해 정답은 맞혔으나 Hit@5=0 | 질문의 공고명(`2-2차 미리내집`) 기반 메타데이터 필터링 강화 |
| `TEST-006` | direct | `RETRIEVAL_FAILURE` | Hit=0, CP=0.0, CR=0.0, F=1.0, AR=0.87 | 질문에 단지명이 생략되어 같은 공고 내 다수 49㎡/59㎡ 단지 중 다른 단지를 검색 | 서비스 UI에서 단지명 미지정 시 되묻기(Clarification) 안내 |
| `TEST-005` | direct | `RETRIEVAL_FAILURE` | Hit=0, CP=0.0, CR=0.0, F=0.50, AR=0.91 | `입주지정기간` 표/일정 청크가 후보 15개 밖으로 밀림 | BM25 불용어 제거 및 후보 수(`RERANK_CANDIDATE_K`) 15 → 20 확대 검토 |
| `TEST-013` | paraphrase | `IRRELEVANT_ANSWER` | Hit=1, CP=1.0, CR=1.0, F=1.0, AR=0.65 | 검색·충실도 모두 `1.0` 만점이나 13개 연접지역을 길게 나열하며 `AR`이 `0.7`에 근소 미달 | 나열형 답변의 간결화 포맷 적용 |

---

## 6. 주요 트러블슈팅 기록

1. **RAGAS `0.0000`점 & `Connection error` 연쇄 장애**
   - *원인*: `gpt-6-luna`가 `temperature`/`n` 강제 변경을 HTTP 400으로 거부했고, RAGAS 기본 동시 워커(`max_workers=16`) 과부하로 `NaN`이 발생해 `to_safe_float`가 `0.0`으로 왜곡.
   - *해결*: `LangchainLLMWrapper(..., bypass_temperature=True, bypass_n=True)` + `RunConfig(max_workers=4, max_retries=5)` + 스키마 `float | None` 결측 분리 → 이후 `dev`·`test` 80회 **오류 0 · 결측 0**.
2. **PGVector 전역 기본키(`e.id`) 충돌 위험**
   - *원인*: `langchain_pg_embedding.id`가 테이블 전체 단일 PK여서, 같은 `chunk_id` 패턴으로 다른 실험 컬렉션을 적재하면 기존 컬렉션이 덮어써짐.
   - *해결*: DB 기본키를 `{collection}:{chunk_id}`로 분리(`make_db_id`)하고 단일 컬렉션(`yj_public_rental`) + `chunk_version` 필터로 통합 (`id 불일치 0`).
3. **표 헤더 유실로 인한 `DEV-007` 오답 (`Faithfulness 0.33 → 1.00`)**
   - *원인*: `V1`의 800자 분할로 표 헤더(`계 | 일반 | 우선`)가 잘려 `65 | 59 | 6`을 잘못 해석.
   - *해결*: `V2` 표 보존 청킹(2단 헤더 복제 + 앞쪽 병합 셀 채우기)으로 헤더 유실 `1,192 → 0건` 달성.
4. **정답을 맞히고도 `Answer Relevancy`가 `0.65`에 머문 문제**
   - *원인*: 답변이 공고명·단지 주어를 생략하고 숫자부터 바로 답해 RAGAS 역질문 유사도 하락.
   - *해결*: `generator.py` 프롬프트에 주어 명시 규칙 1줄 추가 → `AR` **`0.65 → 0.90` (+25%p)**, `test`에서도 **`0.89`** 유지.
5. **쿼리 리라이트(`C_plus_opt`)의 공고명 비중 희석 발견**
   - *원인*: 이미 공고명·조건이 구체적인 질의에서 원질문+변환문을 합치자 공통어 비중이 커져 유사 차수 공고 구분력이 하락하고 `+1.9초` 지연.
   - *해결*: `dev`·`test` 정량 비교 데이터에 근거해 리라이트는 기각하고 `chunkV2 + C_hybrid_rerank` 채택.

---

## 7. 재현 방법 & 향후 개선 방향

### 재현 실행 순서 (`rag-home-api` 루트 기준)
```powershell
# 1) 표 보존 청킹 V2 생성 및 JSONL 검증
uv run python src/rag_home_api/test/yj/works/02_chunk_docs/V2.py
uv run python src/rag_home_api/test/yj/works/validate_jsonl.py

# 2) PGVector 단일 컬렉션(yj_public_rental) 증분 적재
uv run python src/rag_home_api/test/yj/works/03_build_index/indexer.py

# 3) 최종 고정 평가 실행 (run_eval.py: CHUNK_VERSION="V2", SPLIT="test")
uv run python src/rag_home_api/test/yj/works/04_eval/run_eval.py
```

### 향후 개선 방향
- **청킹 `V3` (표 직전 소제목 결합)**: `DEV-002`에서 관찰된 표 직전 소제목 분리 트레이드오프를 해결하기 위해 표 블록 앞에 직전 소제목 1줄을 함께 부착.
- **반복 질의 캐싱 (`rag/cache.py`)**: 최종 채택된 `C_hybrid_rerank`의 평균 응답 시간(`6.78초`)을 버전 키 기반 캐시로 단축.
- **팀 통합 라우터 연결**: PR #16(`docs/team_integration_spec.md`) 머지 후 `public_housing` 핸들러 래퍼(`RAGResponse` 반환)를 연결해 4인 통합 챗봇 완성.
