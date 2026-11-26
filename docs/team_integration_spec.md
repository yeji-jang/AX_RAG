# 서울 주거 RAG 팀 통합 규약서 (방안 3: 어댑터 선통합 + 멀티컬렉션·라우터 결합)

> 작성: 최기혁 (`kh`) · 기준일: 2026-11-27 (통합 실증·검증 완료 반영)  
> 대상 브랜치: `dev` (PR #15~#20 통합 완료), `feat/yeji-chatbot` (PR #21 `찾아줘 홈즈` Streamlit 챗봇 UI), `feat/kihyeok`

---

## 1. 통합 원칙 (재임베딩 0원 · 팀원 코드 파괴 0%)

4명의 청킹 전략과 메타데이터가 도메인 특성에 맞게 이미 최적화되어 있으므로, **억지로 단일 컬렉션에 합쳐 BM25/벡터 노이즈를 만들지 않고** 아래 2단계로 통합한다.

1. **1단계 · 평가 결과 선통합 (즉시 완료)**:
   - 팀원 기존 평가 스크립트를 수정하지 않고, `src/rag_home_api/test/kh/unify_team_results.py` 어댑터가 4명의 원본 결과 포맷을 읽어 공통 요약 JSON(`data/team_unified/summary_<owner>.json`)으로 정규화한다.
   - Streamlit 대시보드(`5. 시험지(_v{n}) & 팀 통합`)에서 4명 결과를 도메인별 공식 목표치(전월세 `0.60`, 법령·공공임대 `0.70`, 거절률 `≥ 70%`)와 함께 즉시 비교한다.
2. **2단계 · 서비스 파이프라인 결합 (멀티 컬렉션 + 입구 라우터 + 공통 DTO)**:
   - PGVector 안에서는 **도메인별 4개 컬렉션**으로 공존시키되, 전역 PK 충돌과 임베딩 차원 충돌만 원천 차단한다.
   - 서비스 입구에 `src/rag_home_api/router/domain_router.py`를 두고, 각 팀원 핸들러는 공통 `RAGResponse` DTO(`src/rag_home_api/shared/schemas/rag_contract.py`)만 맞춰 반환한다.

---

## 2. PGVector 컬렉션 & PK(`e.id`) 충돌 방지 규약

### 2-1. 임베딩 모델 통일 & 멀티 컬렉션 공존 원칙
- 통합 PGVector DB(`rag_db`)의 기본 공용 임베딩 모델은 **`text-embedding-3-small` (1,536차원)** 으로 통일한다.
- 한국어 특화 로컬 임베딩(`nlpai-lab/KURE-v1`, 1,024차원) 등 이종 차원 모델은 반드시 **독립 컬렉션(`housing_rent_kh_kure_v1`)**으로 분리 적재하여 1,536차원 컬렉션과 차원 충돌 없이 공존시킨다.

### 2-2. 도메인별 컬렉션 레지스트리 (`shared/core/vector_store.py` & `router/`)

| 도메인 키 (`domain`) | 담당자 | 통합 표준명 (`standard`) | 현재 확정 컬렉션명 (`current` / 활성) | 임베딩 | 담당 데이터 (단일 DB 실측 청크 수) |
| :--- | :---: | :--- | :--- | :---: | :--- |
| `law` | 최지훈 (`jh`) | `rag_law_v1` | `jh_1000` | `3-small` (1,536d) | 주택임대차 법령·가이드 (`1000/200`, **167청크**) |
| `public_housing` | 장예지 (`yj`) | `rag_notice_v1` | `yj_public_rental` | `3-small` (1,536d) | 공공임대·청년안심 모집공고 (`chunkV2`, **6,197청크**) |
| `rent_stats` | 최기혁 (`kh`) | `rag_rent_stats_v1` | `housing_rent_kh` *(공통 기준)*<br>**`housing_rent_kh_kure_v1`** *(어댑터 활성)* | `3-small` (1,536d)<br>**`KURE-v1` (1,024d)** | 서울 25구 전월세 시세 요약 블록 (**6,200청크** × 2) |
| `rent_deals` | 선병철 (`bc`) | `rag_rent_deals_v1` | `course_v0_small` | `3-small` (1,536d) | 관악구 전월세 개별 조건 필터 검색 (`V0`, **31,772청크**) |

> 단일 PostgreSQL(PGVector) 공존 실측(2026-11-26): 위 5개 도메인 컬렉션 + 공통 파일럿(`seoul_housing_unified`, 216청크) = **총 6개 컬렉션 · 50,752행 · 고유 PK 50,752개 · 컬렉션 간 PK 충돌 `0건`** 검증 완료.

### 2-3. 전역 PK(`langchain_pg_embedding.id`) 필수 규칙
LangChain `PGVector`는 모든 컬렉션이 단일 테이블 `langchain_pg_embedding`을 공유하므로, 적재 시 반드시 아래 공통 함수를 사용해 **컬렉션 접두어를 PK에 포함**한다.

```python
from rag_home_api.shared.core.vector_store import make_db_id

db_id = make_db_id(collection_name, chunk_id)
# 예: "housing_rent_kh_kure_v1:RENT_강남구_신사동_단독다가구_월세_33이하"
# 예: "yj_public_rental:SH_308944_p024_chunkV2_c01"
```

---

## 3. 공통 응답 계약 (`RAGResponse` DTO)

모든 도메인 핸들러는 `Callable[[str], RAGResponse]` 시그니처를 따른다 (`src/rag_home_api/shared/schemas/rag_contract.py`).

```json
{
  "status": "answered | insufficient_evidence | error | unsupported",
  "domain": "law | public_housing | rent_stats | rent_deals | unsupported",
  "answer": "최종 답변 텍스트 (근거 부족 시 '확인할 수 없음.'으로 시작)",
  "citations": [
    {
      "chunk_id": "RENT_용산구_한남동_단독다가구_전세_33초과_60이하",
      "title": "용산구 한남동 단독다가구 전세 33~60㎡(소형) 실거래 시세",
      "source": "서울_전월세_최근1년_용산구.csv",
      "location": "한남동·단독다가구·전세·33~60㎡(소형)",
      "url": ""
    }
  ],
  "retrieved_contexts": ["..."],
  "chunk_ids": ["..."],
  "latency_ms": 342.1,
  "collection_name": "housing_rent_kh",
  "route_reason": "서울 전월세 실거래 시세·통계·비교 질문 감지",
  "error_message": ""
}
```

### 오류 & 근거 부족 안전 원칙
1. **`insufficient_evidence` (근거 부족)**: 임의 인용 생성 금지 (`citations = []`), `"확인할 수 없음."`으로 시작해 사유 안내.
2. **`error` (검색/LLM 예외)**: `TeamDomainRouter.invoke()`가 자동 포획해 서비스 중단 방지, **`status="error"` 응답은 캐시 저장 절대 금지 (`resp.is_cacheable() == False`)**.
3. **`unsupported` (지원 범위 외)**: 서울 외 지역, 매매·분양, 개인 대출 한도·전세사기 개별 법률 자문 등은 검색 전 입구에서 정직 거절.

---

## 4. 통합 120문항 평가셋 `question_id` & 결과 통합 규약

4명(각 30문항 = 120문항)의 평가셋을 합칠 때 ID 충돌을 막기 위해 아래 접두어를 권장한다.

| 담당자 | 도메인 | 권장 `question_id` 형식 | 공식 RAGAS 목표치 |
| :---: | :--- | :--- | :---: |
| 지훈 (`jh`) | 주택임대차 법령 | `LAW-JH-001` ~ `LAW-JH-030` | `≥ 0.70` |
| 예지 (`yj`) | 공공임대 공고 | `PUB-YJ-001` ~ `PUB-YJ-030` | `≥ 0.70` |
| 최기혁 (`kh`) | 전월세 시세 통계 | `RENT-KH-001` ~ `RENT-KH-030` | `≥ 0.60` |
| 병철 (`bc`) | 전월세 조건 탐색 | `DEAL-BC-001` ~ `DEAL-BC-030` | `≥ 0.60` |

### 결과 자동 통합 실행 명령
```powershell
# 1) 최신 원격 브랜치 동기화 후 4명 결과 자동 정규화 -> data/team_unified/summary_*.json 생성
git fetch origin --prune
uv run python -m rag_home_api.test.kh.unify_team_results --from-git

# 2) Streamlit 대시보드 -> 좌측 메뉴 '5. 시험지(_v{n}) & 팀 통합' -> '팀원 통합 비교'에서 4명 즉시 비교!
uv run --with streamlit streamlit run src/rag_home_api/test/kh/eval_dashboard.py
```

---

## 5. 팀원별 통합 완료 현황 (2026-11-27 기준)

- [x] **장예지 (`yj` · `public_housing`)**: `make_db_id` PK 규약·RAGAS JSON 완비 + `yj_public_rental` (`chunkV2`, 6,197청크) 단일 DB 적재 + `router/public_housing_adapter.py` 무수정 연결 + 통합 라우터 기반 **`찾아줘 홈즈` Streamlit 챗봇 UI (PR #21)** 완성!
- [x] **최지훈 (`jh` · `law`)**: `law_router.py` 분류 규칙을 공용 `domain_router.py`에 반영 + `jh_1000` (167청크) `make_db_id` 적재 및 `--check-index` PASS + `router/law_adapter.py` 무수정 연결(PR #19 초기화 수정 · PR #21 중복 인자 핫픽스 검수 완료)!
- [x] **선병철 (`bc` · `rent_deals`)**: `course_v0_small` (`V0`, 31,772청크) 단일 DB 적재 + `router/rent_deals_adapter.py` 무수정 연결 + 기존 `test` (`Hit@5 17/17 · CP/CR 1.000`) 및 신규 `test_v3` (`15/17`) 확정!
- [x] **최기혁 (`kh` · `rent_stats` · 통합 관리)**: `housing_rent_kh_kure_v1` (`KURE-v1`, 6,200청크) `router/rent_stats_adapter.py` 연결 + STEP 6 봉인 `test` 확정(`C+Rewrite RAGAS 0.9435 · Cache Warm 0.3ms`) + **단일 DB `50,752행` PK 충돌 0건 · 18문항 4도메인 라이브 스모크 `18/18 (100%)` · 120문항 평가셋 무결성 `120/120` · Streamlit Pure-Stdlib 4인 워크벤치(83스텝)** 검증 완료!
