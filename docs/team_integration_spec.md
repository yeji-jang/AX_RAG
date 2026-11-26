# 서울 주거 RAG 팀 통합 규약서 (방안 3: 어댑터 선통합 + 멀티컬렉션·라우터 결합)

> 작성: 최기혁 (`kh`) · 기준일: 2026-11-26  
> 대상 브랜치: `dev`, `feat/yeji`, `feat/cjh`, `feat/SBC-rag-pipeline`, `feat/kihyeok`

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

### 2-1. 임베딩 모델 통일
- 통합 PGVector DB(`rag_db`)의 기본 임베딩 모델은 전원 **`text-embedding-3-small` (1,536차원)** 으로 통일한다.
- `text-embedding-3-large` (3,072차원) 또는 로컬 한국어 임베딩(`KURE-v1`, 1,024차원)은 반드시 별도 비교 실험 컬렉션으로만 운용하고 통합 DB 기본 컬렉션과 섞지 않는다.

### 2-2. 도메인별 컬렉션 레지스트리 (`shared/core/vector_store.py`)

| 도메인 키 (`domain`) | 담당자 | 통합 표준명 (`standard`) | 현재 브랜치 컬렉션명 (`current`) | 임베딩 | 담당 데이터 |
| :--- | :---: | :--- | :--- | :---: | :--- |
| `law` | 지훈 (`jh`) | `rag_law_v1` | `jh_1000` | `3-small` | 주택임대차 법령·가이드 (1000/200) |
| `public_housing` | 예지 (`yj`) | `rag_notice_v1` | `yj_preV2_chunkV1_cs800_ov100` | `3-small` | 공공임대·청년안심 모집공고 (800/100) |
| `rent_stats` | 최기혁 (`kh`) | `rag_rent_stats_v1` | `housing_rent_kh` | `3-small` | 서울 25구 전월세 시세 요약 블록 (6,200청크) |
| `rent_deals` | 병철 (`bc`) | `rag_rent_deals_v1` | `course_v1_small` | `3-small` | 전월세 개별 거래 조건 필터 검색 (1행=1청크) |

### 2-3. 전역 PK(`langchain_pg_embedding.id`) 필수 규칙
LangChain `PGVector`는 모든 컬렉션이 단일 테이블 `langchain_pg_embedding`을 공유하므로, 적재 시 반드시 아래 공통 함수를 사용해 **컬렉션 접두어를 PK에 포함**한다.

```python
from rag_home_api.shared.core.vector_store import make_db_id

db_id = make_db_id(collection_name, chunk_id)
# 예: "housing_rent_kh:RENT_강남구_신사동_단독다가구_월세_33이하"
# 예: "yj_preV2_chunkV1_cs800_ov100:SH_308944_p024_chunkV1_c01"
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

## 5. 팀원별 5분 통합 체크리스트

- **예지 (`yj`)**: 이미 `make_db_id` PK 규약·RAGAS JSON 규격 완비 ✅ → `public_housing` 핸들러 래퍼 10줄(`RAGResponse` 반환)만 연결!
- **지훈 (`jh`)**: `law_router.py` 분류 규칙을 공용 `domain_router.py`에 반영 완료 ✅ → `jh_1000` 적재 시 `make_db_id` 확인 + 법령 핸들러를 `router.register("law", ...)`로 연결 + 최종 평가 시 RAGAS 4지표 JSON 1회 저장!
- **병철 (`bc`)**: 통합 DB에는 `course_v1_small` (`text-embedding-3-small`) 기준 적재 + `make_db_id` 확인 → `rent_deals` 핸들러를 `router.register("rent_deals", ...)`로 연결 + 최종 평가 시 RAGAS 4지표 JSON 1회 저장!
- **최기혁 (`kh`)**: `housing_rent_kh` `rent_stats` 핸들러 연결 + `unify_team_results.py`·Streamlit 통합 비교 관리 + `dev_v2` 튜닝 및 `test` 20문항 최종 성적표 확정!
