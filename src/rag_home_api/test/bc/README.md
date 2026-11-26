# 거래 검색·답변 파이프라인

현재 `test/bc`에는 챗봇 실행 코드와 팀 공유용 문서·평가 질문을 보관한다. FE용 HTTP API와 대화
세션 관리는 아직 구현하지 않았다. 데이터는 과거 거래이며 현재 매물 유무는 확인할 수 없다.

## 파일 역할

| 파일 | 역할 |
| --- | --- |
| `structural_chunk_v0.py` | CSV 정제, 거래 본문·메타데이터·청크 생성 |
| `structural_chunk_v1.py` | 검색용 요약을 추가한 청킹 비교 구현. 현재 실행 경로는 V0 사용 |
| `run_course_index.py` | V0/small 임베딩 적재·재개·저장 검증 |
| `metadata_filters.py` | 필터 필드·연산자 검증과 로컬 조건 비교 |
| `query_filters.py` | 자연어 조건 추출·정규화·검색·답변 실행 |
| `lesson_pipeline.py` | PGVector, BM25, reranker, 답변 체인 |
| `cached_pipeline.py` | 정확 일치 응답 캐시와 팀 서비스용 호출 진입점 |

## 준비와 적재

저장소 루트에서 `uv sync --locked`를 실행한다. 루트 `.env`에
`OPENAI_API_KEY`, `DATABASE_URL`을 설정하고 PostgreSQL의 vector 확장을 준비한다.
DB 실행 방법은 루트 README를 참고한다.

```sh
uv run python src/rag_home_api/test/bc/run_course_index.py --stage check
uv run python src/rag_home_api/test/bc/run_course_index.py --stage index
```

`check`는 CSV를 읽어 청크 수와 해시를 확인한다. `index`는 API·DB를 사용하며,
이미 저장된 동일 청크의 임베딩은 재사용한다. 중단 후 재개할 때는 `--resume`을
추가한다. 적재 기록과 모델 캐시는 Git에서 제외한 `.local/`에 저장한다.

## 질문 실행과 API 연결 지점

캐시를 적용할 팀 서비스에서는 아래 진입점을 사용한다. 서비스 시작 시 한 번 만든
인스턴스를 재사용한다. scope는 서버에서 관리하는 사용자/세션 범위로 지정하고,
DB 데이터·색인이 갱신되면 index_version도 갱신한다.

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path('src/rag_home_api/test/bc').resolve()))
from cached_pipeline import CachedPipeline

pipeline = CachedPipeline()
result = pipeline.answer(
    '신림동에서 보증금 500만원 이하, 월세 45만원 이하인 집을 찾아줘',
    scope='session-123', index_version='course_v0_small:2026-11-26-v1', system='C',
)
print(result['response'], result['cache_hit'])
```

캐시 응답의 `sources`는 id/page_content/metadata 목록이다. `cache_hit`,
`pipeline_executed`, `elapsed_seconds`도 반환한다. TTL 600초·최대 100항목이며,
오류·빈 검색 결과·조건 확인 요청은 저장하지 않는다. 프로세스 재시작 시 비워진다.
대화 이력은 현재 입력에 포함되지 않으며, 추후 연결 시 캐시 키에 반영해야 한다.

아래는 캐시를 거치지 않고 기존 함수를 직접 호출하는 방법이다.

저장소 루트에서 Python으로 아래 함수를 호출할 수 있다.

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path('src/rag_home_api/test/bc').resolve()))
from query_filters import build_filter_extractor, execute_query

question = '신림동에서 보증금 500만원 이하, 월세 45만원 이하인 집을 찾아줘'
extractor = build_filter_extractor()
plan = extractor.invoke(question)
result = execute_query(question, plan, system='C')
print(result['response'])
```

`A`는 Vector, `B`는 Hybrid, `C`는 Hybrid + reranker다. 기본은 `C`이며
조건이 모호하거나 제공 데이터로 확인할 수 없는 요청은 안내를 반환한다.

추후 BE의 채팅 API가 조건 추출과 `execute_query`를 호출하고, FE에는
`status`, `response`, 출처 목록을 JSON으로 반환한다. `documents`는 LangChain
Document 객체이므로 본문과 metadata를 추출해 직렬화해야 한다. 요청마다 답변 체인을
생성해 사용하고, 대화 이력·추가 질문으로 조건을 이어가는 기능은 별도로 구현해야 한다.

## 팀 공유 자료

- `docs/data_and_metrics_guide.md`: CSV 분포·결측치, 높은 수치의 이유, 지표별 현재 강점·약점과 개선 방향

- **`실험기록.md`: 팀 공유용 실험 기록. 일정·설정·변경·결과·판단 순으로 정리**
- **먼저 읽기 — `docs/development_history.md`: 버전·단계별 변경 코드와 수치 변화를 한곳에 정리한 종합 이력**
- `docs/embedding_comparison_report.md`: 임베딩·청킹 비교 기록
- `docs/abc_evaluation_questions.md`, `docs/abc_test_questions_v3.md`: 평가 질문 설명
- `eval/*.json`: 평가 질문 원본

`docs`는 실험 당시의 기록이다. 문서에 등장하는 평가 실행기·채점기·테스트 코드와
결과 데이터·보고서 생성기는 현재 정리된 작업 폴더에 포함하지 않는다.
문서 속 과거 실행 명령은 이력 참고용이며, 현재 실행 방법은 이 README를 따른다.
