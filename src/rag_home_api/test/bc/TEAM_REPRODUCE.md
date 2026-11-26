# 일반 전월세 RAG 재현 가이드 (팀장·LLM 작업용)

이 문서는 GitHub의 `feat/SBC-rag-pipeline` 브랜치 파일만으로 **전처리 → 청킹 → 임베딩 적재 → 질의**를 재현하는 절차다.
LLM 에이전트가 이 문서를 읽고 작업한다면 아래 단계를 순서대로 실행하고, 각 단계의 "성공 기준"을 확인한 뒤 다음 단계로 넘어간다.

> 범위: 테스트·평가 스크립트는 **제외**한다. 이유는 [실행하지 말 것](#실행하지-말-것) 참고.

## 기준 정보

| 항목 | 값 |
| --- | --- |
| 브랜치 | `feat/SBC-rag-pipeline` (커밋 `c1c9ad6` 기준으로 확인) |
| 원본 데이터 | `src/rag_home_api/shared/docs/서울_전월세_최근1년_관악구.csv` (git에 포함) |
| 청킹 방식 | V0 (`structural_chunk_v0.py`) |
| 청크 수 | 31,772 |
| 청크 SHA-256 | `14c3d497ff4a7cee1a9201d0ee2171b33734ed4b74f5f6f0b1021de4fc4dfa73` |
| 임베딩 모델 | `text-embedding-3-small` (1536차원) |
| PGVector 컬렉션 | `course_v0_small` |
| reranker | `Dongjin-kr/ko-reranker` (첫 실행 시 HuggingFace에서 자동 다운로드) |
| 예상 임베딩 비용 | 약 517만 토큰, 약 $0.1 |

모든 명령은 **저장소 루트**에서 실행한다.

## 0. 사전 준비

- Docker, `uv`, Python 3.13
- 본인 OpenAI API 키

```sh
git fetch origin
git switch feat/SBC-rag-pipeline
git pull
```

루트 `.env`에 아래 두 값을 설정한다 (`.env.example` 참고). 키 값은 커밋하지 않는다.

```sh
OPENAI_API_KEY=<본인 키>
DATABASE_URL=postgresql+psycopg://postgres:postgres@localhost:5432/rag_db
```

## 1. DB 실행

```sh
docker compose up -d --wait db
```

성공 기준: `docker compose ps`에서 `db`가 `healthy`.
로컬에 이미 5432 포트를 쓰는 PostgreSQL이 있으면 `.env`에 `POSTGRES_PORT`를 다른 값으로 두고 `DATABASE_URL`의 포트도 맞춘다.

## 2. 의존성 설치

```sh
uv sync --locked
uv pip install -r src/rag_home_api/test/bc/requirements-reranker.txt
```

- reranker 의존성(`sentence-transformers`, `torch`, `transformers`)은 `uv.lock`에 없어서 따로 설치한다.
- `torch`는 OS·GPU에 맞는 빌드가 필요할 수 있다. 설치가 실패하면 https://pytorch.org 의 OS별 설치 명령으로 `torch`를 먼저 설치한 후 다시 실행한다.
- 이후 `uv sync`를 다시 실행하면 lock에 없는 reranker 패키지가 지워질 수 있다. 다시 동기화할 때는 `uv sync --locked --inexact`를 쓰거나 2단계 두 번째 명령을 다시 실행한다.

## 3. 전처리·청킹 확인 (API·DB 미사용)

```sh
uv run python src/rag_home_api/test/bc/run_course_index.py --stage check
```

성공 기준: 출력이 정확히 아래와 같다.

```text
V0: 31,772 chunks / SHA-256 14c3d497ff4a7cee1a9201d0ee2171b33734ed4b74f5f6f0b1021de4fc4dfa73
```

청크 수나 해시가 다르면 CSV 또는 청킹 코드가 다른 것이다. 다음 단계로 넘어가지 말고 원인을 확인한다.

## 4. 임베딩 적재 (OpenAI API·DB 사용, 비용 발생)

```sh
uv run python src/rag_home_api/test/bc/run_course_index.py --stage index
```

- 끝나면 DB의 저장 내용을 원본 청크와 다시 대조한다.
- 성공 기준: 마지막에 `Index verified: 31772` 출력.
- 적재 기록은 `src/rag_home_api/test/bc/.local/index_v0_small/`에 남는다 (git 제외).
- 중간에 끊기면 같은 명령에 `--resume`을 붙여 재개한다. 이미 저장된 청크는 다시 임베딩하지 않는다.
- "기존 적재 기록이 있습니다" 오류가 나면 `--resume`을 붙이거나 `--output`으로 새 폴더를 지정한다.

DB에서 직접 확인하려면:

```sh
docker compose exec -T db psql -U postgres -d rag_db -c "SELECT c.name, count(e.id), min(vector_dims(e.embedding)), max(vector_dims(e.embedding)) FROM langchain_pg_collection c JOIN langchain_pg_embedding e ON e.collection_id=c.uuid GROUP BY c.name;"
```

기대값: `course_v0_small | 31772 | 1536 | 1536`

## 5. 질의 실행

저장소 루트에서 실행한다. 첫 실행은 reranker 모델 다운로드 때문에 오래 걸린다.

```sh
uv run python - <<'PY'
import sys
from pathlib import Path

sys.path.insert(0, str(Path('src/rag_home_api/test/bc').resolve()))
from cached_pipeline import CachedPipeline

pipeline = CachedPipeline()
result = pipeline.answer(
    '신림동에서 보증금 500만원 이하, 월세 45만원 이하인 집을 찾아줘',
    scope='local-test', index_version='course_v0_small:local-reindex', system='C',
)
print(result['status'])
print(result['response'])
print(len(result['sources']), 'sources')
PY
```

성공 기준: `status`가 `success`이고 답변과 출처(sources)가 1개 이상 나온다.

- `system`: `A`=벡터 검색, `B`=하이브리드(벡터+BM25), `C`=하이브리드+reranker
- `index_version`: 직접 적재한 DB이므로 임의의 고정 문자열을 쓴다. 데이터를 다시 적재하면 값을 바꾼다 (캐시 키에 포함됨).
- 질문 임베딩도 반드시 `text-embedding-3-small`이어야 한다. 다른 1536차원 모델을 쓰면 오류 없이 검색 품질만 망가진다.

## 실행하지 말 것

아래 파일은 GitHub에 남아 있지만 현재 코드와 맞지 않아 **import 오류로 실행되지 않는다** (`metadata_filters.py`에서 `MANUAL_DEV_FILTERS`가 제거됨). 정리 예정이다.

- `test_filtered_pipeline.py`, `test_query_filters.py`, `test_score_dev_results.py`
- `run_filtered_dev.py`, `run_query_dev.py`, `score_dev_results.py`, `check_metadata_filter.py`

`run_course.py`(RAGAS 평가)는 import는 되지만 실행 검증을 하지 않았고 API 비용이 크므로 이 절차에서 제외한다.
`embedding_small.py`, `embedding_large.py`는 예전 실험용이다. 4단계의 `run_course_index.py`를 사용한다.

## 검증 상태

| 단계 | 상태 |
| --- | --- |
| 3. 청킹 | GitHub 버전으로 실행해 31,772청크·위 해시 확인 |
| 4. 임베딩 적재 | 코드 import만 확인. 같은 코드로 만든 원작성자 DB는 31,772행·1536차원 확인 |
| 5. 질의 | 코드 import만 확인. 새 환경에서의 실행은 미검증 |

4·5단계에서 오류가 나면 전체 오류 메시지와 실행한 명령을 원작성자에게 공유한다.
