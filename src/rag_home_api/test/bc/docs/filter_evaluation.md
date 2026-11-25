# 조건 필터 검색 실행과 평가

기존 질문은 `abc_evaluation_questions.md`와 문장이 일치하는 Dev D01~D10 및
기존 Test T01~T20, 총 30개다. 새 최종 Test T21~T40은 사용하지 않는다.

## 현재 결과와 구현

| 검색 Hit@5 (A/B/C 공통) | 초기 필터 | 정규화 보완 |
| --- | ---: | ---: |
| Dev | 6/8 | 7/8 |
| 기존 Test | 12/17 | 17/17 |

분모는 과거 참고 근거가 있는 문항 수다. 기존 Test를 분석해 수정한 회귀 평가이며,
새 질문의 일반 성능이나 답변 정확도 100%를 뜻하지 않는다. D08의 ‘5평 후반대’는
구체적인 범위 확인 요청으로 남아 있다. RAGAS와 새 최종 Test는 실행하지 않았다.

- `metadata_filters.py`: 수동 조건 상수와 문자열·숫자 필터 검증/로컬 비교.
- `query_filters.py`: 질문의 조건 JSON 추출, 정보 요청 구분, 유형·건물명·면적 정규화.
- `lesson_pipeline.py`: Vector와 BM25에 같은 필터를 적용하고 A/B/C 검색·답변을 연결.
- `run_query_dev.py`: 기존 Dev 또는 Test만 실행. 정답·근거는 모델 검색 입력에 넣지 않는다.
- `score_dev_results.py`: 생성 후 별도로 정답 조건과 검색 ID를 대조해 Hit@5를 채점.
- `run_course_index.py`: 동일 청크 확인과 필요한 경우의 임베딩 적재.
- `run_filtered_dev.py`: 수동 D01~D04 진단 및 공통 입출력·해시 유틸리티.

질문별 예외나 정답 거래 ID를 검색 조건에 사용하지 않는다. 건물명은 지역·주택
유형으로 좁힌 원본 카탈로그에서 표기와 단지·차수 번호가 맞는 유일한 후보만 연결한다.
후보가 여러 개이거나 면적이 모순되면 확인 요청을 유지한다.

V0, text-embedding-3-small, course_v0_small, BM25 text.split(), 후보 각 20개,
RRF c=60/가중치 0.5:0.5, Dongjin-kr/ko-reranker max_length=512, 최종 5개와
gpt-6-luna 설정을 유지한다. 빈 결과에서는 조건을 풀지 않고 답변 LLM을 생략한다.

## Windows / Mac 공통 실행

저장소 루트에서 실행한다. Mac의 기존 `.env`를 유지하고 OPENAI_API_KEY와
DATABASE_URL을 설정한다. `.env`·모델 캐시·가상환경·DB는 Git에 포함하지 않는다.

```sh
uv sync --locked
uv run python src/rag_home_api/test/bc/run_course_index.py --stage check
```

새 로컬 DB가 필요하면 루트 `compose.yaml`과 `infra/postgres/README.md`를 참고한다.
기존 DB가 5432 포트를 사용 중이면 같은 포트에 다른 DB를 실행하지 않는다.
적재가 필요할 때만 아래 명령을 실행한다. 같은 ID의 본문·메타데이터가 같으면 기존
임베딩을 재사용한다. 새 DB에서는 임베딩 API를 호출한다.

```sh
uv run python src/rag_home_api/test/bc/run_course_index.py --stage index --output src/rag_home_api/test/bc/results/local_index
```

새 결과 폴더를 지정해 기존 30문항을 평가한다. 같은 결과 폴더를 덮어쓰지 않는다.
Windows PowerShell에서도 동일한 명령을 사용한다.

```sh
uv run python src/rag_home_api/test/bc/run_query_dev.py --stage answer --split dev --output src/rag_home_api/test/bc/results/local_run/dev
uv run python src/rag_home_api/test/bc/score_dev_results.py --split dev --input src/rag_home_api/test/bc/results/local_run/dev
uv run python src/rag_home_api/test/bc/run_query_dev.py --stage answer --split test --output src/rag_home_api/test/bc/results/local_run/test
uv run python src/rag_home_api/test/bc/score_dev_results.py --split test --input src/rag_home_api/test/bc/results/local_run/test
```

검색만 확인하려면 `--stage retrieve`, 질문·해시만 확인하려면 `--stage check`를 사용한다.
모델·DB가 필요한 단계는 실제 API 호출 및 reranker 모델 다운로드가 발생할 수 있다.
채점은 실행한 컴퓨터에서 먼저 한다. CRLF 체크아웃의 원본 CSV는 LF 환산 해시도
대조하고 실제 바이트 해시를 함께 기록한다. 평수 필터는 원본 환산값을 반올림하지 않는다.

## 보고서와 측정 기록

최종 HTML은 루트 `artifacts/AX_RAG_comparison_20261125.html`이다.
실행 결과·설정·원래 파일 해시는 `results/comparison_data.json` 하나에 통합했다.
정리한 원본의 UTF-8 바이트 내용은 `archive_by_sha256`에 보존하며, 원래 경로는
측정 당시의 이력이다. 원본 코드 기준 커밋은 24d32900715d0d301561e14f08ecf4c7093ac3ed다.

```sh
uv run python src/rag_home_api/test/bc/build_report.py
```

보고서 생성은 보관한 비교 JSON만 읽고 API·DB·새 최종 Test를 호출하지 않는다.
위 명령은 보관한 Windows 측정 보고서를 재생성하며, 새로운 local_run 결과를
자동으로 기존 기록에 합치지 않는다. 실행 기록과 HTML을 함께 보관한다.

실측 후 추가한 정보 요청 경계 보호는 저장된 추출 JSON 30개를 최종 코드로
재처리해 필터·처리 경로가 동일함을 확인했다. 측정 당시 코드도 통합 JSON에 보존했다.

## 검증

```sh
uv run python -m unittest discover -s src/rag_home_api/test/bc -p "test_*.py" -v
uv lock --check
git diff --check
```

테스트 47개는 외부 API·DB·실제 모델 다운로드 없이 실행한다. 실제 LangChain
검색 구성과 프롬프트는 사용하고 외부 시스템만 테스트 대역으로 바꾼다.

교안 근거는 RAG 3장 4강의 수동 메타데이터 필터와 별도 프롬프트 교안 5장 1강의
JSON 출력 체인이다. 자연어 자동 추출과 도메인 정규화는 프로젝트에 추가한 부분이다.
