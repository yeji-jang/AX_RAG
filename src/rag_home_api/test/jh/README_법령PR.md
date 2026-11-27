# 법령 평가·리랭킹·검색 전용 리라이트 인계

범위는 `test/jh`의 법령 DEV10(8+2)·TEST20(17+3), A/B/C 및 선택 리라이트다. 상세 실험기록·기존 점수·평가 문항·빈 후보풀은 보존한다. 챗봇/공용 API 연결은 팀장 담당이다.

## 1. 기본 실행과 비용 없는 확인

저장소 루트에서 기존 프로젝트 환경의 Python을 사용한다. 기본은 A/B, jh_1000, tok, 리라이트/생성 off다. 과거 법령형 실측 조건은 **route**이므로 재현 시 명시한다.

```powershell
python -X utf8 src/rag_home_api/test/jh/rag/validate_eval.py
python -X utf8 src/rag_home_api/test/jh/rag/run.py --help
python -X utf8 src/rag_home_api/test/jh/rag/run.py --inspect-local --collection jh_1000
python -X utf8 src/rag_home_api/test/jh/rag/run.py --split all --modes A B C --preset route --rewrite --generate
```

마지막 명령도 `--execute`가 없으므로 **모의 오프라인 실행**이다. 실제 현재30문항·법령을 읽되 검색/리랭크/변환/답변은 대역이다. `retrieval_metrics=null`, `answer_evaluation=not_scored`이며 실제 성능 점수가 아니다. 키·DB·기존 로그 없이 동작한다. `--tag 이름`은 선택한 새 results 파일만 만들며 덮어쓰지 않는다.

C/C+rewrite는 B의 **전체 후보**를 같은 ko-reranker(CPU)로 재정렬한 후5개를 선택한다. 리라이트는 검색/재정렬에만 쓰고 답변 생성은 원질문+검증된 실제5청크만 사용한다. 정답표는 모델에 보내지 않는다. `score.py`는 Hit@5와 전체 후보 순위만 집계하며 답변 정답·충실도·안전 보류를 대신 판정하지 않는다.

## 2. 팀 설정과 색인 준비

일반 팀 설정은 저장소 루트의 기존 `.env` 또는 환경변수로 지정한다. `OPENAI_API_KEY`, `DATABASE_URL`이 필요하며 모델은 기존 `EMBEDDING_MODEL=text-embedding-3-small`, `LLM_MODEL=gpt-6-luna`를 사용한다. DB 주소는 PostgreSQL/psycopg 형식이다. 키/주소/로컬 승인 파일은 커밋하지 않는다. 공급자는 공식 `https://api.openai.com/v1`로 고정한다.

A/B는 기존 프로젝트 의존성을 사용한다. C는 추가로 기존 실험의 설치 환경(sentence-transformers6.3.0, CPU torch2.13.0, transformers5.23.0)과 `Dongjin-kr/ko-reranker` 로컬 캐시가 필요하다. 기존 설치/캐시를 우선 사용하며 코드가 패키지 설치나 모델 다운로드를 하지 않는다. 없으면 `[RerankerUnavailable]`로 멈추므로 팀장이 별도로 환경을 준비해야 한다.

기존 DB의 `jh_1000`을 우선 확인한다. 1000/200으로 만든 현재 법령167청크의 `chunk_id`·본문·`row_id`·`article`·`doc_id`·분류가 맞아야 한다. `jh_500`을 선택하면 그 설정으로 별도 검사한다. 혼합 컬렉션의 비법령 행은 검색하지 않는다. 공고 전용 `yj_*` 컬렉션이나 원본 조문 매핑이 빠진 공용 loader 결과를 법령 색인으로 간주하거나 강제로 변환하지 않는다.

```powershell
# 팀장이 실행: PostgreSQL을 읽기만 함. OpenAI/임베딩/모델 호출·적재·승인 등록 없음.
python -X utf8 src/rag_home_api/test/jh/rag/run.py --check-index --collection jh_1000
```

없는 컬렉션은 `[IndexMissing]`, 누락/다른 본문은 `[IndexIncomplete/IndexMismatch]`, 조문 매핑 누락은 `[IndexMetadataMismatch]`, 벡터 차원/값 오류는 `[IndexVectorInvalid]`로 거부한다. 원본 URL이 저장 행에 없으면 **동일 ID·본문·조문을 검증한 로컬 원문 출처만** 사용하며 다른 URL을 만들지 않는다.

색인이 없다면 팀의 승인된 적재 절차에 아래 공개 법령 청크를 전달해 **jh_1000 별도 컬렉션**을 준비한다. 기존 공고 컬렉션은 덮어쓰지 않는다. 내보내기 자체는 API/DB 호출이 없으며 새 임베딩/DB 적재는 이 CLI가 자동 실행하지 않는다.

```powershell
New-Item -ItemType Directory -Force src/rag_home_api/test/jh/out | Out-Null
$env:PYTHONPATH='src/rag_home_api/test/jh/rag'
python -X utf8 -c "from common import law_chunks; import json; from pathlib import Path; p=Path('src/rag_home_api/test/jh/out/law_chunks.jsonl'); p.write_text(''.join(json.dumps({'text':d.page_content, **d.metadata}, ensure_ascii=False)+'\n' for d in law_chunks('jh_1000')), encoding='utf-8')"
```

적재 시 이 JSONL의 `text`와 나머지 메타데이터를 그대로 보존한다. 테이블 전체 PK 충돌을 피하려면 DB id에 컬렉션 접두어를 붙여도 되지만 메타데이터 `chunk_id`는 바꾸지 않는다. 팀 공고 indexer의 `load_chunks(path, collection)`도 이 입력 형태를 읽지만 현재 기본 입력/컬렉션은 공고용으로 고정돼 있으므로 **그 스크립트를 그대로 실행해 법령 적재가 된다고 보장하지 않는다**. 팀의 적재 담당자가 입력·컬렉션·허가된 임베딩 모델을 확인해야 한다. 준비 후 `--check-index`가 통과하기 전에는 RAG 실행하지 않는다.

`--check-index`는 평가 바이트 해시(짧은/전체), 원문·메타데이터 지문 `corpus_sha256`, DB의 실제 ID·본문·저장벡터 지문 `index_sha256`을 출력한다. 과거 문서의 인덱스 해시를 무조건 복사하지 말고 팀 DB 출력과 적재 기록을 검토한다. 1536차원만으로 임베딩 모델을 증명할 수 없으므로 모델 태그가 없는 기존 색인은 팀장이 적재 모델을 확인해야 한다. 점검 출력의 `approved=false`는 자동 승인이 아님을 뜻한다.

## 3. 로컬 승인 후 실제 실행 (이번 인계에서는 실행하지 않음)

```powershell
Copy-Item src/rag_home_api/test/jh/execution.example.json src/rag_home_api/test/jh/execution.local.json
```

예시는 거부 상태와 **가짜 해시만** 담고 있다. 팀장이 공개 가상 질문/법령 전송 범위와 DB 점검 결과를 검토한 뒤 로컬 파일에 정확한 해시·컬렉션·모델, 허용 `splits/modes/preset`, `rewrite/generate`를 지정하고 명시적으로 승인한다. 예시를 복사하거나 점검 명령을 실행하는 것만으로 승인되지 않는다. `execution.local.json`·out/results는 Git 제외다. 개인 승인 설정/키/원본 로그를 공유하지 않는다.

- 파일 없음: `[ApprovalRequired]`, 승인 전: `[ApprovalNotGranted]`, 형식/필수값 누락: `[ApprovalConfigInvalid]`.
- 평가셋/원문/컬렉션/공급자/모델 불일치: `[ApprovalMismatch]`, split/mode/preset/선택 기능 범위 초과: `[ApprovalScopeMismatch]`. 여기까지 키/DB/외부 클라이언트를 만들지 않는다.
- 일반 설정 누락: `[RuntimeConfigMissing]`, 팀 모델 설정 불일치: `[RuntimeModelMismatch]`, 실제 DB 지문 변경: `[IndexApprovalMismatch]`. DB 지문 확인 전에 모델 API를 호출하지 않는다.

```powershell
# 팀장이 해당 범위를 로컬 승인한 뒤에만 사용. 여기서는 실제 호출하지 않았음.
python -X utf8 src/rag_home_api/test/jh/rag/run.py --split dev --modes A B C --preset route --rewrite --generate --execute
# DEV 검토·조건 고정 및 TEST 범위 별도 승인 후
python -X utf8 src/rag_home_api/test/jh/rag/run.py --split test --modes A B C --preset route --rewrite --generate --execute
```

현재 실제 실행기는 문서 재임베딩 대신 검증한 DB 저장벡터를 중복 제거에 재사용하고 질문 벡터/동일 B 후보를 방식 간 공유한다. 생성에는 검증된 공개 법령만 주입하며 근거 부족 상태는 빈 인용을 요구한다. 승인 후에도 실제 모델/색인/계정의 실행 성공 및 이전 정답17/17·안전 보류3/3의 재현은 **미검증**이다. 공유 캐시 구성요소 시간은 독립 서비스 왕복 속도가 아니다. 답변 내용 평가는 별도 로컬 검토가 필요하다.

평가 파일 바이트 해시는 Windows CRLF/Git LF에 따라 달라질 수 있다. 다른 OS에서는 `--inspect-local`의 현재 짧은/전체 해시를 검토해 승인하며 문항을 점수에 맞춰 수정하지 않는다. 기존 상세 실험기록의 실행 명령/점수는 역사 기록으로 보존한다. 과거 실행기·생성 원본·노트북·후속 실험·test_*.py는 PR에 포함하지 않는다. 실제 실행 승인이나 챗봇 연결은 PR merge와 별개이며 이번 작업에서 DB/모델 호출·merge·Notion 수정은 하지 않는다.
