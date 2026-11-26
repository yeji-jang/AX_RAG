# 법령 평가·리랭킹·검색 전용 리라이트

팀 공유 범위는 test/jh뿐이다. 실험기록.md와 기존 점수/평가 문항은 변경하지 않는다. 현재 법령형 DEV10(8+2)·TEST20(17+3)을 사용하며 빈 후보풀 파일도 그대로 두어 기존 바이트/해시 추적을 유지한다.

## 단일 CLI와 기본값

- `rag/run.py`: A/B/C 검색과 선택 리라이트를 한 진입점으로 실행한다. 기본 modes=A B, collection=jh_1000, preset=tok, 최종5개, 리라이트/생성 off다. 과거 고정 평가의 route 프리셋을 선택하려면 `--preset route`를 명시한다.
- C와 C+rewrite는 같은 B 후보 생성과 ko-reranker(CPU) 경로를 사용한다. 같은 질의의 B 전체 후보/재정렬은 공유하며, 상위5를 먼저 자른 뒤 리랭킹하지 않는다. 리라이트는 검색·재정렬 질의에만 쓰고 생성에는 원질문과 검증된 실제 상위5를 전달한다.
- `rag/score.py`: 허용 청크 우선 근거 순위, Hit@5, 전체 후보 적중/순위만 집계한다. 생성 답변 정답률·충실도·적절한 보류를 검색 적중으로 대신 채점하지 않는다.
- `rag/validate_eval.py`: DEV/TEST30개 전체의 구성, 원문 출처·조항·청크·조건 연결과 중복/근거 부족 사유를 검사한다. common.py의 정적 로더는 .env/키/DB를 읽지 않는다.

## 비용 없는 현재30문항 예제

기존 설치 환경을 활성화하고 저장소 루트에서 실행한다. 키/DB 설정과 기존 결과 로그가 없어도 동작한다.

```powershell
python -X utf8 src/rag_home_api/test/jh/rag/validate_eval.py
python -X utf8 src/rag_home_api/test/jh/rag/run.py --help
python -X utf8 src/rag_home_api/test/jh/rag/run.py --split all --modes A B C --rewrite
python -X utf8 src/rag_home_api/test/jh/rag/run.py --split dev --modes C --rewrite --generate
```

기본은 `offline_example`이다. 실제 현재 질문/법령 본문을 읽지만 후보/리랭크/리라이트/선택 답변은 모의 대역이다. `paths`는 완료/오류 건수이고 `retrieval_metrics=null`, `answer_evaluation=not_scored`이며 새 성능 점수로 쓰지 않는다. `--generate`도 기본 모드에서는 모의 응답 예제만 활성화한다.

`--tag 이름`을 선택하면 새 `results/이름.jsonl`만 만들며 기존 파일을 덮거나 선행 로그를 읽지 않는다. results/와 out/는 Git 제외다. tag를 생략하면 결과 파일을 만들지 않는다.

## 실제 실행 승인과 미검증 범위

`--execute`는 외부 실행을 요청하는 명시 옵션이며 그 자체가 승인이 아니다. 현재 평가 해시가 기존 API 승인 목록에 있어야 하고, 별도로 컬렉션/원문 지문이 승인돼야 한다. 현재 law-qa-v1은 공유 실행 코드의 승인 목록에 없고 승인 색인 목록도 비어 있어 다음 명령은 키/DB/외부 클라이언트 생성 전에 종료코드2로 차단된다.

```powershell
python -X utf8 src/rag_home_api/test/jh/rag/run.py --split dev --modes C --rewrite --generate --execute
```

승인 등록을 자동 추가하거나 우회하는 CLI 옵션은 없다. 승인된 데이터/실제 DB 색인/설치 모델을 연결한 실행은 미확인이다. 실제 생성도 `--generate`를 별도로 선택해야 하며 정답표는 모델 입력에 전달하지 않는다. 실제 답변 정답·근거·보류의 내용 평가는 별도 로컬 검토가 필요하다.

A/B/C와 C+rewrite는 질의/후보 캐시를 공유한다. 새 경로의 구성요소 지연은 독립 서비스 왕복이 아니며 과거 실험 수치를 새 실행으로 재현했다고 주장하지 않는다. 평가 바이트 해시는 당시 Windows CRLF 기준으로 Git LF 객체 해시와 다를 수 있다.

과거 run_c.py/run_rewrite_c.py 실행기는 변경 전 사본을 이번 후보의 Git 제외 out/runner-prep-*/historical에 보존했다. 공용 API·라우터·출처 포맷·개인 후속/노트북/원본 로그·test_*.py를 다시 포함하지 않는다. 이번 준비는 실제 외부 전송/비용/PR open/merge/Notion 변경을 하지 않는다.
