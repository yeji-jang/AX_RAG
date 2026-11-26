# RAGAS 실행 확인과 오류 분석

작성일: 2026-11-26

> **후속 상태: 수정 후 재실행 완료(11:04 한국 시간).** 6개 묶음, 총 102개 입력이 완료됐고
> 결과 파일의 `ragas_status`가 `complete`임을 확인했다.
> [최종 실측 분석](./ragas_comparison_analysis.md)을 참고한다. 아래는 중단 당시의 원인·복구 이력이다.

## 확인한 실행 상태

사용자가 터미널에서 `compare_saved_quality.py --ragas`를 실행했고,
첫 평가 묶음의 진행률이 `34/34`, 100%에 도달한 뒤 집계 단계에서 중단됐다.
각 묶음은 17문항 × Context Precision·Context Recall 2개 지표다.
평가 연결은 이루어졌으며, 이번 중단의 원인은 결과 열 이름을 잘못 읽은 코드다.

```text
KeyError: 'context_precision'
```

화면의 DeprecationWarning은 사용 중단 예정 안내이며 이번 예외의 직접 원인이 아니다.
이번 실행은 저장된 검색 결과를 심사하는 작업으로 DB의 새 검색이나 캐시 지연 측정과 구분한다.

## 원인

| 항목 | 값 |
| --- | --- |
| 사용한 지표 클래스 | LLMContextPrecisionWithReference |
| 코드가 가정한 열 | context_precision |
| 설치된 RAGAS가 실제 반환한 열 | llm_context_precision_with_reference |
| Recall의 실제 열 | context_recall |

작성한 집계 코드가 Context Precision 열을 하드코딩해, API 평가 후 DataFrame을 읽는 단계에서 실패했다.
설치된 지표의 실제 `name`을 사용해 만든 RAGAS EvaluationResult를 `to_pandas()`로 변환하는
외부 호출 없는 재현에서도 같은 KeyError를 확인했다.

## 수정 내용

`compare_saved_quality.py`를 수정했다.

1. `summarize_ragas()`에서 `metric.name`으로 실제 점수 열을 읽는다.
2. 열 존재 여부·입력 문항 수·점수 범위·유한성을 검사한다. 결측값을 조용히 제외해 평균을 높이지 않는다.
3. 각 묶음의 원본 점수와 질문 ID를 **요약하기 전에** `.local/ragas_checkpoints/`에 저장한다.
4. 질문·문맥·참고 정답·모델 정의·RAGAS 버전·지표 이름이 같은 완료 묶음은 재실행 시 재사용한다.
5. 원자적 파일 교체로 저장 중 손상 위험을 줄인다.
6. 같은 입력의 완료된 RAGAS 결과를 오프라인 명령으로 다시 덮어 지우지 않는다.

## 수정 검증

외부 평가 호출을 모의 평가로 바꾸고 실제 RAGAS 결과 객체의 `to_pandas()` 경로를 사용했다.
초기/정규화 × A/B/C 6개 묶음 집계가 성공하고, 두 번째 실행에서는 평가 함수를 호출하지 않고
체크포인트를 재사용하는 것을 확인했다. 모의 점수는 실제 성능 결과 파일에 저장하지 않았다.
문법·차이 공백 검사도 통과했다.

이 검증은 코드 오류·재개 동작의 확인이며, 새 RAGAS 실측 완료를 의미하지 않는다.

## 실제 비교 수치의 상태

현재 `.local/quality_comparison.json`에는 오프라인 조건 기반 지표가 있고 새 RAGAS 완료 결과는 없다.
첫 묶음도 원본 점수 저장 전에 프로세스가 종료됐으므로 그 API 결과를 복구할 수 없다.
따라서 수정 후 첫 재실행에서는 첫 묶음을 다시 평가해야 한다. 이후 완료된 묶음은 저장·재사용한다.

```sh
uv run python src/rag_home_api/test/bc/compare_saved_quality.py --ragas
```

터미널에는 `initial_mac/A`부터 `normalized/C`까지 묶음별 시작·완료가 표시된다.
성공 시 최종 파일에 `ragas_status: complete`와 양쪽 A/B/C의 Context Precision·Recall이 저장된다.
평가 완료 후 `docs/ragas_comparison_analysis.md`에 실제 수치·증감·질문별 비교 분석도 자동 작성한다.
미완료 결과나 모의 검증 점수로 해당 분석 문서를 생성하지 않도록 검사한다.
6개 묶음은 총 102개 입력 행, 지표 단위로는 204개 평가 작업이며
LLM 호출 수는 지표 내부의 문맥·진술 분해에 따라 더 많을 수 있다.

실제 완료 값이 없는 상태에서 RAGAS 개선 폭을 추정해 작성하지 않는다.
확보된 조건 기반 Precision·Recall·F1과 구현 상세는
[캐싱 및 검색 지표 비교](./caching_and_quality_comparison.md)에 별도로 정리되어 있다.
