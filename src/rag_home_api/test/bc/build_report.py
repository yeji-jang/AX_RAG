"""교안 재실행 결과와 교안 범위 안의 개선 항목을 별도 HTML로 정리한다."""
import hashlib
import html
import json
import math
import re
import statistics
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from run_course import RESULTS, ROOT, read, save


def esc(value):
    return html.escape(str(value),quote=True)


def table(headers,rows):
    return '<div class="table-wrap"><table><thead><tr>'+''.join(f'<th>{esc(x)}</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join(f'<td>{x}</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table></div>'


def main():
    records=read(RESULTS/'abc_results.json')
    metrics=read(RESULTS/'ragas_results.json')
    filtered=read(RESULTS/'metadata_filter_dev.json')
    config=read(RESULTS/'config.json')
    fresh_test=read(ROOT/'eval/abc_test_questions_v3.json')
    preserved=read(Path(__file__).parent/'previous_run_preservation.json')
    existing_preserved={name:digest for name,digest in preserved.items() if (ROOT/name).exists()}
    assert all(hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest for name,digest in existing_preserved.items())
    assert len(records)==60 and len({(r['system'],r['question_id']) for r in records})==60
    metric_keys=['llm_context_precision_with_reference','context_recall','faithfulness','answer_relevancy']
    summaries=[]
    for system in ['A','B','C']:
        items=[r for r in records if r['system']==system]
        support=[r for r in items if r['reference_answerable']]
        success=[r for r in items if r['status']=='success']
        item={'system':system,'hit_count':sum(r['hit_at_5'] or 0 for r in support),'denominator':17,
              'successes':len(success),'attempts':20,'mean_seconds':statistics.mean(r['seconds'] for r in success),
              'median_seconds':statistics.median(r['seconds'] for r in success),'metrics':{}}
        for key in metric_keys:
            values=[r[key] for r in metrics if r['system']==system and r.get(key) is not None and math.isfinite(r[key])]
            item['metrics'][key]={'mean':statistics.mean(values) if values else None,'n':len(values),'errors':17-len(values)}
        summaries.append(item)
    save('summary.json',summaries)
    before=sum(r['before_hit_at_5'] for r in filtered)
    after=sum(r['after_hit_at_5'] for r in filtered)
    old=read(ROOT/'.local/experiment_20261125/abc_summary.json')
    old_summary={r['system']:r for r in old['systems']}
    old_report=ROOT/'embedding_comparison_report.html'
    style_source=old_report if old_report.exists() else Path(__file__).parent/'course_rag_report.html'
    style=re.search(r'<style>(.*?)</style>',style_source.read_text(),re.S).group(1)
    old_report_link='<p><a href="../embedding_comparison_report.html">기존 HTML 보기</a></p>' if old_report.exists() else ''
    now=datetime.now(ZoneInfo('Asia/Seoul')).strftime('%Y-%m-%d %H:%M KST')
    hit_rows=[[f'<b>{r["system"]}</b>',f'{r["hit_count"]}/17 ({r["hit_count"]/17:.1%})',
               f'{r["successes"]}/20',f'{r["mean_seconds"]:.2f}초',f'{r["median_seconds"]:.2f}초',
               str(old_summary[r['system']]['hit_count'])+'/17'] for r in summaries]
    metric_rows=[[f'<b>{r["system"]}</b>']+[
        ('미측정' if r['metrics'][key]['mean'] is None else f"{r['metrics'][key]['mean']:.3f}")
        +f"<small>n={r['metrics'][key]['n']}/17 · 오류 {r['metrics'][key]['errors']}</small>"
        for key in metric_keys] for r in summaries]
    filter_rows=[[r['question_id'],esc(r['question']),str(r['before_hit_at_5']),str(r['after_hit_at_5']),f"{r['allowed_transactions']:,}"] for r in filtered]
    question_map={q['id']:q for q in read(ROOT/'eval/abc_test_questions.json')['questions']}
    by_question=[]
    for qid in sorted(q for q in question_map if question_map[q]['reference_answerable']):
        by_question.append([qid,esc(question_map[qid]['question'])]+[
            '<span class="pass">적중</span>' if next(r for r in records if r['question_id']==qid and r['system']==system)['hit_at_5'] else '<span class="fail">미적중</span>'
            for system in ['A','B','C']])
    cases=[]
    for r in records:
        contexts=''.join(f'<li><pre>{esc(text)}</pre></li>' for text in r['retrieved_contexts'])
        score='근거 없음 · 별도 검토' if r['hit_at_5'] is None else ('Hit@5 적중' if r['hit_at_5'] else 'Hit@5 미적중')
        cases.append(f'<details class="case" data-system="{r["system"]}"><summary><b>{r["question_id"]} · {r["system"]}</b><span>{esc(r["user_input"])}</span><span>{score}</span></summary><div class="case-body"><h4>실제 LangChain 응답</h4><p>{esc(r["response"])}</p><p class="muted">상태 {r["status"]} · 응답 {r["seconds"]:.2f}초 · CSV 행 {esc(r["source_rows"])}</p><h4>실제 주입한 검색 문맥</h4><ol>{contexts}</ol></div></details>')
    improvements=[
        ['1 · 우선','지역·유형·예산·면적·층 조건을 메타데이터 필터로 지정','<a href="https://app.notion.com/p/ab31769337a183bf90c781a8591058ab">3장 4강 · 메타데이터 필터링</a>',f'Dev 4문항 수동 지정 진단에서 {before}/4 → {after}/4. 자동 조건 추출 성능은 별도 검증 필요.'],
        ['2','정답이 후보에 포함되는지 확인한 후 후보 수·최종 k 조정','<a href="https://app.notion.com/p/8d61769337a183bab51e01fd74c4f48f">3장 3강 · Top-k·threshold</a>','현재 후보는 검색기별 20개, C 입력 최대 20개. Dev에서 후보 포함 여부와 응답 시간을 함께 비교.'],
        ['3','건물명·표현 변형별로 BM25 : Vector 가중치 조정','<a href="https://app.notion.com/p/8841769337a182d7a706817d0c2c6718">5장 3강 · RRF·가중치 튜닝</a>','0.5:0.5를 기준으로 교안의 0.8:0.2·0.2:0.8을 Dev에서 검증. 개선은 아직 측정하지 않음.'],
        ['4','검색 실패·순위 실패·답변 실패를 분리','<a href="https://app.notion.com/p/4611769337a183ae9d5601c00b90dfff">7장 5강 · 검색 실패 분석</a>','정답 거래가 없어서 못 찾은 것인지, 후보에서 누락됐는지, 순위가 밀렸는지 구분.'],
        ['5','범위 밖 질문의 정보 부족 안내와 출처 검증','<a href="https://app.notion.com/p/c2f1769337a182ef90c5817eae140074">3장 7강 · 출처·후처리</a> / <a href="https://app.notion.com/p/8a41769337a18241b00e01d7c6d4e23f">7장 6강 · 생성 실패</a>','현재 매물 여부·관리비·주차·반려동물 등을 추정하지 않도록 별도로 점검.'],
    ]
    roadmap=table(
        ['단계','구체적으로 바꿀 내용','유지할 조건','판단·기록할 결과'],[
        ['0 · 평가셋 구분','개선은 기존 Dev 10문항만 사용. 새 T21~T40은 설정 확정 후 최종 평가에만 사용.','현재 실험의 T01~T20과 측정값은 탐색 기록으로 보존.','Dev는 과거 참고 근거 8문항과 근거 없음 2문항을 따로 집계.'],
        ['1 · 조건 필터','지역·주택 유형·전월세 구분·예산·면적·층·건축연도 조건을 먼저 제한. 가격은 만원, 면적은 반올림 전 환산값으로 비교.','우선 V0·small, 후보 수 20, 가중치 0.5:0.5, 최종 5개는 그대로 유지.','필터로 인해 정답이 제외되지 않는지, 각 결과가 모든 조건을 만족하는지 원본과 대조.'],
        ['2 · 후보 단계 점검','필터 후 후보 → Vector/BM25 후보 → 결합 후보 → 재정렬 상위 5개를 단계별로 기록. 필요하면 후보 수 20·40·60을 Dev에서 비교.','최종 Hit@5는 계속 상위 5개로 계산. 최종 k를 늘린 값을 기존 점수와 혼합하지 않음.','정답이 처음 누락된 단계와 응답 시간·처리 문서 수를 비교. 후보 확대 수치는 제안이며 아직 실험하지 않음.'],
        ['3 · 검색 가중치','동일 후보 수에서 BM25:Vector 0.5:0.5, 0.8:0.2, 0.2:0.8을 비교. 건물명 질문과 조건 중심 질문을 나눠 확인.','필터·생성 모델·프롬프트·reranker를 함께 바꾸지 않음.','전체 Dev와 유형별 적중 수, 악화된 문항, 지연을 함께 기록.'],
        ['4 · 답변·평가 점검','선택한 거래의 금액·면적·계약기간·인용이 원문과 맞는지 확인. 현재 매물 확인 불가와 정보 부족 안내를 따로 검토.','정직한 확인 불가 안내를 없애 RAGAS 점수만 높이지 않음.','Hit@5·RAGAS·핵심 사실 충족·근거 없음 대응을 별도 지표로 기록.'],
        ['5 · 최종 검증','Dev로 정한 모델·본문·필터·후보 수·가중치·프롬프트를 고정한 뒤 새 20문항을 A/B/C로 1회 평가.','새 Test 결과를 본 뒤 고친 실험은 별도의 추가 실험으로 분리.','기본 60건과 참고 근거 17문항×3구성의 RAGAS 51건을 새 결과 폴더에 저장.'],
    ])
    evaluation_sets=table(['역할','문항·파일','앞으로의 사용'],[
        ['개선·튜닝','<code>abc_dev_questions.json</code><small>D01~D10 · 기존 10문항 유지</small>','새 설정과 실패 원인은 여기서 검증. 과거 참고 근거 8 + 근거 없음 2.'],
        ['새 최종 평가','<code>abc_test_questions_v3.json</code><small>T21~T40 · 새 20문항 · 미실행</small>','직접 검색 10 + 표현 변형 7 + 근거 없음 3. 팀 교차 검토 후 고정하고 최종 1회 실행.'],
        ['이전 실험 보존','<code>abc_test_questions.json</code><small>T01~T20 · 이미 실행하고 결과를 확인한 문항</small>','이 HTML의 기존 60건 분석용. 새 독립 최종 평가처럼 재사용하지 않음.'],
    ])
    no_evidence=[r for r in records if not r['reference_answerable']]
    none_rows=[[r['question_id'],r['system'],esc(r['response'])] for r in no_evidence]
    output=f'''<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>교안 방식 RAG 재실행 · 결과와 개선 방향</title><style>{style}</style></head><body>
    <header><div class="wrap"><div class="eyebrow">COURSE-ALIGNED LANGCHAIN EXPERIMENT</div><h1>교안 방식으로 다시 실행한<br>RAG 결과와 개선 방향</h1><p class="lead">기존 20문항의 교안 방식 실험 결과를 보존하고, Dev 10문항으로 진행할 개선 방향과 새 최종 평가셋을 정리했습니다. 새 20문항은 아직 실행하지 않았습니다.</p><p class="meta">{now} · V0 + text-embedding-3-small · 기존 20문항 탐색 기록 · 실제 검색·응답 기록 기반</p>{old_report_link}</div></header>
    <nav><div class="wrap"><a href="#evaluation">평가셋 구분</a><a href="#results">기존 결과</a><a href="#method">교안 구성</a><a href="#ragas">RAGAS</a><a href="#diagnosis">낮은 점수의 원인</a><a href="#improve">개선 로드맵</a><a href="#cases">원본 응답</a></div></nav><main class="wrap">
    <section id="evaluation"><h2>앞으로는 Dev 10문항으로 개선합니다</h2>{evaluation_sets}<p class="notice">기존 20문항으로 수행한 실제 결과를 Dev 10문항의 성능으로 바꿔 적지 않았습니다. 아래 수치는 모두 이미 실행한 T01~T20의 기록입니다. 새 평가셋 <b>T21~T40</b>은 원본 CSV로 정답 근거만 검증했으며 검색·생성·RAGAS를 실행하지 않았습니다.</p><p><a href="../abc_test_questions_v3.md">새 20문항 질문표 보기</a> · <a href="../abc_test_questions_v3.json">새 문항·모범 답안·근거 JSON</a><small>버전 {esc(fresh_test['version'])} · 교차 검토 후 확정할 초안</small></p></section>
    <section id="results"><h2>기존 20문항 실험 결과 · 탐색 기록</h2><div class="cards">{''.join(f'<div class="card"><div class="label">{r["system"]} · 기존 문항 Hit@5</div><div class="value">{r["hit_count"]}/17</div><small>{r["hit_count"]/17:.1%} · 호출 성공 {r["successes"]}/20</small></div>' for r in summaries)}<div class="card"><div class="label">수동 조건 필터 Dev 진단</div><div class="value">{before}/4 → {after}/4</div><small>자동 조건 추출·최종 평가 점수와 구분</small></div></div>
    {table(['구성','교안 재실행 Hit@5','호출 성공','응답 평균','응답 중앙값','이전 SDK 실험'],hit_rows)}
    <p class="notice"><b>데이터 자체의 성능 상한으로 볼 수 없습니다.</b> 기존 20문항 중 과거 참고 근거가 있는 17문항은 모두 원본 CSV에 조건을 만족하는 거래가 있습니다. 같은 데이터에서도 검색·재정렬 방식과 필터 적용에 따라 결과가 달라졌습니다. 다만 교안 재실행도 모든 질문을 해결한 상태는 아닙니다.</p>
    <p class="muted">Hit@5는 과거 참고 거래가 상위 5개 안에 있었는지의 비율이며 실시간 매물 존재 확률이 아닙니다. 17문항 중 한 문항 차이는 약 5.9%p입니다. 기존 실험과는 토큰화·재정렬·생성 형식이 함께 달라 단일 요소의 인과 효과로 해석하지 않습니다. 기존 실험의 실패를 삭제하거나 최선의 재실행만 선택하지 않았습니다.</p>
    <details><summary>기존 20문항의 검색 결과</summary>{table(['ID','질문','A','B','C'],by_question)}</details></section>
    <section id="method"><h2>사용한 구성은 교안 안에 있습니다</h2>
    {table(['단계','이번 실행','해당 교안'],[
      ['임베딩','OpenAIEmbeddings(text-embedding-3-small)','<a href="https://app.notion.com/p/27d1769337a18332be40812937a0eedf">2장 4강</a>'],
      ['저장·A 검색','PGVector → as_retriever()','<a href="https://app.notion.com/p/f621769337a182e0bea701abb8e9a158">2장 5강</a>'],
      ['B · Hybrid','BM25Retriever + EnsembleRetriever, 공백 토큰화, RRF 가중치 0.5:0.5','<a href="https://app.notion.com/p/6a71769337a1825faaeb016d5c56a50d">5장 2강</a>'],
      ['C · 재정렬','HuggingFaceCrossEncoder(Dongjin-kr/ko-reranker) + CrossEncoderReranker + ContextualCompressionRetriever','<a href="https://app.notion.com/p/3311769337a183daa69f81b97774302d">5장 4강</a>'],
      ['생성','Retriever → format_docs → ChatPromptTemplate → init_chat_model → StrOutputParser','<a href="https://app.notion.com/p/2591769337a1833a9546016e52c4a15e">3장 6강</a>'],
      ['평가','LangchainLLMWrapper + LangchainEmbeddingsWrapper + ragas.evaluate','<a href="https://app.notion.com/p/3761769337a182888d2c01b9290ae740">7장 3강</a> / <a href="https://app.notion.com/p/af21769337a18313ba9c81c0586b8073">7장 4강</a>']])}
    <p>생성·채점 모델은 프로젝트에서 사용 중인 <code>{esc(config['generation_model'])}</code>을 유지했습니다. 재정렬은 교안의 로컬 Cross-Encoder이며 LLM 재정렬을 사용하지 않았습니다. 후보는 검색기별 20개, 결합 후 C 입력 최대 20개, 최종 문맥 5개입니다. 교안 예시의 작은 문서 수와 달리 본 실험은 31,772건을 사용하므로 k는 프로젝트 평가 조건에 맞게 지정했습니다.</p>
    <p class="muted">DB는 Docker의 PostgreSQL·pgvector, Python·LangChain·Cross-Encoder는 Mac 가상환경에서 실행했습니다. Cross-Encoder는 mps:0에서 실행 확인했습니다. 교안의 PGVector는 공용 테이블과 <code>course_v0_small</code> 컬렉션을 사용하며, 이전 네 전용 테이블은 그대로 보존했습니다. 기존 임베딩을 복사해 본문 재임베딩은 0건입니다.</p></section>
    <section id="ragas"><h2>교안 Wrapper로 실행한 RAGAS</h2>{table(['구성','Context Precision','Context Recall','Faithfulness','Answer Relevancy'],metric_rows)}
    <p>기존 T01~T20 중 참고 근거가 있는 17문항 × 3구성 = 51건에 네 지표를 적용한 기록입니다. 새 T21~T40의 평가 결과가 아닙니다. 각 지표의 실제 유효 n과 오류 수를 함께 표시했습니다. 실제 생성 체인에 주입한 자료 설명·청크 ID·원문 위치를 포함한 문맥을 평가에 사용했습니다.</p>
    <p class="notice">RAGAS 점수는 정답 확률이 아닙니다. Faithfulness가 높더라도 정직하게 “찾지 못했다”고 답한 것일 수 있습니다. Context Recall이 높아도 서로 다른 집의 속성을 합쳐 조건을 만족한 것으로 오해할 수 있으므로, 한 거래가 모든 조건을 만족하는지는 별도의 Hit@5 조건 검사로 확인했습니다.</p>
    <p class="muted">RAGAS reference는 대표 거래 한 건만 유일한 정답으로 강제하지 않고 질문 조건으로 고정한 과거 참고 정답을 사용했습니다. 다른 유효 거래도 인정하기 위한 설정입니다. 현재형 질문에 확인 불가를 안내하는 답변이 회피성 응답으로 판정되면 Answer Relevancy가 0이 될 수 있으므로, 낮은 값은 실제 답변과 함께 해석해야 합니다. 답변 금액·기간·인용의 최종 적합성은 원문 교차 검토가 남아 있습니다. 생성·채점 모델이 같으므로 공통 모델 편향도 남습니다.</p></section>
    <section id="diagnosis"><h2>데이터보다 먼저 확인할 검색 설정</h2><p>이 데이터는 집 한 건의 지역·유형·보증금·월세·면적·층·연도 등을 가진 구조화된 거래 표입니다. 의미가 비슷한 문서만 찾으면 예산이나 면적이 다른 거래도 상위에 올라올 수 있습니다. 현재 A/B/C 본평가는 필터를 쓰지 않은 기준선입니다.</p>
    <h3>교안 3장 4강의 필터를 직접 지정한 Dev 진단</h3>{table(['ID','질문','필터 전 Hit@5','필터 후 Hit@5','원본 조건 충족 거래 수'],filter_rows)}
    <p class="notice">Dev 4문항에서 <b>{before}/4 → {after}/4</b>로 바뀌었습니다. 이는 질문에 적힌 조건을 사람이 정확히 필터로 지정했을 때의 검색 가능성 점검입니다. 자연어에서 필터를 자동 추출하는 기능은 구현·평가하지 않았고, 이 수치를 서비스 정확도나 Test 점수에 합치지 않았습니다. 자동으로 잘 추출되는지와 정보 부족 대응은 별도로 검증해야 합니다.</p>
    <details><summary>실제 지정한 필터와 찾은 원본 행</summary><pre>{esc(json.dumps(filtered,ensure_ascii=False,indent=2))}</pre></details></section>
    <section id="improve"><h2>앞으로의 개선: 교안 내부에서 가능한가?</h2><p><b>검색 품질 개선은 교안 안의 내용으로 충분히 시도할 수 있습니다.</b> 현재 관찰한 병목에 가장 직접적인 것은 3장 4강의 조건 필터이며, 다음 순서로 Dev에서 검증하는 것이 좋습니다. 아래에서 실제 측정한 것은 필터 수동 지정 진단뿐입니다. 나머지 항목의 개선 폭을 측정한 것으로 주장하지 않습니다.</p>
    {table(['우선순위','개선할 부분','교안 위치','검증 방법·현재 상태'],improvements)}
    <h3>먼저 바꿀 부분: 숫자·유형 조건을 검색 후보에 적용</h3><p>예를 들어 D02는 봉천동·오피스텔·월세·보증금 1,000만원 이하·월세 70만원 이하·2018년 이후·5평 이상이라는 조건을 먼저 확인해야 합니다. 조건이 명시적으로 주어졌다면 교안의 PGVector 필터에 넣고, 그 안에서 의미 검색을 수행하는 방향이 적합합니다.</p><p class="notice"><b>Hybrid에도 같은 조건을 적용해야 합니다.</b> Vector에만 필터를 넣고 BM25는 전체 문서를 검색하면 제외한 집이 결합 결과로 다시 들어올 수 있습니다. BM25 후보 문서도 같은 조건으로 제한하거나, 결합 전에 같은 조건 검사를 적용해야 합니다. 이 부분은 앞으로 구현·검증할 제안이며 현재 A/B/C 결과에는 반영하지 않았습니다.</p>
    <h3>실제로 진행할 순서와 완료 확인 방법</h3>{roadmap}
    <p class="muted">교안의 필터·Retriever·EnsembleRetriever·CrossEncoderReranker·프롬프트 체인으로 진행할 수 있습니다. 다만 부동산 필드에 맞는 적용 코드는 작성해야 합니다. Dev에서는 한 번에 한 요소만 바꾸고, 적중 수뿐 아니라 조건 위반·정보 부족 대응·지연·비용을 함께 확인합니다. 모델을 large로 바꾸거나 무조건 재청킹하는 작업은 검색 실패 원인을 분리한 뒤 검토하는 것이 좋습니다.</p>
    <h3>RAGAS 점수와 평가 목표도 함께 점검</h3><p>현재형 집 구하기 질문은 유지하되, 지금 데이터로 평가할 수 있는 것은 과거 참고 사례와 확인 불가 안내입니다. <a href="https://app.notion.com/p/89e1769337a182f7b77881b22c9ff052">7장 2강의 질문·정답·근거 정합성 점검</a>으로 이 범위를 확인하고, 정직한 확인 불가 답변이 자동 지표에서 어떻게 판정되는지 Dev에서 검토해야 합니다. 답변 관련성 점수를 높이려고 필요한 확인 불가 안내를 지우거나 현재 매물이 있다고 단정해서는 안 됩니다.</p>
    <div class="two-col"><div class="panel"><h3>교안 안에서 할 수 있는 것</h3><ul><li>지역·유형·예산·면적 같은 명확한 조건을 메타데이터 필터로 제한</li><li>Top-k·threshold·검색 후보 수 조정</li><li>BM25/Vector 가중치와 Cross-Encoder 결과 비교</li><li>프롬프트·출처 표기·정보 부족 안내 보완</li><li>Dev 튜닝 → 설정 고정 → 별도의 최종 평가</li></ul></div><div class="panel"><h3>별도 구현·검증이 필요한 것</h3><ul><li>자유로운 한국어 문장에서 예산·면적·별칭을 필터로 자동 변환</li><li>실시간 매물 데이터 수집과 갱신</li><li>현재 공실·입주 가능 여부 확인</li><li>사용자 실제 질문 분포에서의 추가 평가</li></ul><p class="muted">조건을 화면 입력값으로 명시적으로 받으면 교안의 필터 방식으로 연결할 수 있습니다. 자연어 자동 해석까지 검증됐다고 볼 수는 없습니다.</p></div></div></section>
    <section><h2>근거 없음 3문항의 실제 응답</h2>{table(['ID','구성','응답'],none_rows)}<p class="muted">이 9개 응답은 RAGAS 참고 근거 문항 평균에 넣지 않았습니다. 관리비·역 도보시간·반려동물·현재 매물 상태는 제공 데이터에 없으므로 사실을 추정하지 않았는지 별도로 교차 검토해야 합니다. 팀원의 정답 충족·근거 적합·정보 부족 대응 최종 채점은 남아 있습니다.</p></section>
    <section id="cases"><h2>기존 20문항의 A/B/C 원본 응답 60건</h2><div class="filters"><button data-filter="all" class="active">전체</button><button data-filter="A">A</button><button data-filter="B">B</button><button data-filter="C">C</button></div>{''.join(cases)}</section>
    <section><h2>실행·보존 정보</h2><p>현재 남아 있는 기존 코드·질문 파일 {len(existing_preserved)}개의 SHA-256이 보존 기록과 일치함을 확인했습니다. 기존 측정 원본은 이 폴더의 <code>results/</code>에 보관했습니다. 이번 수정은 교안 보고서를 업데이트한 것이며, 현재 경로에 없는 이전 SDK HTML은 복원하지 않았습니다.</p>
    <h3>앞으로 개선할 때 사용할 Dev 실행 명령</h3><pre>.venv/bin/python src/rag_home_api/test/bc/run_course.py --stage dev
.venv/bin/python src/rag_home_api/test/bc/run_course.py --stage dev-ragas
.venv/bin/python src/rag_home_api/test/bc/check_metadata_filter.py
.venv/bin/python src/rag_home_api/test/bc/build_report.py</pre>
    <p class="muted">이번 수정에서는 위 Dev 실험을 새로 실행하지 않았습니다. 향후 실행하면 Dev 10문항×A/B/C 30건과 별도 Dev RAGAS 파일에 저장됩니다. 기존 abc/ragas 단계는 과거 20문항 실험 재현용입니다. 새 T21~T40은 자동 실행 단계에 연결하지 않았으므로, 설정 고정 후 별도 최종 평가 실행을 준비해야 합니다.</p>
    <details><summary>고정 실행 설정</summary><pre>{esc(json.dumps(config,ensure_ascii=False,indent=2))}</pre></details>
    <p class="muted">기존 20문항 실험은 course_20261125_v2_draft를 고정해 수행한 기록입니다. 새 평가셋은 course_20261125_v3_holdout_draft로 분리했으며 RAG 파이프라인을 실행하지 않았습니다. 팀 교차 검토 후 문항·근거를 바꾸면 새 버전으로 고정해야 합니다. 수동 필터 Dev 진단은 기존 20문항 점수와 분리했습니다. 향후 Dev에서 정한 설정을 고정한 뒤 새 Test를 평가하고, 결과를 보고 고친 추가 실험은 별도로 남깁니다. 기존 교안 재실행에서는 본문 재임베딩 요금은 추가되지 않았으며 질의 임베딩·생성·RAGAS 호출에는 API 요금이 발생했습니다. 당시 실제 토큰 사용량은 별도로 수집하지 않아 정확한 추가 청구액은 제시하지 않았습니다.</p></section>
    </main><footer><div class="wrap">교안 방식 재실행 · {now} · 실시간 매물 존재·입주 가능 여부는 확인 불가</div></footer>
    <script>document.querySelectorAll('[data-filter]').forEach(button=>button.addEventListener('click',()=>{{document.querySelectorAll('[data-filter]').forEach(b=>b.classList.toggle('active',b===button));document.querySelectorAll('.case').forEach(row=>{{row.hidden=button.dataset.filter!=='all'&&row.dataset.system!==button.dataset.filter;}});}}));</script>
    </body></html>'''
    path=Path(__file__).parent/'course_rag_report.html'
    path.write_text(output)
    print('교안 방식 HTML 보고서:',path.resolve())
    print(json.dumps(summaries,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
