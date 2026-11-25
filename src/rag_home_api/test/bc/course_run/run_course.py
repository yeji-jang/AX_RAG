"""교안 방식 A/B/C와 RAGAS를 별도 결과 폴더에서 실행한다.

python run_course.py --stage dev        # 개선·튜닝: 기존 Dev 10문항
python run_course.py --stage dev-ragas  # Dev 결과의 RAGAS
abc/ragas 단계는 이미 실행한 기존 20문항 기록의 재현용이다.
새 T21~T40 평가셋은 이 스크립트에서 자동 실행하지 않는다.
"""

import argparse
import csv
import hashlib
import json
import re
import time
from pathlib import Path

from lesson_pipeline import (ROOT, K, MODEL, DEFAULT_CSV_PATH, build_chain,
                             build_retrievers, embedding_model, format_context, llm)

RESULTS = Path(__file__).resolve().parent / 'results'
FIELDS = {'자치구':'region_gu','법정동':'region_dong','건물명':'building_name',
          '주택 유형':'housing_type','전월세 구분':'lease_type','보증금(만원)':'deposit_manwon',
          '월세(만원)':'monthly_rent_manwon','건축연도':'build_year','층':'floor',
          '임대면적(㎡)':'area_m2','임대면적 환산 평수':'area_pyeong'}
NUMBERS = {'deposit_manwon','monthly_rent_manwon','build_year','floor','area_m2','area_pyeong'}


def read(path):
    return json.loads(Path(path).read_text())


def save(name, value):
    RESULTS.mkdir(exist_ok=True)
    (RESULTS/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def matches(row, question):
    for fact in question['key_facts']:
        if ': ' not in fact:
            continue
        key, condition = fact.split(': ',1)
        field = FIELDS[key]
        value = float(row['area_m2'])/3.305785 if field=='area_pyeong' else row[field]
        if field not in NUMBERS:
            if value != condition.removesuffix(' 일치'):
                return False
        else:
            if value=='':
                return False
            value=float(value)
            for limit, comparison in re.findall(r'([\d.]+) (이하|이상|미만|일치)',condition):
                limit=float(limit)
                if not {'이하':value<=limit,'이상':value>=limit,'미만':value<limit,'일치':value==limit}[comparison]:
                    return False
    return not (question['id']=='D08' and not row['contract_period'])


def accepted_transactions(questions):
    with DEFAULT_CSV_PATH.open(encoding='utf-8-sig',newline='') as stream:
        rows=list(csv.DictReader(stream))
    accepted={}
    for question in questions:
        accepted[question['id']]=set() if question.get('absence_reason')=='missing_attribute_in_schema' else {
            row['id'] for row in rows if matches(row,question)}
        assert bool(accepted[question['id']])==question['reference_answerable']
    return accepted


def reference(question):
    # 대표 사례와 다른 유효 거래도 허용하기 위해 질문 조건으로 참고 정답을 고정한다.
    return ('과거 공공 거래 데이터에는 다음 조건을 모두 충족하는 참고 거래가 있습니다. '
            + '; '.join(f for f in question['key_facts'] if ': ' in f) + '.')


def run_abc(split='test'):
    is_dev=split=='dev'
    question_file='eval/abc_dev_questions.json' if is_dev else 'eval/abc_test_questions.json'
    prefix='dev_' if is_dev else ''
    questions=read(ROOT/question_file)['questions']
    accepted=accepted_transactions(questions)
    path=RESULTS/f'{prefix}abc_results.json'
    records=read(path) if path.exists() else []
    done={(r['system'],r['question_id']) for r in records}
    config={'implementation':'교안 LangChain', 'selected_collection':'course_v0_small',
            'embedding_model':'text-embedding-3-small', 'generation_model':MODEL,
            'reranker':'Dongjin-kr/ko-reranker','keyword_tokenizer':'text.split()',
            'vector_candidates':20,'keyword_candidates':20,'rrf_c':60,'weights':[0.5,0.5],
            'reranker_input_max':20,'final_k':K,
            ('dev_sha256' if is_dev else 'test_sha256'):hashlib.sha256((ROOT/question_file).read_bytes()).hexdigest(),
            'pipeline_sha256':hashlib.sha256((Path(__file__).parent/'lesson_pipeline.py').read_bytes()).hexdigest()}
    if (RESULTS/f'{prefix}config.json').exists():
        assert read(RESULTS/f'{prefix}config.json')==config
    save(f'{prefix}config.json',config)
    retrievers=build_retrievers('v0','small')
    chains={name:build_chain(retriever) for name,retriever in retrievers.items()}
    for question in questions:
        for system in ['A','B','C']:
            if (system,question['id']) in done:
                continue
            chain,used_docs=chains[system]
            used_docs.clear()
            start=time.perf_counter()
            try:
                response=chain.invoke(question['question'])
                status,error='success',None
            except Exception as failure:
                response,status,error='', 'error',type(failure).__name__
            elapsed=time.perf_counter()-start
            ids=[doc.metadata['id'] for doc in used_docs]
            hit=int(any(key in accepted[question['id']] for key in ids)) if question['reference_answerable'] else None
            record={'system':system,'question_id':question['id'],'user_input':question['question'],
                    'response':response,'retrieved_contexts':[format_context(doc) for doc in used_docs],
                    'retrieved_ids':[doc.metadata['chunk_id'] for doc in used_docs],
                    'source_rows':[doc.metadata['source_row'] for doc in used_docs],
                    'reference':reference(question) if question['reference_answerable'] else question['answer'],
                    'reference_answerable':question['reference_answerable'],'hit_at_5':hit,
                    'seconds':elapsed,'status':status,'error':error}
            records.append(record)
            save(f'{prefix}abc_results.json',records)
            print(f"교안 {split} {system} {question['id']} {status} Hit@5={hit}",flush=True)
    assert len(records)==len(questions)*3


def run_ragas(pilot=False,split='test'):
    # 교안 7장 3·4강: LangChain 모델을 RAGAS Wrapper로 연결하여 evaluate 실행
    from ragas import EvaluationDataset,evaluate
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.metrics import LLMContextPrecisionWithReference,LLMContextRecall,Faithfulness,ResponseRelevancy
    from ragas.run_config import RunConfig

    evaluator_llm=LangchainLLMWrapper(llm())
    evaluator_embeddings=LangchainEmbeddingsWrapper(embedding_model('small'))
    metrics=[LLMContextPrecisionWithReference(llm=evaluator_llm),
             LLMContextRecall(llm=evaluator_llm),
             Faithfulness(llm=evaluator_llm),
             ResponseRelevancy(llm=evaluator_llm,embeddings=evaluator_embeddings)]
    if pilot:
        rows=[{'user_input':'과거 전세 보증금은 얼마인가요?','response':'과거 전세 보증금은 2억원입니다.',
               'reference':'과거 전세 보증금은 2억원입니다.',
               'retrieved_contexts':['기준일은 2026-11-25다. 과거 자료: 2026-01-01 전세 보증금 2억원.']}]
        keys=[{'system':'pilot','question_id':'pilot'}]
    else:
        prefix='dev_' if split=='dev' else ''
        keys=[r for r in read(RESULTS/f'{prefix}abc_results.json') if r['reference_answerable'] and r['status']=='success']
        rows=[{key:r[key] for key in ['user_input','response','reference','retrieved_contexts']} for r in keys]
    dataset=EvaluationDataset.from_list(rows)
    result=evaluate(dataset=dataset,metrics=metrics,llm=evaluator_llm,embeddings=evaluator_embeddings,
                    run_config=RunConfig(timeout=120,max_retries=1,max_workers=4),
                    raise_exceptions=pilot,show_progress=True)
    frame=result.to_pandas()
    # 교안처럼 결과를 DataFrame/CSV로 저장한다. 원본 질문과 답변도 함께 남긴다.
    frame.insert(0,'question_id',[r['question_id'] for r in keys])
    frame.insert(0,'system',[r['system'] for r in keys])
    RESULTS.mkdir(exist_ok=True)
    name='ragas_pilot' if pilot else ('dev_ragas_results' if split=='dev' else 'ragas_results')
    frame.to_csv(RESULTS/f'{name}.csv',index=False,encoding='utf-8-sig')
    frame.to_json(RESULTS/f'{name}.json',orient='records',force_ascii=False,indent=2)
    print(frame[['system','question_id']+[m.name for m in metrics]].to_string(index=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage',choices=['dev','dev-ragas','abc','ragas','pilot'],default='dev')
    args=parser.parse_args()
    if args.stage in ['abc','dev']:
        run_abc(split='dev' if args.stage=='dev' else 'test')
    else:
        run_ragas(pilot=args.stage=='pilot',split='dev' if args.stage=='dev-ragas' else 'test')
