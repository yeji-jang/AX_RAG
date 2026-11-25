"""교안 3장 4강처럼 필터 유무를 비교하는 Dev 진단. 자동 조건 추출 평가는 아니다."""
from lesson_pipeline import ROOT, vector_store
from run_course import read, save, accepted_transactions

# 아래 값은 Dev 질문에 적힌 지역·유형·예산·면적·층 조건을 사람이 직접 지정한 것이다.
# 거래 ID·모범 답안·대표 근거의 금액을 필터로 사용하지 않는다.
from metadata_filters import MANUAL_DEV_FILTERS as FILTERS


def main():
    questions=[q for q in read(ROOT/'eval/abc_dev_questions.json')['questions'] if q['id'] in FILTERS]
    accepted=accepted_transactions(questions)
    store=vector_store('v0','small')
    results=[]
    for question in questions:
        before=store.similarity_search(query=question['question'],k=5)
        after=store.similarity_search(query=question['question'],k=5,filter=FILTERS[question['id']])
        before_hit=int(any(d.metadata['id'] in accepted[question['id']] for d in before))
        after_hit=int(any(d.metadata['id'] in accepted[question['id']] for d in after))
        record={'question_id':question['id'],'question':question['question'],'manual_filter':FILTERS[question['id']],
                'before_hit_at_5':before_hit,'after_hit_at_5':after_hit,
                'allowed_transactions':len(accepted[question['id']]),
                'before_ids':[d.metadata['chunk_id'] for d in before],
                'after_ids':[d.metadata['chunk_id'] for d in after],
                'after_source_rows':[d.metadata['source_row'] for d in after],
                'after_all_match':all(d.metadata['id'] in accepted[question['id']] for d in after)}
        results.append(record)
        print(question['id'], '필터 전',before_hit,'→ 필터 후',after_hit,flush=True)
    save('metadata_filter_dev.json',results)


if __name__=='__main__':
    main()
