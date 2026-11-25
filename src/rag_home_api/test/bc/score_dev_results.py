"""Dev 검색 결과 사후 채점. 정답은 검색·생성 입력에 전달하지 않는다."""
import argparse
import json
import re
from pathlib import Path

from run_filtered_dev import ROOT, QUESTION_PATH, csv_hashes, read_inputs, sha256, write_json


def score_records(records, questions, accepted, chunk_transactions):
    by_question = {q['id']: q for q in questions}
    by_key = {}
    for record in records:
        key = (record['system'], record['question_id'])
        if key in by_key:
            raise ValueError(f'중복 실행 결과: {key}')
        if key[0] not in ('A', 'B', 'C') or key[1] not in by_question:
            raise ValueError('선택한 기존 질문셋의 A/B/C 검색 결과만 채점합니다.')
        expected_question = by_question[key[1]]['question']
        if record.get('question', record.get('user_input')) != expected_question:
            raise ValueError('결과의 질문이 Dev 명세와 다릅니다.')
        by_key[key] = record
    details, systems = [], {}
    for system in ('A', 'B', 'C'):
        supported = [q for q in questions if q['reference_answerable']]
        counts = {'supported_questions': len(supported), 'hit_at_5_count': 0,
                  'error_or_missing_count': 0, 'clarification_count': 0,
                  'unsupported_count': 0, 'records': 0}
        for q in questions:
            record = by_key.get((system, q['id']))
            status = record['status'] if record else 'missing'
            ids = record.get('retrieved_ids', [])[:5] if record else []
            if any(key not in chunk_transactions for key in ids):
                raise ValueError('원본에 없는 청크 ID가 검색 결과에 있습니다.')
            valid = [chunk_transactions[key] in accepted[q['id']] for key in ids]
            hit = int(status == 'success' and any(valid)) if q['reference_answerable'] else None
            response = record.get('response') or '' if record else ''
            citations = re.findall(r'\[([^\[\]]+)\]', response)
            cited_chunks = [key for key in citations if '_structural_' in key]
            detail = {'system': system, 'question_id': q['id'], 'status': status,
                      'reference_answerable': q['reference_answerable'], 'hit_at_5': hit,
                      'returned_count': len(ids), 'all_returned_meet_reference_conditions':
                      (all(valid) if ids and q['reference_answerable'] else None),
                      'citation_ids': cited_chunks,
                      'all_citations_in_retrieved': all(key in ids for key in cited_chunks),
                      'has_citation': bool(cited_chunks),
                      'semantic_answer_correctness': 'requires_review'}
            details.append(detail)
            counts['hit_at_5_count'] += hit or 0
            counts['error_or_missing_count'] += status in ('error', 'missing')
            counts['clarification_count'] += status == 'needs_clarification'
            counts['unsupported_count'] += status == 'unsupported'
            counts['records'] += record is not None
        counts['hit_at_5'] = counts['hit_at_5_count'] / len(supported) if supported else None
        systems[system] = counts
    return {'systems': systems, 'details': details,
            'scope': '선택한 질문셋의 과거 참고 근거 문항 기준 Hit@5. 확인 요청·호출 오류·누락도 분모에 포함. '
                     '답변 의미 정확도와 정보 부족 대응은 별도 검토하며 RAGAS 점수가 아니다.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True, help='config.json/results.json이 있는 폴더')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--split', choices=('dev', 'test'), default='dev')
    args = parser.parse_args()
    read_inputs()
    from run_query_dev import question_path
    input_path = question_path(args.split)
    data = json.loads(input_path.read_text(encoding='utf-8'))
    from run_course import accepted_transactions
    from lesson_pipeline import documents
    records_path = args.input / 'results.json'
    config = json.loads((args.input / 'config.json').read_text(encoding='utf-8'))
    if config.get('stage') not in ('answer', 'retrieve'):
        raise ValueError('검색·답변 실행 결과가 필요합니다.')
    if config['questions_sha256'] != sha256(input_path):
        # Windows의 CRLF 체크아웃과 Mac의 LF 차이는 내용 일치로 확인한다.
        raise ValueError('질문 파일 해시가 실행 당시와 다릅니다. 실행한 PC에서 먼저 채점하세요.')
    if config.get('csv_sha256') not in csv_hashes().values():
        raise ValueError('원본 해시가 실행 당시와 다릅니다.')
    accepted = accepted_transactions(data['questions'])
    chunk_transactions = {doc.metadata['chunk_id']: doc.metadata['id'] for doc in documents('v0')}
    result = score_records(json.loads(records_path.read_text(encoding='utf-8')),
                           data['questions'], accepted, chunk_transactions)
    result.update({'results_sha256': sha256(records_path), 'scorer_sha256': sha256(Path(__file__)),
                   'questions_sha256': sha256(input_path), 'csv_hashes': csv_hashes(), 'split': args.split})
    output = args.output or args.input / 'metrics.json'
    write_json(output, result, 'x')
    for system, values in result['systems'].items():
        print(f"{system}: Hit@5 {values['hit_at_5_count']}/{values['supported_questions']} "
              f"({values['hit_at_5']:.1%}), errors/missing={values['error_or_missing_count']}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
