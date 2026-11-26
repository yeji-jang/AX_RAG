"""Dev D01~D10 자연어 조건 추출. 기본 check는 API·DB를 호출하지 않는다."""
import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from run_filtered_dev import (ROOT, PROJECT_ROOT, QUESTION_PATH, CSV_PATH,
                              read_inputs, sha256, csv_hashes, write_json)


def question_path(split='dev'):
    if split not in ('dev', 'test'):
        raise ValueError('기존 Dev/Test만 지원합니다. v3 최종 Test는 실행하지 않습니다.')
    return ROOT / f'eval/abc_{split}_questions.json'


def read_questions(question_ids=None, split='dev'):
    read_inputs()  # 기존 실행기와 같은 원본 해시 검증
    data = json.loads(question_path(split).read_text(encoding='utf-8'))
    questions = [{'id': q['id'], 'question': q['question']} for q in data['questions']]
    expected_ids = ([f'D{i:02d}' for i in range(1, 11)] if split == 'dev'
                    else [f'T{i:02d}' for i in range(1, 21)])
    if [q['id'] for q in questions] != expected_ids:
        raise ValueError('기존 Dev/Test 질문 ID 명세를 확인하세요.')
    if question_ids:
        if not set(question_ids) <= {q['id'] for q in questions}:
            raise ValueError('선택한 기존 Dev/Test의 질문만 실행할 수 있습니다.')
        questions = [q for q in questions if q['id'] in question_ids]
    return questions


def run(stage, output, questions, split='dev'):
    if stage not in ('extract', 'retrieve', 'answer'):
        raise ValueError('지원하지 않는 실행 단계입니다.')
    output.mkdir(parents=True, exist_ok=True)
    if any((output / name).exists() for name in ('config.json', 'results.json')):
        raise FileExistsError('기존 결과가 있습니다. 다른 --output을 지정하세요.')
    from lesson_pipeline import MODEL, MODELS, K, CANDIDATES, build_retrievers, format_context
    from query_filters import build_filter_extractor, execute_query, resolve_building_plan, EXTRACTION_PROMPT
    code_names = ('query_filters.py', 'run_query_dev.py', 'metadata_filters.py',
                  'lesson_pipeline.py', 'run_filtered_dev.py', 'structural_chunk_v0.py')
    input_path = question_path(split)
    config = {
        'stage': stage, 'created_at': datetime.now(timezone.utc).isoformat(),
        'split': split, 'question_file': str(input_path.relative_to(PROJECT_ROOT)),
        'question_ids': [q['id'] for q in questions],
        'condition_mode': 'llm_json', 'condition_model': MODEL, 'generation_model': MODEL,
        'embedding_model': MODELS['small'], 'collection': 'course_v0_small',
        'final_k': K, 'vector_candidates': CANDIDATES, 'keyword_candidates': CANDIDATES,
        'keyword_tokenizer': 'text.split()', 'rrf_c': 60, 'weights': [0.5, 0.5],
        'reranker': 'Dongjin-kr/ko-reranker', 'reranker_max_length': 512,
        'reranker_input_max': CANDIDATES, 'csv_sha256': sha256(CSV_PATH),
        'csv_hashes': csv_hashes(), 'questions_sha256': sha256(input_path),
        'expected_csv_sha256': json.loads(input_path.read_text(encoding='utf-8'))['source']['sha256'],
        'csv_hash_policy': '실제 바이트 또는 CRLF를 LF로만 환산한 바이트의 SHA-256 일치',
        'code_sha256': {name: sha256(ROOT / name) for name in code_names},
        'dependency_sha256': {name: sha256(PROJECT_ROOT / name) for name in ('pyproject.toml', 'uv.lock')},
        'extraction_prompt': EXTRACTION_PROMPT,
        'evaluation_scope': '조건 추출 및 연결 진단. 조건 추출 성공은 의미 정확도나 Hit@5를 뜻하지 않는다.',
    }
    write_json(output / 'config.json', config, 'x')
    write_json(output / 'results.json', [], 'x')
    extractor, records, failed = None, [], False
    for question in questions:
        started = time.perf_counter()
        plan, extraction_error = None, None
        try:
            if extractor is None:
                extractor = build_filter_extractor()
            plan = resolve_building_plan(extractor.invoke(question['question']))
        except Exception as error:
            extraction_error = f'{type(error).__name__}: {error}'
        extraction_seconds = time.perf_counter() - started
        retrievers, setup_error = None, None
        setup_started = time.perf_counter()
        if stage != 'extract' and plan is not None and plan.status == 'ready':
            try:
                retrievers = build_retrievers('v0', 'small', search_filter=plan.search_filter)
            except Exception as error:
                setup_error = f'{type(error).__name__}: {error}'
        setup_seconds = time.perf_counter() - setup_started
        for system in ([None] if stage == 'extract' else ['A', 'B', 'C']):
            started = time.perf_counter()
            docs, contexts, response, error_text = [], [], None, extraction_error or setup_error
            status = 'error' if error_text else plan.status
            if not error_text and stage != 'extract':
                try:
                    result = execute_query(question['question'], plan, stage, system, retrievers)
                    docs, response, status = result['documents'], result['response'], result['status']
                    contexts = [format_context(doc) for doc in docs]
                except Exception as error:
                    status, error_text = 'error', f'{type(error).__name__}: {error}'
            failed |= status == 'error'
            records.append({
                'question_id': question['id'], 'question': question['question'], 'system': system,
                'plan': plan.as_dict() if plan else None, 'status': status, 'error': error_text,
                'response': response, 'retrieved_ids': [doc.metadata['chunk_id'] for doc in docs],
                'retrieved_contexts': contexts, 'returned_count': len(docs),
                'extraction_seconds': extraction_seconds, 'setup_seconds': setup_seconds,
                'seconds': time.perf_counter() - started,
            })
            write_json(output / 'results.json', records)
            print(question['id'], system or 'extract', status, flush=True)
    return 1 if failed else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('check', 'extract', 'retrieve', 'answer'), default='check')
    parser.add_argument('--split', choices=('dev', 'test'), default='dev', help='기존 질문셋만 사용; v3 제외')
    parser.add_argument('--question-id', action='append', help='D01~D10 중 선택; 여러 번 지정 가능')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    questions = read_questions(args.question_id, args.split)
    if args.stage == 'check':
        for q in questions:
            print(q['id'], q['question'])
        return 0
    output = args.output or ROOT / f'results/natural_filter_{args.split}' / args.stage
    return run(args.stage, output, questions, args.split)


if __name__ == '__main__':
    raise SystemExit(main())
