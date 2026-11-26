"""D01~D04 수동 필터 전용 실행기. 기본 check는 표준 라이브러리만 사용한다."""
import argparse
import csv
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from metadata_filters import MANUAL_DEV_FILTERS, NUMBER_FIELDS, matches_filter

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parents[3]
QUESTION_PATH = ROOT / 'eval/abc_dev_questions.json'
CSV_PATH = PROJECT_ROOT / 'src/rag_home_api/shared/docs/서울_전월세_최근1년_관악구.csv'


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_inputs():
    dev = json.loads(QUESTION_PATH.read_text(encoding='utf-8'))
    source_hashes = csv_hashes()
    if dev['source']['sha256'] not in source_hashes.values():
        raise ValueError('Dev 명세와 원본 CSV SHA-256이 다릅니다.')
    if source_hashes['raw'] != dev['source']['sha256']:
        print('CSV: CRLF→LF 환산 SHA-256이 Dev 원본과 일치합니다. 원본 파일은 유지합니다.')
    # 정답·근거·key_facts는 검색 경로로 전달하지 않는다.
    questions = [{'id': q['id'], 'question': q['question']}
                 for q in dev['questions'] if q['id'] in MANUAL_DEV_FILTERS]
    if [q['id'] for q in questions] != list(MANUAL_DEV_FILTERS):
        raise ValueError('D01~D04 질문 ID와 순서를 확인하세요.')
    return questions


def csv_hashes():
    data = CSV_PATH.read_bytes()
    return {'raw': hashlib.sha256(data).hexdigest(),
            'lf': hashlib.sha256(data.replace(b'\r\n', b'\n')).hexdigest()}


def candidate_counts():
    counts = dict.fromkeys(MANUAL_DEV_FILTERS, 0)
    with CSV_PATH.open(encoding='utf-8-sig', newline='') as stream:
        for row in csv.DictReader(stream):
            metadata = {key: value.strip() for key, value in row.items()}
            for field in NUMBER_FIELDS - {'area_pyeong'}:
                value = metadata.get(field, '')
                metadata[field] = float(value.replace(',', '')) if value else None
            area = metadata['area_m2']
            metadata['area_pyeong'] = area / 3.305785 if area is not None else None
            for key, search_filter in MANUAL_DEV_FILTERS.items():
                counts[key] += matches_filter(metadata, search_filter)
    return counts


def write_json(path, value, mode='w'):
    with path.open(mode, encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def run(stage, output, questions):
    # 기존 run_course.py 결과와 분리한다.
    output.mkdir(parents=True, exist_ok=True)
    if any((output / name).exists() for name in ('config.json', 'results.json')):
        raise FileExistsError('config.json 또는 results.json이 이미 있습니다. 다른 --output을 지정하세요.')
    code_names = ('metadata_filters.py', 'lesson_pipeline.py', 'run_filtered_dev.py',
                  'structural_chunk_v0.py')
    config = {
        'stage': stage, 'created_at': datetime.now(timezone.utc).isoformat(),
        'collection': 'course_v0_small', 'version': 'v0', 'size': 'small',
        'embedding_model': 'text-embedding-3-small', 'generation_model': 'gpt-6-luna',
        'final_k': 5, 'vector_candidates': 20, 'keyword_candidates': 20,
        'keyword_tokenizer': 'text.split()', 'rrf_c': 60, 'weights': [0.5, 0.5],
        'reranker': 'Dongjin-kr/ko-reranker', 'reranker_max_length': 512,
        'reranker_input_max': 20, 'manual_filters': MANUAL_DEV_FILTERS,
        'csv_sha256': sha256(CSV_PATH), 'questions_sha256': sha256(QUESTION_PATH),
        'csv_hashes': csv_hashes(),
        'expected_csv_sha256': json.loads(QUESTION_PATH.read_text(encoding='utf-8'))['source']['sha256'],
        'csv_hash_policy': '실제 바이트 또는 CRLF를 LF로만 환산한 바이트의 SHA-256 일치',
        'code_sha256': {name: sha256(ROOT / name) for name in code_names},
        'dependency_sha256': {name: sha256(PROJECT_ROOT / name)
                              for name in ('pyproject.toml', 'uv.lock')},
    }
    write_json(output / 'config.json', config, 'x')
    write_json(output / 'results.json', [], 'x')
    records = []
    failed = False
    # check 경로에서는 DB·API·모델 관련 모듈을 import하지 않는다.
    for question in questions:
        search_filter = MANUAL_DEV_FILTERS[question['id']]
        started = time.perf_counter()
        setup_error = None
        try:
            from lesson_pipeline import build_retrievers, build_chain, format_context, K
            retrievers = build_retrievers('v0', 'small', search_filter=search_filter)
        except Exception as error:
            retrievers = {}
            setup_error = f'{type(error).__name__}: {error}'
        setup_seconds = time.perf_counter() - started
        for system in ('A', 'B', 'C'):
            docs, response = [], None
            started = time.perf_counter()
            status, error_text = 'success', None
            try:
                if setup_error:
                    raise RuntimeError(setup_error)
                if stage == 'answer':
                    chain, docs = build_chain(retrievers[system])
                    response = chain.invoke(question['question'])
                else:
                    docs = retrievers[system].invoke(question['question'])[:K]
                if not all(matches_filter(doc.metadata, search_filter) for doc in docs):
                    raise ValueError('검색 결과가 수동 조건을 충족하지 않습니다.')
            except Exception as error:
                status, error_text = 'error', f'{type(error).__name__}: {error}'
                failed = True
            records.append({
                'question_id': question['id'], 'system': system,
                'question': question['question'], 'manual_filter': search_filter,
                'retrieved_ids': [doc.metadata['chunk_id'] for doc in docs],
                'retrieved_contexts': [format_context(doc) for doc in docs],
                'returned_count': len(docs),
                'all_match': (all(matches_filter(doc.metadata, search_filter) for doc in docs)
                              if status == 'success' else None),
                'response': response, 'seconds': time.perf_counter() - started,
                'setup_seconds': setup_seconds, 'status': status, 'error': error_text,
            })
            write_json(output / 'results.json', records)
            print(question['id'], system, status, len(docs), flush=True)
    return 1 if failed else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('check', 'retrieve', 'answer'), default='check')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    questions = read_inputs()
    if args.stage == 'check':
        for key, count in candidate_counts().items():
            print(f'{key}: {count:,}건 (원본 조건 일치 후보 수)')
        return 0
    output = args.output or ROOT / 'results/manual_filter_dev' / args.stage
    return run(args.stage, output, questions)


if __name__ == '__main__':
    raise SystemExit(main())
