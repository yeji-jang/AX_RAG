"""Windows/macOS 공통 V0/small 청킹 확인 및 재개 가능한 PGVector 적재."""
import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parents[3]
CSV_PATH = PROJECT_ROOT / 'src/rag_home_api/shared/docs/서울_전월세_최근1년_관악구.csv'


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def csv_hashes():
    data = CSV_PATH.read_bytes()
    return {'raw': hashlib.sha256(data).hexdigest(),
            'lf': hashlib.sha256(data.replace(b'\r\n', b'\n')).hexdigest()}


def write_json(path, value, mode='w'):
    with path.open(mode, encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def fingerprint(docs):
    digest = hashlib.sha256()
    for doc in docs:
        record = {'id': doc.id, 'page_content': doc.page_content, 'metadata': doc.metadata}
        digest.update((json.dumps(record, ensure_ascii=False, sort_keys=True, allow_nan=False)
                       + '\n').encode('utf-8'))
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('check', 'index'), default='check')
    parser.add_argument('--output', type=Path, default=ROOT / '.local/index_v0_small')
    parser.add_argument('--resume', action='store_true', help='같은 청크 해시의 중단된 적재만 재개')
    args = parser.parse_args()
    if __package__:
        from .lesson_pipeline import documents, save_documents
    else:
        from lesson_pipeline import documents, save_documents
    docs = documents('v0')
    chunk_hash = fingerprint(docs)
    print(f'V0: {len(docs):,} chunks / SHA-256 {chunk_hash}', flush=True)
    if args.stage == 'check':
        return 0
    args.output.mkdir(parents=True, exist_ok=True)
    config_path = args.output / 'config.json'
    status_path = args.output / 'status.json'
    if config_path.exists() or status_path.exists():
        if not args.resume or not config_path.exists():
            raise FileExistsError('기존 적재 기록이 있습니다. 새 --output 또는 --resume을 사용하세요.')
        previous = json.loads(config_path.read_text(encoding='utf-8'))
        if previous['chunks_sha256'] != chunk_hash:
            raise ValueError('청크가 달라 기존 실행을 재개하지 않았습니다.')
    else:
        code_names = ('structural_chunk_v0.py', 'lesson_pipeline.py', 'run_course_index.py')
        write_json(config_path, {
            'created_at': datetime.now(timezone.utc).isoformat(),
            'platform': platform.platform(), 'python': sys.version,
            'version': 'v0', 'embedding_model': 'text-embedding-3-small',
            'collection': 'course_v0_small', 'chunk_count': len(docs),
            'chunks_sha256': chunk_hash, 'csv_hashes': csv_hashes(),
            'code_sha256': {name: sha256(ROOT / name) for name in code_names},
            'code_lf_sha256': {name: hashlib.sha256((ROOT / name).read_bytes().replace(b'\r\n', b'\n')).hexdigest()
                               for name in code_names},
            'lock_sha256': sha256(PROJECT_ROOT / 'uv.lock'),
        }, 'x')
    write_json(status_path, {'status': 'running'})
    try:
        save_documents('v0', 'small')
        # 기존 저장 행까지 본문·메타데이터·벡터 차원과 대조한다.
        import os
        import psycopg
        url = os.environ['DATABASE_URL'].replace('postgresql+psycopg://', 'postgresql://', 1)
        with psycopg.connect(url, connect_timeout=10) as conn:
            with conn.cursor() as cursor:
                cursor.execute("""SELECT e.id, e.document, e.cmetadata, vector_dims(e.embedding)
                    FROM langchain_pg_embedding e JOIN langchain_pg_collection c ON e.collection_id=c.uuid
                    WHERE c.name=%s""", ('course_v0_small',))
                stored = {row[0]: row[1:] for row in cursor.fetchall()}
        for doc in docs:
            actual = stored.get(f'course_v0_small:{doc.id}')
            if actual != (doc.page_content, doc.metadata, 1536):
                raise ValueError(f'저장 내용 또는 임베딩 차원이 다릅니다: {doc.id}')
        if len(stored) != len(docs):
            raise ValueError('컬렉션의 저장 청크 수가 원본 청크 수와 다릅니다.')
        write_json(status_path, {'status': 'success', 'verified_count': len(stored),
                                 'embedding_dimensions': 1536, 'chunks_sha256': chunk_hash})
        print('Index verified:', len(stored), flush=True)
        return 0
    except Exception as error:
        # 자격 증명이 예외 문자열에 섞일 수 있어 유형만 기록한다.
        write_json(status_path, {'status': 'error', 'error_type': type(error).__name__})
        raise


if __name__ == '__main__':
    raise SystemExit(main())
