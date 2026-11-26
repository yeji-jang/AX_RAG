"""
# 3단계 : 청크 문서 임베딩 -> PGVector 적재 

- 입력 : docs/02_chunked/{CHUNK_FILE}
- 컬렉션 : 청크 파일 이름에서 자동 생성 (yj_ + 버전 정보)
- 다른 청크를 적재할 때는 CHUNK_FILE 만 바꿔서 실행
- PGVector에 저장된 기존 content_hash를 조회하여 신규 또는 수정된 청크만 임베딩 및 적재(증분 갱신)
"""

from rag_home_api.test.yj.core.config import embedding_settings
from rag_home_api.test.yj.core.logger import get_logger
from rag_home_api.test.yj.core.vector_store import get_vector_store, get_raw_db_connection

from langchain_core.documents import Document

import json
import time
from pathlib import Path

import psycopg

logger = get_logger(__name__)


#===================================================================
# 1. 입력값으로 넣을 청크문서 경로 설정하고 
#===================================================================
PREPROCESS_VERSION = "V2"
CHUNK_VERSION = "V1"
CHUNK_SIZE = 800
CHUNK_OVERLAP  = 100


CHUNK_FILE = f"공공임대_공고문_pre{PREPROCESS_VERSION}_chunk{CHUNK_VERSION}_cs{CHUNK_SIZE}_ov{CHUNK_OVERLAP}.jsonl"


BATCH_SIZE = 200          # Postgres 요청 인자 수 한도(65,535) 때문에 나눠서 저장
TEST_LIMIT = None     

REBUILD = False  # True: 컬렉션 비우고 전체 재적재 / False: 증분 갱신

IN_CHUNK_PATH = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "02_chunked"
    / CHUNK_FILE
)

COLLECTION = "yj_public_rental"   # 단일 컬렉션으로 통합 (버전은 cmetadata['chunk_version'] 필터로 구분)

#===================================================================
# 2. 청크 JSONL → Document (DB 기본키 id = "{COLLECTION}:{chunk_id}")
#===================================================================
def make_db_id(collection: str, chunk_id: str) -> str:
    """PGVector 테이블 전체 PK(e.id) 충돌 방지를 위해 컬렉션 이름을 접두어로 붙임"""
    return f"{collection}:{chunk_id}"


def load_chunks(path: Path, collection: str = COLLECTION) -> list[Document]:
    docs: list[Document] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            text = data.pop("text")
            db_id = make_db_id(collection, data["chunk_id"])
            docs.append(Document(id=db_id, page_content=text, metadata=data))
    return docs

#===================================================================
# 3. DB에 저장된 {chunk_id: content_hash} 조회
#===================================================================
def fetch_existing_hashes(collection: str, chunk_version: str = CHUNK_VERSION) -> dict[str, str]:
    query = """
        SELECT e.id, e.cmetadata->>'content_hash'
        FROM langchain_pg_embedding e
        JOIN langchain_pg_collection c ON e.collection_id = c.uuid
        WHERE c.name = %s
          AND e.cmetadata->>'chunk_version' = %s
    """
    conn = get_raw_db_connection()
    try:
        cur = conn.cursor()
        cur.execute(query, (collection, chunk_version))
        return {row[0]: row[1] for row in cur.fetchall()}
    except psycopg.errors.UndefinedTable:
        # 처음 실행이라 테이블이 아직 없음 → 전부 신규
        # (다른 에러는 숨기지 않고 그대로 터지게 둠)
        return {}
    finally:
        conn.close()

#===================================================================
# 4. 신규 / 수정 / 그대로 / 삭제 분류
#===================================================================
def diff_chunks(docs: list[Document], existing: dict[str, str]) -> dict:
    new, updated, unchanged = [], [], []
    for d in docs:
        old_hash = existing.get(d.id)
        if old_hash is None:
            new.append(d)
        elif old_hash != d.metadata["content_hash"]:
            updated.append(d)
        else:
            unchanged.append(d)

    file_ids = {d.id for d in docs}
    deleted_ids = [cid for cid in existing if cid not in file_ids]
    return {"new": new, "updated": updated, "unchanged": unchanged, "deleted_ids": deleted_ids}

def migrate_legacy_ids(collection: str) -> int:
    """기존에 chunk_id만으로 저장된 행이 있으면 재임베딩 없이 '{collection}:{chunk_id}'로 PK만 갱신"""
    query = """
        UPDATE langchain_pg_embedding e
        SET id = c.name || ':' || (e.cmetadata->>'chunk_id')
        FROM langchain_pg_collection c
        WHERE e.collection_id = c.uuid
          AND c.name = %s
          AND e.id = (e.cmetadata->>'chunk_id')
    """
    conn = get_raw_db_connection()
    try:
        cur = conn.cursor()
        cur.execute(query, (collection,))
        migrated = cur.rowcount
        conn.commit()
        return migrated
    except psycopg.errors.UndefinedTable:
        return 0
    finally:
        conn.close()


#===================================================================
# 5. 적재 (증분 또는 전체 재적재)
#===================================================================
def index_chunks(
    docs: list[Document],
    rebuild: bool = False,
    allow_delete: bool = True,
    chunk_version: str = CHUNK_VERSION,
) -> dict:
    # 벡터 스토어를 먼저 만들면 LangChain 테이블이 없을 때 자동으로 생성됨
    store = get_vector_store(COLLECTION, reset=rebuild)

    if not rebuild:
        migrated = migrate_legacy_ids(COLLECTION)
        if migrated:
            logger.info(f"기존 PK 형식 변환 완료 : {migrated:,}건 -> '{COLLECTION}:{{chunk_id}}'")

    existing = {} if rebuild else fetch_existing_hashes(COLLECTION, chunk_version=chunk_version)
    diff = diff_chunks(docs, existing)
    to_embed = diff["new"] + diff["updated"]

    start = time.time()

    # 1) 신규 + 수정 : 같은 id면 덮어씀 (upsert)
    for i in range(0, len(to_embed), BATCH_SIZE):
        batch = to_embed[i:i + BATCH_SIZE]
        store.add_documents(batch, ids=[d.id for d in batch])
        logger.info(f"임베딩·적재 {i + len(batch):,} / {len(to_embed):,}")

    # 2) 삭제 : 같은 chunk_version 내에서 청크 파일에서 사라진 chunk_id만 삭제
    deleted = 0
    if allow_delete and diff["deleted_ids"]:
        for i in range(0, len(diff["deleted_ids"]), BATCH_SIZE):
            store.delete(ids=diff["deleted_ids"][i:i + BATCH_SIZE], collection_only=True)
        deleted = len(diff["deleted_ids"])

    return {
        "collection": COLLECTION,
        "chunk_version": chunk_version,
        "embedding_model": embedding_settings.EMBEDDING_MODEL,
        "mode": "rebuild" if rebuild else "incremental",
        "total": len(docs),
        "new": len(diff["new"]),
        "updated": len(diff["updated"]),
        "unchanged": len(diff["unchanged"]),
        "deleted": deleted,
        "embedded": len(to_embed),
        "seconds": round(time.time() - start, 1),
    }


#===================================================================
# 6. 적재 확인 (개수 · 벡터 · id · 차원)
#===================================================================
def verify(collection: str, chunk_version: str = CHUNK_VERSION) -> dict:
    query = """
        SELECT COUNT(*),
               COUNT(*) FILTER (WHERE e.embedding IS NULL),
               COUNT(*) FILTER (WHERE e.id <> (c.name || ':' || (e.cmetadata->>'chunk_id'))),
               MIN(vector_dims(e.embedding))
        FROM langchain_pg_embedding e
        JOIN langchain_pg_collection c ON e.collection_id = c.uuid
        WHERE c.name = %s
          AND e.cmetadata->>'chunk_version' = %s
    """
    conn = get_raw_db_connection()
    try:
        cur = conn.cursor()
        cur.execute(query, (collection, chunk_version))
        total, no_vec, id_mismatch, dims = cur.fetchone()
    finally:
        conn.close()
    return {"db_total": total, "no_vector": no_vec, "id_mismatch": id_mismatch, "dims": dims}


REPORT_DIR = Path(__file__).resolve().parents[2] / "docs" / "03_index_reports"

def write_index_report(result: dict, check: dict, n_file: int, docs: list[Document]) -> Path:
    """실행 결과를 컬렉션·버전별 md 파일에 이어 붙임 (처음이면 제목과 설정을 먼저 씀)"""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORT_DIR / f"{COLLECTION}_chunk{CHUNK_VERSION}.md"

    lines = []
    if not path.exists():
        lines += [
            f"# 인덱싱 기록 : {COLLECTION} ({CHUNK_VERSION})",
            "",
            "| 항목 | 값 |",
            "|---|---|",
            f"| 컬렉션 | `{COLLECTION}` |",
            f"| 청킹 버전 | `{CHUNK_VERSION}` |",
            f"| 청크 파일 | `docs/02_chunked/{CHUNK_FILE}` |",
            f"| 임베딩 모델 | {result['embedding_model']} |",
            f"| 배치 크기 | {BATCH_SIZE} |",
            "",
            "## 실행 기록",
            "",
        ]

    ok = (check["db_total"] == n_file and check["no_vector"] == 0 and check["id_mismatch"] == 0)
    lines += [
        f"### {time.strftime('%Y-%m-%d %H:%M')} · {result['mode']}" + ("" if ok else " 확인 필요"),
        "",
        "| 신규 | 수정 | 그대로 | 삭제 | 임베딩 | 소요(초) |",
        "|---:|---:|---:|---:|---:|---:|",
        f"| {result['new']:,} | {result['updated']:,} | {result['unchanged']:,} | "
        f"{result['deleted']:,} | {result['embedded']:,} | {result['seconds']} |",
        "",
        f"- DB 확인 : 저장 {check['db_total']:,}개 (파일 {n_file:,}개) / 벡터 없음 {check['no_vector']} "
        f"/ id 불일치 {check['id_mismatch']} / 차원 {check['dims']}",
        "",
    ]
    with path.open("a", encoding="utf-8") as f :
            # 이번에 적재한 공고 목록 (공고별 청크 수)
        by_doc = {}
        for d in docs:
            m = d.metadata
            info = by_doc.setdefault(m["doc_id"], {"title": m["title"], "house_type": m["house_type"],
                                                "source": m["source"], "chunks": 0})
            info["chunks"] += 1

        lines += [
            f"<details><summary>적재 대상 공고 {len(by_doc)}건 (청크 {len(docs):,}개)</summary>",
            "",
            "| doc_id | 기관 | 주택유형 | 공고명 | 청크 수 |",
            "|---|---|---|---|---:|",
            *[f"| {did} | {i['source']} | {i['house_type']} | {i['title']} | {i['chunks']} |"
            for did, i in sorted(by_doc.items())],
            "",
            "</details>",
            "",
        ]
        f.write("\n".join(lines) + "\n")
    return path




#===================================================================
# 7. 실행
#===================================================================
if __name__ == "__main__":
    docs = load_chunks(IN_CHUNK_PATH)
    if TEST_LIMIT:
        docs = docs[:TEST_LIMIT]
    logger.info(f"컬렉션 {COLLECTION} / 청크 {len(docs):,}개 / {'전체 재적재' if REBUILD else '증분 갱신'}")

    # 시험 적재(TEST_LIMIT)일 때는 나머지 청크가 '삭제'로 잡히므로 삭제를 막음
    result = index_chunks(docs, rebuild=REBUILD, allow_delete=not TEST_LIMIT)
    logger.info(f"적재 결과 : {result}")

    check = verify(COLLECTION)
    logger.info(f"DB 확인 : {check} (파일 {len(docs):,}개)")

    # 인덱싱 기록을 docs/03_index_reports/{컬렉션}.md 에 저장 (시험 적재도 표시해서 기록)
    if TEST_LIMIT:                          # 변경 후 (이 4줄로 교체)
        result["mode"] += f" (test {TEST_LIMIT})"
    report = write_index_report(result, check, n_file=len(docs), docs=docs)
    logger.info(f"기록 저장 : {report}")

    # 검색 확인 : reset 없이(기본값 False) 현재 chunk_version만 필터링해 불러오기
    store = get_vector_store(COLLECTION)
    for doc, dist in store.similarity_search_with_score(
        "행복주택 당첨자 발표일은 언제인가요?",
        k=5,
        filter={"chunk_version": CHUNK_VERSION},
    ):
        logger.info(f"{dist:.3f}  {doc.id}  {doc.page_content[:50]!r}")