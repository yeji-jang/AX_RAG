import json
from pathlib import Path

from langchain_core.documents import Document

from rag_home_api.config import DATA_DIR, SHARED_DOCS_DIR, settings
from rag_home_api.ingest.loaders import load_housing_unit_chunks, load_jsonl_chunks
from rag_home_api.ingest.schemas import DocType
from rag_home_api.shared.core.logger import get_logger
from rag_home_api.shared.core.vector_store import get_raw_db_connection, get_vector_store

logger = get_logger(__name__)


def _fetch_existing_hashes() -> set[str]:
    """현재 컬렉션에 이미 저장된 content_hash 집합을 조회한다. (증분 색인용)"""
    sql = """
        SELECT e.cmetadata->>'content_hash'
        FROM langchain_pg_embedding e
        JOIN langchain_pg_collection c ON e.collection_id = c.uuid
        WHERE c.name = %s
    """
    try:
        with get_raw_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (settings.collection_name,))
                return {row[0] for row in cur.fetchall() if row[0]}
    except Exception:
        return set()


def _drop_collection_rows() -> None:
    """rebuild=True 시 현재 컬렉션의 기존 임베딩 행만 깔끔하게 삭제한다."""
    sql = """
        DELETE FROM langchain_pg_embedding
        WHERE collection_id IN (
            SELECT uuid FROM langchain_pg_collection WHERE name = %s
        )
    """
    with get_raw_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (settings.collection_name,))
        conn.commit()


def _save_chunk_manifest(docs: list[Document]) -> Path:
    """확정된 chunk_id 목록을 data/chunk_manifest.jsonl로 저장한다. (평가셋 allowed_chunk_ids 매핑용)"""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = DATA_DIR / "chunk_manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as f:
        for d in docs:
            m = d.metadata
            f.write(
                json.dumps(
                    {
                        "chunk_id": m["chunk_id"],
                        "doc_id": m["doc_id"],
                        "doc_type": m["doc_type"],
                        "location": m["location"],
                        "index_version": m["index_version"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    return manifest_path


def build_unified_index(
    rebuild: bool = False,
    batch_size: int = 100,
    max_lines_per_jsonl: int | None = None,
    max_housing_units: int | None = None,
) -> dict:
    """공유 문서(JSONL 3종 + 호실 CSV 2종)를 단일 PGVector 컬렉션에 적재한다."""
    all_docs: list[Document] = []

    # ① JSONL 3종 로딩 & 구조적+재귀적 청킹
    jsonl_sources: list[tuple[Path, DocType]] = [
        (SHARED_DOCS_DIR / "일반전월세_법령가이드.jsonl", "rental_law"),
        (SHARED_DOCS_DIR / "공공임대_공고문.jsonl", "public_notice"),
        (SHARED_DOCS_DIR / "청년안심주택_공고문.jsonl", "youth_housing"),
    ]
    for path, doc_type in jsonl_sources:
        if path.exists():
            loaded = load_jsonl_chunks(path, doc_type, max_lines=max_lines_per_jsonl)
            all_docs.extend(loaded)
            logger.info("JSONL 청킹 완료: %s -> %d개 청크", path.name, len(loaded))

    # ② 공급호실 CSV 2종 조인 & 카드 청킹
    units_csv = SHARED_DOCS_DIR / "public_housing_units.csv"
    scenarios_csv = SHARED_DOCS_DIR / "public_rent_scenarios.csv"
    if units_csv.exists() and scenarios_csv.exists():
        unit_docs = load_housing_unit_chunks(units_csv, scenarios_csv, max_lines=max_housing_units)
        all_docs.extend(unit_docs)
        logger.info("공급호실 CSV 청킹 완료: %d개 청크", len(unit_docs))

    # ③ chunk_id 중복 사전 차단
    unique_ids = {d.metadata["chunk_id"] for d in all_docs}
    if len(unique_ids) != len(all_docs):
        raise ValueError(f"중복 chunk_id 발견! 전체={len(all_docs)}, 고유={len(unique_ids)}")

    # ④ 확정 chunk_id 목록표 저장
    manifest_path = _save_chunk_manifest(all_docs)
    logger.info("매니페스트 저장 완료: %s (%d건)", manifest_path.name, len(all_docs))

    # ⑤ PGVector 먼저 초기화(테이블 생성 보장) 후 rebuild / 증분 필터 분기
    store = get_vector_store(settings.collection_name)
    if rebuild:
        _drop_collection_rows()
        existing_hashes: set[str] = set()
        logger.info("rebuild=True: 기존 컬렉션(%s) 행 삭제 완료", settings.collection_name)
    else:
        existing_hashes = _fetch_existing_hashes()

    new_docs = [d for d in all_docs if d.metadata["content_hash"] not in existing_hashes]
    skipped = len(all_docs) - len(new_docs)

    # ⑥ 배치 적재 (ids에 chunk_id 전달 -> 내용 수정 시 덮어쓰기 지원)
    if new_docs:
        total_batches = (len(new_docs) + batch_size - 1) // batch_size
        for b_idx, start in enumerate(range(0, len(new_docs), batch_size), start=1):
            batch = new_docs[start : start + batch_size]
            store.add_documents(batch, ids=[d.metadata["chunk_id"] for d in batch])
            logger.info("PGVector 적재 배치 %d/%d 완료 (%d건)", b_idx, total_batches, len(batch))

    return {
        "collection_name": settings.collection_name,
        "index_version": settings.index_version,
        "rebuild": rebuild,
        "total_chunks": len(all_docs),
        "inserted_chunks": len(new_docs),
        "skipped_chunks": skipped,
        "manifest_path": str(manifest_path),
    }