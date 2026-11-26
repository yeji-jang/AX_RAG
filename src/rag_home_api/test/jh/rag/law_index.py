"""법령 색인의 읽기 전용 사전검증과 저장 문서 벡터 재사용. 적재/승인 등록은 하지 않는다."""
import hashlib
import json
import math

from langchain_core.embeddings import Embeddings

from common import corpus_fingerprint


def validate_index_rows(rows, documents):
    """DB/모델 없는 검사도 가능한 순수 검증. law-only 하위 집합 전체가 일치해야 한다."""
    canonical = {d.metadata["chunk_id"]: d for d in documents}
    seen, by_text, fingerprint = set(), {}, []
    for row in rows:
        meta = row.get("cmetadata") or {}
        if not isinstance(meta, dict):
            raise ValueError("[IndexMetadataMismatch] 법령 메타데이터 객체가 필요합니다.")
        cid = meta.get("chunk_id")
        doc = canonical.get(cid)
        if doc is None or cid in seen or row.get("document") != doc.page_content:
            raise ValueError("[IndexMismatch] 누락·중복 또는 원문과 다른 법령 청크입니다.")
        for field in ("row_id", "article", "doc_id", "category"):
            if meta.get(field) != doc.metadata[field]:
                raise ValueError(f"[IndexMetadataMismatch] 법령 {field} 원본 매핑이 일치하지 않습니다.")
        for field in ("title", "source_url"):
            if meta.get(field) is not None and meta[field] != doc.metadata[field]:
                raise ValueError(f"[IndexMetadataMismatch] 법령 {field} 원본 매핑이 일치하지 않습니다.")
        if meta.get("embedding_model") not in (None, "text-embedding-3-small"):
            raise ValueError("[IndexModelMismatch] 저장된 임베딩 모델 태그가 다릅니다.")
        vector = row.get("vector")
        try:
            if isinstance(vector, str):
                vector = json.loads(vector)
            if (not isinstance(vector, list) or len(vector) != 1536 or
                    any(type(v) not in (int, float) or not math.isfinite(v) for v in vector) or
                    not any(vector)):
                raise ValueError
        except (TypeError, ValueError):
            raise ValueError("[IndexVectorInvalid] 유효한 1536차원 저장 벡터가 필요합니다.") from None
        if doc.page_content in by_text and by_text[doc.page_content] != vector:
            raise ValueError("[IndexVectorMismatch] 같은 원문의 저장 벡터가 서로 다릅니다.")
        seen.add(cid)
        by_text[doc.page_content] = vector
        fingerprint.append([cid, doc.page_content, vector])
    if not canonical or seen != set(canonical):
        raise ValueError("[IndexIncomplete] 현재 법령 청크 전체가 색인에 필요합니다.")
    # 과거 실제 실행과 같은 ID·본문·저장 벡터 지문. 원본 메타데이터는 별도로 검증한다.
    index_sha = hashlib.sha256(json.dumps(sorted(fingerprint), ensure_ascii=False,
                                        sort_keys=True).encode()).hexdigest()
    return {"index_sha256": index_sha, "corpus_sha256": corpus_fingerprint(documents),
            "law_chunks": len(seen), "dimensions": 1536, "vectors_by_text": by_text}


def read_index(connection, collection, documents):
    """모델/API 없이 기존 PGVector를 읽음. 테이블·컬렉션·임베딩을 생성하지 않는다."""
    from sqlalchemy import create_engine, text
    if not connection.startswith(("postgresql://", "postgresql+psycopg://")):
        raise ValueError("[DatabaseConfigInvalid] PostgreSQL DATABASE_URL이 필요합니다.")
    connection = connection.replace("postgresql://", "postgresql+psycopg://", 1)
    engine = None
    try:
        engine = create_engine(connection, connect_args={"connect_timeout": 8})
        with engine.connect() as conn:
            conn.execute(text("SET TRANSACTION READ ONLY"))
            uid = conn.execute(text("SELECT uuid FROM langchain_pg_collection WHERE name=:name"),
                               {"name": collection}).scalar_one_or_none()
            if uid is None:
                raise ValueError("[IndexMissing] 승인할 법령 컬렉션이 DB에 없습니다. README의 준비 절차를 확인하세요.")
            rows = [dict(row) for row in conn.execute(text(
                "SELECT document, cmetadata, embedding::text AS vector "
                "FROM langchain_pg_embedding WHERE collection_id=:uid "
                "AND cmetadata->>'category'='일반 전월세 법령'"), {"uid": uid}).mappings()]
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"[IndexReadFailed] DB 주소·권한·pgvector 테이블을 확인하세요 ({type(exc).__name__}).") from None
    finally:
        if engine is not None:
            engine.dispose()
    return validate_index_rows(rows, documents)


class StoredDocumentEmbeddings(Embeddings):
    """질문만 공식 임베딩 클라이언트에 전달하고 문서 벡터는 검증한 DB 값만 사용한다."""
    def __init__(self, query_embeddings, vectors_by_text):
        self.query_embeddings = query_embeddings
        self.vectors = vectors_by_text
        self.queries = {}

    def embed_documents(self, texts):
        if any(text not in self.vectors for text in texts):
            raise ValueError("[UnverifiedDocument] 저장 벡터가 검증되지 않은 문서입니다.")
        return [self.vectors[text] for text in texts]

    def embed_query(self, text):
        if text not in self.queries:
            vector = self.query_embeddings.embed_query(text)
            if (not isinstance(vector, list) or len(vector) != 1536 or
                    any(type(v) not in (int, float) or not math.isfinite(v) for v in vector) or
                    not any(vector)):
                raise ValueError("[QueryVectorInvalid] 질문 임베딩 차원이 저장 색인과 다릅니다.")
            self.queries[text] = vector
        return self.queries[text]
