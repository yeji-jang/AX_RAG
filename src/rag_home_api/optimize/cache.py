"""[STEP 5-2: 버전 인식 캐싱] SQLite 기반 RAG 응답 캐시 (적중 <100ms · 버전 변경 무효화 · error 미저장).

작성자: 최기혁 (kh)
안전 규칙 (rag_eval_guide.html 3장):
  1. 캐시 키에 collection_name · index_version · embedding_model · llm_model · prompt_version · preset_signature · user_scope 를 모두 포함해
     문서·인덱스·프롬프트·권한이 바뀌면 자동으로 캐시 미스(무효화)가 발생한다.
  2. status == "error" 또는 빈 응답은 절대 캐시에 저장하지 않는다.
  3. TTL 만료 시 자동 미스 처리한다.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from contextlib import closing
from pathlib import Path
from typing import Any

from rag_home_api.config import DATA_DIR

DEFAULT_CACHE_DB = DATA_DIR / "cache" / "rag_cache.sqlite3"


def normalize_query(query: str) -> str:
    return " ".join(query.strip().split())


def build_cache_key(
    *,
    query: str,
    collection_name: str,
    index_version: str,
    embedding_model: str,
    llm_model: str,
    prompt_version: str,
    preset_dict: dict[str, Any],
    user_scope: str = "public",
) -> str:
    """버전·설정·권한 경계를 모두 반영한 결정적 SHA-256 캐시 키를 생성한다."""
    # 캐시 토글 자체는 키 비교에서 제외해 ON/OFF 동일 설정 재사용
    preset_clean = {k: v for k, v in preset_dict.items() if k not in ("use_cache", "cache_ttl_seconds")}
    payload = {
        "q": normalize_query(query),
        "collection": collection_name,
        "index_version": index_version,
        "embedding_model": embedding_model,
        "llm_model": llm_model,
        "prompt_version": prompt_version,
        "preset": preset_clean,
        "user_scope": user_scope,
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class VersionedRAGCache:
    """SQLite 기반 영속 버전 인식 캐시."""

    def __init__(self, db_path: Path = DEFAULT_CACHE_DB, ttl_seconds: int = 3600) -> None:
        self.db_path = db_path
        self.ttl_seconds = ttl_seconds
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with closing(sqlite3.connect(self.db_path)) as conn:
            with conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS rag_cache (
                        cache_key TEXT PRIMARY KEY,
                        payload_json TEXT NOT NULL,
                        created_at REAL NOT NULL,
                        expires_at REAL NOT NULL
                    )
                    """
                )

    def get(self, cache_key: str) -> dict[str, Any] | None:
        now = time.time()
        with closing(sqlite3.connect(self.db_path)) as conn:
            row = conn.execute(
                "SELECT payload_json, expires_at FROM rag_cache WHERE cache_key = ?",
                (cache_key,),
            ).fetchone()
            if not row:
                return None
            payload_json, expires_at = row
            if now > float(expires_at):
                with conn:
                    conn.execute("DELETE FROM rag_cache WHERE cache_key = ?", (cache_key,))
                return None
            return json.loads(payload_json)

    def set(self, cache_key: str, payload: dict[str, Any], ttl_seconds: int | None = None) -> bool:
        """정상 응답만 저장한다. status='error' 또는 응답 텍스트가 비어 있으면 저장을 거부한다."""
        if payload.get("status") == "error":
            return False
        if not str(payload.get("response", "")).strip():
            return False

        now = time.time()
        ttl = self.ttl_seconds if ttl_seconds is None else ttl_seconds
        with closing(sqlite3.connect(self.db_path)) as conn:
            with conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO rag_cache (cache_key, payload_json, created_at, expires_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (cache_key, json.dumps(payload, ensure_ascii=False), now, now + ttl),
                )
        return True

    def clear(self) -> int:
        with closing(sqlite3.connect(self.db_path)) as conn:
            with conn:
                cur = conn.execute("DELETE FROM rag_cache")
                return cur.rowcount

    def count(self) -> int:
        with closing(sqlite3.connect(self.db_path)) as conn:
            row = conn.execute("SELECT COUNT(*) FROM rag_cache").fetchone()
            return int(row[0]) if row else 0
