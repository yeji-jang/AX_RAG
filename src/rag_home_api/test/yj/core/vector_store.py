"""
init_chat_model 기반 LLM 팩토리와 OpenAIEmbeddings, PGVector 팩토리 함수를 정의
"""

from typing import Protocol, Any
import psycopg
from langchain.chat_models import init_chat_model
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.vectorstores import VectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector
from langchain_postgres.vectorstores import DistanceStrategy
from rag_home_api.test.yj.core.config import embedding_settings
from rag_home_api.test.yj.core.logger import get_logger


logger = get_logger(__name__)


def get_llm() -> BaseChatModel :
    """설정된 프로바이더에 맞는 채팅 모델 인스턴스를 추상 타입으로 반환"""
    return init_chat_model(
        model=embedding_settings.LLM_MODEL,
        model_provider=embedding_settings.LLM_PROVIDER,
        api_key=embedding_settings.API_KEY,
        reasoning_effort=embedding_settings.REASONING_EFFORT,
    )


def get_embeddings() -> Embeddings:
    """임베딩 모델 인스턴스를 공통 Embeddings 인터페이스로 반환"""
    return OpenAIEmbeddings(
        model=embedding_settings.EMBEDDING_MODEL,
        api_key=embedding_settings.API_KEY,
    )


def get_vector_store(
        collection_name: str, 
        reset : bool = False 
        ) -> VectorStore:
    """
    지정된 컬렉션 이름의 벡터 스토어를 공통 VectorStore 인터페이스로 반환
    reset=True : 이 컬렉션만 비우고 시작 (적재용) / 검색·평가에서는 False
    """
    return PGVector(
        embeddings=get_embeddings(),
        collection_name=collection_name,
        connection=embedding_settings.DATABASE_URL,
        distance_strategy=DistanceStrategy.COSINE,
        use_jsonb=True,
        pre_delete_collection=reset,
    )

class DBConnection(Protocol):
    def cursor(self) -> Any: ... 
    def commit(self) -> None: ...
    def rollback(self) -> None: ...
    def close(self) -> None: ...


def get_raw_db_connection() -> DBConnection:
    """증분 해시 조회 및 BM25 코퍼스 로드를 위한 DB 커넥션을 반환"""
    raw_dsn = embedding_settings.DATABASE_URL.replace(
        "postgresql+psycopg://", 
        "postgresql://")
    return psycopg.connect(raw_dsn)