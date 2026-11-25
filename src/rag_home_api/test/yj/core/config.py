"""
- 전역에서 참조할 LLM 모델 및 임베딩 설정, 
- PGVector 접속 정보를 관리
"""

import os
from dotenv import load_dotenv
from pathlib import Path


load_dotenv(Path(__file__).resolve().parents[1] / ".env")
load_dotenv()

# 임베딩 모델 지정
embedding_model = "text-embedding-3-small"
llm_provider = "openai"
llm_model = "gpt-6-luna"
reasoning_effort = "low"


# PGVector 설정
database_url = "" # 실제값은 .env에 포함
public_rental_notice_collection = "public_rental_notice_collection"

class Embedding_PGVector_Settings : 
    # 임베딩 LLM 모델 설정
    API_KEY : str = os.getenv("OPENAI_API_KEY", "")
    EMBEDDING_MODEL : str = os.getenv("EMBEDDING_MODEL", embedding_model)
    LLM_PROVIDER : str = os.getenv("LLM_PROVIDER", llm_provider)
    LLM_MODEL : str = os.getenv("LLM_MODEL", llm_model)
    REASONING_EFFORT : str = os.getenv("REASONING_EFFORT", reasoning_effort)


    # PostgreSQL (PGVector) 접속 정보
    DATABASE_URL : str = os.getenv(
        "DATABASE_URL",
        database_url
    )

    # 도메인 컬렉션 분리
    PUBLIC_RENTAL_NOTICE_COLLECTION : str = os.getenv(
        "PUBLIC_RENTAL_NOTICE_COLLECTION",
        public_rental_notice_collection
    )


embedding_settings = Embedding_PGVector_Settings()