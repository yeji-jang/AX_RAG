"""프로젝트 공통 설정. .env 값을 읽어 한곳에서 관리한다."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# 프로젝트 최상위 폴더 (노트북·스크립트 어디서 실행해도 같은 경로를 쓰기 위함)
ROOT = Path(__file__).resolve().parents[2]  # src/rag_home_api/config.py → 프로젝트 최상위
DATA_DIR = ROOT / "data"
EVAL_DIR = ROOT / "eval"
RESULTS_DIR = ROOT / "results"
SHARED_DOCS_DIR = ROOT / "src" / "rag_home_api" / "shared" / "docs"

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", extra="ignore")

    openai_api_key: str
    data_go_kr_service_key: str = ""  # 공공데이터포털 활용신청 후 채움

    # A/B/C 비교 시 모두 같은 값을 써야 한다 (지침 5장)
    # 현재 키로 쓸 수 있는 생성 모델은 gpt-6-luna 하나뿐 (2026-11-23 확인)
    llm_model: str = "gpt-6-luna"
    rerank_model: str = "gpt-6-luna"
    embedding_model: str = "text-embedding-3-small"
    database_url: str
    collection_name: str = "seoul_housing_unified"
    index_version: str = "idx-v3-c600-o100-table"  # 구조적 표 보존 + 공고 구분자 + 풍부한 Prefix

    # 검색 B (Hybrid RRF) 가중치·상수 (dev 10문항으로만 튜닝 후 고정)
    bm25_weight: float = 0.55
    vector_weight: float = 0.45
    rrf_k: int = 60
    
settings = Settings()
