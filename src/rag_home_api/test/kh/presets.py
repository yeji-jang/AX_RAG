"""실험 파라미터 프리셋 정의 (A / B / 경량필터 / C / Query Rewrite / Cache / HyDE / 청킹·임베딩 분리).

작성자: 최기혁 (kh)
원칙:
  - 모든 실험 변수는 ExperimentPreset 객체의 스위치 1개만 켜서 독립 비교(단일 변인 통제)한다.
  - 청킹 전략이나 임베딩 모델을 바꿀 때는 기존 컬렉션을 절대 덮어쓰지 않고
    별도 collection_name 에 적재한 뒤, 여기의 collection_name · embedding_model 만 지정해 안전하게 비교한다.
"""

from dataclasses import asdict, dataclass
from typing import Literal

PipelineMode = Literal["A_vector", "B_hybrid", "C_hybrid_rerank"]


@dataclass(frozen=True)
class ExperimentPreset:
    name: str
    pipeline_mode: PipelineMode
    # 검색 파라미터 (STEP 3)
    top_k: int = 5
    candidate_k: int = 12
    bm25_weight: float = 0.55
    vector_weight: float = 0.45
    rrf_k: int = 60
    # 최적화 스위치 (STEP 5 — 반드시 1개씩 독립 ON/OFF!)
    use_metadata_filter: bool = False
    use_rewrite: bool = False
    use_cache: bool = False
    use_hyde: bool = False
    # 청킹·임베딩·프롬프트 버전 격리 (안전한 별도 컬렉션 비교 & 캐시 무효화 키)
    collection_name: str = "housing_rent_kh"
    embedding_model: str = "text-embedding-3-small"
    chunk_strategy: str = "rent-v2-11cols-src"
    prompt_version: str = "rent-ans-v2-subject"
    cache_ttl_seconds: int = 3600

    def to_dict(self) -> dict:
        return asdict(self)


# ============================================================
# [기본 5종] A vs B(가중치 3종) vs B + 경량 메타데이터 필터
# ============================================================
RENT_AB_PRESETS: list[ExperimentPreset] = [
    ExperimentPreset(name="A_vector", pipeline_mode="A_vector", top_k=5, candidate_k=12),
    ExperimentPreset(
        name="B_rrf_55_45",
        pipeline_mode="B_hybrid",
        top_k=5,
        candidate_k=12,
        bm25_weight=0.55,
        vector_weight=0.45,
        rrf_k=60,
    ),
    ExperimentPreset(
        name="B_rrf_70_30",
        pipeline_mode="B_hybrid",
        top_k=5,
        candidate_k=12,
        bm25_weight=0.70,
        vector_weight=0.30,
        rrf_k=60,
    ),
    ExperimentPreset(
        name="B_rrf_80_20",
        pipeline_mode="B_hybrid",
        top_k=5,
        candidate_k=12,
        bm25_weight=0.80,
        vector_weight=0.20,
        rrf_k=60,
    ),
    ExperimentPreset(
        name="B_rrf_70_30_filter",
        pipeline_mode="B_hybrid",
        top_k=5,
        candidate_k=12,
        bm25_weight=0.70,
        vector_weight=0.30,
        rrf_k=60,
        use_metadata_filter=True,
    ),
]

# ============================================================
# [확장 스위치 프리셋] C(Re-ranking) · Rewrite · Cache · HyDE 독립 토글
# ============================================================
RENT_ADVANCED_PRESETS: list[ExperimentPreset] = [
    # 1) 검색 C (Re-ranking) 단독 추가 (B_rrf_70_30_filter 대비 변수 1개: pipeline_mode)
    ExperimentPreset(
        name="C_rrf_70_30_filter_rerank",
        pipeline_mode="C_hybrid_rerank",
        top_k=5,
        candidate_k=12,
        bm25_weight=0.70,
        vector_weight=0.30,
        rrf_k=60,
        use_metadata_filter=True,
    ),
    # 2) B + Filter + Query Rewrite 단독 추가 (C 없이 Rewrite 효과만 독립 검증)
    ExperimentPreset(
        name="B_filter_rewrite",
        pipeline_mode="B_hybrid",
        top_k=5,
        candidate_k=12,
        bm25_weight=0.70,
        vector_weight=0.30,
        rrf_k=60,
        use_metadata_filter=True,
        use_rewrite=True,
    ),
    # 3) C + Filter + Query Rewrite 결합 (2단계 FN + 3단계 FP 동시 해결)
    ExperimentPreset(
        name="C_filter_rewrite",
        pipeline_mode="C_hybrid_rerank",
        top_k=5,
        candidate_k=12,
        bm25_weight=0.70,
        vector_weight=0.30,
        rrf_k=60,
        use_metadata_filter=True,
        use_rewrite=True,
    ),
    # 4) + 버전 인식 캐싱 ON (지연 <100ms · 버전 무효화 검증)
    ExperimentPreset(
        name="C_filter_rewrite_cache",
        pipeline_mode="C_hybrid_rerank",
        top_k=5,
        candidate_k=12,
        bm25_weight=0.70,
        vector_weight=0.30,
        rrf_k=60,
        use_metadata_filter=True,
        use_rewrite=True,
        use_cache=True,
    ),
    # 5) + HyDE ON (Vector 전용 가상문서 + 출처 격리 검증)
    ExperimentPreset(
        name="C_filter_rewrite_hyde",
        pipeline_mode="C_hybrid_rerank",
        top_k=5,
        candidate_k=12,
        bm25_weight=0.70,
        vector_weight=0.30,
        rrf_k=60,
        use_metadata_filter=True,
        use_rewrite=True,
        use_hyde=True,
    ),
    # 6) [안전한 청킹 비교용] 평수·쉬운 별칭 병기 청킹 컬렉션 (rent_chunker --collection housing_rent_kh_v3_pyeong --include-pyeong 적재 후 사용)
    ExperimentPreset(
        name="B_chunk_v3_pyeong",
        pipeline_mode="B_hybrid",
        top_k=5,
        candidate_k=12,
        bm25_weight=0.70,
        vector_weight=0.30,
        rrf_k=60,
        use_metadata_filter=True,
        collection_name="housing_rent_kh_v3_pyeong",
        chunk_strategy="rent-v3-pyeong",
    ),
    # 7) [안전한 임베딩 비교용] text-embedding-3-large 전용 컬렉션
    ExperimentPreset(
        name="B_emb_3large",
        pipeline_mode="B_hybrid",
        top_k=5,
        candidate_k=12,
        bm25_weight=0.70,
        vector_weight=0.30,
        rrf_k=60,
        use_metadata_filter=True,
        collection_name="housing_rent_kh_3large",
        embedding_model="text-embedding-3-large",
        chunk_strategy="rent-v2-11cols-src",
    ),
    # 8) [안전한 로컬 GPU 임베딩 비교용] 한국어 특화 KURE-v1 전용 컬렉션 (A 단독 / B 가중치 스윕 4종 / B 70:30+Filter)
    ExperimentPreset(
        name="A_vector_kure_v1",
        pipeline_mode="A_vector",
        top_k=5,
        candidate_k=12,
        collection_name="housing_rent_kh_kure_v1",
        embedding_model="nlpai-lab/KURE-v1",
        chunk_strategy="rent-v2-11cols-src",
    ),
    ExperimentPreset(
        name="B_rrf_10_90_kure_v1",
        pipeline_mode="B_hybrid",
        top_k=5,
        candidate_k=12,
        bm25_weight=0.10,
        vector_weight=0.90,
        rrf_k=60,
        collection_name="housing_rent_kh_kure_v1",
        embedding_model="nlpai-lab/KURE-v1",
        chunk_strategy="rent-v2-11cols-src",
    ),
    ExperimentPreset(
        name="B_rrf_25_75_kure_v1",
        pipeline_mode="B_hybrid",
        top_k=5,
        candidate_k=12,
        bm25_weight=0.25,
        vector_weight=0.75,
        rrf_k=60,
        collection_name="housing_rent_kh_kure_v1",
        embedding_model="nlpai-lab/KURE-v1",
        chunk_strategy="rent-v2-11cols-src",
    ),
    ExperimentPreset(
        name="B_rrf_50_50_kure_v1",
        pipeline_mode="B_hybrid",
        top_k=5,
        candidate_k=12,
        bm25_weight=0.50,
        vector_weight=0.50,
        rrf_k=60,
        collection_name="housing_rent_kh_kure_v1",
        embedding_model="nlpai-lab/KURE-v1",
        chunk_strategy="rent-v2-11cols-src",
    ),
    ExperimentPreset(
        name="B_rrf_55_45_kure_v1",
        pipeline_mode="B_hybrid",
        top_k=5,
        candidate_k=12,
        bm25_weight=0.55,
        vector_weight=0.45,
        rrf_k=60,
        collection_name="housing_rent_kh_kure_v1",
        embedding_model="nlpai-lab/KURE-v1",
        chunk_strategy="rent-v2-11cols-src",
    ),
    # 8-b) [원인 ①·② 수정 검증] BM25 숫자 중복 버그 수정 + rrf_k=15 (단독 리스트 페널티 해소)
    ExperimentPreset(
        name="B_rrf_25_75_k15_kure_v1",
        pipeline_mode="B_hybrid",
        top_k=5,
        candidate_k=12,
        bm25_weight=0.25,
        vector_weight=0.75,
        rrf_k=15,
        collection_name="housing_rent_kh_kure_v1",
        embedding_model="nlpai-lab/KURE-v1",
        chunk_strategy="rent-v2-11cols-src",
    ),
    ExperimentPreset(
        name="B_rrf_50_50_k15_kure_v1",
        pipeline_mode="B_hybrid",
        top_k=5,
        candidate_k=12,
        bm25_weight=0.50,
        vector_weight=0.50,
        rrf_k=15,
        collection_name="housing_rent_kh_kure_v1",
        embedding_model="nlpai-lab/KURE-v1",
        chunk_strategy="rent-v2-11cols-src",
    ),
    # 8-c) [사전 필터링(Pre-filtering) 단일 변인] FIX B 25:75 (k=15) + use_metadata_filter=True
    ExperimentPreset(
        name="B_rrf_25_75_k15_filter_kure_v1",
        pipeline_mode="B_hybrid",
        top_k=5,
        candidate_k=12,
        bm25_weight=0.25,
        vector_weight=0.75,
        rrf_k=15,
        use_metadata_filter=True,
        collection_name="housing_rent_kh_kure_v1",
        embedding_model="nlpai-lab/KURE-v1",
        chunk_strategy="rent-v2-11cols-src",
    ),
    ExperimentPreset(
        name="B_rrf_25_75_k15_filter_v2_kure_v1",
        pipeline_mode="B_hybrid",
        top_k=5,
        candidate_k=12,
        bm25_weight=0.25,
        vector_weight=0.75,
        rrf_k=15,
        use_metadata_filter=True,
        collection_name="housing_rent_kh_kure_v1",
        embedding_model="nlpai-lab/KURE-v1",
        chunk_strategy="rent-v2-11cols-src",
    ),
    ExperimentPreset(
        name="B_emb_kure_v1",
        pipeline_mode="B_hybrid",
        top_k=5,
        candidate_k=12,
        bm25_weight=0.70,
        vector_weight=0.30,
        rrf_k=60,
        use_metadata_filter=True,
        collection_name="housing_rent_kh_kure_v1",
        embedding_model="nlpai-lab/KURE-v1",
        chunk_strategy="rent-v2-11cols-src",
    ),
]

ALL_RENT_PRESETS: list[ExperimentPreset] = [*RENT_AB_PRESETS, *RENT_ADVANCED_PRESETS]
PRESET_REGISTRY: dict[str, ExperimentPreset] = {p.name: p for p in ALL_RENT_PRESETS}


# ============================================================
# [레거시 호환] 11/24 미니 통합 실험용 프리셋 (exp_02_abc.py)
# ============================================================
NOW_AB_PRESETS: list[ExperimentPreset] = [
    ExperimentPreset(name="A_vector", pipeline_mode="A_vector", top_k=3),
    ExperimentPreset(name="B_rrf_50_50", pipeline_mode="B_hybrid", top_k=3, bm25_weight=0.50, vector_weight=0.50),
    ExperimentPreset(name="B_rrf_55_45", pipeline_mode="B_hybrid", top_k=3, bm25_weight=0.55, vector_weight=0.45),
    ExperimentPreset(name="B_rrf_70_30", pipeline_mode="B_hybrid", top_k=3, bm25_weight=0.70, vector_weight=0.30),
]
