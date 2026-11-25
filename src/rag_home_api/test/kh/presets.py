"""실험 파라미터 프리셋 정의.

- 지금은 A(Vector) / B(Hybrid RRF 가중치) 실험에 사용하고,
- 이후 C(Rerank) / Query Rewrite / Cache / HyDE 단계가 붙을 때
  같은 ExperimentPreset 객체의 스위치만 켜서 재사용한다.
"""

from dataclasses import asdict, dataclass
from typing import Literal

PipelineMode = Literal["A_vector", "B_hybrid", "C_hybrid_rerank"]


@dataclass(frozen=True)
class ExperimentPreset:
    name: str
    pipeline_mode: PipelineMode
    # 검색 파라미터 (지금 STEP 3에서 주로 튜닝하는 값)
    top_k: int = 5
    candidate_k: int = 12
    bm25_weight: float = 0.55
    vector_weight: float = 0.45
    rrf_k: int = 60
    # 최적화 스위치 (STEP 5에서 1개씩 독립적으로 ON -> 효과 분리 측정 보장!)
    use_metadata_filter: bool = False
    use_rewrite: bool = False
    use_cache: bool = False
    use_hyde: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


# ============================================================
# [전월세 AB 심화 비교] A vs B 가중치 3종 vs B + 경량 메타데이터 필터
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
# [지금 바로 실행] A(Vector) vs B(Hybrid RRF 가중치 3종) 비교 프리셋
# ============================================================
NOW_AB_PRESETS: list[ExperimentPreset] = [
    ExperimentPreset(name="A_vector", pipeline_mode="A_vector", top_k=3),
    ExperimentPreset(name="B_rrf_50_50", pipeline_mode="B_hybrid", top_k=3, bm25_weight=0.50, vector_weight=0.50),
    ExperimentPreset(name="B_rrf_55_45", pipeline_mode="B_hybrid", top_k=3, bm25_weight=0.55, vector_weight=0.45),
    ExperimentPreset(name="B_rrf_70_30", pipeline_mode="B_hybrid", top_k=3, bm25_weight=0.70, vector_weight=0.30),
]

# ============================================================
# [이후 발제 로드맵 예약] STEP 3(C) -> STEP 5(Rewrite/Cache/HyDE)
# ============================================================
STAGE1_ABC_PRESETS: list[ExperimentPreset] = [
    ExperimentPreset(name="A_vector", pipeline_mode="A_vector"),
    ExperimentPreset(name="B_hybrid", pipeline_mode="B_hybrid"),
    ExperimentPreset(name="C_hybrid_rerank", pipeline_mode="C_hybrid_rerank"),
]

STAGE2_REWRITE_PRESETS: list[ExperimentPreset] = [
    ExperimentPreset(name="C_plus_rewrite", pipeline_mode="C_hybrid_rerank", use_rewrite=True),
]

STAGE3_CACHE_PRESETS: list[ExperimentPreset] = [
    ExperimentPreset(name="C_rewrite_cache", pipeline_mode="C_hybrid_rerank", use_rewrite=True, use_cache=True),
]

STAGE4_HYDE_PRESETS: list[ExperimentPreset] = [
    ExperimentPreset(name="C_rewrite_hyde", pipeline_mode="C_hybrid_rerank", use_rewrite=True, use_hyde=True),
]