from typing import TypedDict

import numpy as np

# -------------------------
# Config typing
# -------------------------


class DataConfig(TypedDict):
    ratings_path: str
    movies_path: str
    users_path: str
    mode: str
    injection: bool


class PreprocessingConfig(TypedDict):
    m_threshold: int
    u_threshold: int


class CollaborativeConfig(TypedDict):
    k_principal: int


class ContentConfig(TypedDict):
    movie_conf_alpha: float
    movie_conf_beta: float
    profile_alpha: float
    shrinkage: str
    movies_conf: bool


class HybridConfig(TypedDict):
    weight: float


class PopularityConfig(TypedDict):
    movie_conf_alpha: float
    movie_conf_beta: float
    default_n: int


class FactorsConfig(TypedDict):
    dtype_mx: np.dtype = np.float16
    dtype_indices: np.dtype = np.uint16


class MemoryConfig(TypedDict):
    mmap_mode: str | None = "r"


class ModelsConfig(TypedDict):
    collaborative: CollaborativeConfig
    content: ContentConfig
    hybrid: HybridConfig
    popularity: PopularityConfig
    factors: FactorsConfig
    memory: MemoryConfig


class Config(TypedDict):
    data: DataConfig
    preprocessing: PreprocessingConfig
    models: ModelsConfig
