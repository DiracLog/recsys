from logging import getLogger
import os
from pathlib import Path

import joblib

from src.api.movie_search import MovieSearch

from src.config.config_loader import load_config
from src.config.config_schema import Config
from src.utils.loading_utils import prepare_data_for_dev

from src.models.data_schemas import TrainingData, Catalog
from src.models.collaborative_recommender import CollaborativeRecommender
from src.models.content_recommender import ContentBasedRecommender
from src.models.hybrid_recommender import HybridRecommender
from src.models.popularity_recommender import PopularityRecommender
from src.utils.saving_utils import resolve_latest
from dotenv import load_dotenv

from src.utils.utils import construct_path


logger = getLogger(__name__)


def prepare_state(
    load_from_artifacts: bool = False,
) -> tuple[HybridRecommender, PopularityRecommender, MovieSearch, list[dict], Catalog]:
    load_dotenv()
    if not load_from_artifacts:
        if os.getenv("LOAD_ARTIFACTS") == "1":
            path = construct_path(Path("artifacts") / "runs")
            return prepare_from_artifacts(path)
    else:
        prepare_from_artifacts(path)
    return prepare_fresh()


def build_and_fit_recommenders(
    cfg: Config, catalog: Catalog, data: TrainingData
) -> tuple[HybridRecommender, PopularityRecommender, MovieSearch]:
    cntnt_recommender = ContentBasedRecommender(
        profile_alpha=cfg["models"]["content"]["profile_alpha"],
        movie_conf_alpha=cfg["models"]["content"]["movie_conf_alpha"],
        movie_conf_beta=cfg["models"]["content"]["movie_conf_beta"],
        shrinkage=cfg["models"]["content"]["shrinkage"],
        movies_conf=cfg["models"]["content"]["movies_conf"],
        dtype_indices=cfg["models"]["factors"]["dtype_indices"],
        dtype_mx=cfg["models"]["factors"]["dtype_mx"],
    )

    clb_recommender = CollaborativeRecommender(
        k_principal=cfg["models"]["collaborative"]["k_principal"],
        dtype_indices=cfg["models"]["factors"]["dtype_indices"],
        dtype_mx=cfg["models"]["factors"]["dtype_mx"],
    )

    hybrid = HybridRecommender(
        weight=cfg["models"]["hybrid"]["weight"], content=cntnt_recommender, collaborative=clb_recommender
    )

    hybrid.fit(catalog=catalog, data=data)

    popularity = PopularityRecommender(
        movie_conf_alpha=cfg["models"]["popularity"]["movie_conf_alpha"],
        movie_conf_beta=cfg["models"]["popularity"]["movie_conf_beta"],
    )

    popularity.fit(catalog=catalog, data=data)

    movie_search = MovieSearch()
    movie_search.fit(catalog=catalog)

    return hybrid, popularity, movie_search


def prepare_fresh() -> tuple[HybridRecommender, PopularityRecommender, MovieSearch, list[dict], Catalog]:
    cfg = load_config()
    catalog, train_data = prepare_data_for_dev(cfg=cfg)
    hybrid, popularity, movie_search = build_and_fit_recommenders(cfg=cfg, catalog=catalog, data=train_data)
    pop_precomputed = popularity.recommend(n=100, catalog=catalog)
    pop_result = popularity.enrich(pop_precomputed, catalog=catalog)
    return hybrid, popularity, movie_search, pop_result, catalog


def prepare_from_artifacts(
    path: Path,
) -> tuple[HybridRecommender, PopularityRecommender, MovieSearch, list[dict], Catalog]:
    cfg = load_config()
    path = resolve_latest(runs_dir=path)
    logger.info("Resolved path: %s", path)
    logger.info("Path exists: %s", path.exists())
    logger.info("Is dir: %s", path.is_dir())
    if path.exists():
        logger.info("Contents: %s", [p.name for p in path.iterdir()])
    catalog_path = path / "catalog"
    catalog = Catalog.load(catalog_path)
    logger.info("Catalog path: %s, exists: %s", catalog_path, catalog_path.exists())
    if catalog_path.exists():
        logger.info("Catalog contents: %s", [p.name for p in catalog_path.iterdir()])
    popularity = PopularityRecommender.load(path / "popularity")
    hybrid = HybridRecommender.load(path / "hybrid", mmap_mode=cfg["models"]["memory"]["mmap_mode"])

    # MovieSearch fits from catalog in seconds — not worth artifacting
    movie_search = MovieSearch()
    movie_search.fit(catalog=catalog)

    pop_precomputed = joblib.load(path / "pop_precomputed.joblib")

    return hybrid, popularity, movie_search, pop_precomputed, catalog


if __name__ == "__main__":
    hybrid, popularity, movie_search, pop_precomputed, catalog = prepare_state()
    print(pop_precomputed[1:5])
