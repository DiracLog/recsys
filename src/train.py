import logging

from src.logging.setup_logging import setup_logging

from datetime import datetime
import time
from pathlib import Path

import joblib

from src.config.config_loader import load_config
from src.utils.loading_utils import prepare_data_for_dev, prepare_data_for_dev_ml32
from src.models.collaborative_recommender import CollaborativeRecommender
from src.models.content_recommender import ContentBasedRecommender
from src.models.data_schemas import SCHEMA_VERSION
from src.models.hybrid_recommender import HybridRecommender
from src.models.popularity_recommender import PopularityRecommender
from src.utils.saving_utils import write_run_meta, set_latest

from src.utils.utils import construct_path


import os

setup_logging(log_file="train.log")

logger = logging.getLogger(__name__)


def main():
    # construct everything for further running from artifacts
    cfg = load_config()
    run_id = os.getenv("RUN_ID") or datetime.now().strftime("%Y-%m-%d-%H%M%S")
    run_dir = construct_path(Path("artifacts") / "runs" / run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Run ID: {run_id}, directory: {run_dir}")
    if cfg["data"]["mode"] == "ml32":
        catalog, train_data = prepare_data_for_dev_ml32(cfg=cfg)
    else:
        catalog, train_data = prepare_data_for_dev(cfg=cfg)
    catalog.save(run_dir / "catalog")

    timings = {}

    t = time.time()
    popularity = PopularityRecommender(
        movie_conf_alpha=cfg["models"]["popularity"]["movie_conf_alpha"],
        movie_conf_beta=cfg["models"]["popularity"]["movie_conf_beta"],
    )
    popularity.fit(catalog, train_data)
    timings["popularity"] = time.time() - t
    popularity.save(run_dir / "popularity")

    cntnt_recommender = ContentBasedRecommender(
        profile_alpha=cfg["models"]["content"]["profile_alpha"],
        movie_conf_alpha=cfg["models"]["content"]["movie_conf_alpha"],
        movie_conf_beta=cfg["models"]["content"]["movie_conf_beta"],
        shrinkage=cfg["models"]["content"]["shrinkage"],
        movies_conf=cfg["models"]["content"]["movies_conf"],
    )

    clb_recommender = CollaborativeRecommender(
        k_principal=cfg["models"]["collaborative"]["k_principal"],
    )

    hybrid = HybridRecommender(
        weight=cfg["models"]["hybrid"]["weight"], content=cntnt_recommender, collaborative=clb_recommender
    )

    hybrid.fit(catalog=catalog, data=train_data)

    hybrid.save(run_dir / "hybrid")
    clb_recommender.save(run_dir / "clb_recommender")
    cntnt_recommender.save(run_dir / "cntnt_recommender")

    pop_precomputed = popularity.recommend(catalog=catalog, n=100)
    pop_precomputed = popularity.enrich(pop_precomputed, catalog=catalog)
    joblib.dump(pop_precomputed, run_dir / "pop_precomputed.joblib")

    # meta.json for info about latest artifacts
    write_run_meta(run_dir, run_id, timings, train_data, cfg, SCHEMA_VERSION)

    # update latest symlink / latest.txt for windows
    set_latest(construct_path(Path("artifacts") / "runs"), run_id)

    logger.info("Run complete, artifacts saved")


if __name__ == "__main__":
    main()
