from __future__ import annotations

import numpy as np
import polars as pl

from src.config.config_loader import load_config
from src.utils.loading_utils import prepare_data_for_dev_ml32
from src.models.abstract import BaseRecommender
from src.models.data_schemas import Catalog, TrainingData
from src.utils.utils import conf


class PopularityRecommender(BaseRecommender):
    """
    Popularity recommender, returns top k most popular movies, adjusted by the confidences
    """

    _params = ("movie_conf_alpha", "movie_conf_beta")
    _state = ("resulting_scores_", "movies_counts_", "movie_ids_")

    def __init__(self, movie_conf_alpha: float, movie_conf_beta: float):
        # hyperparams
        self._movie_conf_alpha = movie_conf_alpha
        self._movie_conf_beta = movie_conf_beta

        # Internal
        self._full_result = None

        # complete during fit
        self.resulting_scores_: np.ndarray | None = None
        self.movies_counts_: np.ndarray | None = None
        self.movie_ids_: np.ndarray | None = None

    def fit(self, catalog: Catalog, data: TrainingData, **kwargs) -> PopularityRecommender:
        """

        Parameters
        ----------
        catalog: Catalog
            non-ratings data
        data: TrainingData
            data for fit

        Returns
        -------
            self for chaining
        """
        movies_df = data.ratings_df.group_by("movie_id").agg(
            mean_rating=pl.col("rating").mean(),
            count=pl.col("rating").count(),
        )
        global_mean = data.ratings_df["rating"].mean()

        movie_ids = movies_df["movie_id"].to_numpy()
        movies_means_np = movies_df["mean_rating"].to_numpy()
        movies_counts_np = movies_df["count"].to_numpy()

        weights = conf(movies_counts_np, alpha=self._movie_conf_alpha, beta=self._movie_conf_beta)
        self.resulting_scores_ = movies_means_np * weights + global_mean * (1 - weights)
        self.movies_counts_ = movies_counts_np
        self.movie_ids_ = movie_ids

        return self

    def recommend(self, catalog: Catalog | None, user_id: int | None = None, n: int = 10) -> list[dict[str, float]]:
        """Get top n recommendations for given user_id.
        For popularity model, user_id is not used, but kept for interface consistency."""

        if self.resulting_scores_ is None:
            raise RuntimeError("Fit the model first")

        idxs = np.argsort(self.resulting_scores_)[::-1][:n]
        self._full_result = [
            {
                "movie_id": int(self.movie_ids_[i]),
                "score": float(self.resulting_scores_[i]),
                "num_ratings": int(self.movies_counts_[i]),
            }
            for i in idxs
        ]

        final = [
            {
                "movie_id": item["movie_id"],
                "score": item["score"],
            }
            for item in self._full_result
        ]

        if not self._validate_full_response(final):
            raise RuntimeError("Result format is invalid")
        return final


if __name__ == "__main__":
    cfg = load_config()
    catalog, train_data = prepare_data_for_dev_ml32(cfg=cfg)
    recommender = PopularityRecommender(
        movie_conf_alpha=cfg["models"]["popularity"]["movie_conf_alpha"],
        movie_conf_beta=cfg["models"]["popularity"]["movie_conf_beta"],
    )

    recommender.fit(catalog, train_data)
    rec = recommender.recommend(catalog=None, user_id=None, n=5)
    print(recommender.enrich(recs=rec, catalog=catalog))
