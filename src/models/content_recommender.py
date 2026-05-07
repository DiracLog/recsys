from __future__ import annotations

import time
import warnings
from logging import getLogger

from pathlib import Path

import joblib
import numpy as np

from src.config.config_loader import load_config
from src.utils.loading_utils import prepare_data_for_dev_ml32
from src.models.abstract import BaseRecommender
from src.models.data_schemas import Catalog, TrainingData
from src.utils.utils import conf
import numpy.typing as npt
from src.logging.setup_logging import setup_logging

logger = getLogger(__name__)

setup_logging(log_file="log_content.log")


class ContentBasedRecommender(BaseRecommender):
    """Content-based recommender with two regularization knobs.

    Profile shrinkage:
        Pulls sparse user profiles toward the global mean via pseudo-count α.
        w=0.5 at n_i = α, profile dominates as n_i → inf, global as n_i → 0.

    Movie confidence:
        Sigmoid weight on log-count
        α sets the midpoint (conf = 0.5 at n = α), β sets the steepness.
        Suppresses low-count movies without a hard threshold.
    """

    _VALID_SHRINKAGE = {"bayesian", "none"}
    _params = ("profile_alpha", "movie_conf_alpha", "movie_conf_beta", "shrinkage", "movies_conf")
    _state = (
        "user_profiles_",
        "rated_indices_",
        "rated_indptr_",
        "user_ids_",
        "movie_ids_",
        "counts_",
        "movie_conf_vect_",
        "global_profile_",
        "movie_id_to_idx_",
    )

    def __init__(
        self,
        profile_alpha: float,
        movie_conf_alpha: float,
        movie_conf_beta: float,
        shrinkage: str,
        movies_conf: bool,
        dtype_mx: np.dtype = np.float16,
        dtype_indices: np.dtype = np.uint16,
    ):
        # hyperparameters — private
        self._profile_alpha = profile_alpha
        self._movie_conf_alpha = movie_conf_alpha
        self._movie_conf_beta = movie_conf_beta
        self._movie_conf = movies_conf
        self._dtype_mx = dtype_mx
        self._dtype_indices = dtype_indices
        if shrinkage not in self._VALID_SHRINKAGE:
            raise ValueError(f"shrinkage must be one of {self._VALID_SHRINKAGE}, got {shrinkage!r}")
        self._shrinkage = shrinkage

        # sklearn convention: trailing underscore for fitted attributes
        self.user_profiles_ = None

        self.global_profile_ = None
        self.rated_indices_ = None
        self.rated_indptr_ = None
        self.user_ids_ = None
        self.movie_ids_ = None
        self.counts_ = None
        self.movie_conf_vect_ = None
        self.movie_id_to_idx_ = None

    def fit(self, catalog: Catalog, data: TrainingData, **kwargs) -> ContentBasedRecommender:
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
        t = time.time()
        R = data.ratings_matrix
        self.movie_ids_ = data.movie_ids
        self.user_ids_ = data.user_ids
        # copy for coupling and mutation def
        # 4x memory reduction
        self.rated_indices_ = (R.indices.copy()).astype(self._dtype_indices)
        self.rated_indptr_ = R.indptr.copy()
        logger.info(f"Ratings matrix shape: {R.shape}")

        R_coo = R.tocoo()
        R_coo.data = R_coo.data + data.user_means[R_coo.row]
        weighted_sum = R_coo @ catalog.genre_matrix
        logger.info(f"Raw Ratings matrix shape: {R_coo.shape}")
        logger.info(f"Genre matrix shape: {catalog.genre_matrix.shape}")

        # weights to make 0 unbiased
        counts = R.getnnz(axis=1).reshape(-1, 1)
        user_profiles = weighted_sum / counts

        self.counts_ = counts
        self.user_profiles_ = user_profiles.astype(self._dtype_mx)
        movie_counts = R.getnnz(axis=0)
        self.global_profile_ = self.user_profiles_.mean(axis=0)
        if self._movie_conf:
            self.movie_conf_vect_ = conf(movie_counts, alpha=self._movie_conf_alpha, beta=self._movie_conf_beta)
        self.movie_id_to_idx_ = {mid: i for i, mid in enumerate(self.movie_ids_)}
        logger.info(f"Built user profiles, ellapsed time: {time.time() - t:.2f}")

        return self

    def recommend(self, catalog: Catalog, user_id: int, n: int = 10) -> list[dict]:
        """
        Get top n recommendations per existing user

        Parameters
        ----------
        user_id: int
            user's id to constuct predict for
        catalog: Catalog
            non-ratings data
        n: int
            number of recommendations to return

        Returns
        -------

        """
        if self.user_profiles_ is None:
            raise RuntimeError("Recommender is not fit")

        idx = np.where(self.user_ids_ == user_id)[0]
        if len(idx) == 0:
            raise ValueError("Unknown user_id")
        user_idx = idx[0]

        user_vector = self.user_profiles_[user_idx]
        n_i = self.counts_[user_idx]

        if self._shrinkage == "bayesian":
            w = self._n_i_weight(n_i)
            user_vector = w * user_vector + (1 - w) * self.global_profile_
        elif self._shrinkage == "none":
            pass
        else:
            raise ValueError(f"Unknown shrinkage: {self._shrinkage}")

        user_norm = np.linalg.norm(user_vector)

        movie_vectors = catalog.genre_matrix
        movie_norms = np.linalg.norm(movie_vectors, axis=1)

        denominator = user_norm * movie_norms
        numerator = movie_vectors @ user_vector

        similarities = np.divide(numerator, denominator, out=np.zeros_like(movie_norms), where=denominator != 0)

        if self.movie_conf_vect_ is not None:
            # add perspective how reliably users agree on that movie

            similarities *= self.movie_conf_vect_

        # exclude seen
        user_mask = self.rated_indices_[self.rated_indptr_[user_idx] : self.rated_indptr_[user_idx + 1]]
        similarities[user_mask] = -np.inf

        top_indices = np.argsort(similarities)[::-1][:n]

        return [
            {"movie_id": int(self.movie_ids_[i]), "score": float(similarities[i])}
            for i in top_indices
            if similarities[i] != -np.inf
        ]

    def score_all(self, catalog: Catalog, user_id: int) -> np.ndarray:
        """
        Get scores for all movies per existing user

        Parameters
        ----------
        user_id: int
            user's id to constuct predict for
        catalog: Catalog
            non-ratings data

        Returns
        -------
        method to retun all scores for given user_id
        """

        if self.user_profiles_ is None:
            raise RuntimeError("Recommender is not fit")

        idx = np.where(self.user_ids_ == user_id)[0]
        if len(idx) == 0:
            raise ValueError("Unknown user_id")
        user_pos = idx[0]
        user_vector = self.user_profiles_[user_pos, :]
        movie_vectors = catalog.genre_matrix
        user_norm = np.linalg.norm(user_vector)
        movie_norms = np.linalg.norm(movie_vectors, axis=1)
        denominator = user_norm * movie_norms
        similarities = np.divide(
            movie_vectors @ user_vector, denominator, out=np.zeros_like(movie_norms), where=denominator != 0
        )
        return similarities

    def score_all_from_ratings(
        self, catalog: Catalog, ratings: dict
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.bool_]]:
        """
        Get scores for all movies per user's ratings

        Parameters
        ----------
        ratings: dict
            user's ratings
        catalog: Catalog
            non-ratings data

        Returns
        -------
        tuple[npt.NDArray[np.float64], npt.NDArray[np.bool_]]  # (scores, rated_mask)
        """

        if self.user_profiles_ is None:
            raise RuntimeError("Recommender is not fit")

        movie_vectors = catalog.genre_matrix
        rated_mask = np.zeros(len(self.movie_ids_), dtype=bool)
        r = np.zeros(len(self.movie_ids_), dtype=bool)
        for movie_id, rating in ratings.items():
            if movie_id in self.movie_id_to_idx_:
                i = self.movie_id_to_idx_[movie_id]
                r[i] = rating
                rated_mask[i] = True
        weighted_sum = r @ movie_vectors

        # weights to make 0
        counts = rated_mask.sum()
        if counts == 0:
            warnings.warn(
                "No valid ratings provided; returning zero scores.",
                UserWarning,
                stacklevel=2,
            )
            return np.zeros(len(self.movie_ids_)), rated_mask

        user_vector = weighted_sum / counts

        user_norm = np.linalg.norm(user_vector)
        movie_norms = np.linalg.norm(movie_vectors, axis=1)
        denominator = user_norm * movie_norms

        # bayesian shrinkage per surprise library
        if self._shrinkage == "bayesian":
            n_i = counts
            w = self._n_i_weight(n_i)
            user_vector = w * user_vector + (1 - w) * self.global_profile_
        elif self._shrinkage == "none":
            pass
        else:
            raise ValueError(f"Unknown shrinkage: {self._shrinkage}")

        similarities = np.divide(
            movie_vectors @ user_vector, denominator, out=np.zeros_like(movie_norms), where=denominator != 0
        )

        return similarities, rated_mask

    def _n_i_weight(self, n_i: int) -> float:
        return n_i / (n_i + self._profile_alpha)

    @classmethod
    def load(cls, path: str | Path, mmap_mode: str | None = None) -> ContentBasedRecommender:
        data = joblib.load(path, mmap_mode=mmap_mode)
        obj = cls(**data["params"])

        for k, v in data["state"].items():
            setattr(obj, k, v)

        return obj

    def save(self, path: str | Path) -> None:
        """Save params which needed to restore fitted recommender"""
        path = Path(path)

        joblib.dump(
            {
                "params": {
                    "profile_alpha": self._profile_alpha,
                    "movie_conf_alpha": self._movie_conf_alpha,
                    "movie_conf_beta": self._movie_conf_beta,
                    "movies_conf": self._movie_conf,
                    "shrinkage": self._shrinkage,
                    "dtype_mx": self._dtype_mx,
                    "dtype_indices": self._dtype_indices,
                },
                "state": {
                    "user_profiles_": self.user_profiles_,
                    "rated_indices_": self.rated_indices_,
                    "rated_indptr_": self.rated_indptr_,
                    "user_ids_": self.user_ids_,
                    "movie_ids_": self.movie_ids_,
                    "counts_": self.counts_,
                    "movie_conf_vect_": self.movie_conf_vect_,
                    "global_profile_": self.global_profile_,
                    "movie_id_to_idx_": self.movie_id_to_idx_,
                },
            },
            path,
        )


if __name__ == "__main__":
    cfg = load_config()
    (
        catalog,
        train_data,
    ) = prepare_data_for_dev_ml32(cfg=cfg)

    recommender = ContentBasedRecommender(
        profile_alpha=cfg["models"]["content"]["profile_alpha"],
        movie_conf_alpha=cfg["models"]["content"]["movie_conf_alpha"],
        movie_conf_beta=cfg["models"]["content"]["movie_conf_beta"],
        shrinkage=cfg["models"]["content"]["shrinkage"],
        movies_conf=cfg["models"]["content"]["movies_conf"],
    )
    recommender.fit(catalog=catalog, data=train_data)
    # run_dir = construct_path(Path("artifacts") / "runs" / "content_test")
    # run_dir.mkdir(parents=True, exist_ok=True)
    # recommender.save(run_dir / "cntnt_recommender")
    recommender = recommender.load("artifacts/runs/2026-04-29-152605/cntnt_recommender")

    recommendations = recommender.recommend(catalog=catalog, user_id=1)

    # recommendations = recommender.recommend(catalog=catalog, user_id=1)
    # recommendations = recommender.enrich(recommendations, catalog=catalog)
