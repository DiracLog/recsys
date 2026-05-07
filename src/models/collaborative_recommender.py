from __future__ import annotations

import warnings
from logging import getLogger

import time

import numpy as np

from src.config.config_loader import load_config
from src.models.abstract import BaseRecommender
from src.models.data_schemas import TrainingData, Catalog

from scipy.sparse.linalg import svds, ArpackNoConvergence

from src.utils.loading_utils import prepare_data

from src.logging.setup_logging import setup_logging

logger = getLogger(__name__)
setup_logging(log_file="collab.log")


class CollaborativeRecommender(BaseRecommender):
    """
    Collaborative recommendation system bases on SVD
    Recommendations for new users are derived from projecting their rating into latent space
    """

    _params = ("k_principal", "dtype_mx", "dtype_indices")
    _state = (
        "user_means_",
        "user_ids_",
        "rated_indices_",
        "rated_indptr_",
        "movie_ids_",
        "U_",
        "S_",
        "Vh_",
        "movie_id_to_idx_",
    )

    def __init__(self, k_principal: int = 50, dtype_mx: np.dtype = np.float16, dtype_indices: np.dtype = np.uint16):
        """
        Collaborative recommender - uses trunc SVD for creating latent user-movie space

        Parameters
        ----------
        k_principal : int, optional
            _description_, by default 50
        dtype : np.dtype, optional
            type of saving for np arrays, by default np.float32, options: np.float64, np.float16
        dtype_indices: np.dtype, optional
            type for saving indices of rated movies, by default np.uint16, options: np.uint32, np.uint64, np.uint16,
              np.uint8
        """
        # internal
        self._k_principal = k_principal
        self._dtype_mx = dtype_mx
        self._dtype_indices = dtype_indices

        # populated w/ fit
        self.user_means_ = None
        self.user_ids_ = None
        self.rated_indices_ = None
        self.rated_indptr_ = None  # additional ptrs for restoring flatten array
        self.movie_ids_ = None
        self.U_ = None
        self.S_ = None
        self.Vh_ = None
        self.movie_id_to_idx_ = None

    def fit(self, catalog: Catalog | None, data: TrainingData, **kwargs) -> CollaborativeRecommender:
        """

        Parameters
        ----------
        kwargs: dict
            Parameters like how many principal components to keep
        data: TrainingData
            ratings data for fit
        catalog: Catalog
            non-ratings static data
        Returns
        -------

        """
        R = data.ratings_matrix
        # dummy check for k < rank
        if self._k_principal >= min(R.shape):
            raise ValueError(f"k_principal must be <= min(R.shape), got {self._k_principal} >= {min(R.shape)}")
        # copy for coupling and mutation def
        # up to ~65k movies (2^16)
        self.rated_indices_ = (R.indices.copy()).astype(self._dtype_indices)
        self.rated_indptr_ = R.indptr.copy()
        start = time.time()
        logger.info("Started SVD decomposition")
        rng_f = np.random.default_rng(42)
        try:
            U, S, Vh = svds(R, k=self._k_principal, solver="arpack", rng=rng_f)
        except ArpackNoConvergence:
            # for ill-cond
            U, S, Vh = svds(R, k=self._k_principal, solver="lobpcg", rng=rng_f)
        logger.info(f"Finished SVD decomposition, elapsed {time.time() - start:.2f} seconds")
        self.user_means_ = data.user_means
        self.user_ids_ = data.user_ids
        self.movie_ids_ = data.movie_ids
        self.movie_id_to_idx_ = {mid: i for i, mid in enumerate(self.movie_ids_)}
        # reverse sorting in svds
        # np.ascontiguousarray forces buffer having. Without this,
        # U[:, ::-1] is a view into the original SVD output -> original alive and we point to it
        self.U_ = np.ascontiguousarray(U[:, ::-1]).astype(self._dtype_mx)
        self.S_ = S[::-1].copy()
        self.Vh_ = np.ascontiguousarray(Vh[::-1, :]).astype(self._dtype_mx)
        return self

    def recommend(self, catalog: Catalog | None, user_id: int, n: int = 10) -> list[dict]:
        """

        Parameters
        ----------
        user_id: int
            user's id to construct predict for
        n: int
            top n results to return
        catalog: Catalog
            non-ratings static data

        Returns
        -------
        List[dict]
        list of recommendations dicts, {movie_id:..., score:...}

        """
        if self.U_ is None:
            raise RuntimeError("Recommender is not fit")

        user_idx = np.where(self.user_ids_ == user_id)[0][0]

        ratings = (self.U_[user_idx] * self.S_) @ self.Vh_
        ratings = ratings + self.user_means_[user_idx]
        user_mask = self.rated_indices_[self.rated_indptr_[user_idx] : self.rated_indptr_[user_idx + 1]]
        ratings[user_mask] = -np.inf
        idxs = np.argsort(ratings)[::-1]
        final_movies = self.movie_ids_[idxs[:n]]
        final_ratings = ratings[idxs[:n]]
        return [{"movie_id": int(m), "score": s} for m, s in zip(final_movies, final_ratings) if s != -np.inf]

    def score_all(self, user_id: int) -> np.ndarray:
        """

        Parameters
        ----------
        user_id

        Returns
        -------
        method to retun all scores for given user_id
        """
        if self.U_ is None:
            raise RuntimeError("Recommender is not fit")

        idx = np.where(self.user_ids_ == user_id)[0]
        if len(idx) == 0:
            raise ValueError("Unknown user_id")
        user_pos = idx[0]  # scalar, peels the dimension
        ratings = (self.U_[user_pos] * self.S_) @ self.Vh_
        ratings = ratings + self.user_means_[user_pos]
        return ratings

    def score_all_from_ratings(self, ratings: dict) -> tuple[np.ndarray, np.ndarray]:
        """

        Parameters
        ----------
        ratings: dict
            user's ratings

        Returns
        -------
        so first R ~ UsigmaV^T, for new user r ~ u_r * sigmaV^T
        u_r = rVsigma-1, r_restored = r * V * VT,
        """
        if self.U_ is None:
            raise RuntimeError("Recommender is not fit")

        r = np.zeros(len(self.movie_ids_))
        rated_mask = np.zeros(len(self.movie_ids_), dtype=bool)

        for movie_id, rating in ratings.items():
            if movie_id in self.movie_id_to_idx_:
                i = self.movie_id_to_idx_[movie_id]
                r[i] = rating
                rated_mask[i] = True

        if not rated_mask.any():
            warnings.warn("No valid ratings provided; returning zero scores.", UserWarning, stacklevel=2)
            return np.zeros(len(self.movie_ids_)), rated_mask

        user_mean = r[rated_mask].mean()
        r_centered = np.where(rated_mask, r - user_mean, 0)

        predicted = r_centered @ self.Vh_.T @ self.Vh_
        return predicted + user_mean, rated_mask


if __name__ == "__main__":
    ...
    cfg = load_config()
    catalog, train_data = prepare_data(cfg=cfg)

    recommender = CollaborativeRecommender(
        k_principal=cfg["models"]["collaborative"]["k_principal"],
    )

    recommender.fit(catalog=catalog, data=train_data)
    recommendations = recommender.recommend(catalog=catalog, user_id=1)
    recommendations = recommender.enrich(recommendations, catalog=catalog)

    print(recommendations)
