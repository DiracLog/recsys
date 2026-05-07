from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np

from src.config.config_loader import load_config
from src.utils.loading_utils import prepare_data
from src.models.abstract import BaseRecommender
from src.models.collaborative_recommender import CollaborativeRecommender
from src.models.content_recommender import ContentBasedRecommender
from src.models.data_schemas import TrainingData, Catalog, SCHEMA_VERSION, _check_version


def normalize_data(data):
    dmin = np.min(data)
    dmax = np.max(data)
    if dmax == dmin:
        return np.zeros_like(data)
    return (data - dmin) / (dmax - dmin)


class HybridRecommender(BaseRecommender):
    """
    Combined recommender based on weighted sum of collaborative and content
    """

    _params = ("weight",)

    def __init__(self, content: ContentBasedRecommender, collaborative: CollaborativeRecommender, weight: float = 0.5):
        self._content = content
        self._collaborative = collaborative
        self._weight = weight

    def fit(self, catalog: Catalog, data: TrainingData, **kwargs) -> HybridRecommender:
        self._content.fit(catalog=catalog, data=data)
        self._collaborative.fit(catalog=catalog, data=data)

        if not np.array_equal(self._content.movie_ids_, self._collaborative.movie_ids_):
            raise RuntimeError("Content and collaborative have inconsistent movie_ids")

        if not np.array_equal(self._content.user_ids_, self._collaborative.user_ids_):
            raise RuntimeError("Content and collaborative have inconsistent user_ids")

        return self

    def recommend(self, catalog: Catalog, user_id: int, n: int = 10) -> list[dict]:
        """

        Parameters
        ----------
        user_id: int
            user's id to constuct predict for
        n:
            top n results to return
        catalog: Catalog
            non rating static data

        Returns
        -------
        dict with movie_id and score for top n recommendations
        """

        content_scores = self._content.score_all(user_id=user_id, catalog=catalog)
        content_norm = normalize_data(content_scores)

        if user_id in self._collaborative.user_ids_:
            collab_scores = self._collaborative.score_all(user_id)
            collab_norm = normalize_data(collab_scores)
            final = self._weight * collab_norm + (1 - self._weight) * content_norm
        else:
            final = content_norm

        # mask for rated movies
        idx = np.where(self._content.user_ids_ == user_id)[0]
        if len(idx) == 0:
            raise ValueError("Unknown user_id")

        user_pos = idx[0]
        mask = self._content.rated_indices_[
            self._content.rated_indptr_[user_pos] : self._content.rated_indptr_[user_pos + 1]
        ]
        final[mask] = -np.inf

        top_idx = np.argsort(final)[::-1][:n]
        movie_ids = self._content.movie_ids_

        return [{"movie_id": int(movie_ids[i]), "score": float(final[i])} for i in top_idx if final[i] != -np.inf]

    def recommend_from_ratings(self, catalog: Catalog, ratings: dict, n: int = 10) -> list[dict]:
        """

        Parameters
        ----------
        ratings: dict
            user's ratings
        n: int
            number of recommendations to return
        catalog: Catalog
            non rating static data

        Returns
        -------
        list of dict with movie_id and score for top n recommendations
        """
        if not ratings:
            raise ValueError("ratings must not be empty")
        content_scores, rated_mask = self._content.score_all_from_ratings(ratings=ratings, catalog=catalog)
        content_norm = normalize_data(content_scores)

        collab_scores, _ = self._collaborative.score_all_from_ratings(ratings)
        collab_norm = normalize_data(collab_scores)

        final = self._weight * collab_norm + (1 - self._weight) * content_norm
        final = np.asarray(final)
        final[rated_mask] = -np.inf

        top_idx = np.argsort(final)[::-1][:n]
        movie_ids = self._content.movie_ids_
        return [{"movie_id": int(movie_ids[i]), "score": float(final[i])} for i in top_idx if final[i] != -np.inf]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "version": SCHEMA_VERSION,
                "params": {k: getattr(self, f"_{k}") for k in self._params},
                "state": {k: getattr(self, k) for k in self._state},
            },
            path / "model.joblib",
        )
        self._content.save(path / "content")
        self._collaborative.save(path / "collaborative")

    @classmethod
    def load(cls, path: str | Path, mmap_mode: str | None = None) -> HybridRecommender:
        path = Path(path)
        # pass mmap_mode to submodels to avoid double loading into RAM,
        data = joblib.load(path / "model.joblib", mmap_mode=mmap_mode)
        _check_version(data)
        content = ContentBasedRecommender.load(path / "content", mmap_mode=mmap_mode)
        collaborative = CollaborativeRecommender.load(path / "collaborative", mmap_mode=mmap_mode)
        # point to the same, avoid double copying to save RAM
        # no longer works with mmap, kept for consistency with non-mmap loading
        collaborative.rated_indices_ = content.rated_indices_
        obj = cls(content=content, collaborative=collaborative, **data["params"])
        return obj

    @property
    def is_fitted(self):
        return self._content.user_profiles_ is not None and self._collaborative.U_ is not None


if __name__ == "__main__":
    cfg = load_config()
    catalog, train_data = prepare_data(cfg=cfg)
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

    recommender = HybridRecommender(weight=0.5, content=cntnt_recommender, collaborative=clb_recommender)
    recommender.fit(catalog, train_data)
    rec = recommender.recommend(catalog=catalog, user_id=1, n=5)
    print(recommender.enrich(recs=rec, catalog=catalog))
