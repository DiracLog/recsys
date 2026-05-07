from __future__ import annotations

from abc import ABC, abstractmethod

from pathlib import Path
from typing import Self, TYPE_CHECKING

import joblib
from src.models.data_schemas import _check_version, SCHEMA_VERSION

if TYPE_CHECKING:
    from src.models.data_schemas import Catalog, TrainingData


class BaseRecommender(ABC):
    """
    Abstract class - common interface for all recommenders.
    allows saving fit recommenders and upload and use them by path
    """

    _params: tuple[str, ...] = ()
    _state: tuple[str, ...] = ()

    @abstractmethod
    def fit(self, catalog: Catalog, data: TrainingData, **kwargs) -> Self:
        """Fit recommender, self for chaining."""

    @abstractmethod
    def recommend(self, catalog: Catalog, user_id: int, n: int) -> list[dict]:
        """Return top n recommendations for a user id"""

    def save(self, path: str | Path) -> None:
        """Serialize fitted artifacts + hyperparameters to directory path."""
        joblib.dump(
            {
                "params": {k: getattr(self, f"_{k}") for k in self._params},
                "state": {k: getattr(self, k) for k in self._state},
                "version": SCHEMA_VERSION,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path, mmap_mode: str | None = None) -> Self:
        """Reconstruct a fitted recommender from directory path"""
        data = joblib.load(path, mmap_mode=mmap_mode)
        _check_version(data)
        obj = cls(**data["params"])
        for k, v in data["state"].items():
            setattr(obj, k, v)
        return obj

    @staticmethod
    def enrich(recs: list[dict], catalog: Catalog) -> list[dict]:
        for rec in recs:
            rec["title"] = catalog.title_of(rec["movie_id"])
        return recs

    @staticmethod
    def _validate_response(response: dict) -> bool:
        check_length = len(response) == 2
        check_items = set(response.keys()) == {"movie_id", "score"}
        return check_length and check_items

    def _validate_full_response(self, response: list[dict]) -> bool:
        res = True
        for rec in response:
            res = res and self._validate_response(rec)
        return res
