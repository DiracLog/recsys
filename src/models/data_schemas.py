from __future__ import annotations
from dataclasses import dataclass, field

import json
from pathlib import Path

import numpy as np
import polars as pl
from scipy.sparse import csr_matrix

SCHEMA_VERSION = 1


def _check_version(meta: dict) -> None:
    """Raise if artifact version doesn't match current code version."""
    v = meta.get("version")
    if v != SCHEMA_VERSION:
        raise RuntimeError(f"Artifact version {v!r} does not match code version {SCHEMA_VERSION}. Retrain required.")


@dataclass
class TrainingData:
    """Training-time data — consumed by fit(), not needed at inference.

    Produced fresh each retrain cycle. NOT serialized with the fitted model.

    Parameters
    ----------
    ratings_matrix : pd.DataFrame
        Sparse ratings matrix
    user_ids : np.ndarray
        positional index keys after fltrd
    movie_ids : np.ndarray
        positional index keys after fltrd
    user_means : np.ndarray
        Per-user mean rating (for de-centering)
    ratings_df : pl.DataFrame
        Raw ratings in long form — columns: user_id, movie_id, rating.
    """

    ratings_matrix: csr_matrix
    user_ids: np.ndarray
    movie_ids: np.ndarray
    user_means: np.ndarray  # for de-centering predictions back
    ratings_df: pl.DataFrame  # raw ratings (long format)


@dataclass
class Catalog:
    """Static reference data — movies + genres.
    Shared across all recommenders. Built once during preprocessing

    Parameters
    ----------
    movies_df : pl.DataFrame
        Columns: movie_id, name, genres
    genre_matrix : np.ndarray
        Shape (n_movies, n_genres), multi-hot encoding. Row i maps to movie_ids[i].
    movie_ids : np.ndarray
        Ordered movie_ids defining the row order of `genre_matrix`.
    """

    movies_df: pl.DataFrame
    genre_matrix: np.ndarray  # multi-hot genres
    movie_ids: np.ndarray = None

    _movie_indx: dict[int, int] = field(init=False, repr=False)
    _title_lookup: dict[int, str] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.movie_ids is None:
            self.movie_ids = np.unique(self.movies_df["movie_id"].to_numpy())
        self._movie_indx = {int(k): v for v, k in enumerate(self.movie_ids)}  # movie_id -> genre matrix_movie_id
        self._title_lookup = {k: v for k, v in zip(self.movies_df["movie_id"].cast(pl.Int64), self.movies_df["name"])}

    # ----- accesors ----

    def row_of(self, movie_id: int) -> int | None:
        """Row index of movie_id in genre matrix, none if unknown"""
        return self._movie_indx.get(movie_id, None)

    def title_of(self, movie_id: int) -> str | None:
        """Name of movie by movie_id, none if unknown"""
        return self._title_lookup.get(movie_id, None)

    # ---- persistence ----

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        self.movies_df.write_parquet(path / "movies.parquet", compression="zstd")
        np.savez(
            path / "arrays.npz",
            genre_matrix=self.genre_matrix,
            movie_ids=self.movie_ids,
        )
        (path / "meta.json").write_text(
            json.dumps(
                {
                    "version": SCHEMA_VERSION,
                    "n_movies": int(len(self.movie_ids)),
                    "n_genres": int(self.genre_matrix.shape[1]),
                },
                indent=2,
            )
        )

    @classmethod
    def load(cls, path: str | Path) -> Catalog:
        path = Path(path)
        meta = json.loads((path / "meta.json").read_text())
        _check_version(meta)

        movies_df = pl.read_parquet(path / "movies.parquet")
        arrays = np.load(path / "arrays.npz", allow_pickle=False)
        return cls(
            movies_df=movies_df,
            genre_matrix=arrays["genre_matrix"],
            movie_ids=arrays["movie_ids"],
        )
