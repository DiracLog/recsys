"""Tests for Preprocessor.

Strategy
--------
Since build_recommender_data() loads from paths internally, each test writes
synthetic .dat files to tmp_path (pytest-managed temp dir) and points the
Preprocessor at them. No real MovieLens data required.
"""

import numpy as np
import polars as pl
import pytest
import scipy

from src.data.loaders import load_data
from src.data.preprocessor import Preprocessor
from src.models.data_schemas import TrainingData, Catalog


# ---------------------------------------------------------------------------
# File-writing fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_dat_files(tmp_path):
    """Write three .dat files in MovieLens format (:: delimited).

    Design
    ------
    - 6 users, 6 movies
    - User 1: 4 ratings (should survive u_threshold=3)
    - User 2: 5 ratings (survives)
    - User 3: 5 ratings (survives)
    - User 4: 2 ratings (filtered out with u_threshold=3)
    - User 5: 4 ratings (survives)
    - User 6: 4 ratings (survives)
    - Movies 10, 20, 30, 40: many ratings (survive m_threshold=3)
    - Movie 50: 2 ratings (filtered)
    - Movie 60: 1 rating (filtered)
    """
    ratings_path = tmp_path / "ratings.dat"
    movies_path = tmp_path / "movies.dat"
    users_path = tmp_path / "users.dat"

    # Ratings: user_id::movie_id::rating::timestamp
    ratings_rows = [
        # user 1
        (1, 10, 5, 1000),
        (1, 20, 4, 1000),
        (1, 30, 3, 1000),
        (1, 40, 2, 1000),
        # user 2
        (2, 10, 4, 1000),
        (2, 20, 5, 1000),
        (2, 30, 4, 1000),
        (2, 40, 3, 1000),
        (2, 50, 2, 1000),
        # user 3
        (3, 10, 3, 1000),
        (3, 20, 4, 1000),
        (3, 30, 5, 1000),
        (3, 40, 4, 1000),
        (3, 50, 3, 1000),
        # user 4 — only 2 ratings, filtered
        (4, 10, 2, 1000),
        (4, 60, 1, 1000),
        # user 5
        (5, 10, 5, 1000),
        (5, 20, 4, 1000),
        (5, 30, 3, 1000),
        (5, 40, 2, 1000),
        # user 6
        (6, 10, 4, 1000),
        (6, 20, 5, 1000),
        (6, 30, 4, 1000),
        (6, 40, 3, 1000),
    ]
    ratings_path.write_text("\n".join("::".join(str(x) for x in row) for row in ratings_rows))

    # Movies: movie_id::name::genres
    movies_rows = [
        (10, "Alpha", "Action|Comedy"),
        (20, "Beta", "Action"),
        (30, "Gamma", "Drama"),
        (40, "Delta", "Drama|Comedy"),
        (50, "Epsilon", "Comedy"),
        (60, "Zeta", "Action|Drama"),
    ]
    movies_path.write_text("\n".join("::".join(str(x) for x in row) for row in movies_rows))

    # Users: user_id::gender::age::occupation::zip
    users_rows = [
        (1, "M", 25, 4, "00000"),
        (2, "F", 30, 7, "11111"),
        (3, "M", 35, 1, "22222"),
        (4, "F", 40, 2, "33333"),
        (5, "M", 45, 3, "44444"),
        (6, "F", 50, 5, "55555"),
    ]
    users_path.write_text("\n".join("::".join(str(x) for x in row) for row in users_rows))

    return {
        "ratings_path": ratings_path,
        "movies_path": movies_path,
        "users_path": users_path,
    }


@pytest.fixture
def preprocessor(synthetic_dat_files) -> Preprocessor:
    return Preprocessor(
        m_threshold=3,
        u_threshold=3,
    )


@pytest.fixture
def built_data(synthetic_dat_files) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    return load_data(
        path_m=synthetic_dat_files["movies_path"],
        path_r=synthetic_dat_files["ratings_path"],
        path_u=synthetic_dat_files["users_path"],
    )


@pytest.fixture
def built_data_cls(preprocessor, built_data) -> tuple[TrainingData, Catalog]:
    movies, ratings, users = built_data
    train, catalog = preprocessor.preprocess(movies_df=movies, ratings_df=ratings)
    return train, catalog


# ---------------------------------------------------------------------------
# build_recommender_data — end-to-end shape / contract
# ---------------------------------------------------------------------------


class TestSchema:
    def test_types(self, built_data_cls):
        train, catalog = built_data_cls

        assert isinstance(train.ratings_matrix, scipy.sparse.csr_matrix)
        assert isinstance(train.user_ids, np.ndarray)
        assert isinstance(train.movie_ids, np.ndarray)
        assert isinstance(train.user_means, np.ndarray)
        assert isinstance(train.ratings_df, pl.DataFrame)

    def test_shapes_consistency(self, built_data_cls):
        train, _ = built_data_cls

        n_users, n_movies = train.ratings_matrix.shape

        assert n_users == len(train.user_ids)
        assert n_movies == len(train.movie_ids)
        assert len(train.user_means) == n_users


class TestPreprocessAndBuild:
    def test_returns_recommender_data(self, built_data_cls):
        train, catalog = built_data_cls
        assert isinstance(train, TrainingData)
        assert isinstance(catalog, Catalog)

    def test_rating_matrix_is_sparse(self, built_data_cls):
        train, _ = built_data_cls
        assert isinstance(train.ratings_matrix, scipy.sparse.csr_matrix)

    def test_user_means_is_series(self, built_data_cls):
        train, _ = built_data_cls
        assert isinstance(train.user_means, np.ndarray)

    def test_genre_matrix_is_ndarray(self, built_data_cls):
        _, catalog = built_data_cls
        assert isinstance(catalog.genre_matrix, np.ndarray)


# ---------------------------------------------------------------------------
# Filtering correctness
# ---------------------------------------------------------------------------


class TestFiltering:
    def test_user_below_threshold_filtered(self, built_data_cls):
        train, _ = built_data_cls
        assert 4 not in train.user_ids

    def test_users_above_threshold_survive(self, built_data_cls):
        train, _ = built_data_cls
        for uid in [1, 2, 3, 5, 6]:
            assert uid in train.user_ids

    def test_movie_below_threshold_filtered(self, built_data_cls):
        train, _ = built_data_cls
        assert 60 not in train.movie_ids

    def test_movies_above_threshold_survive(self, built_data_cls):
        train, _ = built_data_cls
        for mid in [10, 20, 30, 40]:
            assert mid in train.movie_ids

    def test_catalog_alignment(self, built_data_cls):
        train, catalog = built_data_cls
        assert np.array_equal(catalog.movie_ids, train.movie_ids)

    def test_ratings_df_consistency(self, built_data_cls):
        train, _ = built_data_cls

        assert set(train.ratings_df["user_id"]) <= set(train.user_ids)
        assert set(train.ratings_df["movie_id"]) <= set(train.movie_ids)


# ---------------------------------------------------------------------------
# Dimension alignment
# ---------------------------------------------------------------------------


class TestDimensions:
    def test_matrix_shape_expected(self, built_data_cls):
        train, _ = built_data_cls

        # 5 users × 4 movies
        assert train.ratings_matrix.shape == (5, 4)

    def test_genre_matrix_alignment(self, built_data_cls):
        train, catalog = built_data_cls

        assert catalog.genre_matrix.shape[0] == len(train.movie_ids)


# ---------------------------------------------------------------------------
# Centering
# ---------------------------------------------------------------------------


class TestCentering:
    def test_rows_sum_to_zero_on_nonzero_entries(self, built_data_cls):
        train, _ = built_data_cls

        mat = train.ratings_matrix.tocsr()

        for i in range(mat.shape[0]):
            row = mat.getrow(i)
            if row.nnz == 0:
                continue

            assert abs(row.data.sum()) < 1e-9

    def test_user_means_range(self, built_data_cls):
        train, _ = built_data_cls

        assert np.all(train.user_means > 0)
        assert np.all(train.user_means <= 5)


# ---------------------------------------------------------------------------
# Genre encoding
# ---------------------------------------------------------------------------
class TestInvariants:
    def test_no_empty_users(self, built_data_cls):
        train, _ = built_data_cls

        mat = train.ratings_matrix.tocsr()

        for i in range(mat.shape[0]):
            assert mat.getrow(i).nnz > 0

    def test_no_empty_movies(self, built_data_cls):
        train, _ = built_data_cls

        mat = train.ratings_matrix.tocsc()

        for j in range(mat.shape[1]):
            assert mat.getcol(j).nnz > 0


class TestGenreEncoding:
    def test_genre_matrix_is_binary(self, built_data_cls):
        _, catalog = built_data_cls
        assert set(np.unique(catalog.genre_matrix)) <= {0, 1}

    def test_genre_matrix_has_expected_genres(self, built_data_cls):
        """Surviving movies have: Action, Comedy, Drama → 3 genres."""
        _, catalog = built_data_cls
        print(111111)
        print(catalog.genre_matrix)
        assert catalog.genre_matrix.shape[1] == 3


class TestRoundTrip:
    def test_centering_matches_manual(self, built_data_cls):
        train, _ = built_data_cls

        df = train.ratings_df.clone()

        # manual centering
        user_means = df.group_by("user_id").agg(pl.col("rating").mean().alias("user_mean"))

        df = df.join(user_means, on="user_id", how="left")
        df = df.join(user_means, on="user_id", how="left")

        df = df.with_columns((pl.col("rating") - pl.col("user_mean")).alias("centered"))

        # map ids → indices
        user_pos = {u: i for i, u in enumerate(train.user_ids)}
        movie_pos = {m: j for j, m in enumerate(train.movie_ids)}

        mat = train.ratings_matrix.tocsr()

        expected = pl.Series(
            "expected", [mat[user_pos[r["user_id"]], movie_pos[r["movie_id"]]] for r in df.iter_rows(named=True)]
        )
        result = df.select(pl.col("centered").is_close(expected, abs_tol=1e-9))
        assert result["centered"].all()
