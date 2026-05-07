"""Shared fixtures for recommender tests.

Design notes
------------
- `sample_data` is session-scoped: the tiny DataFrames are immutable in our
  tests and rebuilding them per test is pure waste.
- Fitted recommenders are function-scoped: SVD / profile-building is cheap
  on 5x5 data, and per-test isolation protects against tests that might
  accidentally mutate fitted state.
"""

import pytest

import polars as pl

from src.data.loaders import load_data
from src.data.preprocessor import Preprocessor
from src.models.collaborative_recommender import CollaborativeRecommender
from src.models.content_recommender import ContentBasedRecommender
from src.models.data_schemas import TrainingData, Catalog
from src.models.hybrid_recommender import HybridRecommender
from src.models.popularity_recommender import PopularityRecommender


# ---------------------------------------------------------------------------
# Data fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def model_path(tmp_path):
    return tmp_path / "model.joblib"


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
def built_data(synthetic_dat_files) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    return load_data(
        path_m=synthetic_dat_files["movies_path"],
        path_r=synthetic_dat_files["ratings_path"],
        path_u=synthetic_dat_files["users_path"],
    )


@pytest.fixture
def preprocessor(synthetic_dat_files) -> Preprocessor:
    return Preprocessor(
        m_threshold=3,
        u_threshold=3,
    )


@pytest.fixture
def built_data_cls(preprocessor, built_data) -> tuple[TrainingData, Catalog]:
    movies, ratings, _ = built_data
    train, catalog = preprocessor.preprocess(movies_df=movies, ratings_df=ratings)
    return train, catalog


# ----------------------- Recommender fixtures


@pytest.fixture
def popularity_recommender(built_data_cls) -> PopularityRecommender:
    train, catalog = built_data_cls
    return PopularityRecommender(movie_conf_alpha=2.0, movie_conf_beta=1.0).fit(catalog=catalog, data=train)


@pytest.fixture
def content_recommender(built_data_cls) -> ContentBasedRecommender:
    train, catalog = built_data_cls
    return ContentBasedRecommender(
        profile_alpha=2.0,
        movie_conf_alpha=2.0,
        movie_conf_beta=1.0,
        shrinkage="bayesian",
        movies_conf=True,
    ).fit(catalog=catalog, data=train)


@pytest.fixture
def content_no_shrinkage(built_data_cls) -> ContentBasedRecommender:
    train, catalog = built_data_cls
    return ContentBasedRecommender(
        profile_alpha=2.0,
        movie_conf_alpha=2.0,
        movie_conf_beta=1.0,
        shrinkage="none",
        movies_conf=False,
    ).fit(catalog=catalog, data=train)


@pytest.fixture
def collab_recommender(built_data_cls) -> CollaborativeRecommender:
    train, catalog = built_data_cls
    return CollaborativeRecommender(k_principal=3).fit(catalog=catalog, data=train)


@pytest.fixture
def hybrid_recommender(built_data_cls) -> HybridRecommender:
    train, catalog = built_data_cls
    content = ContentBasedRecommender(
        profile_alpha=2.0,
        movie_conf_alpha=2.0,
        movie_conf_beta=1.0,
        shrinkage="bayesian",
        movies_conf=True,
    )
    collab = CollaborativeRecommender(k_principal=3)
    return HybridRecommender(content=content, collaborative=collab, weight=0.5).fit(data=train, catalog=catalog)
