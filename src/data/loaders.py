from pathlib import Path

import polars as pl
import io


# SCHEMA 1
CANONICAL_MOVIES_COLS = ["movie_id", "name", "genres"]
CANONICAL_RATING_COLS = ["user_id", "movie_id", "rating", "timestamp"]
CANONICAL_USER_COLS = ["user_id", "gender", "age", "occupation", "zip_code"]

# ML32_MOVIES_COLS = ["movieId", "title", "genres"]
# ML32_RATING_COLS = ["userId", "movieId", "rating", "timestamp"]


def assert_schema_movies(df: pl.DataFrame):
    missing = set(CANONICAL_MOVIES_COLS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {list(missing)[:3]} ...")
    return True


def assert_schema_ratings(df: pl.DataFrame):
    missing = set(CANONICAL_RATING_COLS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {list(missing)[:3]} ...")
    return True


def assert_schema_users(df: pl.DataFrame):
    missing = set(CANONICAL_USER_COLS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {list(missing)[:3]} ...")
    return True


def load_movielens_movies(path: Path) -> pl.DataFrame:
    raw = path.read_bytes().replace(b"::", b"\t")
    return pl.read_csv(
        io.BytesIO(raw), separator="\t", has_header=False, new_columns=CANONICAL_MOVIES_COLS, encoding="latin1"
    )


def load_movielens_ratings(path: Path) -> pl.DataFrame:
    raw = path.read_bytes().replace(b"::", b"\t")
    return pl.read_csv(
        io.BytesIO(raw), separator="\t", has_header=False, new_columns=CANONICAL_RATING_COLS, encoding="latin1"
    )


def load_movielens_users(path: Path) -> pl.DataFrame:
    raw = path.read_bytes().replace(b"::", b"\t")
    return pl.read_csv(
        io.BytesIO(raw),
        separator="\t",
        new_columns=CANONICAL_USER_COLS,
        has_header=False,
        encoding="latin1",
        schema_overrides={"column_5": pl.Utf8},
    )


def load_ml32_movies(path: Path) -> pl.DataFrame:
    data = pl.read_csv(path, encoding="latin1", new_columns=CANONICAL_MOVIES_COLS)
    return data[CANONICAL_MOVIES_COLS]


def load_ml32_ratings(path: Path) -> pl.DataFrame:
    data = pl.read_csv(path, encoding="latin1", new_columns=CANONICAL_RATING_COLS)
    return data[CANONICAL_RATING_COLS]


def load_data(path_m, path_r, path_u) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    movies_data = load_movielens_movies(path_m)
    ratings_data = load_movielens_ratings(path_r)
    users_data = load_movielens_users(path_u)

    assert assert_schema_movies(movies_data)
    assert assert_schema_ratings(ratings_data)
    assert assert_schema_users(users_data)
    return movies_data, ratings_data, users_data


def load_data_ml32(path_m, path_r) -> tuple[pl.DataFrame, pl.DataFrame, None]:
    movies_data = load_ml32_movies(path_m)
    ratings_data = load_ml32_ratings(path_r)

    # schema checks
    assert assert_schema_movies(movies_data)
    assert assert_schema_ratings(ratings_data)
    return movies_data, ratings_data, None


if __name__ == "__main__":
    path_m = Path("data/ml-32m/movies.csv")
    path_r = Path("data/ml-32m/ratings.csv")
    # test 32M
    movies, ratings, _ = load_data_ml32(path_m, path_r)
    print(movies.head())
    print(ratings.head())
    # test 1M
    movies, ratings, users = load_data(
        Path("data/ml-1m/movies.dat"), Path("data/ml-1m/ratings.dat"), Path("data/ml-1m/users.dat")
    )
    print(movies.head())
    print(ratings.head())
    print(users.head())
