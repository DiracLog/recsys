import numpy as np
from scipy.sparse import csr_matrix

from src.config.config_loader import load_config
from src.data.loaders import load_data_ml32
from src.models.data_schemas import TrainingData, Catalog
from src.utils.utils import construct_path

import polars as pl


class Preprocessor:
    def __init__(
        self,
        m_threshold: int = 10,
        u_threshold: int = 20,
    ):
        """
        Initializes the Preprocessor.

        Parameters
        ----------
        m_threshold : int, optional
            threshold for minimum number of ratings per movie, by default 10
        u_threshold : int, optional
            threshold for minimum number of ratings per user, by default 20
        """
        self._m_threshold = m_threshold
        self._u_threshold = u_threshold
        self._user_means = None

    def encode_genres(self, movie_data: pl.DataFrame) -> tuple[pl.DataFrame, np.ndarray]:
        movie_data = movie_data.with_columns(
            pl.col("genres").str.split("|").list.eval(pl.element().str.strip_chars()).alias("genres_lst")
        )
        from src.utils.preprocessing_utils import multilabel_binarize

        encoded, _ = multilabel_binarize(movie_data["genres_lst"].to_list())
        return movie_data, encoded

    def preprocess(self, movies_df: pl.DataFrame, ratings_df: pl.DataFrame) -> tuple[TrainingData, Catalog]:
        """
        loads data, filters by min_thresholds, constrcut sparse user-item matrix

        Parameters
        ----------
        movies_df : pl.DataFrame
            movies dataframe
        ratings_df : pl.DataFrame
            ratings dataframe

        Returns
        -------
        tuple[TrainingData, Catalog]
            returns TrainingData and Catalog objects
        """
        valid_users = (
            ratings_df.group_by("user_id")
            .agg(pl.col("rating").count().alias("count"))
            .filter(pl.col("count") > self._u_threshold)
            .get_column("user_id")
            .to_numpy()
        )

        valid_movies = (
            ratings_df.group_by("movie_id")
            .agg(pl.col("rating").count().alias("count"))
            .filter(pl.col("count") > self._m_threshold)
            .get_column("movie_id")
            .to_numpy()
        )

        filtered_movies = movies_df.filter(pl.col("movie_id").is_in(valid_movies))

        filtered_ratings = ratings_df.filter(
            (pl.col("movie_id").is_in(valid_movies)) & (pl.col("user_id").is_in(valid_users))
        )

        filtered_movies = filtered_movies.sort("movie_id")
        _, encoded = self.encode_genres(filtered_movies)

        movie_ids = filtered_ratings["movie_id"].unique().sort().to_numpy()
        user_ids = filtered_ratings["user_id"].unique().sort().to_numpy()
        movie_idx = np.searchsorted(movie_ids, filtered_ratings["movie_id"].to_numpy())
        user_idx = np.searchsorted(user_ids, filtered_ratings["user_id"].to_numpy())
        raw = csr_matrix(
            (filtered_ratings["rating"].to_numpy(writable=True), (user_idx, movie_idx)),
            shape=(len(user_ids), len(movie_ids)),
        )

        sums = raw.sum(axis=1).A1
        counts = raw.getnnz(axis=1)

        self._user_means = sums / counts

        raw = raw.tocoo()
        centered_data = raw.data - self._user_means[raw.row]

        ratings_matrix = csr_matrix((centered_data, (raw.row, raw.col)), shape=raw.shape)

        train_data = TrainingData(
            ratings_matrix=ratings_matrix,
            user_ids=user_ids,
            movie_ids=movie_ids,
            ratings_df=filtered_ratings,
            user_means=self._user_means,
        )
        catalog = Catalog(movies_df=filtered_movies, genre_matrix=encoded)
        return train_data, catalog


if __name__ == "__main__":
    cfg = load_config()
    path_m = construct_path(cfg["data"]["movies_path"])
    path_r = construct_path(cfg["data"]["ratings_path"])

    movies_data, ratings_data, users_data = load_data_ml32(path_m, path_r)

    pr = Preprocessor()
    pr.preprocess(movies_df=movies_data, ratings_df=ratings_data)
    train_data, catalog = pr.preprocess(movies_df=movies_data, ratings_df=ratings_data)
    print(train_data.__annotations__)
    print(train_data.ratings_df.head())
    assert isinstance(train_data, TrainingData)
