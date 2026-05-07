# tests/test_loader.py
from pathlib import Path

import polars as pl
import pytest

from src.data.loaders import load_data, assert_schema_movies, assert_schema_ratings, assert_schema_users


class TestLoadData:
    def test_invalid_path_raises(self):
        with pytest.raises(FileNotFoundError):
            load_data(path_r=Path("nonexistent.dat"), path_u=Path("nonexistent.dat"), path_m=Path("nonexistent.dat"))

    def test_happy_load(self, built_data):
        movies_data, ratings_data, users_data = built_data
        assert len(movies_data) != 0
        assert len(ratings_data) != 0
        assert len(users_data) != 0

    def test_types(self, built_data):
        movies_data, ratings_data, users_data = built_data
        assert isinstance(movies_data, pl.DataFrame)
        assert isinstance(ratings_data, pl.DataFrame)
        assert isinstance(users_data, pl.DataFrame)

    def test_schemas(self, built_data):
        movies_data, ratings_data, users_data = built_data
        assert assert_schema_movies(movies_data)
        assert assert_schema_ratings(ratings_data)
        assert assert_schema_users(users_data)
