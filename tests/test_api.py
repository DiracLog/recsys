"""
Tests for FastAPI serving endpoints.
Run with: pytest .\tests\test_api.py -v-v

Requires:
  - pip install httpx
  - MODEL_RUN_ID env variable set to a valid run_id with saved artifacts
  - Artifacts accessible at the expected path
"""

import numpy as np
import polars as pl
import pytest
from fastapi.testclient import TestClient


from src.api.app import app
from src.api.schemas import MovieResult
from src.models.data_schemas import Catalog


class FakeHybrid:
    is_fitted = True

    def recommend_from_ratings(self, ratings, n=10, catalog=None):
        return [{"movie_id": i, "score": 1.0 / i} for i in range(1, n + 1)]

    def enrich(self, recs, catalog):
        for r in recs:
            r["title"] = catalog.title_of(r["movie_id"]) or "unknown"
        return recs


pop_precomputed = [
    {"movie_id": i, "title": f"Movie {i}", "score": 1.0 - i * 0.01}
    for i in range(100)  # 100 items so slicing has room
]


class FakeMovieSearch:
    def search(self, q: str, limit=10):
        print(q)
        res = [MovieResult(movie_id=1 * i, score=1.0 - i * 0.01, title="X" * i) for i in range(limit)]
        return res


catalog = Catalog(
    movies_df=pl.DataFrame(
        {
            "movie_id": [1, 2, 3, 4, 5],
            "name": [
                "Alpha Rising",
                "Beta Continuum",
                "Gamma Protocol",
                "Delta Hours",
                "Epsilon Road",
            ],
            "genres": ["Action", "Action|Drama", "Drama", "Drama|Comedy", "Comedy"],
        }
    ),
    genre_matrix=np.array(
        [
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 1, 1],
            [0, 0, 1],
        ],
        dtype=np.float64,
    ),
)


@pytest.fixture
def client():
    app.state.pop_precomputed = pop_precomputed
    app.state.hybrid = FakeHybrid()
    app.state.pop_precomputed = pop_precomputed
    app.state.movie_search = FakeMovieSearch()
    app.state.catalog = catalog
    return TestClient(app)


# ============================================================
# 1. Health endpoint
# ============================================================


class TestHealth:
    def test_health_returns_200(self, client):
        resp = client.get("/health")
        print(resp.json())
        assert resp.status_code == 200

    def test_health_status_healthy(self, client):
        resp = client.get("/healthcheck")
        assert resp.json()["status"] == "healthy"

    def test_health(self, client):
        resp = client.get("/health")
        assert resp.json()["status"] == "ok"


# ============================================================
# 2. Model info endpoint
# ============================================================


class TestPopular:
    def test_popular_default(self, client):
        resp = client.get("/popular")
        assert len(resp.json()["results"]) == 10

    def test_popular_not_default(self, client):
        n = 12
        resp = client.get("/popular", params={"n": 12})  #
        assert len(resp.json()["results"]) == n


class TestRecommend:
    def test_happy_path(self, client):
        resp = client.post("/recommend", json={"ratings": {10: 5.0, 20: 4.0}, "n": 5})
        assert resp.status_code == 200
        assert len(resp.json()["results"]) == 5

    def test_empty_falls_back_to_popular(self, client):
        resp = client.post("/recommend", json={"ratings": {}, "n": 3})
        assert resp.status_code == 200
        assert len(resp.json()["results"]) != 0

    def test_error_handling(self, client):
        resp = client.post("/recommend", json={"ratings": {10: 6.0, 20: 4.0}, "n": 5})
        assert resp.status_code == 422


class TestSearch:
    def test_happy_path(self, client):
        resp = client.get("/search", params={"q": "a" * 20, "limit": 3})
        print(resp.json())
        assert resp.status_code == 200
        assert len(resp.json()["results"]) == 3

    def test_error_limit(self, client):
        resp = client.get("/search", params={"q": "a" * 20, "limit": 15})
        assert resp.status_code == 422

    def test_error_query(self, client):
        resp = client.get("/search", params={"q": "a" * 120, "limit": 5})
        assert resp.status_code == 422

    def test_empty_query(self, client):
        resp = client.get("/search", params={"q": "", "limit": 5})
        assert resp.status_code == 422
