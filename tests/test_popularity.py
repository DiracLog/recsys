"""Tests for PopularityRecommender."""

import pytest

from src.models.popularity_recommender import PopularityRecommender


class TestFit:
    def test_fit_returns_self(self, built_data_cls):
        rec = PopularityRecommender(movie_conf_alpha=2.0, movie_conf_beta=1.0)
        train, catalog = built_data_cls
        assert rec.fit(catalog=catalog, data=train) is rec

    def test_fit_populates_scores(self, popularity_recommender, built_data_cls):
        train, catalog = built_data_cls
        n_movies = len(train.movie_ids)
        assert popularity_recommender.resulting_scores_.shape == (n_movies,)
        assert popularity_recommender.movies_counts_.shape == (n_movies,)


class TestRecommend:
    def test_unfit_raises(self):
        rec = PopularityRecommender(movie_conf_alpha=2.0, movie_conf_beta=1.0)
        with pytest.raises(RuntimeError, match="Fit the model first"):
            rec.recommend(n=3, catalog=None)

    def test_user_id_is_ignored(self, popularity_recommender):
        """Popularity is user-agnostic — same output for any user."""
        recs_a = popularity_recommender.recommend(user_id=1, n=5, catalog=None)
        recs_b = popularity_recommender.recommend(user_id=999, n=5, catalog=None)
        recs_none = popularity_recommender.recommend(n=5, catalog=None)
        assert [r["movie_id"] for r in recs_a] == [r["movie_id"] for r in recs_b]
        assert [r["movie_id"] for r in recs_a] == [r["movie_id"] for r in recs_none]

    def test_respects_n(self, popularity_recommender):
        recs = popularity_recommender.recommend(n=2, catalog=None)
        assert len(recs) == 2

    def test_returns_enriched_shape(self, popularity_recommender):
        """Popularity currently returns enriched dicts directly."""
        recs = popularity_recommender.recommend(n=3, catalog=None)
        for rec in recs:
            assert {"movie_id", "score"} <= set(rec.keys())

    def test_sorted_descending(self, popularity_recommender):
        recs = popularity_recommender.recommend(n=5, catalog=None)
        scores = [r["score"] for r in recs]
        assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# Test load/save
# ---------------------------------------------------------------------------


class TestLoadSave:
    def test_save_load_preserves_recommendations(self, popularity_recommender, tmp_path):
        path = tmp_path / "model.joblib"

        original_recs = popularity_recommender.recommend(n=5, catalog=None)

        popularity_recommender.save(path)
        restored = popularity_recommender.load(path)

        restored_recs = restored.recommend(n=5, catalog=None)

        assert original_recs == restored_recs

    def test_save_load_preserves_params(self, popularity_recommender, tmp_path):
        path = tmp_path / "model.joblib"
        # treat as file not path to avoid perm
        popularity_recommender.save(path)
        restored = popularity_recommender.load(path)

        assert restored._movie_conf_alpha == popularity_recommender._movie_conf_alpha
        assert restored._movie_conf_beta == popularity_recommender._movie_conf_beta

        assert (restored.movies_counts_ == popularity_recommender.movies_counts_).all()

        assert (restored.resulting_scores_ == popularity_recommender.resulting_scores_).all()
