"""Tests for ContentBasedRecommender."""

import numpy as np
import pytest

from src.models.content_recommender import ContentBasedRecommender


# ---------------------------------------------------------------------------
# Construction & validation
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_invalid_shrinkage_raises(self):
        with pytest.raises(ValueError, match="shrinkage"):
            ContentBasedRecommender(
                profile_alpha=2.0,
                movie_conf_alpha=2.0,
                movie_conf_beta=1.0,
                shrinkage="bayessian",  # typo
                movies_conf=True,
            )

    def test_valid_shrinkage_values(self):
        for mode in ["bayesian", "none"]:
            rec = ContentBasedRecommender(
                profile_alpha=2.0,
                movie_conf_alpha=2.0,
                movie_conf_beta=1.0,
                shrinkage=mode,
                movies_conf=False,
            )
            assert rec._shrinkage == mode


# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------


class TestFit:
    def test_fit_returns_self(self, built_data_cls):
        train, catalog = built_data_cls
        rec = ContentBasedRecommender(
            profile_alpha=2.0,
            movie_conf_alpha=2.0,
            movie_conf_beta=1.0,
            shrinkage="none",
            movies_conf=False,
        )
        assert rec.fit(catalog=catalog, data=train) is rec

    def test_fit_populates_artifacts(self, content_recommender, built_data_cls):
        train, catalog = built_data_cls
        n_users = len(train.user_ids)
        n_movies = len(train.movie_ids)
        n_genres = catalog.genre_matrix.shape[1]

        assert content_recommender.user_profiles_.shape == (n_users, n_genres)
        assert content_recommender.user_ids_.shape == (n_users,)
        assert content_recommender.movie_ids_.shape == (n_movies,)

    def test_fit_builds_movie_conf_when_enabled(self, content_recommender, built_data_cls):
        train, catalog = built_data_cls
        assert content_recommender._movie_conf is not None
        assert content_recommender.movie_conf_vect_.shape == (train.ratings_matrix.shape[1],)

    def test_movie_conf_toggle_stored(self, built_data_cls):
        train, catalog = built_data_cls
        rec_on = ContentBasedRecommender(
            profile_alpha=2.0,
            movie_conf_alpha=2.0,
            movie_conf_beta=1.0,
            shrinkage="none",
            movies_conf=True,
        ).fit(catalog=catalog, data=train)
        rec_off = ContentBasedRecommender(
            profile_alpha=2.0,
            movie_conf_alpha=2.0,
            movie_conf_beta=1.0,
            shrinkage="none",
            movies_conf=False,
        ).fit(catalog=catalog, data=train)
        assert rec_on._movie_conf is True
        assert rec_off._movie_conf is False


# ---------------------------------------------------------------------------
# Recommend
# ---------------------------------------------------------------------------


class TestRecommend:
    def test_unfit_raises(self, built_data_cls):
        train, catalog = built_data_cls
        rec = ContentBasedRecommender(
            profile_alpha=2.0,
            movie_conf_alpha=2.0,
            movie_conf_beta=1.0,
            shrinkage="none",
            movies_conf=False,
        )
        with pytest.raises(RuntimeError, match="not fit"):
            rec.recommend(user_id=1, n=3, catalog=catalog)

    def test_unknown_user_raises(self, content_recommender, built_data_cls):
        train, catalog = built_data_cls
        with pytest.raises(ValueError, match="Unknown user_id"):
            content_recommender.recommend(user_id=999, n=3, catalog=catalog)

    def test_returns_correct_shape(self, content_recommender, built_data_cls):
        train, catalog = built_data_cls
        recs = content_recommender.recommend(user_id=1, n=3, catalog=catalog)
        assert len(recs) <= 3
        for rec in recs:
            assert set(rec.keys()) >= {"movie_id", "score"}
            assert isinstance(rec["movie_id"], int)

    def test_respects_n(self, content_recommender, built_data_cls):
        train, catalog = built_data_cls
        recs = content_recommender.recommend(user_id=1, n=2, catalog=catalog)
        assert len(recs) <= 2

    def test_excludes_rated_movies(self, content_recommender, built_data_cls):
        train, catalog = built_data_cls
        user_id = 1
        rated = set(train.ratings_matrix[user_id].indices)
        recs = content_recommender.recommend(user_id=user_id, n=10, catalog=catalog)
        recommended = {r["movie_id"] for r in recs}
        assert recommended.isdisjoint(rated)


# ---------------------------------------------------------------------------
# Enrich (inherited from base)
# ---------------------------------------------------------------------------


class TestEnrich:
    def test_enrich_adds_title(self, content_recommender, built_data_cls):
        train, catalog = built_data_cls
        recs = content_recommender.recommend(user_id=1, n=3, catalog=catalog)
        enriched = content_recommender.enrich(recs, catalog=catalog)
        for rec in enriched:
            assert "title" in rec
            assert rec["title"] is not None


# ---------------------------------------------------------------------------
# Load/Save
# ---------------------------------------------------------------------------
class TestLoadSave:
    def test_save_load_preserves_recommendations(self, content_recommender, built_data_cls, tmp_path):
        train, catalog = built_data_cls
        path = tmp_path / "model.joblib"

        original_recs = content_recommender.recommend(n=5, catalog=catalog, user_id=1)

        content_recommender.save(path)
        restored = content_recommender.load(path)

        restored_recs = restored.recommend(n=5, catalog=catalog, user_id=1)

        assert original_recs == restored_recs

    def test_save_load_preserves_params(self, content_recommender, tmp_path, built_data_cls):
        train, catalog = built_data_cls
        path = tmp_path / "model.joblib"
        # treat as file not path to avoid perm
        content_recommender.save(path)
        restored = content_recommender.load(path)

        assert restored._movie_conf_alpha == content_recommender._movie_conf_alpha
        assert restored._movie_conf_beta == content_recommender._movie_conf_beta

        assert np.isclose(restored.rated_indices_, restored.rated_indices_).all()
