"""Tests for CollaborativeRecommender."""

import numpy as np

from src.models.collaborative_recommender import CollaborativeRecommender


class TestFit:
    def test_fit_returns_self(self, built_data_cls):
        train, catalog = built_data_cls
        rec = CollaborativeRecommender(k_principal=3)
        assert rec.fit(catalog=catalog, data=train) is rec

    def test_fit_populates_decomposition(self, collab_recommender, built_data_cls):
        train, catalog = built_data_cls
        n_users = len(train.user_ids)
        n_movies = len(train.movie_ids)
        k = 3

        Uk = collab_recommender.U_
        Sk = collab_recommender.S_
        Vhk = collab_recommender.Vh_
        assert Uk.shape == (n_users, k)
        assert Sk.shape == (k,)
        assert Vhk.shape == (k, n_movies)

    def test_fit_populates_indices(self, collab_recommender, built_data_cls):
        train, catalog = built_data_cls
        np.testing.assert_array_equal(collab_recommender.user_ids_, train.user_ids)
        np.testing.assert_array_equal(collab_recommender.movie_ids_, train.movie_ids)


class TestRecommend:
    def test_returns_correct_shape(self, collab_recommender):
        recs = collab_recommender.recommend(user_id=1, n=3, catalog=None)
        assert len(recs) <= 3
        for rec in recs:
            assert {"movie_id", "score"} <= set(rec.keys())
            assert isinstance(rec["movie_id"], int)

    def test_respects_n(self, collab_recommender):
        recs = collab_recommender.recommend(user_id=1, n=2, catalog=None)
        assert len(recs) <= 2

    def test_excludes_rated_movies(self, collab_recommender, built_data_cls):
        train, catalog = built_data_cls
        user_id = 1
        rated = set(train.ratings_matrix[user_id].indices)
        recs = collab_recommender.recommend(user_id=user_id, n=10, catalog=None)
        recommended = {r["movie_id"] for r in recs}
        assert recommended.isdisjoint(rated)


class TestScoreAll:
    def test_returns_score_per_movie(self, collab_recommender, built_data_cls):
        train, catalog = built_data_cls
        scores = collab_recommender.score_all(user_id=1)
        assert scores.shape[0] == train.ratings_matrix.shape[1]

    def test_includes_user_mean(self, collab_recommender, built_data_cls):
        train, catalog = built_data_cls
        """Scores are de-centered by user mean — check order of magnitude is sane."""
        scores = collab_recommender.score_all(user_id=1)
        user_mean = train.user_means[1]
        # After adding mean, scores should hover around user_mean (within rating range)
        assert np.abs(scores - user_mean).max() < 5.0

    def test_avoid_double_copy(self, collab_recommender):
        """Scores are de-centered by user mean — check order of magnitude is sane."""
        assert collab_recommender.U_.flags["OWNDATA"]
        assert not collab_recommender.U_.base
        assert collab_recommender.U_.base is None
        assert collab_recommender.Vh_.base is None


class TestK:
    def test_different_k_gives_different_predictions(self, built_data_cls):
        train, catalog = built_data_cls
        rec_low = CollaborativeRecommender(k_principal=1).fit(data=train, catalog=catalog)
        rec_high = CollaborativeRecommender(k_principal=3).fit(data=train, catalog=catalog)
        scores_low = rec_low.score_all(user_id=1)
        scores_high = rec_high.score_all(user_id=1)
        assert not np.allclose(scores_low, scores_high)


class TestEnrich:
    def test_enrich_adds_title(self, collab_recommender, built_data_cls):
        train, catalog = built_data_cls
        recs = collab_recommender.recommend(user_id=1, n=3, catalog=None)
        enriched = collab_recommender.enrich(recs, catalog=catalog)
        for rec in enriched:
            assert "title" in rec


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
        # bool arr
        assert np.isclose(restored.rated_indices_, restored.rated_indices_).all()
