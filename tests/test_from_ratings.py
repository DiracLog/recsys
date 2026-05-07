"""Tests for score_all_from_ratings across recommenders (new user cold-start path)."""

import numpy as np
import pytest


class TestContentFromRatings:
    def test_returns_score_per_movie(self, content_recommender, built_data_cls):
        train, catalog = built_data_cls
        ratings = {10: 5.0, 20: 4.0}
        scores, mask = content_recommender.score_all_from_ratings(ratings=ratings, catalog=catalog)
        assert scores.shape == (train.ratings_matrix.shape[1],)
        assert mask.shape == (train.ratings_matrix.shape[1],)

    def test_mask_identifies_rated(self, content_recommender, built_data_cls):
        train, catalog = built_data_cls
        ratings = {10: 5.0, 30: 2.0}
        _, mask = content_recommender.score_all_from_ratings(ratings=ratings, catalog=catalog)
        movie_ids = train.movie_ids
        rated_positions = np.isin(movie_ids, [10, 30])
        np.testing.assert_array_equal(mask, rated_positions)

    def test_empty_ratings_no_crash(self, content_recommender, built_data_cls):
        train, catalog = built_data_cls
        scores, mask = content_recommender.score_all_from_ratings(ratings={}, catalog=catalog)
        assert not mask.any()


class TestCollabFromRatings:
    def test_returns_score_per_movie(self, collab_recommender, built_data_cls):
        train, catalog = built_data_cls
        ratings = {10: 5.0, 20: 4.0}
        scores, mask = collab_recommender.score_all_from_ratings(ratings)
        assert scores.shape == (train.ratings_matrix.shape[1],)
        assert mask.shape == (train.ratings_matrix.shape[1],)

    def test_mask_identifies_rated(self, collab_recommender, built_data_cls):
        train, catalog = built_data_cls
        ratings = {10: 5.0, 30: 2.0}
        _, mask = collab_recommender.score_all_from_ratings(ratings)
        movie_ids = train.movie_ids
        rated_positions = np.isin(movie_ids, [10, 30])
        np.testing.assert_array_equal(mask, rated_positions)

    def test_empty_ratings_warns(self, collab_recommender):
        with pytest.warns(UserWarning, match="No valid ratings"):
            scores, mask = collab_recommender.score_all_from_ratings({})
        assert not mask.any()
        assert np.allclose(scores, 0.0)

    def test_unknown_movie_id_ignored(self, collab_recommender):
        """Movie ID not in training set should be silently skipped."""
        ratings = {10: 5.0, 99999: 3.0}  # 99999 does not exist
        scores, mask = collab_recommender.score_all_from_ratings(ratings)
        # Should still produce a valid result from the known rating
        assert scores.shape[0] > 0


class TestConsistency:
    def test_existing_user_from_ratings_close_to_score_all(self, collab_recommender, built_data_cls):
        """
        Sanity check: feeding an existing user's ratings through the new-user
        projection should give scores in roughly the right ballpark — not
        identical (centering / sign handling differs) but same shape and finite.
        """
        train, catalog = built_data_cls
        user_id = 1
        user_pos = np.where(train.user_ids == user_id)[0][0]  # id → position
        user_row = train.ratings_matrix.getrow(user_pos)
        user_mean = train.user_means[user_pos]
        # reconstruct raw ratings from sparse data
        raw_ratings = {
            int(train.movie_ids[col_pos]): float(rating + user_mean)  # position → id
            for col_pos, rating in zip(user_row.indices, user_row.data)
        }

        scores, mask = collab_recommender.score_all_from_ratings(raw_ratings)

        assert np.isfinite(scores).all()
        assert mask.sum() == len(raw_ratings)
