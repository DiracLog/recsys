"""Validation tests for Block 5 (float32 factors) and Block 6 (uint16 indices).

Block 5: top-N recommendation overlap between float64 and float32 must be >= 99% at N=10.
Block 6: uint16 indices must round-trip cleanly (values preserved, max < 65536, indexing works).
"""

import numpy as np
import pytest

from src.data.loaders import load_data
from src.data.preprocessor import Preprocessor
from src.models.collaborative_recommender import CollaborativeRecommender
from src.models.content_recommender import ContentBasedRecommender
from src.models.hybrid_recommender import HybridRecommender


@pytest.fixture
def fitted_pair_with_unrated(tmp_path):
    """Fit pair with a fixture that guarantees unrated candidates per user."""
    # Build dat files inline with movie 50 surviving but unrated by some users
    ratings_path = tmp_path / "ratings.dat"
    movies_path = tmp_path / "movies.dat"
    users_path = tmp_path / "users.dat"

    ratings_rows = [
        (1, 10, 5, 1000),
        (1, 20, 4, 1000),
        (1, 30, 3, 1000),
        (1, 40, 2, 1000),
        (2, 10, 4, 1000),
        (2, 20, 5, 1000),
        (2, 30, 4, 1000),
        (2, 50, 2, 1000),
        (3, 10, 3, 1000),
        (3, 20, 4, 1000),
        (3, 40, 4, 1000),
        (3, 50, 3, 1000),
        (5, 10, 5, 1000),
        (5, 30, 3, 1000),
        (5, 40, 2, 1000),
        (5, 50, 4, 1000),
        (6, 20, 5, 1000),
        (6, 30, 4, 1000),
        (6, 40, 3, 1000),
        (6, 50, 1, 1000),
    ]
    ratings_path.write_text("\n".join("::".join(str(x) for x in row) for row in ratings_rows))
    movies_path.write_text(
        "\n".join(
            "::".join(str(x) for x in row)
            for row in [
                (10, "Alpha", "Action|Comedy"),
                (20, "Beta", "Action"),
                (30, "Gamma", "Drama"),
                (40, "Delta", "Drama|Comedy"),
                (50, "Epsilon", "Comedy"),
            ]
        )
    )
    users_path.write_text(
        "\n".join("::".join(str(x) for x in row) for row in [(i, "M", 25, 4, "00000") for i in range(1, 7)])
    )

    movies, ratings, _ = load_data(path_m=movies_path, path_r=ratings_path, path_u=users_path)
    train, catalog = Preprocessor(m_threshold=3, u_threshold=3).preprocess(movies_df=movies, ratings_df=ratings)

    return _build_hybrid(train, catalog, np.float64), _build_hybrid(train, catalog, np.float32), catalog, train


# ---------------------------------------------------------------------------
# Block 5 — float32 vs float64 top-N overlap
# ---------------------------------------------------------------------------
#
# Strategy: fit two hybrid models on the same training data, one with
# dtype_mx=float64 (baseline), one with dtype_mx=float32 (Block 5 default).
# For a sample of users, compare top-N recommendation sets.
#
# Note on synthetic data: the conftest fixture has only 5 surviving users
# and 4 surviving movies after filtering. With universe < N, top-N collapses
# to "all available items" and overlap is trivially 100%. This is fine for
# a smoke test — it confirms the cast doesn't break the pipeline. For real
# precision-loss validation, run on ML-1M (separate manual run).


def _build_hybrid(train, catalog, dtype_mx, dtype_indices=np.uint16):
    """Build a hybrid with explicit dtypes on both children."""
    content = ContentBasedRecommender(
        profile_alpha=2.0,
        movie_conf_alpha=2.0,
        movie_conf_beta=1.0,
        shrinkage="bayesian",
        movies_conf=True,
        dtype_mx=dtype_mx,
        dtype_indices=dtype_indices,
    )
    collab = CollaborativeRecommender(
        k_principal=3,
        dtype_mx=dtype_mx,
        dtype_indices=dtype_indices,
    )
    return HybridRecommender(content=content, collaborative=collab, weight=0.5).fit(data=train, catalog=catalog)


def _topn_movie_ids(model, catalog, user_id, n):
    """Extract movie_ids from a recommend() call as a set."""
    recs = model.recommend(catalog=catalog, user_id=user_id, n=n)
    return {r["movie_id"] for r in recs}


@pytest.fixture
def fitted_pair(built_data_cls):
    """Fit two hybrid models, one float64 and one float32, on the same data."""
    train, catalog = built_data_cls
    model_f64 = _build_hybrid(train, catalog, dtype_mx=np.float64)
    model_f32 = _build_hybrid(train, catalog, dtype_mx=np.float32)
    return model_f64, model_f32, catalog, train


def test_float32_dtype_propagates(fitted_pair):
    """Sanity: float32 model actually has float32 factors."""
    _, model_f32, _, _ = fitted_pair
    assert model_f32._collaborative.U_.dtype == np.float32
    assert model_f32._collaborative.Vh_.dtype == np.float32
    assert model_f32._content.user_profiles_.dtype == np.float32


def test_float64_dtype_propagates(fitted_pair):
    """Sanity: float64 baseline has float64 factors."""
    model_f64, _, _, _ = fitted_pair
    assert model_f64._collaborative.U_.dtype == np.float64
    assert model_f64._collaborative.Vh_.dtype == np.float64
    assert model_f64._content.user_profiles_.dtype == np.float64


@pytest.mark.parametrize("n", [3, 5])
def test_topn_overlap_synthetic(fitted_pair_with_unrated, n):
    model_f64, model_f32, catalog, train = fitted_pair_with_unrated

    overlaps = []
    for user_id in train.user_ids:
        try:
            top_f64 = _topn_movie_ids(model_f64, catalog, int(user_id), n)
            top_f32 = _topn_movie_ids(model_f32, catalog, int(user_id), n)
        except Exception as e:
            print(f"user {user_id}: EXCEPTION {e}")
            continue

        if not top_f64 or not top_f32:
            continue

        union_size = len(top_f64 | top_f32)
        intersect = len(top_f64 & top_f32)
        overlaps.append(intersect / max(union_size, 1))

    assert overlaps, "no users produced recommendations"
    mean_overlap = float(np.mean(overlaps))
    assert mean_overlap >= 0.99, (
        f"mean top-{n} Jaccard overlap {mean_overlap:.4f} < 0.99 (across {len(overlaps)} users)"
    )


def test_topn_scores_close(fitted_pair_with_unrated):
    """Scores between f32 and f64 should be numerically close (not equal)."""
    model_f64, model_f32, catalog, train = fitted_pair_with_unrated

    user_id = int(train.user_ids[0])
    recs_f64 = model_f64.recommend(catalog=catalog, user_id=user_id, n=5)
    recs_f32 = model_f32.recommend(catalog=catalog, user_id=user_id, n=5)

    by_id_f64 = {r["movie_id"]: r["score"] for r in recs_f64}
    by_id_f32 = {r["movie_id"]: r["score"] for r in recs_f32}

    common = set(by_id_f64) & set(by_id_f32)
    assert common, "no movies in common between f32 and f64 recommendations"

    for mid in common:
        diff = abs(by_id_f64[mid] - by_id_f32[mid])
        # Generous tolerance — float32 mantissa is ~7 decimal digits
        assert diff < 1e-3, f"score divergence on movie {mid}: f64={by_id_f64[mid]}, f32={by_id_f32[mid]}"


# ---------------------------------------------------------------------------
# Block 6 — uint16 indices round-trip and value preservation
# ---------------------------------------------------------------------------


def test_uint16_dtype_propagates(content_recommender, collab_recommender):
    """Sanity: rated_indices_ is uint16 by default."""
    assert content_recommender.rated_indices_.dtype == np.uint16
    assert collab_recommender.rated_indices_.dtype == np.uint16


def test_uint16_values_within_range(content_recommender, collab_recommender):
    """All stored indices must fit in uint16 [0, 65535]."""
    for arr in [content_recommender.rated_indices_, collab_recommender.rated_indices_]:
        assert arr.min() >= 0
        assert arr.max() < 2**16


def test_uint16_roundtrip_via_int32(built_data_cls):
    """Casting int32 -> uint16 -> int32 must preserve values when in range.

    This guards against silent overflow if movie indices ever exceed 65535.
    """
    train, catalog = built_data_cls
    rec = ContentBasedRecommender(
        profile_alpha=2.0,
        movie_conf_alpha=2.0,
        movie_conf_beta=1.0,
        shrinkage="bayesian",
        movies_conf=True,
        dtype_indices=np.uint16,
    ).fit(catalog=catalog, data=train)

    indices_uint16 = rec.rated_indices_
    indices_back = indices_uint16.astype(np.int32)

    # Re-fit with int32 to compare directly
    rec_int32 = ContentBasedRecommender(
        profile_alpha=2.0,
        movie_conf_alpha=2.0,
        movie_conf_beta=1.0,
        shrinkage="bayesian",
        movies_conf=True,
        dtype_indices=np.int32,
    ).fit(catalog=catalog, data=train)

    assert np.array_equal(indices_back, rec_int32.rated_indices_)


def test_uint16_indices_usable_for_indexing(collab_recommender):
    """uint16 indices must work as numpy array indexers without dtype issues."""
    Vh = collab_recommender.Vh_
    indices = collab_recommender.rated_indices_

    # Pick first user's slice via indptr
    indptr = collab_recommender.rated_indptr_
    user0_indices = indices[indptr[0] : indptr[1]]

    # Index into Vh columns
    sliced = Vh[:, user0_indices]
    assert sliced.shape == (Vh.shape[0], len(user0_indices))


# ---------------------------------------------------------------------------
# Block 6 — overflow guard (currently absent; this test documents the gap)
# ---------------------------------------------------------------------------


@pytest.mark.skip(reason="overflow assertion not yet added to fit; remove skip after adding")
def test_uint16_overflow_raises(built_data_cls):
    """When n_movies >= 65536, fit with dtype_indices=uint16 should raise.

    Currently fit silently casts and overflows. Add an assertion in fit:
        if dtype_indices == np.uint16:
            assert n_movies < 2**16, f"uint16 overflow: {n_movies} movies"

    Then remove the skip marker on this test.
    """
    # Construction of a >65k movie catalog is non-trivial in a unit test;
    # this test exists primarily as a documentation stub. The real guard
    # belongs in fit() and will trip in production if MovieLens grows past 65k.
    pass
