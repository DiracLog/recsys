"""Forward-compat tests for float16 factors.

float16 is a possible Block 7 candidate if RAM constraints tighten further
(e.g., Render free tier 256 MB target). These tests confirm the pipeline
accepts float16 as a dtype_mx and does not silently produce garbage.

Acceptance for float16 is loosened vs float32:
- Top-N Jaccard overlap >= 0.85 at N=3 (vs 0.99 for float32)
- Score divergence within 5e-2 (vs 1e-3 for float32)

Rationale: float16 has ~3 decimal digits of mantissa precision. SVD
reconstructions and similarity scores can drift more, especially for
items with close scores. Threshold chosen to catch silent breakage
without false positives from legitimate precision loss.
"""

import numpy as np
import pytest

from src.models.collaborative_recommender import CollaborativeRecommender
from src.models.content_recommender import ContentBasedRecommender
from src.models.hybrid_recommender import HybridRecommender


# Reuse helpers from test_precision_validation.py if test files share a dir;
# otherwise duplicate the small helper here.
def _build_hybrid(train, catalog, dtype_mx, dtype_indices=np.uint16):
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
    recs = model.recommend(catalog=catalog, user_id=user_id, n=n)
    return {r["movie_id"] for r in recs}


# ---------------------------------------------------------------------------
# Construction & dtype propagation
# ---------------------------------------------------------------------------


def test_float16_dtype_propagates(built_data_cls):
    """Sanity: float16 model has float16 factors after fit."""
    train, catalog = built_data_cls
    model = _build_hybrid(train, catalog, dtype_mx=np.float16)

    assert model._collaborative.U_.dtype == np.float16
    assert model._collaborative.Vh_.dtype == np.float16
    assert model._content.user_profiles_.dtype == np.float16


def test_float16_user_means_stays_float64(built_data_cls):
    """user_means must NOT be cast — accuracy-sensitive, small footprint."""
    train, catalog = built_data_cls
    model = _build_hybrid(train, catalog, dtype_mx=np.float16)

    # user_means lives on the collaborative recommender (or wherever it's stored).
    # Adjust attribute access to your actual location.
    assert model._collaborative.user_means_.dtype == np.float64


def test_float16_fit_does_not_crash(built_data_cls):
    """End-to-end fit must complete without exception on synthetic data."""
    train, catalog = built_data_cls
    model = _build_hybrid(train, catalog, dtype_mx=np.float16)
    assert model._collaborative.U_.size > 0
    assert model._content.user_profiles_.size > 0


# ---------------------------------------------------------------------------
# Numerical sanity (loose thresholds vs float64 baseline)
# ---------------------------------------------------------------------------


@pytest.fixture
def fitted_f16_pair_with_unrated(tmp_path):
    """Same construction as fitted_pair_with_unrated but float16 vs float64."""
    from src.data.loaders import load_data
    from src.data.preprocessor import Preprocessor

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

    model_f64 = _build_hybrid(train, catalog, dtype_mx=np.float64)
    model_f16 = _build_hybrid(train, catalog, dtype_mx=np.float16)
    return model_f64, model_f16, catalog, train


def test_float16_topn_overlap_loose(fitted_f16_pair_with_unrated):
    """Top-N Jaccard overlap >= 0.85 — loose threshold, catches silent breakage."""
    model_f64, model_f16, catalog, train = fitted_f16_pair_with_unrated

    overlaps = []
    for user_id in train.user_ids:
        try:
            top_f64 = _topn_movie_ids(model_f64, catalog, int(user_id), n=3)
            top_f16 = _topn_movie_ids(model_f16, catalog, int(user_id), n=3)
        except Exception:
            continue
        if not top_f64 or not top_f16:
            continue
        union = len(top_f64 | top_f16)
        intersect = len(top_f64 & top_f16)
        overlaps.append(intersect / max(union, 1))

    assert overlaps, "no users produced recommendations"
    mean_overlap = float(np.mean(overlaps))
    assert mean_overlap >= 0.85, (
        f"float16 top-3 Jaccard overlap {mean_overlap:.4f} < 0.85 — "
        f"either real precision loss or silent breakage (across {len(overlaps)} users)"
    )


def test_float16_scores_loosely_close(fitted_f16_pair_with_unrated):
    """Scores between f16 and f64 within 5e-2 — loose tolerance."""
    model_f64, model_f16, catalog, train = fitted_f16_pair_with_unrated

    user_id = int(train.user_ids[0])
    recs_f64 = model_f64.recommend(catalog=catalog, user_id=user_id, n=5)
    recs_f16 = model_f16.recommend(catalog=catalog, user_id=user_id, n=5)

    by_id_f64 = {r["movie_id"]: r["score"] for r in recs_f64}
    by_id_f16 = {r["movie_id"]: r["score"] for r in recs_f16}

    common = set(by_id_f64) & set(by_id_f16)
    assert common, "no movies in common between f16 and f64"

    for mid in common:
        diff = abs(by_id_f64[mid] - by_id_f16[mid])
        assert diff < 5e-2, (
            f"score divergence on movie {mid}: f64={by_id_f64[mid]:.4f}, f16={by_id_f16[mid]:.4f}, diff={diff:.4f}"
        )


# ---------------------------------------------------------------------------
# Memory footprint sanity
# ---------------------------------------------------------------------------


def test_float16_factor_size_halves_vs_float32(built_data_cls):
    """float16 arrays must use exactly half the bytes of float32."""
    train, catalog = built_data_cls
    model_f32 = _build_hybrid(train, catalog, dtype_mx=np.float32)
    model_f16 = _build_hybrid(train, catalog, dtype_mx=np.float16)

    for attr in ["U_", "Vh_"]:
        f32_bytes = getattr(model_f32._collaborative, attr).nbytes
        f16_bytes = getattr(model_f16._collaborative, attr).nbytes
        assert f16_bytes * 2 == f32_bytes, f"{attr}: expected {f32_bytes // 2} bytes for float16, got {f16_bytes}"

    f32_profiles = model_f32._content.user_profiles_.nbytes
    f16_profiles = model_f16._content.user_profiles_.nbytes
    assert f16_profiles * 2 == f32_profiles
