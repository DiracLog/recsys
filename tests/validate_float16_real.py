"""Standalone float16 vs float64 validation on real (ML-32M dev) data.

Run manually, not as part of pytest:
    uv run python scripts/validate_float16_real.py

Builds two hybrid models on the same training data, one with dtype_mx=float16,
one with float64. Reports top-N overlap and score divergence over a sample
of users. Output informs the Block 7 (float16) go/no-go decision.

Threshold guidance:
    >= 0.95 mean Jaccard at N=10  → float16 safe to deploy
    0.85–0.95                      → marginal; consider float16 only if RAM-critical
    < 0.85                         → too lossy; stay on float32

Why standalone, not pytest:
    Real-data fits take minutes. Not appropriate in CI. This is a one-off
    decision check before committing to Block 7.
"""

import time
from collections import defaultdict

import numpy as np

from src.api.dependencies import prepare_state
from src.data.loaders import load_data_ml32
from src.models.collaborative_recommender import CollaborativeRecommender
from src.models.content_recommender import ContentBasedRecommender
from src.models.hybrid_recommender import HybridRecommender


SAMPLE_SIZE = 1000
N_VALUES = [10, 50, 100]
SEED = 42


def build_hybrid(train, catalog, dtype_mx, weight=0.5, k_principal=50):
    """Construct and fit a hybrid with the given factor dtype."""
    content = ContentBasedRecommender(
        profile_alpha=2.0,
        movie_conf_alpha=2.0,
        movie_conf_beta=1.0,
        shrinkage="bayesian",
        movies_conf=True,
        dtype_mx=dtype_mx,
    )
    collab = CollaborativeRecommender(
        k_principal=k_principal,
        dtype_mx=dtype_mx,
    )
    return HybridRecommender(content=content, collaborative=collab, weight=weight).fit(data=train, catalog=catalog)


def topn_movie_ids(model, catalog, user_id, n):
    """Run recommend, return set of movie_ids and score-by-id dict."""
    recs = model.recommend(catalog=catalog, user_id=int(user_id), n=n)
    ids = {r["movie_id"] for r in recs}
    scores = {r["movie_id"]: r["score"] for r in recs}
    return ids, scores


def jaccard(a, b):
    if not a and not b:
        return 1.0
    return len(a & b) / max(len(a | b), 1)


def main():
    print("=" * 72)
    print("float16 vs float64 — real-data validation")
    print("=" * 72)

    # ---------------------------------------------------------------
    # Load training data via prepare_state (one fit, then we extract)
    # ---------------------------------------------------------------
    print("\n[1/4] Loading data via prepare_state (fresh fit)...")
    t0 = time.time()
    hybrid_default, _popularity, _movie_search, _pop_pre, catalog = prepare_state(load_from_artifacts=False)
    print(f"      done in {time.time() - t0:.1f}s")

    # We need the underlying training data to refit at different dtypes.
    # If prepare_state doesn't expose it directly, reconstruct via the
    # already-fitted model (rated_indices_, indptr live on the recommenders).
    # Easiest: re-run preprocessor here. Adjust to your project's API.
    from src.data.preprocessor import Preprocessor
    from pathlib import Path
    import os
    from dotenv import load_dotenv

    load_dotenv()
    data_root = Path(os.environ["ML32_PATH"])  # adjust env var to your setup
    movies_df, ratings_df, _ = load_data_ml32(
        path_m=data_root / "movies.csv",
        path_r=data_root / "ratings.csv",
    )
    train, catalog = Preprocessor(m_threshold=5, u_threshold=5).preprocess(movies_df=movies_df, ratings_df=ratings_df)
    print(f"      training data: {len(train.user_ids)} users × {len(train.movie_ids)} movies")

    # ---------------------------------------------------------------
    # Fit float64 baseline and float16 candidate
    # ---------------------------------------------------------------
    print("\n[2/4] Fitting float64 baseline...")
    t0 = time.time()
    model_f64 = build_hybrid(train, catalog, dtype_mx=np.float64)
    print(f"      done in {time.time() - t0:.1f}s")

    print("\n[3/4] Fitting float16 candidate...")
    t0 = time.time()
    model_f16 = build_hybrid(train, catalog, dtype_mx=np.float16)
    print(f"      done in {time.time() - t0:.1f}s")

    # ---------------------------------------------------------------
    # Sample users and compare top-N
    # ---------------------------------------------------------------
    print(f"\n[4/4] Comparing top-N over {SAMPLE_SIZE} random users...")

    rng = np.random.default_rng(SEED)
    n_users = len(train.user_ids)
    sample_size = min(SAMPLE_SIZE, n_users)
    sample_idx = rng.choice(n_users, size=sample_size, replace=False)
    sample_user_ids = train.user_ids[sample_idx]

    overlaps_by_n = defaultdict(list)
    score_diffs = []
    skipped = 0
    errors = 0

    for i, uid in enumerate(sample_user_ids):
        if i % 100 == 0 and i > 0:
            print(f"      {i}/{sample_size}...")

        try:
            for n in N_VALUES:
                ids_f64, scores_f64 = topn_movie_ids(model_f64, catalog, uid, n)
                ids_f16, scores_f16 = topn_movie_ids(model_f16, catalog, uid, n)

                if not ids_f64 or not ids_f16:
                    skipped += 1
                    continue

                overlaps_by_n[n].append(jaccard(ids_f64, ids_f16))

                if n == 10:  # only collect score diffs at N=10
                    common = ids_f64 & ids_f16
                    for mid in common:
                        score_diffs.append(abs(scores_f64[mid] - scores_f16[mid]))
        except Exception as e:
            errors += 1
            if errors < 5:
                print(f"      user {uid}: {e}")

    # ---------------------------------------------------------------
    # Report
    # ---------------------------------------------------------------
    print("\n" + "=" * 72)
    print("Results")
    print("=" * 72)
    print(f"Sample size:    {sample_size}")
    print(f"Skipped:        {skipped} (empty recommendations)")
    print(f"Errors:         {errors}")

    print("\nTop-N Jaccard overlap (float16 vs float64):")
    for n in N_VALUES:
        vals = overlaps_by_n[n]
        if not vals:
            print(f"  N={n:3d}: NO DATA")
            continue
        arr = np.array(vals)
        print(
            f"  N={n:3d}:  mean={arr.mean():.4f}  "
            f"median={np.median(arr):.4f}  "
            f"p10={np.percentile(arr, 10):.4f}  "
            f"min={arr.min():.4f}  "
            f"({len(vals)} users)"
        )

    if score_diffs:
        sd = np.array(score_diffs)
        print(
            f"\nScore divergence on common items (N=10): "
            f"mean={sd.mean():.5f}  max={sd.max():.5f}  p99={np.percentile(sd, 99):.5f}"
        )

    # ---------------------------------------------------------------
    # Verdict
    # ---------------------------------------------------------------
    print("\n" + "=" * 72)
    print("Verdict")
    print("=" * 72)
    if not overlaps_by_n[10]:
        print("INSUFFICIENT DATA — investigate skipped/error counts above")
        return

    mean10 = float(np.mean(overlaps_by_n[10]))
    if mean10 >= 0.95:
        print(f"GO. Top-10 overlap {mean10:.4f} >= 0.95 — float16 safe to ship.")
    elif mean10 >= 0.85:
        print(f"MARGINAL. Top-10 overlap {mean10:.4f} in [0.85, 0.95). Use float16 only if RAM-critical.")
    else:
        print(f"NO-GO. Top-10 overlap {mean10:.4f} < 0.85. Stay on float32.")


if __name__ == "__main__":
    main()
