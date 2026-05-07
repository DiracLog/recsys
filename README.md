# Movie Recommender

A hybrid movie recommendation system over MovieLens, built from primitives. Combines content-based similarity, collaborative filtering via sparse SVD, and Bayesian-shrunk popularity into a convex blend. FastAPI service, optimized to serve ML-32M in **190 MB cold / 223 MB warm RSS** — a 5.6× reduction from a naïve baseline.

The focus is engineering, not algorithm novelty. Off-the-shelf libraries (`surprise`, `lightfm`, `implicit`) handle the math; this project foregrounds the parts they abstract away — sparse linear algebra, memory layout, deployment constraints, and validation methodology.

> Live demo: deployment in progress. Will be added when public URL is available.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        FastAPI service                          │
│       /recommend       /popular       /search       /health     │
└────────────┬────────────────┬──────────────────┬────────────────┘
             │                │                  │
             ▼                ▼                  ▼
       ┌───────────┐    ┌────────────┐     ┌──────────────┐
       │  Hybrid   │    │ Popularity │     │ MovieSearch  │
       │ ───────── │    │  (Bayesian │     │  (catalog    │
       │  α·Content│    │   shrunk)  │     │   index)     │
       │ +β·Collab │    └────────────┘     └──────────────┘
       │ +γ·Pop    │
       └─────┬─────┘
             │
   ┌─────────┴───────────┐
   ▼                     ▼
┌──────────┐      ┌──────────────┐
│ Content  │      │ Collaborative│
│ ──────── │      │ ──────────── │
│ genre    │      │ sparse SVD   │
│ cosine + │      │ (svds)       │
│ Bayesian │      │ + fold-in    │
│ shrinkage│      │ for new users│
└─────┬────┘      └─────┬────────┘
      │                 │
      └────────┬────────┘
               ▼
        ┌────────────────┐
        │ TrainingData   │
        │ (sparse CSR,   │
        │  user_means,   │
        │  rated_indices)│
        └────────────────┘
```

**Data layer:** Polars internally, numpy/scipy at the model boundary. Catalog stays in a domain-specific schema for the API.

**Serving:** FastAPI + uvicorn, single worker. Factor matrices memory-mapped from disk via `joblib`'s `mmap_mode`.

---

## Memory optimization journey

The optimization arc reduced post-load RSS by **5.6×**, validated for recommendation quality at each step.

| Stage | RSS | Δ | What changed |
|---|---|---|---|
| Original | 1066 MB | — | Naïve dense matrices, view-pinning bugs |
| Sparse + flat indices | 484 MB | −582 | CSR layout, `ascontiguousarray` to break view pins |
| Polars + dependency cleanup | 433 MB | −51 | Polars in data layer, dropped pandas + sklearn |
| float32 + uint16 (Blocks 5+6) | 315 MB | −118 | Half-precision factors, narrow indices |
| float16 (Block 7) | 287 MB | −28 | Quarter-precision factors |
| **mmap (Block 8) — cold** | **190 MB** | −97 | File-backed factor arrays |
| mmap warm (100 distinct users) | 223 MB | — | Working set under load |

**Validation:** at each precision step, top-N recommendation overlap was measured against a float64 baseline on real ML-32M data. Float16 maintained 99.4% top-10 overlap — well above the 0.95 ship threshold.

### Component breakdown (final state)

| Component | Size | Shape | Dtype |
|---|---|---|---|
| Python + deps baseline | 101 MB | — | — |
| catalog (Polars, dict) | 9.6 MB | — | — |
| content.user_profiles | 7.5 MB | (195544, 20) | float16 |
| content.rated_indices_ | 60.5 MB | (31720336,) | uint16 |
| collab.U | 18.6 MB | (195544, 50) | float16 |
| collab.Vh | 2.9 MB | (50, 30521) | float16 |
| collab.rated_indices_ (aliased) | 60.5 MB | (31720336,) | uint16 |
| popularity.scores | 0.2 MB | (30521,) | float64 |

`rated_indices_` is shared between content and collaborative recommenders via aliasing post-load. Disk stores duplicates; RAM dedups.

---

## Methodology

The optimization work followed a pattern that mattered as much as the result:

1. **Measure first.** Each block began with a baseline benchmark logged to `measurements/benchmarks.md`.
2. **Hypothesize the source.** Component-by-component sizing via `pympler.asizeof` and `tracemalloc`, not guesswork.
3. **Implement on a dedicated branch.** One block per branch, merged through `perf/optimization` before landing on `main`.
4. **Validate output.** Recommendation quality preserved via top-N overlap tests on real data — not synthetic toy fixtures.
5. **Decide based on numbers.** Some blocks (Block 7 float16) were validated and shipped; others (aggressive filtering) were scoped out as future work.

The full block-by-block git history on `main` documents the trajectory.

---

## Stack

- **Python 3.12**, **uv** for dependency management
- **Polars** for the data layer (CSV ingest, groupby, sorted unique)
- **scipy.sparse** for CSR/COO matrices and `svds` (ARPACK with LOBPCG fallback)
- **numpy** for factor arithmetic, fold-in, and scoring
- **FastAPI** + **uvicorn** for serving
- **joblib** for artifact persistence with mmap-compatible save format
- **pympler** + **tracemalloc** + **psutil** for memory measurement

Notable absences: pandas (replaced by Polars + numpy), scikit-learn (replaced by ~10 lines for `MultiLabelBinarizer`), sentence-transformers (initially scoped, removed when unused).

---

## Why not just use `surprise` / `lightfm`?

Off-the-shelf recommender libraries handle the math elegantly. They abstract away the parts this project intentionally exposes:

- **Memory layout:** `surprise` materializes ratings into pandas DataFrames; ML-32M is 2–3 GB before the model even fits. This project ships at 190 MB.
- **Sparse SVD on real-scale data:** `surprise` uses dense SVD internally. `scipy.sparse.linalg.svds` with `ascontiguousarray` to break view-pinning is what makes 32M ratings tractable.
- **Hybrid blending:** convex combination of content + collaborative + Bayesian-shrunk popularity, not provided by any single library off-the-shelf.
- **Cold-start fold-in:** new users get content-based profiles built on the fly; `surprise.predict` requires users in the training set.
- **Production serving:** `surprise` isn't a service. The FastAPI layer, dependency injection, and lifespan management are project-specific.
- **Deploy constraints:** mmap, float16 validation, uint16 indices — none of these are library decisions; they're engineering decisions made under a 256 MB target.

For a quick baseline on small data, use `surprise`. For understanding what those abstractions hide, build it.

---

## Project layout

```
src/
├── api/                 # FastAPI app, lifespan, dependency wiring
├── data/                # Loaders (Polars), Preprocessor, schemas
├── models/              # Recommender base, hybrid + 3 children, MovieSearch
├── train.py             # Fit pipeline, artifact persistence
└── utils/               # Memory measurement, benchmark writer

tests/                   # pytest suite (precision, dtype, round-trip)
scripts/                 # One-off validators (e.g., float16 on ML-32M)
artifacts/runs/          # Fitted models per run, with `latest` symlink
measurements/            # Append-only benchmark log
config.example.yaml      # Documented config template
```

---

## Running locally

```bash
# Install deps via uv
uv sync

# Fit on MovieLens 1M or 32M (configure in config.yaml)
uv run python -m src.train

# Serve
uv run serve
# or directly:
uv run uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```

Configuration lives in `config.yaml`. A documented template is checked in as `config.example.yaml`. Knobs include factor precision, index dtype, mmap mode, hybrid blend weight, and SVD components.

### API

```bash
# Top-N popular movies
curl localhost:8000/popular?n=10

# Personalized recommendations for an existing user
curl "localhost:8000/recommend?user_id=42&n=10"

# Search movies by title
curl "localhost:8000/search?q=matrix"

# Health
curl localhost:8000/health
```

---

## Validation

**Functional tests** (`tests/`):

- Precision validation: float32 vs float64 top-N overlap on synthetic and 1M data
- Forward-compat: float16 dtype propagation, byte-size halving, score divergence within tolerance
- uint16 round-trip: cast → cast back, bitwise equality
- Hybrid: weighted blend, fold-in for cold-start users
- API: response schema, error handling

**Real-data validation scripts** (`scripts/`):

- `validate_float16_real.py`: top-N Jaccard overlap between float16 and float64 hybrids on ML-32M, with go/no-go verdict at thresholds 0.95 (GO), 0.85 (MARGINAL).

Latest float16 verdict on ML-32M:

```
Top-10 overlap mean=0.9940 (1000 users)
Verdict: GO. float16 safe to ship.
```

---

## Future work

Features deferred to keep scope contained:

- **1-10 rating scale with half-step input.** Frontend-only conversion to internal 1-5 scale.
- **Anime dataset support.** Demonstrates data-layer extensibility; reuses preprocessor and training pipeline.
- **External rating sync (IMDB CSV import).** Adds personalization without auth/DB complexity. Full OAuth integration with Shikimori or similar deferred indefinitely — DB and auth work has high cost relative to portfolio signal.
- **Aggressive filtering.** Drop users with <5 ratings and movies with <10 ratings for further memory reduction. Compounds with everything above.

---

## Acknowledgments

- MovieLens datasets ([GroupLens, University of Minnesota](https://grouplens.org/datasets/movielens/))
- `scipy.sparse.linalg.svds` for sparse truncated SVD
- The `surprise` library docs for confirming algorithmic conventions

---

## About

- GitHub: [DiracLog](https://github.com/DiracLog)
- LinkedIn: [Maksym Korolchuk](https://www.linkedin.com/in/maksym-k-751ab0214/)

## License

MIT.
