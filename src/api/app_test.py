"""Version for app test, with measurement utils and
artifact saving/loading for debugging memory issues during loading."""

import os
import time
import tracemalloc
from contextlib import asynccontextmanager
from logging import getLogger
from pathlib import Path

import psutil
from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.api.dependencies import prepare_state
from src.api.movie_search import MovieSearch
from src.api.schemas import (
    MovieResult,
    PopularResponse,
    RecommendRequest,
    RecommendResponse,
    SearchResponse,
)
from src.logging.setup_logging import setup_logging
from src.models.data_schemas import Catalog
from src.models.hybrid_recommender import HybridRecommender
from src.utils.utils_measurement import catalog_size_mb, measure_components, write_benchmark_section

STATIC_DIR = Path(__file__).parent / "static"

setup_logging(log_file="api_seving.log")
logger = getLogger(__name__)


@asynccontextmanager
async def lifespan(application: FastAPI):
    proc = psutil.Process(os.getpid())
    baseline_mb = proc.memory_info().rss / 1024**2
    logger.info(f"Python+deps baseline: {baseline_mb:.1f}MB")

    tracemalloc.start()
    hybrid, popularity, movie_search, pop_precomputed, catalog = prepare_state()

    from pympler import muppy, summary

    all_objects = muppy.get_objects()
    sum1 = summary.summarize(all_objects)
    summary.print_(sum1, limit=20)

    application.state.hybrid = hybrid
    application.state.popularity = popularity
    application.state.movie_search = movie_search
    application.state.pop_precomputed = pop_precomputed
    application.state.catalog = catalog

    current, peak = tracemalloc.get_traced_memory()
    rss_mb = proc.memory_info().rss / 1024**2
    logger.info(f"Traced memory: current={current / 1024**2:.2f} MB, peak={peak / 1024**2:.2f} MB")
    logger.info(f"Post-load RSS: {rss_mb:.1f}MB")
    logger.info(f"diff: {rss_mb - baseline_mb:.1f}MB")

    import gc

    gc.collect()
    rss_after_gc = proc.memory_info().rss / 1024**2
    logger.info(f"RSS after gc.collect: {rss_after_gc:.1f}MB (was {rss_mb:.1f}MB)")

    aliased = id(hybrid._content.rated_indices_) == id(hybrid._collaborative.rated_indices_)
    logger.info(f"rated_indices_ aliased: {aliased}")
    import numpy as np

    def array_count(obj):
        if isinstance(obj, np.ndarray):
            return 1
        if isinstance(obj, list):
            return sum(array_count(x) for x in obj)
        if isinstance(obj, dict):
            return sum(array_count(v) for v in obj.values())
        return 0

    logger.info(f"content.rated_indices_: {array_count(hybrid._content.rated_indices_)} arrays")
    logger.info(f"collab.rated_indices_: {array_count(hybrid._collaborative.rated_indices_)} arrays")
    logger.info(f"content all attrs: {array_count(vars(hybrid._content))}")
    logger.info(f"collab all attrs: {array_count(vars(hybrid._collaborative))}")

    rows = measure_components(
        [
            ("movies", movie_search),
            ("catalog", catalog),
            ("hybrid.content.user_profiles", hybrid._content.user_profiles_),
            ("hybrid.content.rated_indices_", hybrid._content.rated_indices_),
            ("hybrid.collab.U", hybrid._collaborative.U_),
            ("hybrid.collab.Vh", hybrid._collaborative.Vh_),
            ("hybrid.collab.rated_indices_", hybrid._collaborative.rated_indices_),
            ("popularity.scores", popularity.resulting_scores_),
        ],
        catalog_size_fn=catalog_size_mb,
    )

    for name, size_mb, shape, dtype in rows:
        if size_mb is None:
            logger.warning(f"{name} FAILED: {dtype}")
        else:
            logger.info(f"{name:<42} {size_mb:>7.1f} MB   shape={shape}  dtype={dtype}")

    # Warm-up: simulate request load to measure mmap residency growth
    rng = np.random.default_rng(42)

    all_user_ids = hybrid._content.user_ids_
    # before request load
    rss_cold = proc.memory_info().rss / 1024**2
    logger.info("RSS post-load (cold): %sMB", rss_cold)
    # 1 request
    hybrid.recommend(catalog=catalog, user_id=int(all_user_ids[0]), n=10)
    rss_after_1 = proc.memory_info().rss / 1024**2

    # 100 diverse requests
    sample_idx = rng.choice(len(all_user_ids), size=100, replace=False)
    for idx in sample_idx:
        hybrid.recommend(catalog=catalog, user_id=int(all_user_ids[idx]), n=10)
    rss_after_100 = proc.memory_info().rss / 1024**2
    logger.info(f"RSS after 1 request: {rss_after_1:.1f}MB")
    logger.info(f"RSS after 100 requests: {rss_after_100:.1f}MB")
    title = os.environ.get("BENCHMARK_TITLE", "Run")
    write_benchmark_section(
        title=title,
        baseline_mb=baseline_mb,
        post_load_rss_mb=rss_mb,
        traced_current_mb=current / 1024**2,
        traced_peak_mb=peak / 1024**2,
        component_rows=rows,
        rss_after_1=rss_after_1,
        rss_after_100=rss_after_100,
    )

    yield


app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.middleware("http")
async def memory_middleware(request: Request, call_next):
    logger.info(f"{request.method} {request.url}")
    proc = psutil.Process(os.getpid())

    rss_before = proc.memory_info().rss
    current_before, _ = tracemalloc.get_traced_memory()
    t0 = time.time()

    response = await call_next(request)

    rss_after = proc.memory_info().rss
    current_after, peak_after = tracemalloc.get_traced_memory()

    logger.info(
        f"{request.url.path} | "
        f"RSS delta={(rss_after - rss_before) / 1024**2:.2f} MB | "
        f"Py delta={(current_after - current_before) / 1024**2:.2f} MB | "
        f"Peak={peak_after / 1024**2:.2f} MB | "
        f"{(time.time() - t0) * 1000:.1f} ms"
    )
    return response


@app.get("/")
async def root():
    return FileResponse(STATIC_DIR / "index.html")


def get_hybrid(request: Request) -> HybridRecommender:
    return request.app.state.hybrid


def get_catalog(request: Request) -> Catalog:
    return request.app.state.catalog


def get_movie_search(request: Request) -> MovieSearch:
    return request.app.state.movie_search


@app.get("/health")
async def health() -> dict[str, str]:
    model = getattr(app.state, "hybrid", None)
    if model and model.is_fitted:
        return {"status": "healthy"}
    return {"status": "not ready"}


@app.get("/popular")
async def popular(n: int = Query(default=10, ge=1, le=100)) -> PopularResponse:
    try:
        return PopularResponse(results=app.state.pop_precomputed[:n])
    except (AttributeError, KeyError, ValueError) as e:
        raise HTTPException(status_code=422, detail=str(e))


@app.post("/recommend")
async def recommend(
    req: RecommendRequest,
    hybrid: HybridRecommender = Depends(get_hybrid),
    catalog: Catalog = Depends(get_catalog),
) -> RecommendResponse:
    if not req.ratings:
        return RecommendResponse(results=app.state.pop_precomputed[: req.n])

    recs = hybrid.recommend_from_ratings(ratings=req.ratings, n=req.n, catalog=catalog)
    recs = hybrid.enrich(recs=recs, catalog=catalog)
    recs = [MovieResult(movie_id=j["movie_id"], title=j["title"], score=j["score"]) for j in recs]
    return RecommendResponse(results=recs)


@app.get("/search")
async def search(
    q: str = Query(min_length=1, max_length=100),
    limit: int = Query(10, ge=1, le=10),
    movie_search: MovieSearch = Depends(get_movie_search),
) -> SearchResponse:
    result = movie_search.search(q, limit=limit)
    return SearchResponse(results=result)
