from logging import getLogger

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Depends, Request, Query

from src.api.dependencies import prepare_state
from src.api.movie_search import MovieSearch
from src.api.schemas import PopularResponse, RecommendRequest, RecommendResponse, MovieResult, SearchResponse
from src.models.data_schemas import Catalog
from src.models.hybrid_recommender import HybridRecommender

from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from src.logging.setup_logging import setup_logging

STATIC_DIR = Path(__file__).parent / "static"


setup_logging(log_file="api_seving.log")
logger = getLogger(__name__)


@asynccontextmanager
async def lifespan(application: FastAPI):
    hybrid, popularity, movie_search, pop_precomputed, catalog = prepare_state()
    application.state.hybrid = hybrid
    application.state.popularity = popularity
    application.state.movie_search = movie_search
    application.state.pop_precomputed = pop_precomputed
    application.state.catalog = catalog
    yield


app = FastAPI(lifespan=lifespan)


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


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
    return {"status": "ok"}


@app.get("/healthcheck")
async def health_check() -> dict[str, str]:
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
    req: RecommendRequest, hybrid: HybridRecommender = Depends(get_hybrid), catalog: Catalog = Depends(get_catalog)
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


#  uv run uvicorn src.api.app_test:app --reload
