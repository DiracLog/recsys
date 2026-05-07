from enum import Enum

from pydantic import BaseModel, Field
from typing import Annotated


class RecommendStrategy(str, Enum):
    EPSILON_GREEDY = "epsilon_greedy"  # default: 80/20 explore
    WEIGHTED_SAMPLE = "weighted_sample"  # reroll: diverse
    # future: TRENDING = "trending"


class RecommendRequest(BaseModel):
    ratings: dict[
        Annotated[int, Field(gt=0)],
        Annotated[float, Field(ge=1.0, le=5.0)],
    ]
    n: int = Field(ge=1, le=20, default=5)
    reroll: RecommendStrategy = RecommendStrategy.EPSILON_GREEDY


class MovieResult(BaseModel):
    movie_id: int
    title: str
    score: float
    year: int | None = None  # for front serve
    genres: list[str] = []


class RecommendResponse(BaseModel):
    results: list[MovieResult]


class PopularResponse(BaseModel):
    results: list[MovieResult]


class SearchResponse(BaseModel):
    results: list[MovieResult]
