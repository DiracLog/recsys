from rapidfuzz import process, fuzz

from src.api.schemas import MovieResult
from src.models.data_schemas import Catalog


class MovieSearch:
    def __init__(self):
        self._titles = None
        self._title_to_id = None

    def fit(self, catalog: Catalog):
        self._titles = catalog.movies_df["name"]
        self._title_to_id = dict(zip(self._titles, catalog.movies_df["movie_id"]))

    def search(self, query: str, limit: int = 10) -> list[MovieResult]:
        results = process.extract(query, self._titles, scorer=fuzz.WRatio, limit=limit)
        return [
            MovieResult(movie_id=self._title_to_id[title], title=title, score=float(score))
            for title, score, _ in results
        ]
