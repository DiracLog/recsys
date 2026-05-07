from pathlib import Path
from src.models.hybrid_recommender import HybridRecommender
from src.utils.utils import construct_path

path = construct_path(Path("artifacts/runs/latest/hybrid"))
h = HybridRecommender.load(path, mmap_mode="r")

print("U_ type:", type(h._collaborative.U_))
print("U_ flags.OWNDATA:", h._collaborative.U_.flags.owndata)
print("U_ base:", type(h._collaborative.U_.base) if h._collaborative.U_.base is not None else None)
