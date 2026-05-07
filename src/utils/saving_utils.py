import json
import sys
from pathlib import Path


from src.config.config_schema import Config
from src.models.data_schemas import TrainingData


def set_latest(runs_dir: Path, run_id: str) -> None:
    """
    Sets simlink to latest run or stores name of latest run in latest.txt

    Parameters
    ----------
    runs_dir : Path
        pth to the directory containing the runs
    run_id: string
        Unique identifier of the run

    """
    if sys.platform == "win32":
        (runs_dir / "latest.txt").write_text(run_id)
    else:
        latest = runs_dir / "latest"
        if latest.is_symlink() or latest.exists():
            latest.unlink()
        latest.symlink_to(run_id)  # relative


def resolve_latest(runs_dir: Path) -> Path:
    """Gets the path of patest run (simlink or real path for win32)"""
    if sys.platform == "win32":
        run_id = (runs_dir / "latest.txt").read_text().strip()
        return runs_dir / run_id
    return runs_dir / "latest"


def write_run_meta(
    run_dir: Path, run_id: str, timings: dict[str, float], train_data: TrainingData, cfg: Config, SCHEMA_VERSION: int
):
    """Write metadata of latest run for further reference"""

    data = {
        "run_id": run_id,
        "schema_version": SCHEMA_VERSION,
        "fit_seconds": timings,
        "data": {
            "n_users": int(len(train_data.user_ids)),
            "n_movies": int(len(train_data.movie_ids)),
            "n_ratings": int(train_data.ratings_matrix.getnnz()),  # not na counts
        },
        "hyperparams": {
            "content": cfg["models"]["content"],
            "collaborative": cfg["models"]["collaborative"],
            "popularity": cfg["models"]["popularity"],
            "hybrid": cfg["models"]["hybrid"],
            "preprocessor": cfg["preprocessing"],
        },
    }

    with (run_dir / "metadata.json").open("w") as f:
        json.dump(data, f, indent=2, default=str)
