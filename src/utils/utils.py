from pathlib import Path

import numpy as np

from src.paths.get_project_root import get_project_root


def conf(n: int | list[float], alpha: float = 10000.0, beta: float = 0.4):
    """
    Confidence weight for rating counts. Returns values in (0, 1), sigmoid in log space

    Movies with many ratings get weight close to 1,
    movies with few ratings get weight close to 0.

    Parameters
    ----------
    n : int or array-like
        Number of ratings
    alpha : float
        Half-confidence point — the count at which confidence equals 0.5
    beta : float
        Controls steepness. Lower = more gradual transition.

    Returns
    -------
    np.ndarray
        confidence weights in (0, 1)
    """
    n = np.asarray(n)
    return (n**beta) / (n**beta + alpha**beta)


def construct_path(path: Path) -> Path:
    """

    Parameters
    ----------
    path : path to construct

    Returns
    -------
    path from guaranteed root
    """
    return get_project_root() / path
