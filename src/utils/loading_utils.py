from src.config.config_schema import Config
from src.data.loaders import load_data, load_data_ml32
from src.data.preprocessor import Preprocessor
from src.models.data_schemas import Catalog, TrainingData
from src.utils.utils import construct_path

from typing import Callable


def prepare_data(cfg: Config) -> Callable[[Config], tuple[Catalog, TrainingData]]:
    """
    wrapper for different preprocessors

    Parameters
    ----------
    cfg : Config
        config object with paths to data and other parameters

    Returns
    -------
    Callable[[Config], tuple[Catalog, TrainingData]]
        function that takes cfg and returns Catalog and TrainingData objects

    Raises
    ------
    ValueError
        if no loader for specified mode is implemented
    """
    if cfg["data"]["mode"] == "ml1":
        return prepare_data_for_dev(cfg)
    elif cfg["data"]["mode"] == "ml32":
        return prepare_data_for_dev_ml32(cfg)
    else:
        raise ValueError(f"Loader for mode {cfg['data']['mode']} is not implmeneted yet.")


def prepare_data_for_dev(
    cfg: Config,
) -> tuple[Catalog, TrainingData]:
    """
    prepare_data_for_dev(ML1), loads data from paths specified in cfg

    Parameters
    ----------
    cfg : Config
        config object with paths to data and other parameters

    Returns
    -------
    tuple[Catalog, TrainingData]
        returns Catalog and TrainingData objects
    """
    movies_path = construct_path(cfg["data"]["movies_path"])
    ratings_path = construct_path(cfg["data"]["ratings_path"])
    users_path = construct_path(cfg["data"]["users_path"])

    movies_data, ratings_data, users_data = load_data(path_m=movies_path, path_r=ratings_path, path_u=users_path)
    preprocessor = Preprocessor()
    train_data, catalog = preprocessor.preprocess(movies_data, ratings_data)
    return catalog, train_data


def prepare_data_for_dev_ml32(cfg: Config) -> tuple[Catalog, TrainingData]:
    """
    prepare_data_for_dev_ml32(ML32), loads data from paths specified in cfg (polars)

    Parameters
    ----------
    cfg : Config
        config object with paths to data and other parameters

    Returns
    -------
    tuple[Catalog, TrainingData]
        returns Catalog and TrainingData objects
    """
    path_m = construct_path(cfg["data"]["movies_path"])
    path_r = construct_path(cfg["data"]["ratings_path"])
    movies_data, ratings_data, _ = load_data_ml32(path_m, path_r)
    preprocessor = Preprocessor()
    train_data, catalog = preprocessor.preprocess(movies_data, ratings_data)
    return catalog, train_data
