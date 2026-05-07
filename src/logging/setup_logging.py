import logging

from src.paths.get_project_root import get_project_root


def setup_logging(log_file: str = None) -> None:
    handlers = [logging.StreamHandler()]
    if log_file:
        log_path = get_project_root() / "logs" / log_file
        log_path.parent.mkdir(exist_ok=True, parents=True)
        handlers.append(logging.FileHandler(log_path))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
        force=True,
    )
