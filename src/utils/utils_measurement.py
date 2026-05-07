from datetime import datetime
from pathlib import Path
from pympler import asizeof

from src.models.data_schemas import Catalog

BENCHMARKS_PATH = Path("measurements/benchmarks.md")


def catalog_size_mb(cat: Catalog) -> float:
    size = 0.0

    size += cat.movies_df.estimated_size()
    size += cat.genre_matrix.nbytes
    size += cat.movie_ids.nbytes
    size += asizeof.asizeof(cat._movie_indx)  # was getsizeof
    size += asizeof.asizeof(cat._title_lookup)  # was getsizeof
    return size / 1024**2


def measure_components(named_objects, catalog_size_fn):
    rows = []
    for name, obj in named_objects:
        try:
            if name == "catalog":
                size_mb = catalog_size_fn(obj)
            else:
                size_mb = asizeof.asizeof(obj) / 1024**2
        except Exception as e:
            rows.append((name, None, "FAILED", str(e)))
            continue
        shape = getattr(obj, "shape", "—")
        dtype = getattr(obj, "dtype", "—")
        rows.append((name, size_mb, shape, dtype))
    return rows


def write_benchmark_section(
    title: str,
    baseline_mb: float,
    post_load_rss_mb: float,
    traced_current_mb: float,
    traced_peak_mb: float,
    component_rows: list,
    rss_after_1: float,
    rss_after_100: float,
    path: Path = BENCHMARKS_PATH,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = [
        f"\n## {title}",
        f"_Run: {timestamp}_\n",
        "| Component                          | Size      | Shape          | Dtype   |",
        "|------------------------------------|-----------|----------------|---------|",
        f"| Python+deps baseline               | {baseline_mb:>7.1f} MB | —              | —       |",
        f"| Traced memory (current)            | {traced_current_mb:>7.2f} MB | —              | —       |",
        f"| Traced memory (peak)               | {traced_peak_mb:>7.2f} MB | —              | —       |",
        f"| Post-load RSS                      | {post_load_rss_mb:>7.1f} MB | —              | —       |",
        f"| RSS diff                           | {post_load_rss_mb - baseline_mb:>7.1f} MB | —              | —     |",
        f"| RSS after gc.collect (1st)         | {rss_after_1:>7.1f} MB | —              | —       |",
        f"| RSS after gc.collect (100th)       | {rss_after_100:>7.1f} MB | —              | —       |",
    ]

    for name, size_mb, shape, dtype in component_rows:
        if size_mb is None:
            lines.append(f"| {name:<34} | FAILED    | {shape}        | —       |")
        else:
            lines.append(f"| {name:<34} | {size_mb:>7.1f} MB | {str(shape):<14} | {str(dtype):<7} |")

    with path.open("a") as f:
        f.write("\n".join(lines) + "\n")
