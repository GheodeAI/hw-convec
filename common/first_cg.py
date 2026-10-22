"""Read first-CG environment values from their monthly Parquet partitions."""
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


MONTHS = range(3, 11)


def monthly_value_files(run):
    directory = Path(run) / "first_cg/data"
    files = [directory / f"first_cg_values_m{month:02d}.parquet" for month in MONTHS]
    missing = [str(p) for p in files if not p.is_file()]
    if missing:
        raise FileNotFoundError("Incomplete first-CG monthly data: " + ", ".join(missing))
    return files


def read_first_cg_values(run, columns=None, filters=None):
    """Project columns and optionally filter before loading the monthly dataset."""
    return pd.read_parquet(monthly_value_files(run), columns=columns, filters=filters)


def validate_first_cg_values(run):
    """Validate row count and unique event/hour keys without loading all fields."""
    files = monthly_value_files(run)
    anchors = pq.read_metadata(Path(run) / "first_cg/data/confirmed_anchors.parquet").num_rows
    rows = sum(pq.read_metadata(p).num_rows for p in files)
    if rows != anchors * 9:
        raise ValueError("Missing first-CG month products")
    keys = read_first_cg_values(run, columns=["kind", "id", "relative_hour"])
    if keys.duplicated(["kind", "id", "relative_hour"]).any():
        raise ValueError("Duplicate first-CG rows")
    return rows
