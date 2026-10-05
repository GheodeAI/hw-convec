"""Shared output helpers for compact, predictable file names."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable


SHORT_TOKENS = (
    ("compound_noncompound", "cmp_ncmp"),
    ("noncompound", "ncmp"),
    ("compound", "cmp"),
    ("climatological_anomaly_curves", "clim_anom"),
    ("heatwave_nonheatwave", "hw_nonhw"),
    ("wetbulb_heatwave", "wb_hw"),
    ("heatwave", "hw"),
    ("convection", "conv"),
    ("lightning", "ltg"),
    ("enhancement", "enh"),
    ("environment", "env"),
    ("regional", "reg"),
    ("gridcell", "grid"),
    ("spatial_stats", "spatial"),
    ("historical_spatial", "hist_spatial"),
    ("annual_timeseries", "annual"),
    ("event_location_density", "loc_density"),
    ("mean_days_intensity", "mean_days_int"),
    ("first_lightning", "first_ltg"),
    ("early_middle_late_post", "phases"),
    ("probability", "prob"),
    ("characteristics", "chars"),
    ("difference", "diff"),
    ("summary", "sum"),
    ("anomalies", "anom"),
    ("anomaly", "anom"),
    ("events", "evts"),
    ("event", "evt"),
    ("variables", "vars"),
    ("variable", "var"),
    ("occurrence", "occ"),
    ("intensity", "int"),
    ("ratio", "ratio"),
)


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def year_span(start_year: int | str, end_year: int | str) -> str:
    return f"{start_year}-{end_year}"


def month_tag(months: Iterable[int], prefix: str = "m") -> str:
    values = sorted({int(month) for month in months})
    if not values:
        return f"{prefix}all"
    if values == list(range(values[0], values[-1] + 1)):
        return f"{prefix}{values[0]:02d}-{values[-1]:02d}"
    return prefix + "".join(f"{month:02d}" for month in values)


def compact_stem(stem: str) -> str:
    """Shorten verbose output stems while preserving their main tokens."""
    short = stem
    short = re.sub(r"_months((?:\d{2})+)", lambda m: "_" + _compact_month_digits(m.group(1)), short)
    short = re.sub(r"(?<!\d)(19|20)\d{2}_(19|20)\d{2}(?!\d)", lambda m: m.group(0).replace("_", "-"), short)
    for old, new in SHORT_TOKENS:
        short = short.replace(old, new)
    short = re.sub(r"_+", "_", short)
    return short.strip("_")


def figure_stem(stem: str) -> str:
    """Shorten figure stems and drop method/period tags used by data products."""
    short = compact_stem(stem)
    short = re.sub(r"(^|_)90p20(?=_|$)", r"\1", short)
    short = re.sub(r"(^|_)2006[_-]2025(?=_|$)", r"\1", short)
    short = re.sub(r"(^|_)m04-09(?=_|$)", r"\1", short)
    short = re.sub(r"_+", "_", short)
    return short.strip("_")


def _compact_month_digits(digits: str) -> str:
    months = [int(digits[index : index + 2]) for index in range(0, len(digits), 2)]
    return month_tag(months)


def output_path(directory: str | Path, stem: str, suffix: str) -> Path:
    return ensure_dir(directory) / f"{compact_stem(stem)}{suffix}"
