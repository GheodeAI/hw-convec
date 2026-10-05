#!/usr/bin/env python3
"""Build regional compound wetbulb heatwave and EUCLID lightning events."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import xarray as xr
from scipy import ndimage


PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from common.paths import COMPOUND_EUCLID_DIR, COMPOUND_TABLES_DIR, DATA_DIR, WETBULB_HEATWAVES_DIR
from scripts.pipeline.build_gridcell_compound_events_euclid import load_euclid_lightning_events
from scripts.pipeline.build_regional_compound_events import (
    connectivity_structure,
    heatwave_path,
    load_grid,
    load_heatwave_mask,
)


DEFAULT_EUCLID_DIR = DATA_DIR / "Euclid" / "monthly_nc"
DEFAULT_OUTPUT_DIR = COMPOUND_EUCLID_DIR
DEFAULT_TABLES_DIR = COMPOUND_TABLES_DIR
DEFAULT_METHODS = {
    "90p": WETBULB_HEATWAVES_DIR / "90p",
    "90p20": WETBULB_HEATWAVES_DIR / "90p20",
    "90p22": WETBULB_HEATWAVES_DIR / "90p22",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge spatially contiguous and temporally overlapping gridded heatwaves "
            "into regional heatwave events, then flag compound events with EUCLID "
            "hourly gridded lightning in the regional event window."
        )
    )
    parser.add_argument("--euclid-dir", default=str(DEFAULT_EUCLID_DIR))
    parser.add_argument(
        "--euclid-pattern",
        default="euclid_flash_cg_ic_*.nc",
        help="Monthly EUCLID NetCDF filename glob relative to each yearly directory.",
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--output-prefix",
        default="euclid",
        help="Prefix used in output filenames.",
    )
    parser.add_argument(
        "--tables-dir",
        default=str(DEFAULT_TABLES_DIR),
        help="Output directory for non-event summary tables.",
    )
    parser.add_argument("--start-year", type=int, default=2006)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument(
        "--post-window-hours",
        type=float,
        default=24.0,
        help="Hours after the heatwave end day included in the compound-event window.",
    )
    parser.add_argument(
        "--min-lightning-gridcell-fraction",
        type=float,
        default=0.10,
        help=(
            "Minimum fraction of unique regional heatwave footprint grid cells that "
            "must contain lightning during the event window (default: 0.10)."
        ),
    )
    parser.add_argument(
        "--spatial-connectivity",
        type=int,
        choices=(4, 8),
        default=8,
        help="Spatial adjacency on each daily grid. Temporal adjacency is same-cell consecutive days.",
    )
    parser.add_argument(
        "--method",
        action="append",
        nargs=2,
        metavar=("LABEL", "DIR"),
        help="Method label and directory containing wetbulb_heatwave_<year>.nc. Defaults to 90p, 90p20, 90p22.",
    )
    return parser.parse_args()


def method_paths(args: argparse.Namespace) -> dict[str, Path]:
    if args.method:
        return {label: Path(path) for label, path in args.method}
    return DEFAULT_METHODS


def lightning_groups(lightning: pd.DataFrame) -> dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    groups = {}
    for cell_id, group in lightning.groupby("cell_id", sort=False):
        ordered = group.sort_values("TIME_EVENT")
        groups[int(cell_id)] = (
            ordered["TIME_EVENT"].to_numpy(dtype="datetime64[ns]"),
            ordered["n_lightning_gridcells"].to_numpy(dtype=np.int64),
            ordered["total_flash_count"].to_numpy(dtype=np.int64),
        )
    return groups


def component_has_euclid_lightning(
    cell_ids: np.ndarray,
    start_time: pd.Timestamp,
    window_end: pd.Timestamp,
    groups: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> tuple[int, int, int, int, pd.Timestamp | None]:
    n_lightning_cell_hours = 0
    n_lightning_gridcells = 0
    n_lightning_event_gridcells = 0
    total_flash_count = 0
    first_time = None
    start64 = start_time.to_datetime64()
    end64 = window_end.to_datetime64()

    for cell_id in cell_ids:
        group = groups.get(int(cell_id))
        if group is None:
            continue

        times, gridcell_counts, flash_counts = group
        left = np.searchsorted(times, start64, side="left")
        right = np.searchsorted(times, end64, side="right")
        if left == right:
            continue

        # Each heatwave-footprint cell contributes at most once to the spatial
        # coverage criterion, irrespective of how many lightning hours it has.
        n_lightning_event_gridcells += 1
        n_lightning_cell_hours += int(right - left)
        n_lightning_gridcells += int(gridcell_counts[left:right].sum())
        total_flash_count += int(flash_counts[left:right].sum())
        hit_time = pd.Timestamp(times[left]).tz_localize("UTC")
        if first_time is None or hit_time < first_time:
            first_time = hit_time

    return (
        n_lightning_cell_hours,
        n_lightning_gridcells,
        n_lightning_event_gridcells,
        total_flash_count,
        first_time,
    )


def regional_events_for_method(
    method: str,
    method_dir: Path,
    start_year: int,
    end_year: int,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    lightning_by_cell: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]],
    post_window_hours: float,
    spatial_connectivity: int,
    min_lightning_gridcell_fraction: float,
) -> pd.DataFrame:
    mask, times = load_heatwave_mask(method_dir, start_year, end_year)
    labels, n_labels = ndimage.label(mask, structure=connectivity_structure(spatial_connectivity))
    objects = ndimage.find_objects(labels)
    print(f"{method}: regional heatwave events={n_labels}", flush=True)

    rows = []
    post_window = pd.Timedelta(hours=post_window_hours)
    nlon = len(longitudes)
    for label_id, slices in enumerate(objects, start=1):
        if slices is None:
            continue
        time_slice, lat_slice, lon_slice = slices
        sub = labels[slices] == label_id
        t_idx, y_idx, x_idx = np.nonzero(sub)
        absolute_t = t_idx + time_slice.start
        absolute_y = y_idx + lat_slice.start
        absolute_x = x_idx + lon_slice.start
        cell_ids = np.unique(absolute_y.astype(np.int64) * nlon + absolute_x.astype(np.int64))

        start_time = pd.Timestamp(times[int(absolute_t.min())]).tz_localize("UTC")
        end_time = pd.Timestamp(times[int(absolute_t.max())]).tz_localize("UTC") + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        window_end = end_time + post_window
        (
            n_lightning,
            n_lightning_gridcells,
            n_lightning_event_gridcells,
            total_flash_count,
            first_convective_time,
        ) = component_has_euclid_lightning(cell_ids, start_time, window_end, lightning_by_cell)
        lightning_event_gridcell_fraction = n_lightning_event_gridcells / len(cell_ids)
        n_required_lightning_event_gridcells = max(
            1,
            int(np.ceil(min_lightning_gridcell_fraction * len(cell_ids) - 1.0e-12)),
        )
        has_lightning = n_lightning_event_gridcells >= n_required_lightning_event_gridcells
        first_time_text = ""
        lag_hours = np.nan
        if first_convective_time is not None:
            first_time_text = first_convective_time.strftime("%Y-%m-%dT%H:%M:%SZ")
            lag_hours = (first_convective_time - start_time).total_seconds() / 3600.0

        rows.append(
            {
                "regional_heatwave_event_id": f"{method}_regional_event{label_id:06d}",
                "method": method,
                "start_time": start_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "end_time": end_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "duration": int(absolute_t.max() - absolute_t.min() + 1),
                "n_gridcell_days": int(sub.sum()),
                "n_gridcells": int(len(cell_ids)),
                "min_latitude": float(latitudes[absolute_y].min()),
                "max_latitude": float(latitudes[absolute_y].max()),
                "min_longitude": float(longitudes[absolute_x].min()),
                "max_longitude": float(longitudes[absolute_x].max()),
                "has_lightning": has_lightning,
                "has_precip": False,
                "has_any_convection": has_lightning,
                "n_lightning": n_lightning,
                "n_precip": 0,
                "n_lightning_gridcells": n_lightning_gridcells,
                "n_lightning_event_gridcells": n_lightning_event_gridcells,
                "lightning_event_gridcell_fraction": lightning_event_gridcell_fraction,
                "n_required_lightning_event_gridcells": n_required_lightning_event_gridcells,
                "min_lightning_gridcell_fraction": min_lightning_gridcell_fraction,
                "total_flash_count": total_flash_count,
                "first_convective_time": first_time_text,
                "lag_hours": lag_hours,
            }
        )

    return pd.DataFrame(rows)


def write_method_outputs(
    events: pd.DataFrame,
    output_dir: Path,
    method: str,
    start_year: int,
    end_year: int,
    output_prefix: str,
) -> tuple[int, int]:
    event_path = output_dir / f"regional_{output_prefix}_heatwave_lightning_events_{method}_{start_year}_{end_year}.csv"
    compound_path = output_dir / f"compound_regional_{output_prefix}_heatwave_lightning_events_{method}_{start_year}_{end_year}.csv"
    compound = events[events["has_any_convection"]].copy()
    events.to_csv(event_path, index=False, encoding="utf-8", na_rep="", lineterminator="\n")
    compound.to_csv(compound_path, index=False, encoding="utf-8", na_rep="", lineterminator="\n")
    print(f"Wrote {event_path} ({len(events)} rows)", flush=True)
    print(f"Wrote {compound_path} ({len(compound)} rows)", flush=True)
    return len(events), len(compound)


def main() -> None:
    args = parse_args()
    if not 0.0 < args.min_lightning_gridcell_fraction <= 1.0:
        raise ValueError("--min-lightning-gridcell-fraction must be in (0, 1].")
    methods = method_paths(args)
    output_dir = Path(args.output_dir)
    tables_dir = Path(args.tables_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    latitudes, longitudes = load_grid(next(iter(methods.values())), args.start_year)
    lightning = load_euclid_lightning_events(
        Path(args.euclid_dir),
        args.start_year,
        args.end_year,
        latitudes,
        longitudes,
        args.euclid_pattern,
    )
    lightning_by_cell = lightning_groups(lightning)
    print(f"Loaded EUCLID hourly lightning records mapped to heatwave grid: {len(lightning)}", flush=True)
    print(f"EUCLID heatwave-grid cells with lightning: {len(lightning_by_cell)}", flush=True)

    summary_rows = []
    for method, method_dir in methods.items():
        events = regional_events_for_method(
            method,
            method_dir,
            args.start_year,
            args.end_year,
            latitudes,
            longitudes,
            lightning_by_cell,
            args.post_window_hours,
            args.spatial_connectivity,
            args.min_lightning_gridcell_fraction,
        )
        n_events, n_compound = write_method_outputs(
            events,
            output_dir,
            method,
            args.start_year,
            args.end_year,
            args.output_prefix,
        )
        summary_rows.append(
            {
                "method": method,
                "all_regional_heatwave_events": n_events,
                "compound_regional_heatwave_events": n_compound,
                "compound_regional_heatwave_event_fraction": n_compound / n_events if n_events else np.nan,
                "compound_regional_heatwave_event_percent": n_compound / n_events * 100.0 if n_events else np.nan,
            }
        )

    summary = pd.DataFrame(summary_rows)
    summary_path = tables_dir / f"regional_{args.output_prefix}_heatwave_lightning_summary_{args.start_year}_{args.end_year}.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8", lineterminator="\n")
    print(summary.to_string(index=False), flush=True)
    print(f"Wrote {summary_path}", flush=True)


if __name__ == "__main__":
    main()
