#!/usr/bin/env python3
"""Build grid-cell compound wetbulb heatwave and EUCLID lightning events."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import xarray as xr


PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from common.paths import COMPOUND_EUCLID_DIR, DATA_DIR, WETBULB_HEATWAVES_DIR
from scripts.pipeline.build_gridcell_compound_events import (
    build_heatwave_events,
    heatwave_path,
    nearest_index,
    open_sample_grid,
)


DEFAULT_EUCLID_DIR = DATA_DIR / "Euclid" / "monthly_nc"
DEFAULT_OUTPUT_DIR = COMPOUND_EUCLID_DIR
DEFAULT_METHODS = {
    "90p20": WETBULB_HEATWAVES_DIR / "90p20",
    "90p22": WETBULB_HEATWAVES_DIR / "90p22",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Combine 90p+absolute Tw heatwave events with EUCLID hourly gridded "
            "lightning, mapped to the nearest ERA5 heatwave grid cell."
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
    parser.add_argument("--start-year", type=int, default=2006)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument(
        "--post-window-hours",
        type=float,
        default=24.0,
        help="Hours after the heatwave end day included in the compound-event window.",
    )
    parser.add_argument(
        "--method",
        action="append",
        nargs=2,
        metavar=("LABEL", "DIR"),
        help=(
            "Heatwave method label and directory containing wetbulb_heatwave_<year>.nc. "
            "Can be supplied multiple times. Defaults to 90p20 and 90p22."
        ),
    )
    parser.add_argument(
        "--all-events-output",
        default="euclid_heatwave_lightning_event_stats_2006_2025.csv",
        help="CSV name for all heatwave events with EUCLID lightning-window statistics.",
    )
    parser.add_argument(
        "--compound-output",
        default="compound_euclid_heatwave_lightning_events_2006_2025.csv",
        help="CSV name for final compound heatwave-EUCLID-lightning events only.",
    )
    return parser.parse_args()


def method_paths(args: argparse.Namespace) -> dict[str, Path]:
    if args.method:
        return {label: Path(path) for label, path in args.method}
    return DEFAULT_METHODS


def euclid_month_paths(euclid_dir: Path, start_year: int, end_year: int, pattern: str) -> list[Path]:
    paths: list[Path] = []
    for year in range(start_year, end_year + 1):
        year_dir = euclid_dir / str(year)
        paths.extend(sorted(year_dir.glob(pattern)))
    if not paths:
        raise FileNotFoundError(f"No EUCLID monthly NetCDF files found in {euclid_dir}")
    return paths


def load_euclid_lightning_events(
    euclid_dir: Path,
    start_year: int,
    end_year: int,
    era5_latitudes: np.ndarray,
    era5_longitudes: np.ndarray,
    pattern: str = "euclid_flash_cg_ic_*.nc",
) -> pd.DataFrame:
    frames = []
    lat_min, lat_max = float(np.min(era5_latitudes)), float(np.max(era5_latitudes))
    lon_min, lon_max = float(np.min(era5_longitudes)), float(np.max(era5_longitudes))

    for path in euclid_month_paths(euclid_dir, start_year, end_year, pattern):
        print(f"Reading EUCLID lightning: {path}", flush=True)
        with xr.open_dataset(path) as ds:
            ds = ds.sel(latitude=slice(lat_min, lat_max), longitude=slice(lon_min, lon_max))
            flash = ds["flash_count"].values
            time_idx, lat_idx, lon_idx = np.nonzero(flash > 0)
            if len(time_idx) == 0:
                continue

            euclid_latitudes = ds["latitude"].values
            euclid_longitudes = ds["longitude"].values
            era5_lat_idx = nearest_index(era5_latitudes, euclid_latitudes[lat_idx])
            era5_lon_idx = nearest_index(era5_longitudes, euclid_longitudes[lon_idx])
            cell_id = era5_lat_idx.astype(np.int32) * len(era5_longitudes) + era5_lon_idx.astype(np.int32)

            frame = pd.DataFrame(
                {
                    "cell_id": cell_id,
                    "TIME_EVENT": pd.to_datetime(ds["valid_time"].values[time_idx], utc=True),
                    "flash_count": flash[time_idx, lat_idx, lon_idx].astype(np.int32),
                }
            )
            frames.append(
                frame.groupby(["cell_id", "TIME_EVENT"], as_index=False)
                .agg(
                    n_lightning_gridcells=("flash_count", "size"),
                    total_flash_count=("flash_count", "sum"),
                )
            )

    if not frames:
        return pd.DataFrame(columns=["cell_id", "TIME_EVENT", "n_lightning_gridcells", "total_flash_count"])

    lightning = (
        pd.concat(frames, ignore_index=True)
        .groupby(["cell_id", "TIME_EVENT"], as_index=False)
        .agg(
            n_lightning_gridcells=("n_lightning_gridcells", "sum"),
            total_flash_count=("total_flash_count", "sum"),
        )
        .sort_values(["cell_id", "TIME_EVENT"])
        .reset_index(drop=True)
    )
    return lightning


def attach_euclid_statistics(events: pd.DataFrame, lightning: pd.DataFrame, post_window_hours: float) -> pd.DataFrame:
    events = events.copy()
    events["has_lightning"] = False
    events["has_precip"] = False
    events["has_any_convection"] = False
    events["n_lightning"] = 0
    events["n_precip"] = 0
    events["n_lightning_gridcells"] = 0
    events["total_flash_count"] = 0
    # Must be tz-aware from the start: the per-row assignments below write
    # UTC-localized timestamps, which pandas >=2 refuses to put into a tz-naive
    # datetime64[ns] column (older pandas coerced silently).
    events["first_convective_time"] = pd.Series(pd.NaT, index=events.index, dtype="datetime64[ns, UTC]")

    lightning_groups = {
        cell_id: group[["TIME_EVENT", "n_lightning_gridcells", "total_flash_count"]].sort_values("TIME_EVENT")
        for cell_id, group in lightning.groupby("cell_id", sort=False)
    }

    post_window = pd.Timedelta(hours=post_window_hours)
    for cell_id, idx in events.groupby("cell_id").groups.items():
        group = lightning_groups.get(cell_id)
        if group is None or group.empty:
            continue

        times = group["TIME_EVENT"].to_numpy(dtype="datetime64[ns]")
        gridcell_counts = group["n_lightning_gridcells"].to_numpy(dtype=np.int64)
        flash_counts = group["total_flash_count"].to_numpy(dtype=np.int64)

        for row_idx in idx:
            start = events.at[row_idx, "start_time"].to_datetime64()
            window_end = (events.at[row_idx, "end_time"] + post_window).to_datetime64()
            left = np.searchsorted(times, start, side="left")
            right = np.searchsorted(times, window_end, side="right")
            if left == right:
                continue

            first_time = pd.Timestamp(times[left]).tz_localize("UTC")
            n_lightning_hours = int(right - left)
            n_lightning_gridcells = int(gridcell_counts[left:right].sum())
            total_flash_count = int(flash_counts[left:right].sum())

            events.at[row_idx, "has_lightning"] = True
            events.at[row_idx, "has_any_convection"] = True
            events.at[row_idx, "n_lightning"] = n_lightning_hours
            events.at[row_idx, "n_lightning_gridcells"] = n_lightning_gridcells
            events.at[row_idx, "total_flash_count"] = total_flash_count
            events.at[row_idx, "first_convective_time"] = first_time

    events["first_convective_time"] = pd.to_datetime(events["first_convective_time"], errors="coerce", utc=True)
    events["lag_hours"] = (
        (events["first_convective_time"] - events["start_time"])
        .dt.total_seconds()
        .div(3600.0)
    )
    return events


def format_output(events: pd.DataFrame) -> pd.DataFrame:
    out = events.copy()
    out["heatwave_event_id"] = (
        out["method"].astype(str)
        + "_lat"
        + out["grid_latitude"].map(lambda value: f"{value:.2f}")
        + "_lon"
        + out["grid_longitude"].map(lambda value: f"{value:.2f}")
        + "_event"
        + out["event_number"].map(lambda value: f"{int(value):04d}")
    )
    out["start_time"] = out["start_time"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    out["end_time"] = out["end_time"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    out["first_convective_time"] = out["first_convective_time"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    columns = [
        "heatwave_event_id",
        "start_time",
        "end_time",
        "duration",
        "max_tw",
        "has_lightning",
        "has_precip",
        "has_any_convection",
        "n_lightning",
        "n_precip",
        "n_lightning_gridcells",
        "total_flash_count",
        "first_convective_time",
        "lag_hours",
    ]
    return out[columns].sort_values(["start_time", "heatwave_event_id"]).reset_index(drop=True)


def write_method_outputs(
    all_events: pd.DataFrame,
    compound: pd.DataFrame,
    output_dir: Path,
    start_year: int,
    end_year: int,
    output_prefix: str,
) -> None:
    methods = sorted(all_events["heatwave_event_id"].str.extract(r"^(90p(?:[0-9]+)?)")[0].dropna().unique())
    for method in methods:
        all_method = all_events[all_events["heatwave_event_id"].str.startswith(method)].copy()
        compound_method = compound[compound["heatwave_event_id"].str.startswith(method)].copy()
        all_path = output_dir / f"{output_prefix}_heatwave_lightning_event_stats_{method}_{start_year}_{end_year}.csv"
        compound_path = output_dir / f"compound_{output_prefix}_heatwave_lightning_events_{method}_{start_year}_{end_year}.csv"
        all_method.to_csv(all_path, index=False, encoding="utf-8", na_rep="", lineterminator="\n")
        compound_method.to_csv(compound_path, index=False, encoding="utf-8", na_rep="", lineterminator="\n")
        print(f"Wrote {method} EUCLID heatwave event stats: {all_path} ({len(all_method)} rows)", flush=True)
        print(f"Wrote {method} EUCLID compound events: {compound_path} ({len(compound_method)} rows)", flush=True)


def main() -> None:
    args = parse_args()
    methods = method_paths(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    latitudes, longitudes = open_sample_grid(methods, args.start_year)
    lightning = load_euclid_lightning_events(
        Path(args.euclid_dir),
        args.start_year,
        args.end_year,
        latitudes,
        longitudes,
        args.euclid_pattern,
    )
    print(f"Loaded EUCLID hourly lightning records mapped to heatwave grid: {len(lightning)}", flush=True)

    heatwaves = build_heatwave_events(methods, args.start_year, args.end_year, latitudes, longitudes)
    print(f"Heatwave events before EUCLID matching: {len(heatwaves)}", flush=True)

    matched = attach_euclid_statistics(heatwaves, lightning, args.post_window_hours)
    all_events = format_output(matched)
    compound = all_events[all_events["has_any_convection"]].copy().reset_index(drop=True)

    all_path = output_dir / args.all_events_output
    compound_path = output_dir / args.compound_output
    all_events.to_csv(all_path, index=False, encoding="utf-8", na_rep="", lineterminator="\n")
    compound.to_csv(compound_path, index=False, encoding="utf-8", na_rep="", lineterminator="\n")
    write_method_outputs(all_events, compound, output_dir, args.start_year, args.end_year, args.output_prefix)

    print(f"Wrote all EUCLID heatwave event stats: {all_path} ({len(all_events)} rows)", flush=True)
    print(f"Wrote final EUCLID compound events: {compound_path} ({len(compound)} rows)", flush=True)
    if not compound.empty:
        print("EUCLID compound event totals by method:", flush=True)
        print(compound.assign(method=compound["heatwave_event_id"].str.extract(r"^(90p(?:[0-9]+)?)")[0]).groupby("method").size().to_string(), flush=True)


if __name__ == "__main__":
    main()
