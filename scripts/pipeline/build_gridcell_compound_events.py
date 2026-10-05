import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import xarray as xr


PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from common.paths import COMPOUND_ESWD_DIR, ESWD_DIR, WETBULB_HEATWAVES_DIR

DEFAULT_ESWD = ESWD_DIR / "eswd_selected_2006_2025.csv"
DEFAULT_OUTPUT_DIR = COMPOUND_ESWD_DIR
DEFAULT_METHODS = {
    "90p20": WETBULB_HEATWAVES_DIR / "90p20",
    "90p22": WETBULB_HEATWAVES_DIR / "90p22",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Combine 90p+absolute Tw heatwave events with ESWD LIGHTNING/PRECIP "
            "records mapped to the nearest ERA5 grid cell."
        )
    )
    parser.add_argument("--eswd", default=str(DEFAULT_ESWD), help="Filtered ESWD CSV with ID/TYPE_EVENT/TIME_EVENT/LATITUDE/LONGITUDE.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
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
        default="eswd_heatwave_convection_event_stats_2006_2025.csv",
        help="CSV name for all heatwave events with convection-window statistics.",
    )
    parser.add_argument(
        "--compound-output",
        default="compound_eswd_heatwave_convection_events_2006_2025.csv",
        help="CSV name for final compound heatwave-convection events only.",
    )
    return parser.parse_args()


def method_paths(args):
    if args.method:
        return {label: Path(path) for label, path in args.method}
    return DEFAULT_METHODS


def heatwave_path(method_dir, year):
    return Path(method_dir) / f"wetbulb_heatwave_{year}.nc"


def open_sample_grid(methods, start_year):
    first_dir = next(iter(methods.values()))
    path = heatwave_path(first_dir, start_year)
    if not path.exists():
        raise FileNotFoundError(f"Missing heatwave file: {path}")
    with xr.open_dataset(path) as ds:
        return ds["latitude"].values, ds["longitude"].values


def nearest_index(values, targets):
    values = np.asarray(values)
    targets = np.asarray(targets)
    if values[0] > values[-1]:
        ascending = values[::-1]
        idx = np.searchsorted(ascending, targets)
        idx = np.clip(idx, 1, len(ascending) - 1)
        left = ascending[idx - 1]
        right = ascending[idx]
        asc_idx = np.where(np.abs(targets - left) <= np.abs(targets - right), idx - 1, idx)
        return len(values) - 1 - asc_idx

    idx = np.searchsorted(values, targets)
    idx = np.clip(idx, 1, len(values) - 1)
    left = values[idx - 1]
    right = values[idx]
    return np.where(np.abs(targets - left) <= np.abs(targets - right), idx - 1, idx)


def load_convection_events(eswd_path, latitudes, longitudes):
    usecols = ["TYPE_EVENT", "TIME_EVENT", "LATITUDE", "LONGITUDE"]
    df = pd.read_csv(eswd_path, usecols=usecols)
    df["TIME_EVENT"] = pd.to_datetime(df["TIME_EVENT"], errors="coerce", utc=True)
    df["LATITUDE"] = pd.to_numeric(df["LATITUDE"], errors="coerce")
    df["LONGITUDE"] = pd.to_numeric(df["LONGITUDE"], errors="coerce")
    df = df.dropna(subset=["TIME_EVENT", "LATITUDE", "LONGITUDE"])
    df = df[df["TYPE_EVENT"].isin(["LIGHTNING", "PRECIP"])].copy()

    lat_min, lat_max = float(np.min(latitudes)), float(np.max(latitudes))
    lon_min, lon_max = float(np.min(longitudes)), float(np.max(longitudes))
    in_domain = (
        df["LATITUDE"].between(lat_min, lat_max)
        & df["LONGITUDE"].between(lon_min, lon_max)
    )
    outside = int((~in_domain).sum())
    df = df[in_domain].copy()

    lat_idx = nearest_index(latitudes, df["LATITUDE"].to_numpy())
    lon_idx = nearest_index(longitudes, df["LONGITUDE"].to_numpy())
    df["lat_idx"] = lat_idx.astype(np.int32)
    df["lon_idx"] = lon_idx.astype(np.int32)
    df["cell_id"] = df["lat_idx"] * len(longitudes) + df["lon_idx"]
    df = df.sort_values(["cell_id", "TIME_EVENT"]).reset_index(drop=True)
    return df, outside


def yearly_heatwave_fragments(method_label, method_dir, year, nlon):
    path = heatwave_path(method_dir, year)
    if not path.exists():
        raise FileNotFoundError(f"Missing heatwave file: {path}")

    with xr.open_dataset(path) as ds:
        event_id = ds["heatwave_event_id"].values
        if not np.any(event_id > 0):
            return pd.DataFrame()

        tw = ds["tw_daily_max"].values
        times = pd.to_datetime(ds["time"].values)

    nt, nlat, nlon_file = event_id.shape
    if nlon_file != nlon:
        raise ValueError(f"Unexpected longitude size in {path}: {nlon_file} != {nlon}")

    event_flat = event_id.reshape(nt, -1)
    tw_flat = tw.reshape(nt, -1)
    time_idx, cell_idx = np.nonzero(event_flat > 0)

    data = pd.DataFrame(
        {
            "method": method_label,
            "cell_id": cell_idx.astype(np.int32),
            "event_number": event_flat[time_idx, cell_idx].astype(np.int16),
            "time": times[time_idx],
            "tw_daily_max": tw_flat[time_idx, cell_idx].astype(np.float32),
        }
    )
    grouped = (
        data.groupby(["method", "cell_id", "event_number"], sort=False)
        .agg(
            start_time=("time", "min"),
            end_day=("time", "max"),
            duration=("time", "size"),
            max_tw=("tw_daily_max", "max"),
        )
        .reset_index()
    )
    return grouped


def build_heatwave_events(methods, start_year, end_year, latitudes, longitudes):
    fragments = []
    nlon = len(longitudes)
    for method_label, method_dir in methods.items():
        for year in range(start_year, end_year + 1):
            print(f"Reading heatwave events: {method_label} {year}", flush=True)
            yearly = yearly_heatwave_fragments(method_label, method_dir, year, nlon)
            if not yearly.empty:
                fragments.append(yearly)

    if not fragments:
        raise RuntimeError("No heatwave events found.")

    events = pd.concat(fragments, ignore_index=True)
    events = (
        events.groupby(["method", "cell_id", "event_number"], sort=False)
        .agg(
            start_time=("start_time", "min"),
            end_day=("end_day", "max"),
            duration=("duration", "sum"),
            max_tw=("max_tw", "max"),
        )
        .reset_index()
    )
    events["lat_idx"] = (events["cell_id"] // len(longitudes)).astype(np.int32)
    events["lon_idx"] = (events["cell_id"] % len(longitudes)).astype(np.int32)
    events["grid_latitude"] = latitudes[events["lat_idx"].to_numpy()]
    events["grid_longitude"] = longitudes[events["lon_idx"].to_numpy()]
    events["start_time"] = pd.to_datetime(events["start_time"], utc=True)
    # Daily heatwave dates represent full days. Use the end of the final heatwave day.
    events["end_time"] = pd.to_datetime(events["end_day"], utc=True) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    events = events.drop(columns=["end_day"])
    return events


def attach_convection_statistics(events, convection, post_window_hours):
    events = events.copy()
    events["has_lightning"] = False
    events["has_precip"] = False
    events["has_any_convection"] = False
    events["n_lightning"] = 0
    events["n_precip"] = 0
    events["first_convective_time"] = pd.NaT

    conv_groups = {
        cell_id: group[["TIME_EVENT", "TYPE_EVENT"]].sort_values("TIME_EVENT")
        for cell_id, group in convection.groupby("cell_id", sort=False)
    }

    post_window = pd.Timedelta(hours=post_window_hours)
    for cell_id, idx in events.groupby("cell_id").groups.items():
        conv = conv_groups.get(cell_id)
        if conv is None or conv.empty:
            continue

        conv_times = conv["TIME_EVENT"].to_numpy(dtype="datetime64[ns]")
        conv_types = conv["TYPE_EVENT"].to_numpy()
        for row_idx in idx:
            start = events.at[row_idx, "start_time"].to_datetime64()
            window_end = (events.at[row_idx, "end_time"] + post_window).to_datetime64()
            left = np.searchsorted(conv_times, start, side="left")
            right = np.searchsorted(conv_times, window_end, side="right")
            if left == right:
                continue

            types = conv_types[left:right]
            n_lightning = int(np.count_nonzero(types == "LIGHTNING"))
            n_precip = int(np.count_nonzero(types == "PRECIP"))
            first_time = pd.Timestamp(conv_times[left]).tz_localize("UTC")

            events.at[row_idx, "n_lightning"] = n_lightning
            events.at[row_idx, "n_precip"] = n_precip
            events.at[row_idx, "has_lightning"] = n_lightning > 0
            events.at[row_idx, "has_precip"] = n_precip > 0
            events.at[row_idx, "has_any_convection"] = (n_lightning + n_precip) > 0
            events.at[row_idx, "first_convective_time"] = first_time

    events["first_convective_time"] = pd.to_datetime(events["first_convective_time"], errors="coerce", utc=True)
    events["lag_hours"] = (
        (events["first_convective_time"] - events["start_time"])
        .dt.total_seconds()
        .div(3600.0)
    )
    return events


def format_output(events):
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
        "first_convective_time",
        "lag_hours",
    ]
    return out[columns].sort_values(["start_time", "heatwave_event_id"]).reset_index(drop=True)


def write_method_outputs(all_events, compound, output_dir, start_year, end_year):
    methods = sorted(all_events["heatwave_event_id"].str.extract(r"^(90p[0-9]+)")[0].dropna().unique())
    for method in methods:
        all_method = all_events[all_events["heatwave_event_id"].str.startswith(method)].copy()
        compound_method = compound[compound["heatwave_event_id"].str.startswith(method)].copy()
        all_path = output_dir / f"eswd_heatwave_convection_event_stats_{method}_{start_year}_{end_year}.csv"
        compound_path = output_dir / f"compound_eswd_heatwave_convection_events_{method}_{start_year}_{end_year}.csv"
        all_method.to_csv(all_path, index=False, encoding="utf-8", na_rep="", lineterminator="\n")
        compound_method.to_csv(compound_path, index=False, encoding="utf-8", na_rep="", lineterminator="\n")
        print(f"Wrote {method} heatwave event stats: {all_path} ({len(all_method)} rows)", flush=True)
        print(f"Wrote {method} compound events: {compound_path} ({len(compound_method)} rows)", flush=True)


def main():
    args = parse_args()
    methods = method_paths(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    latitudes, longitudes = open_sample_grid(methods, args.start_year)
    convection, outside_count = load_convection_events(args.eswd, latitudes, longitudes)
    print(f"Loaded convection records in heatwave grid domain: {len(convection)}", flush=True)
    print(f"Excluded convection records outside heatwave grid domain: {outside_count}", flush=True)

    heatwaves = build_heatwave_events(methods, args.start_year, args.end_year, latitudes, longitudes)
    print(f"Heatwave events before convection matching: {len(heatwaves)}", flush=True)

    matched = attach_convection_statistics(heatwaves, convection, args.post_window_hours)
    all_events = format_output(matched)
    compound = all_events[all_events["has_any_convection"]].copy().reset_index(drop=True)

    all_path = output_dir / args.all_events_output
    compound_path = output_dir / args.compound_output
    all_events.to_csv(all_path, index=False, encoding="utf-8", na_rep="", lineterminator="\n")
    compound.to_csv(compound_path, index=False, encoding="utf-8", na_rep="", lineterminator="\n")
    write_method_outputs(all_events, compound, output_dir, args.start_year, args.end_year)

    print(f"Wrote all heatwave event stats: {all_path} ({len(all_events)} rows)", flush=True)
    print(f"Wrote final compound events: {compound_path} ({len(compound)} rows)", flush=True)
    if not compound.empty:
        print("Compound counts by has_lightning/has_precip:", flush=True)
        print(compound.groupby(["has_lightning", "has_precip"]).size().to_string(), flush=True)


if __name__ == "__main__":
    main()
