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

from common.paths import COMPOUND_ESWD_DIR, COMPOUND_TABLES_DIR, ESWD_DIR, WETBULB_HEATWAVES_DIR

DEFAULT_ESWD = ESWD_DIR / "eswd_selected_2006_2025.csv"
DEFAULT_OUTPUT_DIR = COMPOUND_ESWD_DIR
DEFAULT_TABLES_DIR = COMPOUND_TABLES_DIR
DEFAULT_METHODS = {
    "90p": WETBULB_HEATWAVES_DIR / "90p",
    "90p20": WETBULB_HEATWAVES_DIR / "90p20",
    "90p22": WETBULB_HEATWAVES_DIR / "90p22",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Merge spatially contiguous and temporally overlapping gridded heatwaves "
            "into regional heatwave events, then flag compound events with ESWD "
            "LIGHTNING/PRECIP in the regional event window."
        )
    )
    parser.add_argument("--eswd", default=str(DEFAULT_ESWD))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--tables-dir",
        default=str(DEFAULT_TABLES_DIR),
        help="Output directory for non-event summary tables.",
    )
    parser.add_argument("--start-year", type=int, default=2006)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--post-window-hours", type=float, default=24.0)
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


def method_paths(args):
    if args.method:
        return {label: Path(path) for label, path in args.method}
    return DEFAULT_METHODS


def heatwave_path(method_dir, year):
    return Path(method_dir) / f"wetbulb_heatwave_{year}.nc"


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


def load_grid(sample_method_dir, start_year):
    with xr.open_dataset(heatwave_path(sample_method_dir, start_year)) as ds:
        return ds["latitude"].values, ds["longitude"].values


def load_convection(eswd_path, latitudes, longitudes):
    df = pd.read_csv(eswd_path, usecols=["TYPE_EVENT", "TIME_EVENT", "LATITUDE", "LONGITUDE"])
    df["TIME_EVENT"] = pd.to_datetime(df["TIME_EVENT"], errors="coerce", utc=True)
    df["LATITUDE"] = pd.to_numeric(df["LATITUDE"], errors="coerce")
    df["LONGITUDE"] = pd.to_numeric(df["LONGITUDE"], errors="coerce")
    df = df.dropna(subset=["TIME_EVENT", "LATITUDE", "LONGITUDE"])
    df = df[df["TYPE_EVENT"].isin(["LIGHTNING", "PRECIP"])].copy()

    in_domain = (
        df["LATITUDE"].between(float(np.min(latitudes)), float(np.max(latitudes)))
        & df["LONGITUDE"].between(float(np.min(longitudes)), float(np.max(longitudes)))
    )
    outside = int((~in_domain).sum())
    df = df[in_domain].copy()

    df["lat_idx"] = nearest_index(latitudes, df["LATITUDE"].to_numpy()).astype(np.int32)
    df["lon_idx"] = nearest_index(longitudes, df["LONGITUDE"].to_numpy()).astype(np.int32)
    df["cell_id"] = df["lat_idx"] * len(longitudes) + df["lon_idx"]
    df = df.sort_values(["cell_id", "TIME_EVENT"]).reset_index(drop=True)
    return df, outside


def connectivity_structure(spatial_connectivity):
    structure = np.zeros((3, 3, 3), dtype=bool)
    structure[1, 1, 1] = True
    structure[0, 1, 1] = True
    structure[2, 1, 1] = True
    structure[1, 0, 1] = True
    structure[1, 2, 1] = True
    structure[1, 1, 0] = True
    structure[1, 1, 2] = True
    if spatial_connectivity == 8:
        structure[1, 0, 0] = True
        structure[1, 0, 2] = True
        structure[1, 2, 0] = True
        structure[1, 2, 2] = True
    return structure


def load_heatwave_mask(method_dir, start_year, end_year):
    masks = []
    times = []
    for year in range(start_year, end_year + 1):
        path = heatwave_path(method_dir, year)
        if not path.exists():
            raise FileNotFoundError(f"Missing heatwave file: {path}")
        print(f"Reading heatwave mask {path}", flush=True)
        with xr.open_dataset(path) as ds:
            masks.append(ds["heatwave_day"].values.astype(bool))
            times.append(pd.to_datetime(ds["time"].values))
    return np.concatenate(masks, axis=0), pd.DatetimeIndex(np.concatenate(times))


def convection_groups(convection):
    return {
        cell_id: group[["TIME_EVENT", "TYPE_EVENT"]].sort_values("TIME_EVENT")
        for cell_id, group in convection.groupby("cell_id", sort=False)
    }


def component_has_convection(cell_ids, start_time, window_end, conv_groups):
    n_lightning = 0
    n_precip = 0
    first_time = None
    start64 = start_time.to_datetime64()
    end64 = window_end.to_datetime64()

    for cell_id in cell_ids:
        conv = conv_groups.get(int(cell_id))
        if conv is None or conv.empty:
            continue
        times = conv["TIME_EVENT"].to_numpy(dtype="datetime64[ns]")
        left = np.searchsorted(times, start64, side="left")
        right = np.searchsorted(times, end64, side="right")
        if left == right:
            continue
        types = conv["TYPE_EVENT"].to_numpy()[left:right]
        n_lightning += int(np.count_nonzero(types == "LIGHTNING"))
        n_precip += int(np.count_nonzero(types == "PRECIP"))
        hit_time = pd.Timestamp(times[left]).tz_localize("UTC")
        if first_time is None or hit_time < first_time:
            first_time = hit_time

    return n_lightning, n_precip, first_time


def regional_events_for_method(
    method,
    method_dir,
    start_year,
    end_year,
    latitudes,
    longitudes,
    conv_groups,
    post_window_hours,
    spatial_connectivity,
):
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
        n_lightning, n_precip, first_convective_time = component_has_convection(
            cell_ids,
            start_time,
            window_end,
            conv_groups,
        )
        has_lightning = n_lightning > 0
        has_precip = n_precip > 0
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
                "has_precip": has_precip,
                "has_any_convection": has_lightning or has_precip,
                "n_lightning": n_lightning,
                "n_precip": n_precip,
                "first_convective_time": first_time_text,
                "lag_hours": lag_hours,
            }
        )

    return pd.DataFrame(rows)


def main():
    args = parse_args()
    methods = method_paths(args)
    output_dir = Path(args.output_dir)
    tables_dir = Path(args.tables_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    latitudes, longitudes = load_grid(next(iter(methods.values())), args.start_year)
    convection, outside = load_convection(args.eswd, latitudes, longitudes)
    conv_groups = convection_groups(convection)
    print(f"Loaded convection records in heatwave grid domain: {len(convection)}", flush=True)
    print(f"Excluded convection records outside heatwave grid domain: {outside}", flush=True)

    summary_rows = []
    for method, method_dir in methods.items():
        events = regional_events_for_method(
            method,
            method_dir,
            args.start_year,
            args.end_year,
            latitudes,
            longitudes,
            conv_groups,
            args.post_window_hours,
            args.spatial_connectivity,
        )
        event_path = output_dir / f"eswd_regional_heatwave_convection_events_{method}_{args.start_year}_{args.end_year}.csv"
        compound_path = output_dir / f"compound_eswd_regional_heatwave_convection_events_{method}_{args.start_year}_{args.end_year}.csv"
        compound = events[events["has_any_convection"]].copy()
        events.to_csv(event_path, index=False, encoding="utf-8", na_rep="", lineterminator="\n")
        compound.to_csv(compound_path, index=False, encoding="utf-8", na_rep="", lineterminator="\n")
        summary_rows.append(
            {
                "method": method,
                "all_regional_heatwave_events": len(events),
                "compound_regional_heatwave_events": len(compound),
                "compound_regional_heatwave_event_fraction": len(compound) / len(events) if len(events) else np.nan,
                "compound_regional_heatwave_event_percent": len(compound) / len(events) * 100.0 if len(events) else np.nan,
            }
        )
        print(f"Wrote {event_path} ({len(events)} rows)", flush=True)
        print(f"Wrote {compound_path} ({len(compound)} rows)", flush=True)

    summary = pd.DataFrame(summary_rows)
    summary_path = tables_dir / f"regional_compound_heatwave_convection_summary_{args.start_year}_{args.end_year}.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8", lineterminator="\n")
    print(summary.to_string(index=False), flush=True)
    print(f"Wrote {summary_path}", flush=True)


if __name__ == "__main__":
    main()
