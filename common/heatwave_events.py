"""Grid-cell and regional heatwave events read from the yearly wet-bulb
heatwave files; shared by the EUCLID compound-event builders."""
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


def heatwave_path(method_dir, year):
    return Path(method_dir) / f"wetbulb_heatwave_{year}.nc"


def open_sample_grid(methods, start_year):
    first_dir = next(iter(methods.values()))
    path = heatwave_path(first_dir, start_year)
    if not path.exists():
        raise FileNotFoundError(f"Missing heatwave file: {path}")
    with xr.open_dataset(path) as ds:
        return ds["latitude"].values, ds["longitude"].values


def load_grid(sample_method_dir, start_year):
    with xr.open_dataset(heatwave_path(sample_method_dir, start_year)) as ds:
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
