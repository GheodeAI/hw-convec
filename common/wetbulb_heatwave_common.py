import logging
import os

import numpy as np
import pandas as pd
import xarray as xr

from common.paths import ERA5_DIR, WETBULB_HEATWAVES_DIR, WETBULB_THRESHOLDS_DIR

DEFAULT_INPUT_DIR = str(ERA5_DIR)
DEFAULT_PROCESSED_DIR = str(WETBULB_THRESHOLDS_DIR)
DEFAULT_OUTPUT_DIR = str(WETBULB_HEATWAVES_DIR / "90p")
DEFAULT_THRESHOLD_START_YEAR = 1991
DEFAULT_THRESHOLD_END_YEAR = 2020
DEFAULT_HEATWAVE_START_YEAR = 2006
DEFAULT_HEATWAVE_END_YEAR = 2025
DEFAULT_MIN_RUN_LENGTH = 3
DEFAULT_ABSOLUTE_TW_THRESHOLD = None
DEFAULT_MAX_GAP_DAYS = 1
DEFAULT_WINDOW_DAYS = 31
DEFAULT_DASK_SCHEDULER = "single-threaded"
DEFAULT_CHUNKS = "time=248,latitude=100,longitude=100"


def default_threshold_output(output_dir, baseline_start_year, baseline_end_year):
    return os.path.join(output_dir, f"wetbulb_threshold_90p_{baseline_start_year}_{baseline_end_year}.nc")


def default_threshold_input(processed_dir, baseline_start_year, baseline_end_year):
    return os.path.join(processed_dir, f"wetbulb_threshold_90p_{baseline_start_year}_{baseline_end_year}.nc")


def instant_files(input_dir, start_year, end_year):
    files = []
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            path = os.path.join(input_dir, str(year), "instant", f"{year}_{month:02d}.nc")
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing input file: {path}")
            files.append(path)
    return files


def saturation_vapor_pressure_hpa(temperature_c):
    """Magnus formula over water, input in degC and output in hPa."""
    return 6.112 * np.exp((17.67 * temperature_c) / (temperature_c + 243.5))


def relative_humidity_from_dewpoint(temperature_k, dewpoint_k):
    temperature_c = temperature_k - 273.15
    dewpoint_c = dewpoint_k - 273.15
    rh = 100.0 * saturation_vapor_pressure_hpa(dewpoint_c) / saturation_vapor_pressure_hpa(temperature_c)
    return rh.clip(0.0, 100.0)


def wet_bulb_stull_c(temperature_k, dewpoint_k):
    """
    Approximate 2 m wet-bulb temperature using Stull (2011).

    The formula expects air temperature in degC and relative humidity in percent,
    and returns wet-bulb temperature in degC.
    """
    temperature_c = temperature_k - 273.15
    rh = relative_humidity_from_dewpoint(temperature_k, dewpoint_k)

    return (
        temperature_c * np.arctan(0.151977 * np.sqrt(rh + 8.313659))
        + np.arctan(temperature_c + rh)
        - np.arctan(rh - 1.676331)
        + 0.00391838 * rh**1.5 * np.arctan(0.023101 * rh)
        - 4.686035
    )


def wet_bulb_romps_c(temperature_k, dewpoint_k, surface_pressure_pa):
    """
    Thermodynamic wet-bulb temperature via the Rankine-Kirchhoff approach of
    Romps (2026, JAMC), using the author's official `heatindex` package
    (pip install heatindex; https://github.com/davidromps/heatindex). Unlike
    Stull (2011), this explicitly depends on pressure, so it is the one to
    check over elevated terrain where actual surface pressure departs most
    from the fixed-pressure assumption implicit in the simpler formula.

    temperature_k, dewpoint_k: same as wet_bulb_stull_c.
    surface_pressure_pa: ERA5 surface_pressure ('sp'), in Pa -- NOT mean sea
        level pressure ('msl'); using msl would defeat the point of this
        comparison over elevated regions.

    Returns wet-bulb temperature in degC, matching wet_bulb_stull_c's convention.
    """
    rh_fraction = relative_humidity_from_dewpoint(temperature_k, dewpoint_k) / 100.0

    def _wetbulb(pressure, temperature, rh):
        import heatindex as hi
        return hi.wetbulb(
            np.asarray(pressure, dtype=np.float64),
            np.asarray(temperature, dtype=np.float64),
            np.asarray(rh, dtype=np.float64),
        ) - 273.15

    return xr.apply_ufunc(
        _wetbulb, surface_pressure_pa, temperature_k, rh_fraction,
        dask="parallelized", output_dtypes=[np.float64],
    )


def open_instant_dataset(input_dir, start_year, end_year, chunks=None):
    files = instant_files(input_dir, start_year, end_year)

    return xr.open_mfdataset(
        files,
        combine="by_coords",
        chunks=chunks,
        parallel=False,
        data_vars="minimal",
        coords="minimal",
        compat="override",
    )


def compute_daily_max_wetbulb(ds, method="stull"):
    if "t2m" not in ds or "d2m" not in ds:
        raise KeyError("Input files must contain t2m and d2m variables.")

    time_name = "valid_time" if "valid_time" in ds.coords else "time"
    if method == "stull":
        tw = wet_bulb_stull_c(ds["t2m"], ds["d2m"]).rename("tw")
        method_note = "Stull 2011 approximation from ERA5 t2m and d2m"
    elif method == "romps":
        if "sp" not in ds:
            raise KeyError("Romps wet-bulb requires ERA5 surface_pressure ('sp') in the instant dataset.")
        tw = wet_bulb_romps_c(ds["t2m"], ds["d2m"], ds["sp"]).rename("tw")
        method_note = "Romps 2026 thermodynamic wet-bulb (Rankine-Kirchhoff via heatindex) from ERA5 t2m, d2m, sp"
    else:
        raise ValueError(f"Unknown wet-bulb method: {method!r} (expected 'stull' or 'romps')")
    tw.attrs.update(
        long_name="2 m wet-bulb temperature",
        units="degC",
        method=method_note,
    )

    daily = tw.resample({time_name: "1D"}).max().rename("tw_daily_max")
    daily = daily.rename({time_name: "time"})
    daily.attrs.update(
        long_name="Daily maximum 2 m wet-bulb temperature",
        units="degC",
    )
    return daily


def noleap_month_day_index(times):
    dates = pd.DatetimeIndex(times)
    return np.array(
        [
            pd.Timestamp(2001, date.month, date.day).dayofyear
            if not (date.month == 2 and date.day == 29)
            else 59
            for date in dates
        ],
        dtype=np.int16,
    )


def build_threshold_90p(daily_tw, baseline_start_year, baseline_end_year, window_days=31):
    baseline = daily_tw.sel(time=slice(f"{baseline_start_year}-01-01", f"{baseline_end_year}-12-31"))
    baseline = baseline.sel(time=~((baseline.time.dt.month == 2) & (baseline.time.dt.day == 29)))
    if hasattr(baseline.data, "chunks"):
        baseline = baseline.chunk({"time": -1})

    target_days = np.arange(1, 366, dtype=np.int16)
    baseline_md = xr.DataArray(
        noleap_month_day_index(baseline.time.values),
        coords={"time": baseline.time},
        dims="time",
        name="month_day",
    )

    thresholds = []
    half_window = window_days // 2
    for target_day in target_days:
        distance = np.abs(((baseline_md - target_day + 182) % 365) - 182)
        sample = baseline.where(distance <= half_window, drop=True)
        threshold = sample.quantile(0.9, dim="time", skipna=True).drop_vars("quantile")
        thresholds.append(threshold)

    threshold_by_day = xr.concat(
        thresholds,
        dim=xr.DataArray(target_days, dims="dayofyear", name="dayofyear"),
    ).rename("threshold_90p")
    dates = pd.date_range("2001-01-01", "2001-12-31", freq="D")
    threshold_by_day = threshold_by_day.assign_coords(
        month=("dayofyear", dates.month.astype(np.int16)),
        day=("dayofyear", dates.day.astype(np.int16)),
    )
    threshold_by_day.attrs.update(
        long_name="Calendar-day 90th percentile threshold of daily maximum wet-bulb temperature",
        units="degC",
        baseline_period=f"{baseline_start_year}-{baseline_end_year}",
        percentile="90",
        window_days=str(window_days),
        note="Each target calendar day uses samples within +/-15 days across baseline years; leap day is interpolated.",
    )
    return threshold_by_day


def threshold_for_each_time(daily_tw, threshold_by_day):
    md = noleap_month_day_index(daily_tw.time.values)
    threshold = threshold_by_day.sel(dayofyear=xr.DataArray(md, dims="time", coords={"time": daily_tw.time}))

    leap_day = (daily_tw.time.dt.month == 2) & (daily_tw.time.dt.day == 29)
    if bool(leap_day.any()):
        feb28 = threshold_by_day.sel(dayofyear=59)
        mar01 = threshold_by_day.sel(dayofyear=60)
        leap_threshold = (feb28 + mar01) / 2.0
        threshold = xr.where(leap_day, leap_threshold, threshold)

    threshold = threshold.rename("threshold_90p")
    threshold.attrs.update(threshold_by_day.attrs)
    return threshold


def run_length_mask_1d(exceedance, min_run_length):
    exceedance = np.asarray(exceedance, dtype=bool)
    heatwave = np.zeros(exceedance.shape, dtype=bool)

    start = None
    for i, value in enumerate(exceedance):
        if value and start is None:
            start = i
        if (not value or i == len(exceedance) - 1) and start is not None:
            end = i + 1 if value and i == len(exceedance) - 1 else i
            if end - start >= min_run_length:
                heatwave[start:end] = True
            start = None

    return heatwave


def heatwave_mask_with_gap_1d(exceedance, min_run_length, max_gap_days):
    exceedance = np.asarray(exceedance, dtype=bool)
    heatwave = np.zeros(exceedance.shape, dtype=bool)
    n_times = exceedance.size
    i = 0

    while i < n_times:
        if not exceedance[i]:
            i += 1
            continue

        run_start = i
        while i < n_times and exceedance[i]:
            i += 1
        run_end = i

        if run_end - run_start < min_run_length:
            continue

        event_start = run_start
        event_end = run_end
        j = run_end

        while j < n_times:
            gap_start = j
            while j < n_times and not exceedance[j]:
                j += 1
            gap_length = j - gap_start

            if j >= n_times or gap_length > max_gap_days:
                break

            while j < n_times and exceedance[j]:
                j += 1
            event_end = j

        heatwave[event_start:event_end] = True
        i = j

    return heatwave


def run_event_id_1d(heatwave):
    heatwave = np.asarray(heatwave, dtype=bool)
    event_id = np.zeros(heatwave.shape, dtype=np.int16)

    current_id = 0
    in_event = False
    for i, value in enumerate(heatwave):
        if value and not in_event:
            current_id += 1
            in_event = True
        elif not value:
            in_event = False
        if value:
            event_id[i] = current_id

    return event_id


def run_duration_1d(heatwave):
    heatwave = np.asarray(heatwave, dtype=bool)
    duration = np.zeros(heatwave.shape, dtype=np.int16)

    start = None
    for i, value in enumerate(heatwave):
        if value and start is None:
            start = i
        if (not value or i == len(heatwave) - 1) and start is not None:
            end = i + 1 if value and i == len(heatwave) - 1 else i
            duration[start:end] = end - start
            start = None

    return duration


def detect_heatwaves(exceedance, min_run_length, max_gap_days=0):
    # Consecutive-day detection needs the full time series for each grid cell.
    if hasattr(exceedance.data, "chunks"):
        exceedance = exceedance.chunk({"time": -1})

    detector = heatwave_mask_with_gap_1d if max_gap_days > 0 else run_length_mask_1d
    detector_kwargs = {"min_run_length": min_run_length}
    if max_gap_days > 0:
        detector_kwargs["max_gap_days"] = max_gap_days

    heatwave_day = xr.apply_ufunc(
        detector,
        exceedance,
        kwargs=detector_kwargs,
        input_core_dims=[["time"]],
        output_core_dims=[["time"]],
        vectorize=True,
        dask="parallelized",
        dask_gufunc_kwargs={"allow_rechunk": True},
        output_dtypes=[bool],
    ).transpose("time", ...)
    heatwave_day = heatwave_day.rename("heatwave_day")
    heatwave_day.attrs.update(
        long_name=(
            f"Heatwave days, starting with exceedance runs >= {min_run_length} consecutive days"
            f" and merging later exceedance periods separated by <= {max_gap_days} below-threshold days"
        ),
        min_run_length_days=str(min_run_length),
        max_gap_days=str(max_gap_days),
    )

    heatwave_event_id = xr.apply_ufunc(
        run_event_id_1d,
        heatwave_day,
        input_core_dims=[["time"]],
        output_core_dims=[["time"]],
        vectorize=True,
        dask="parallelized",
        dask_gufunc_kwargs={"allow_rechunk": True},
        output_dtypes=[np.int16],
    ).transpose("time", ...)
    heatwave_event_id = heatwave_event_id.rename("heatwave_event_id")
    heatwave_event_id.attrs.update(
        long_name="Sequential heatwave event ID for each grid cell; 0 means not a heatwave day",
        min_run_length_days=str(min_run_length),
        max_gap_days=str(max_gap_days),
    )

    heatwave_duration = xr.apply_ufunc(
        run_duration_1d,
        heatwave_day,
        input_core_dims=[["time"]],
        output_core_dims=[["time"]],
        vectorize=True,
        dask="parallelized",
        dask_gufunc_kwargs={"allow_rechunk": True},
        output_dtypes=[np.int16],
    ).transpose("time", ...)
    heatwave_duration = heatwave_duration.rename("heatwave_duration")
    heatwave_duration.attrs.update(
        long_name="Duration of the heatwave event containing each day; 0 means not a heatwave day",
        duration_units="days",
        min_run_length_days=str(min_run_length),
        max_gap_days=str(max_gap_days),
    )

    return heatwave_day, heatwave_event_id, heatwave_duration


def parse_chunks(chunks_text):
    if chunks_text.lower() in ("none", "false", "0"):
        return None
    chunks = {}
    for item in chunks_text.split(","):
        key, value = item.split("=")
        chunks[key.strip()] = int(value)
    return chunks


def spatial_chunks(chunks):
    if chunks is None:
        return None
    return {key: value for key, value in chunks.items() if key in ("latitude", "longitude")}


def remove_if_overwrite(path, overwrite):
    if os.path.exists(path):
        if overwrite:
            os.remove(path)
            return True
        return False
    return True


def write_netcdf_by_variable(ds, output, encoding, overwrite=True):
    if not remove_if_overwrite(output, overwrite):
        logging.info("Skipping existing file %s", output)
        return

    xr.Dataset(coords=ds.coords, attrs=ds.attrs).to_netcdf(output, mode="w")

    for name in ds.data_vars:
        logging.info("Writing variable %s", name)
        variable_encoding = {name: encoding[name]} if name in encoding else None
        ds[[name]].to_netcdf(output, mode="a", encoding=variable_encoding)


def write_netcdf(ds, output, encoding, overwrite=True):
    if not remove_if_overwrite(output, overwrite):
        logging.info("Skipping existing file %s", output)
        return
    ds.to_netcdf(output, encoding=encoding)


def configure_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
