import argparse
import logging
import os
import warnings
from pathlib import Path
import sys

import dask
import numpy as np
import xarray as xr

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from common.wetbulb_heatwave_common import (
    DEFAULT_ABSOLUTE_TW_THRESHOLD,
    DEFAULT_CHUNKS,
    DEFAULT_DASK_SCHEDULER,
    DEFAULT_HEATWAVE_END_YEAR,
    DEFAULT_HEATWAVE_START_YEAR,
    DEFAULT_INPUT_DIR,
    DEFAULT_MAX_GAP_DAYS,
    DEFAULT_MIN_RUN_LENGTH,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PROCESSED_DIR,
    DEFAULT_THRESHOLD_END_YEAR,
    DEFAULT_THRESHOLD_START_YEAR,
    DEFAULT_WINDOW_DAYS,
    compute_daily_max_wetbulb,
    configure_logging,
    default_threshold_input,
    detect_heatwaves,
    open_instant_dataset,
    parse_chunks,
    spatial_chunks,
    threshold_for_each_time,
    write_netcdf,
    write_netcdf_by_variable,
)


warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Calculate yearly wet-bulb heatwave files from ERA5 files and a precomputed threshold."
    )
    parser.add_argument("--input-dir", default=DEFAULT_INPUT_DIR, help="Directory containing <year>/instant/*.nc")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--start-year", type=int, default=DEFAULT_HEATWAVE_START_YEAR)
    parser.add_argument("--end-year", type=int, default=DEFAULT_HEATWAVE_END_YEAR)
    parser.add_argument("--baseline-start-year", type=int, default=DEFAULT_THRESHOLD_START_YEAR)
    parser.add_argument("--baseline-end-year", type=int, default=DEFAULT_THRESHOLD_END_YEAR)
    parser.add_argument("--min-run-length", type=int, default=DEFAULT_MIN_RUN_LENGTH)
    parser.add_argument(
        "--absolute-tw-threshold",
        type=float,
        default=DEFAULT_ABSOLUTE_TW_THRESHOLD,
        help="Single absolute daily maximum wet-bulb threshold in degC. If omitted, use the percentile-only method.",
    )
    parser.add_argument(
        "--absolute-tw-thresholds",
        nargs="+",
        type=float,
        default=[20],
        help="Run several absolute-threshold methods and write beside <output-dir>, for example heatwaves/90p22/90p24/90p26.",
    )
    parser.add_argument(
        "--max-gap-days",
        type=int,
        default=None,
        help="Merge heatwave exceedance periods separated by at most this many below-threshold days.",
    )
    parser.add_argument("--window-days", type=int, default=DEFAULT_WINDOW_DAYS)
    parser.add_argument(
        "--chunks",
        default=DEFAULT_CHUNKS,
        help="xarray/dask chunks such as time=248,latitude=100,longitude=100; use none to disable.",
    )
    parser.add_argument(
        "--dask-scheduler",
        default=DEFAULT_DASK_SCHEDULER,
        choices=("single-threaded", "threads", "synchronous"),
        help="Use single-threaded to reduce peak memory while writing large outputs.",
    )
    parser.add_argument(
        "--write-by-variable",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write one variable at a time to reduce peak memory.",
    )
    parser.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Overwrite existing output files. By default, files that already exist are skipped.",
    )
    parser.add_argument(
        "--threshold-input",
        default=None,
        help="Threshold NetCDF path. Default: <processed-dir>/wetbulb_threshold_90p_<baseline-start>_<baseline-end>.nc"
             " (or _romps suffixed, see --wetbulb-method).",
    )
    parser.add_argument(
        "--wetbulb-method",
        choices=("stull", "romps"),
        default="stull",
        help="'stull' (default, unchanged) or 'romps' (Romps 2026 pressure-dependent thermodynamic wet-bulb,"
             " via the heatindex package; requires ERA5 surface_pressure). Must match the threshold file's method"
             " -- build_wetbulb_threshold.py --wetbulb-method romps first. Output goes beside <output-dir> with a"
             " '_romps' suffix so it never collides with the Stull-based directories.",
    )
    return parser.parse_args()


def build_year_dataset(
    year,
    daily_tw,
    threshold,
    heatwave_threshold,
    exceedance_day,
    heatwave_day,
    heatwave_event_id,
    heatwave_duration,
    args,
    threshold_input,
):
    start = f"{year}-01-01"
    end = f"{year}-12-31"
    variables = {
        "tw_daily_max": daily_tw.sel(time=slice(start, end)).astype("float32"),
        "threshold_90p": threshold.sel(time=slice(start, end)).astype("float32"),
        "exceedance_day": exceedance_day.sel(time=slice(start, end)).astype(bool),
        "heatwave_day": heatwave_day.sel(time=slice(start, end)).astype(bool),
        "heatwave_event_id": heatwave_event_id.sel(time=slice(start, end)).astype("int16"),
        "heatwave_duration": heatwave_duration.sel(time=slice(start, end)).astype("int16"),
    }
    if heatwave_threshold is not None:
        variables["heatwave_threshold"] = heatwave_threshold.sel(time=slice(start, end)).astype("float32")

    if args.absolute_tw_threshold is None:
        heatwave_definition = f"At least {args.min_run_length} consecutive percentile-threshold exceedance days at each grid cell"
    else:
        heatwave_definition = (
            f"Exceedance means daily Twmax > max(local calendar-day 90th percentile, "
            f"{args.absolute_tw_threshold:g} degC). Events start with at least "
            f"{args.min_run_length} consecutive exceedance days and merge later "
            f"exceedance periods separated by at most {args.max_gap_days} below-threshold days."
        )

    formula = (
        "Stull 2011 approximation using 2m temperature and 2m dewpoint-derived relative humidity"
        if args.wetbulb_method == "stull" else
        "Romps 2026 thermodynamic wet-bulb (Rankine-Kirchhoff via heatindex) using ERA5 t2m, d2m, sp"
    )
    ds = xr.Dataset(variables)
    ds.attrs.update(
        source="ERA5 single levels",
        wet_bulb_formula=formula,
        threshold_file=os.path.basename(threshold_input),
        baseline_period=f"{args.baseline_start_year}-{args.baseline_end_year}",
        percentile="90",
        window_days=str(args.window_days),
        max_gap_days=str(args.max_gap_days),
        heatwave_definition=heatwave_definition,
        note="Heatwave events are detected on the full requested time period before slicing by year, so events crossing year boundaries are preserved.",
    )
    if args.absolute_tw_threshold is not None:
        ds.attrs["absolute_tw_threshold_degC"] = str(args.absolute_tw_threshold)
    return ds


def format_threshold_suffix(threshold):
    return f"{threshold:g}".replace(".", "p")


def method_output_dir(base_output_dir, absolute_tw_threshold, wetbulb_method="stull"):
    base = Path(base_output_dir)
    suffix = "" if wetbulb_method == "stull" else f"_{wetbulb_method}"
    return str(base.with_name(f"90p{format_threshold_suffix(absolute_tw_threshold)}{suffix}"))


def yearly_encoding(absolute_method):
    encoding = {
        "tw_daily_max": {"zlib": True, "complevel": 4, "_FillValue": np.float32(np.nan)},
        "threshold_90p": {"zlib": True, "complevel": 4, "_FillValue": np.float32(np.nan)},
        "exceedance_day": {"zlib": True, "complevel": 4},
        "heatwave_day": {"zlib": True, "complevel": 4},
        "heatwave_event_id": {"zlib": True, "complevel": 4},
        "heatwave_duration": {"zlib": True, "complevel": 4},
    }
    if absolute_method:
        encoding["heatwave_threshold"] = {"zlib": True, "complevel": 4, "_FillValue": np.float32(np.nan)}
    return encoding


def method_args(args, absolute_tw_threshold, output_dir, max_gap_days):
    method = argparse.Namespace(**vars(args))
    method.absolute_tw_threshold = absolute_tw_threshold
    method.output_dir = output_dir
    method.max_gap_days = max_gap_days
    return method


def build_method_jobs(args):
    if args.absolute_tw_thresholds:
        gap_days = DEFAULT_MAX_GAP_DAYS if args.max_gap_days is None else args.max_gap_days
        return [
            method_args(args, threshold, method_output_dir(args.output_dir, threshold, args.wetbulb_method), gap_days)
            for threshold in args.absolute_tw_thresholds
        ]

    gap_days = DEFAULT_MAX_GAP_DAYS if args.max_gap_days is None else args.max_gap_days
    return [method_args(args, args.absolute_tw_threshold, args.output_dir, gap_days)]


def build_exceedance(daily_tw, threshold, absolute_tw_threshold):
    if absolute_tw_threshold is None:
        exceedance_day = (daily_tw > threshold).rename("exceedance_day")
        exceedance_day.attrs.update(
            long_name="Daily maximum wet-bulb temperature exceeds the local calendar-day 90th percentile threshold",
        )
        return exceedance_day, None

    heatwave_threshold = xr.where(
        threshold > absolute_tw_threshold,
        threshold,
        absolute_tw_threshold,
    ).rename("heatwave_threshold")
    heatwave_threshold.attrs.update(
        long_name="Combined heatwave threshold for daily maximum wet-bulb temperature",
        units="degC",
        method=(
            "Maximum of local calendar-day 90th percentile threshold and "
            f"{absolute_tw_threshold:g} degC absolute threshold"
        ),
    )
    exceedance_day = (daily_tw > heatwave_threshold).rename("exceedance_day")
    exceedance_day.attrs.update(
        long_name=(
            "Daily maximum wet-bulb temperature exceeds both the local calendar-day "
            "90th percentile threshold and the absolute wet-bulb threshold"
        ),
        absolute_tw_threshold_degC=str(absolute_tw_threshold),
    )
    return exceedance_day, heatwave_threshold


def detect_and_write_method(method, daily_tw, threshold, threshold_input):
    os.makedirs(method.output_dir, exist_ok=True)
    exceedance_day, heatwave_threshold = build_exceedance(daily_tw, threshold, method.absolute_tw_threshold)

    logging.info(
        "Detecting heatwaves for output %s, absolute threshold=%s, max gap=%s",
        method.output_dir,
        method.absolute_tw_threshold,
        method.max_gap_days,
    )
    heatwave_day, heatwave_event_id, heatwave_duration = detect_heatwaves(
        exceedance_day,
        method.min_run_length,
        max_gap_days=method.max_gap_days,
    )

    encoding = yearly_encoding(method.absolute_tw_threshold is not None)
    for year in range(method.start_year, method.end_year + 1):
        output = os.path.join(method.output_dir, f"wetbulb_heatwave_{year}.nc")
        if os.path.exists(output) and not method.overwrite:
            logging.info("Skipping existing file %s", output)
            continue

        logging.info("Writing yearly file %s", output)
        year_ds = build_year_dataset(
            year,
            daily_tw,
            threshold,
            heatwave_threshold,
            exceedance_day,
            heatwave_day,
            heatwave_event_id,
            heatwave_duration,
            method,
            threshold_input,
        )
        if method.write_by_variable:
            write_netcdf_by_variable(year_ds, output, encoding, overwrite=method.overwrite)
        else:
            write_netcdf(year_ds, output, encoding, overwrite=method.overwrite)


def main():
    configure_logging()
    args = parse_args()

    if args.end_year < args.start_year:
        raise ValueError("--end-year must be greater than or equal to --start-year")
    if args.min_run_length < 1:
        raise ValueError("--min-run-length must be at least 1.")
    if args.max_gap_days is not None and args.max_gap_days < 0:
        raise ValueError("--max-gap-days must be 0 or greater.")

    threshold_input = args.threshold_input
    if threshold_input is None:
        threshold_input = default_threshold_input(
            DEFAULT_PROCESSED_DIR,
            args.baseline_start_year,
            args.baseline_end_year,
        )
        if args.wetbulb_method == "romps":
            threshold_input = threshold_input.replace(".nc", "_romps.nc")
    if not os.path.exists(threshold_input):
        raise FileNotFoundError(
            f"Missing threshold file: {threshold_input}. Run scripts/pipeline/build_wetbulb_threshold.py first."
        )

    chunks = parse_chunks(args.chunks)
    ds = open_instant_dataset(args.input_dir, args.start_year, args.end_year, chunks=chunks)

    with dask.config.set(scheduler=args.dask_scheduler):
        logging.info("Calculating daily maximum wet-bulb temperature for %s-%s (%s)", args.start_year, args.end_year, args.wetbulb_method)
        daily_tw = compute_daily_max_wetbulb(ds, method=args.wetbulb_method)

        logging.info("Reading threshold file %s", threshold_input)
        threshold_by_day = xr.open_dataset(threshold_input, chunks=spatial_chunks(chunks))["threshold_90p"]
        threshold = threshold_for_each_time(daily_tw, threshold_by_day)

        # Materialize these two shared, expensive arrays exactly once, here, before
        # the per-method/per-year write loop below. Without this they stay lazy
        # dask graphs, and every single .to_netcdf() call for every variable of
        # every year of every method would silently recompute daily_tw and
        # threshold from the raw ERA5 files from scratch (values are unaffected --
        # this only changes when the computation happens, not what is computed).
        logging.info("Computing daily wet-bulb temperature and threshold once (shared across all years/methods)")
        daily_tw, threshold = dask.compute(daily_tw, threshold)
        logging.info("Done computing daily wet-bulb temperature and threshold")

        for method in build_method_jobs(args):
            detect_and_write_method(method, daily_tw, threshold, threshold_input)

    logging.info("Done")


if __name__ == "__main__":
    main()
