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
    DEFAULT_CHUNKS,
    DEFAULT_DASK_SCHEDULER,
    DEFAULT_INPUT_DIR,
    DEFAULT_PROCESSED_DIR,
    DEFAULT_THRESHOLD_END_YEAR,
    DEFAULT_THRESHOLD_START_YEAR,
    DEFAULT_WINDOW_DAYS,
    build_threshold_90p,
    compute_daily_max_wetbulb,
    configure_logging,
    default_threshold_output,
    open_instant_dataset,
    parse_chunks,
    write_netcdf,
)

warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Calculate 1991-2020 wet-bulb 90th percentile thresholds from ERA5 files."
    )
    parser.add_argument("--input-dir", default=DEFAULT_INPUT_DIR, help="Directory containing <year>/instant/*.nc")
    parser.add_argument("--output-dir", default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--baseline-start-year", type=int, default=DEFAULT_THRESHOLD_START_YEAR)
    parser.add_argument("--baseline-end-year", type=int, default=DEFAULT_THRESHOLD_END_YEAR)
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
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Overwrite existing output files. By default, files that already exist are skipped.",
    )
    parser.add_argument(
        "--threshold-output",
        default=None,
        help="Threshold NetCDF path. Default: <processed-dir>/wetbulb_threshold_90p_<baseline-start>_<baseline-end>.nc"
             " (or _romps suffixed, see --wetbulb-method).",
    )
    parser.add_argument(
        "--wetbulb-method",
        choices=("stull", "romps"),
        default="stull",
        help="'stull' (default, unchanged) or 'romps' (Romps 2026 pressure-dependent thermodynamic wet-bulb,"
             " via the heatindex package; requires ERA5 surface_pressure).",
    )
    return parser.parse_args()


def build_threshold_dataset(threshold_by_day, args):
    ds = xr.Dataset({"threshold_90p": threshold_by_day.astype("float32")})
    formula = (
        "Stull 2011 approximation using 2m temperature and 2m dewpoint-derived relative humidity"
        if args.wetbulb_method == "stull" else
        "Romps 2026 thermodynamic wet-bulb (Rankine-Kirchhoff via heatindex) using ERA5 t2m, d2m, sp"
    )
    ds.attrs.update(
        source="ERA5 single levels",
        wet_bulb_formula=formula,
        baseline_period=f"{args.baseline_start_year}-{args.baseline_end_year}",
        percentile="90",
        window_days=str(args.window_days),
        window_description="Target calendar day +/- 15 days for a 31-day window, across all baseline years.",
        leap_day_handling="Threshold file stores 365 no-leap calendar days; leap-day daily files use the average of Feb 28 and Mar 1 thresholds.",
    )
    return ds


def main():
    configure_logging()
    args = parse_args()

    if args.baseline_end_year < args.baseline_start_year:
        raise ValueError("--baseline-end-year must be greater than or equal to --baseline-start-year")
    if args.window_days % 2 != 1:
        raise ValueError("--window-days must be an odd number, for example 31.")

    os.makedirs(args.output_dir, exist_ok=True)
    threshold_output = args.threshold_output
    if threshold_output is None:
        threshold_output = default_threshold_output(
            args.output_dir,
            args.baseline_start_year,
            args.baseline_end_year,
        )
        if args.wetbulb_method == "romps":
            # Never silently collide with the Stull-based default filename.
            threshold_output = threshold_output.replace(".nc", "_romps.nc")
    else:
        os.makedirs(os.path.dirname(os.path.abspath(threshold_output)), exist_ok=True)

    threshold_encoding = {
        "threshold_90p": {"zlib": True, "complevel": 4, "_FillValue": np.float32(np.nan)},
    }

    if os.path.exists(threshold_output) and not args.overwrite:
        logging.info("Skipping existing file %s", threshold_output)
        return

    chunks = parse_chunks(args.chunks)
    ds = open_instant_dataset(args.input_dir, args.baseline_start_year, args.baseline_end_year, chunks=chunks)

    with dask.config.set(scheduler=args.dask_scheduler):
        logging.info("Calculating daily maximum wet-bulb temperature for threshold baseline (%s)", args.wetbulb_method)
        daily_tw = compute_daily_max_wetbulb(ds, method=args.wetbulb_method)

        logging.info("Calculating 90th percentile thresholds with %s-day calendar window", args.window_days)
        threshold_by_day = build_threshold_90p(
            daily_tw,
            args.baseline_start_year,
            args.baseline_end_year,
            window_days=args.window_days,
        )

        logging.info("Writing threshold file %s", threshold_output)
        threshold_ds = build_threshold_dataset(threshold_by_day, args)
        write_netcdf(threshold_ds, threshold_output, threshold_encoding, overwrite=args.overwrite)

    logging.info("Done")


if __name__ == "__main__":
    main()
