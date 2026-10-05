# %%
import os
import time
import calendar
import argparse
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import sys

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from common.paths import ERA5_DIR

# %%
BASE_DIR = str(ERA5_DIR)

AREA = [72, -10, 36, 66.5]  # North, West, South, East
DEFAULT_START_YEAR = 2006
DEFAULT_END_YEAR = 2025
DEFAULT_MAX_WORKERS = 2
DEFAULT_MAX_RETRIES = 4

TIMES = [
    "00:00", "03:00", "06:00", "09:00",
    "12:00", "15:00", "18:00", "21:00"
]

DATASETS = {
    "instant": {
        "dataset": "reanalysis-era5-single-levels",
        "variables": [
            "10m_u_component_of_wind",
            "10m_v_component_of_wind",
            "2m_dewpoint_temperature",
            "2m_temperature",
            "surface_pressure",
            "mean_sea_level_pressure",
            "total_column_water_vapour",
            "geopotential",
            "volumetric_soil_water_layer_1",
            "boundary_layer_height",
        ],
        "pressure_levels": None,
    },
    "accum": {
        "dataset": "reanalysis-era5-single-levels",
        "variables": [
            "total_precipitation",
            "surface_latent_heat_flux",
            "surface_sensible_heat_flux",
            "surface_net_solar_radiation",
        ],
        "pressure_levels": None,
    },
    "pressure": {
        "dataset": "reanalysis-era5-pressure-levels",
        "variables": [
            "temperature",
            "specific_humidity",
            "u_component_of_wind",
            "v_component_of_wind",
            "vertical_velocity",
            "geopotential",
        ],
        "pressure_levels": [
            "925",
            "850",
            "700",
            "500",
            "300",
        ],
    },
}


def build_request(year, month, days, config):
    request = {
        "product_type": ["reanalysis"],
        "variable": config["variables"],
        "year": [f"{year}"],
        "month": [f"{month:02d}"],
        "day": [f"{day:02d}" for day in days],
        "time": TIMES,
        "data_format": "netcdf",
        "download_format": "unarchived",
        "area": AREA,
    }

    if config["pressure_levels"] is not None:
        request["pressure_level"] = config["pressure_levels"]

    return request


def file_ok(path, min_size_mb=1):
    if not os.path.exists(path):
        return False
    return os.path.getsize(path) > min_size_mb * 1024 * 1024


def make_filename(save_dir, year, month, days):
    if len(days) == 1:
        return os.path.join(save_dir, f"{year}_{month:02d}_{days[0]:02d}.nc")
    return os.path.join(save_dir, f"{year}_{month:02d}.nc")


def download_one(group_name, year, month, days, max_retries=DEFAULT_MAX_RETRIES):
    import cdsapi

    config = DATASETS[group_name]

    save_dir = os.path.join(BASE_DIR, str(year), group_name)
    os.makedirs(save_dir, exist_ok=True)

    filename = make_filename(save_dir, year, month, days)

    if file_ok(filename):
        return f"SKIP: {filename}"

    request = build_request(year, month, days, config)

    for attempt in range(1, max_retries + 1):
        try:
            logging.info("Downloading %s, attempt %s/%s", filename, attempt, max_retries)

            client = cdsapi.Client()
            client.retrieve(config["dataset"], request).download(filename)

            if file_ok(filename):
                return f"OK: {filename}"
            else:
                raise RuntimeError("Downloaded file is missing or too small.")

        except Exception as e:
            logging.warning("FAILED: %s", filename)
            logging.warning("%s", e)

            if attempt < max_retries:
                time.sleep(60 * attempt)
            else:
                return f"ERROR: {filename}"


def make_tasks(years, groups, granularity):
    tasks = []

    for year in years:
        for month in range(1, 13):
            days_in_month = calendar.monthrange(year, month)[1]

            if granularity == "month":
                day_groups = [list(range(1, days_in_month + 1))]
            else:
                day_groups = [[day] for day in range(1, days_in_month + 1)]

            for days in day_groups:
                for group_name in groups:
                    tasks.append((group_name, year, month, days))

    return tasks


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download ERA5 data from CDS. Monthly requests are usually much faster in queue-heavy periods."
    )
    parser.add_argument("--start-year", type=int, default=DEFAULT_START_YEAR)
    parser.add_argument("--end-year", type=int, default=DEFAULT_END_YEAR)
    parser.add_argument(
        "--granularity",
        choices=("month", "day"),
        default="month",
        help="Use month to reduce CDS queue overhead; use day to keep the old output layout.",
    )
    parser.add_argument(
        "--groups",
        nargs="+",
        choices=tuple(DATASETS),
        default=tuple(DATASETS),
        help="Data groups to download.",
    )
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS)
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)
    args = parser.parse_args()

    if args.end_year < args.start_year:
        parser.error("--end-year must be greater than or equal to --start-year")
    if args.max_workers < 1:
        parser.error("--max-workers must be at least 1")
    if args.max_retries < 1:
        parser.error("--max-retries must be at least 1")

    return args


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    args = parse_args()
    years = range(args.start_year, args.end_year + 1)
    tasks = make_tasks(years, args.groups, args.granularity)

    logging.info(
        "Prepared %s requests: years=%s-%s, groups=%s, granularity=%s, workers=%s",
        len(tasks),
        args.start_year,
        args.end_year,
        ",".join(args.groups),
        args.granularity,
        args.max_workers,
    )

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [
            executor.submit(download_one, group, year, month, days, args.max_retries)
            for group, year, month, days in tasks
        ]

        for future in as_completed(futures):
            logging.info("%s", future.result())


if __name__ == "__main__":
    main()
