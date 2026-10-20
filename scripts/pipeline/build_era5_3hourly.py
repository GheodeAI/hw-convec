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

from common.paths import ERA5_3HOURLY_DIR
from common.wetbulb_heatwave_common import (
    DEFAULT_CHUNKS,
    DEFAULT_DASK_SCHEDULER,
    DEFAULT_HEATWAVE_END_YEAR,
    DEFAULT_HEATWAVE_START_YEAR,
    DEFAULT_INPUT_DIR,
    configure_logging,
    parse_chunks,
    relative_humidity_from_dewpoint,
    wet_bulb_stull_c,
)


warnings.filterwarnings("ignore")

DEFAULT_OUTPUT_DIR = str(ERA5_3HOURLY_DIR)

GRAVITY = 9.80665
CP_DRY_AIR = 1004.0
LV = 2.5e6
EPSILON = 0.622
EARTH_RADIUS = 6371000.0


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build yearly 3-hourly ERA5 derived datasets split into single-level "
            "and pressure-level NetCDF files."
        )
    )
    parser.add_argument("--input-dir", default=DEFAULT_INPUT_DIR, help="Directory containing <year>/<group>/*.nc")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--start-year", type=int, default=DEFAULT_HEATWAVE_START_YEAR)
    parser.add_argument("--end-year", type=int, default=DEFAULT_HEATWAVE_END_YEAR)
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
    return parser.parse_args()


def group_files(input_dir, year, group_name):
    files = []
    for month in range(1, 13):
        path = os.path.join(input_dir, str(year), group_name, f"{year}_{month:02d}.nc")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing input file: {path}")
        files.append(path)
    return files


def open_group_dataset(input_dir, year, group_name, chunks=None):
    return xr.open_mfdataset(
        group_files(input_dir, year, group_name),
        combine="by_coords",
        chunks=chunks,
        parallel=False,
        data_vars="minimal",
        coords="minimal",
        compat="override",
    )


def standardize_time(ds):
    if "valid_time" in ds.coords:
        return ds.rename({"valid_time": "time"})
    return ds


def saturation_vapor_pressure_pa(temperature_k):
    temperature_c = temperature_k - 273.15
    return 611.2 * np.exp((17.67 * temperature_c) / (temperature_c + 243.5))


def specific_humidity_from_dewpoint(dewpoint_k, pressure_pa):
    vapor_pressure = saturation_vapor_pressure_pa(dewpoint_k)
    return EPSILON * vapor_pressure / (pressure_pa - (1.0 - EPSILON) * vapor_pressure)


def moist_static_energy(temperature_k, specific_humidity, geopotential):
    return CP_DRY_AIR * temperature_k + LV * specific_humidity + geopotential


def lcl_height_bolton_m(temperature_k, dewpoint_k):
    temperature_c = temperature_k - 273.15
    dewpoint_c = dewpoint_k - 273.15
    return 125.0 * (temperature_c - dewpoint_c)


def horizontal_derivative_x(data):
    meters_per_degree_lon = (np.pi / 180.0) * EARTH_RADIUS * np.cos(np.deg2rad(data.latitude))
    return data.differentiate("longitude") / meters_per_degree_lon


def horizontal_derivative_y(data):
    meters_per_degree_lat = (np.pi / 180.0) * EARTH_RADIUS
    return data.differentiate("latitude") / meters_per_degree_lat


def convergence(u, v):
    return (-(horizontal_derivative_x(u) + horizontal_derivative_y(v))).rename("convergence")


def moisture_flux_convergence(uq, vq):
    return (-(horizontal_derivative_x(uq) + horizontal_derivative_y(vq))).rename("moisture_flux_convergence")


def set_attrs(da, long_name, units, **extra_attrs):
    da.attrs.update({"long_name": long_name, "units": units, **extra_attrs})
    return da


def build_single_level_dataset(instant):
    ds = standardize_time(instant)

    q2m = specific_humidity_from_dewpoint(ds["d2m"], ds["sp"]).rename("q2m")
    tw = wet_bulb_stull_c(ds["t2m"], ds["d2m"]).rename("tw_3h")
    mse_surface = moist_static_energy(ds["t2m"], q2m, ds["z"]).rename("mse_surface")
    soil_moisture = ds["swvl1"].rename("soil_moisture")

    data_vars = {
        "tw_3h": set_attrs(tw, "3-hourly 2 m wet-bulb temperature", "degC"),
        "mse_surface": set_attrs(mse_surface, "Surface moist static energy", "J kg**-1"),
        "soil_moisture": set_attrs(
            soil_moisture,
            "Volumetric soil water layer 1",
            "m**3 m**-3",
        ),
        "cape": set_attrs(ds["cape"], "Convective available potential energy", "J kg**-1"),
        "cin": set_attrs(ds["cin"], "Convective inhibition", "J kg**-1"),
        "tcwv": set_attrs(ds["tcwv"], "Total column vertically-integrated water vapour", "kg m**-2"),
    }

    out = xr.Dataset(data_vars)
    out.attrs.update(
        source="ERA5 single levels from the Copernicus Climate Data Store",
        temporal_resolution="3-hourly",
        note="Derived variables are calculated at each 3-hourly time step; no daily aggregation is applied.",
    )
    return out.astype({name: "float32" for name in out.data_vars})


def pressure_tcwv(q):
    q_sorted = q.sortby("pressure_level")
    return (q_sorted.integrate("pressure_level") * 100.0 / GRAVITY).rename("tcwv")


def pressure_wind_shear(ds, bottom, top):
    levels = set(float(level) for level in ds.pressure_level.values)
    if float(bottom) not in levels or float(top) not in levels:
        return None
    du = ds["u"].sel(pressure_level=top) - ds["u"].sel(pressure_level=bottom)
    dv = ds["v"].sel(pressure_level=top) - ds["v"].sel(pressure_level=bottom)
    return np.hypot(du, dv).rename(f"wind_shear_{bottom}_{top}")


def build_wind_shear_dataset(pressure):
    ds = standardize_time(pressure)
    data_vars = {}
    for bottom, top in ((925, 300), (850, 300), (850, 500)):
        shear = pressure_wind_shear(ds, bottom, top)
        if shear is not None:
            data_vars[shear.name] = set_attrs(shear, f"{bottom}-{top} hPa wind shear magnitude", "m s**-1")
    return xr.Dataset(data_vars).astype({name: "float32" for name in data_vars})


def build_pressure_level_dataset(pressure):
    ds = standardize_time(pressure)
    moisture_flux_u = (ds["q"] * ds["u"]).rename("moisture_flux_u")
    moisture_flux_v = (ds["q"] * ds["v"]).rename("moisture_flux_v")
    mfc = moisture_flux_convergence(moisture_flux_u, moisture_flux_v).rename("moisture_flux_convergence")
    omega = ds["w"].rename("omega")

    data_vars = {
        "z": ds["z"],
        "omega": set_attrs(omega, "Pressure-level vertical velocity", "Pa s**-1"),
        "moisture_flux_convergence": set_attrs(mfc, "Pressure-level moisture flux convergence", "s**-1"),
    }

    out = xr.Dataset(data_vars)
    out.attrs.update(
        source="ERA5 pressure levels from CDS",
        temporal_resolution="3-hourly",
        note="Derived variables are calculated at each 3-hourly time step; no daily aggregation is applied.",
    )
    return out.astype({name: "float32" for name in out.data_vars})


def netcdf_encoding(ds):
    return {
        name: {"zlib": True, "complevel": 4, "_FillValue": np.float32(np.nan)}
        for name in ds.data_vars
    }


def write_netcdf_file(ds, output, overwrite=False):
    if os.path.exists(output):
        if not overwrite:
            logging.info("Skipping existing NetCDF file %s", output)
            return
        os.remove(output)

    ds.to_netcdf(output, encoding=netcdf_encoding(ds))


def output_paths(output_dir, year):
    single_dir = os.path.join(output_dir, "single_level")
    pressure_dir = os.path.join(output_dir, "pressure_levels")
    os.makedirs(single_dir, exist_ok=True)
    os.makedirs(pressure_dir, exist_ok=True)
    return (
        os.path.join(single_dir, f"{year}.nc"),
        os.path.join(pressure_dir, f"{year}.nc"),
    )


def main():
    configure_logging()
    args = parse_args()

    if args.end_year < args.start_year:
        raise ValueError("--end-year must be greater than or equal to --start-year")

    chunks = parse_chunks(args.chunks)

    with dask.config.set(scheduler=args.dask_scheduler):
        for year in range(args.start_year, args.end_year + 1):
            single_output, pressure_output = output_paths(args.output_dir, year)
            need_single = args.overwrite or not os.path.exists(single_output)
            need_pressure = args.overwrite or not os.path.exists(pressure_output)

            if not need_single and not need_pressure:
                logging.info("Skipping existing files for %s", year)
                continue

            open_datasets = []
            try:
                pressure = None
                if need_pressure:
                    logging.info("Opening ERA5 pressure-level files for %s", year)
                    pressure = open_group_dataset(args.input_dir, year, "pressure", chunks=chunks)
                    open_datasets.append(pressure)

                if need_single:
                    logging.info("Opening ERA5 single-level files for %s", year)
                    instant = open_group_dataset(args.input_dir, year, "instant", chunks=chunks)
                    open_datasets.append(instant)

                    logging.info("Building single-level dataset for %s", year)
                    single_ds = build_single_level_dataset(instant)
                    logging.info("Writing %s", single_output)
                    write_netcdf_file(single_ds, single_output, overwrite=args.overwrite)

                if need_pressure:
                    logging.info("Building pressure-level dataset for %s", year)
                    pressure_ds = build_pressure_level_dataset(pressure)
                    logging.info("Writing %s", pressure_output)
                    write_netcdf_file(pressure_ds, pressure_output, overwrite=args.overwrite)
            finally:
                for ds in open_datasets:
                    ds.close()

    logging.info("Done")


if __name__ == "__main__":
    main()
