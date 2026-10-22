import argparse
import os
import struct
from pathlib import Path as FsPath
import sys

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.path import Path
import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


PROJECT_DIR = FsPath(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from common.figures import figure_path, save_figure
from common.paths import COUNTRIES_SHAPEFILE, HEATWAVE_DIR, WETBULB_HEATWAVES_DIR

DEFAULT_PROCESSED_DIR = str(WETBULB_HEATWAVES_DIR)
DEFAULT_TABLE_DIR = str(HEATWAVE_DIR)
DEFAULT_SHAPEFILE = str(COUNTRIES_SHAPEFILE)
from common.style import BLUE, FULL_WIDTH, apply_style, label_panels

apply_style()
DEFAULT_CG_DIR = str(PROJECT_DIR / "data" / "Euclid" / "monthly_nc_cg")
THRESHOLD_DIRS = {
    "90p": "90p",
    "90p+20": "90p20",
    "90p+22": "90p22",
    "90p+23": "90p23",
    "90p+24": "90p24",
}
SEASON_START_MONTH = 4
SEASON_END_MONTH = 9


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Plot 2006-2025 annual Europe-land wet-bulb heatwave statistics: "
            "mean heatwave days, mean event count per grid cell, mean event duration, "
            "and longest event duration."
        )
    )
    parser.add_argument(
        "--threshold",
        choices=sorted(THRESHOLD_DIRS),
        default="90p+20",
        help="Heatwave definition to plot. 90p uses only the percentile threshold; 90p+XX uses max(90p, XX degC).",
    )
    parser.add_argument("--start-year", type=int, default=2006)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument(
        "--processed-dir",
        default=DEFAULT_PROCESSED_DIR,
        help="Directory containing heatwave method folders such as 90p, 90p20, and 90p22.",
    )
    parser.add_argument(
        "--input-dir",
        default=None,
        help="Override heatwave input directory. If omitted, it is inferred from --threshold.",
    )
    parser.add_argument("--table-dir", default=DEFAULT_TABLE_DIR, help="Directory for output CSV tables.")
    parser.add_argument("--shapefile", default=DEFAULT_SHAPEFILE, help="Country boundary shapefile path.")
    parser.add_argument(
        "--cg-domain",
        action="store_true",
        help="Calculate statistics over the ERA5-EUCLID CG latitude-longitude intersection.",
    )
    parser.add_argument("--cg-dir", default=DEFAULT_CG_DIR, help="EUCLID CG monthly NetCDF directory.")
    parser.add_argument("--lat-min", type=float, default=None)
    parser.add_argument("--lat-max", type=float, default=None)
    parser.add_argument("--lon-min", type=float, default=None)
    parser.add_argument("--lon-max", type=float, default=None)
    parser.add_argument("--no-land-mask", action="store_true", help="Use all grid cells in the file, including ocean.")
    parser.add_argument("--no-csv", action="store_true", help="Do not save the annual statistics CSV.")
    return parser.parse_args()


def heatwave_file(input_dir, year):
    return os.path.join(input_dir, f"wetbulb_heatwave_{year}.nc")


def input_dir_from_threshold(args):
    if args.input_dir is not None:
        return args.input_dir
    return os.path.join(args.processed_dir, THRESHOLD_DIRS[args.threshold])


def first_cg_file(cg_dir, start_year, end_year):
    cg_dir = FsPath(cg_dir)
    for year in range(start_year, end_year + 1):
        files = sorted((cg_dir / str(year)).glob("euclid_flash_cg_*.nc"))
        if files:
            return files[0]
    raise FileNotFoundError(f"No CG monthly NetCDF files found in {cg_dir}")


def cg_intersection_bounds(input_dir, cg_dir, start_year, end_year):
    with xr.open_dataset(heatwave_file(input_dir, start_year)) as era5, xr.open_dataset(
        first_cg_file(cg_dir, start_year, end_year)
    ) as cg:
        lat_min = max(float(era5.latitude.min()), float(cg.latitude.min()))
        lat_max = min(float(era5.latitude.max()), float(cg.latitude.max()))
        lon_min = max(float(era5.longitude.min()), float(cg.longitude.min()))
        lon_max = min(float(era5.longitude.max()), float(cg.longitude.max()))
    if lat_min >= lat_max or lon_min >= lon_max:
        raise ValueError("ERA5 and CG domains do not overlap.")
    return lat_min, lat_max, lon_min, lon_max


def select_bbox(ds, bounds):
    lat_min, lat_max, lon_min, lon_max = bounds
    if float(ds.latitude[0]) > float(ds.latitude[-1]):
        lat_slice = slice(lat_max, lat_min)
    else:
        lat_slice = slice(lat_min, lat_max)
    return ds.sel(latitude=lat_slice, longitude=slice(lon_min, lon_max))


def override_bounds(bounds, args):
    if not any(value is not None for value in (args.lat_min, args.lat_max, args.lon_min, args.lon_max)):
        return bounds
    if bounds is None:
        raise ValueError("Explicit latitude-longitude bounds require --cg-domain for this timeseries script.")
    lat_min, lat_max, lon_min, lon_max = bounds
    return (
        lat_min if args.lat_min is None else args.lat_min,
        lat_max if args.lat_max is None else args.lat_max,
        lon_min if args.lon_min is None else args.lon_min,
        lon_max if args.lon_max is None else args.lon_max,
    )


def event_start_mask(event_id):
    has_event = event_id > 0
    previous_id = event_id.shift(time=1, fill_value=0)
    return has_event & (event_id != previous_id)


def iter_shapefile_polygon_parts(path):
    with open(path, "rb") as shp:
        shp.seek(100)
        while True:
            header = shp.read(8)
            if len(header) < 8:
                break

            _, content_length_words = struct.unpack(">2i", header)
            content = shp.read(content_length_words * 2)
            if len(content) < 44:
                continue

            shape_type = struct.unpack("<i", content[:4])[0]
            if shape_type == 0:
                continue
            if shape_type not in (5, 15, 25):
                continue

            num_parts, num_points = struct.unpack("<2i", content[36:44])
            parts_start = 44
            points_start = parts_start + 4 * num_parts
            parts = list(struct.unpack(f"<{num_parts}i", content[parts_start:points_start]))
            parts.append(num_points)

            for start, end in zip(parts[:-1], parts[1:]):
                point_offset = points_start + start * 16
                point_count = end - start
                coords = np.frombuffer(content, dtype="<f8", count=point_count * 2, offset=point_offset)
                if coords.size:
                    yield coords.reshape(-1, 2)


def build_land_mask(ds, shapefile):
    if not shapefile or not os.path.exists(shapefile):
        raise FileNotFoundError(f"Missing shapefile for land mask: {shapefile}")

    lon = ds.longitude.values
    lat = ds.latitude.values
    lon2d, lat2d = np.meshgrid(lon, lat)
    points = np.column_stack([lon2d.ravel(), lat2d.ravel()])
    is_land = np.zeros(points.shape[0], dtype=bool)

    lon_min, lon_max = float(lon.min()), float(lon.max())
    lat_min, lat_max = float(lat.min()), float(lat.max())

    for coords in iter_shapefile_polygon_parts(shapefile):
        if (
            coords[:, 0].max() < lon_min
            or coords[:, 0].min() > lon_max
            or coords[:, 1].max() < lat_min
            or coords[:, 1].min() > lat_max
        ):
            continue
        is_land |= Path(coords).contains_points(points, radius=1.0e-9)

    mask = is_land.reshape(lat.size, lon.size)
    return xr.DataArray(mask, coords={"latitude": ds.latitude, "longitude": ds.longitude}, dims=("latitude", "longitude"))


def area_weights(ds):
    weights = np.cos(np.deg2rad(ds.latitude))
    return weights / weights.mean()


def weighted_spatial_mean(data, weights):
    return data.weighted(weights).mean(("latitude", "longitude"), skipna=True)


def weighted_event_duration_mean(duration_at_starts, weights):
    event_weights = xr.ones_like(duration_at_starts, dtype="float32") * weights
    return duration_at_starts.weighted(event_weights).mean(("time", "latitude", "longitude"), skipna=True)


def weighted_time_space_mean(data, weights):
    data_weights = xr.ones_like(data, dtype="float32") * weights
    return data.weighted(data_weights).mean(("time", "latitude", "longitude"), skipna=True)


def threshold_for_intensity(ds):
    if "heatwave_threshold" in ds:
        return ds["heatwave_threshold"]
    return ds["threshold_90p"]


def year_stats(ds, land_mask=None):
    weights = area_weights(ds)
    season = ds.time.dt.month.isin(range(SEASON_START_MONTH, SEASON_END_MONTH + 1))
    heatwave_days = ds["heatwave_day"].where(season, False).sum("time")
    starts = event_start_mask(ds["heatwave_event_id"]).where(season, False)
    event_count = starts.sum("time")
    duration_at_starts = ds["heatwave_duration"].where(starts)
    max_duration = ds["heatwave_duration"].where((ds["heatwave_event_id"] > 0) & season).max(skipna=True)
    heatwave_mask = ds["heatwave_day"].where(season, False).astype(bool)
    intensity = (ds["tw_daily_max"] - threshold_for_intensity(ds)).where(heatwave_mask)

    if land_mask is not None:
        heatwave_days = heatwave_days.where(land_mask)
        event_count = event_count.where(land_mask)
        duration_at_starts = duration_at_starts.where(land_mask)
        max_duration = ds["heatwave_duration"].where((ds["heatwave_event_id"] > 0) & season & land_mask).max(skipna=True)
        intensity = intensity.where(land_mask)

    return {
        "mean_heatwave_days": float(weighted_spatial_mean(heatwave_days, weights).item()),
        "mean_heatwave_intensity": float(weighted_time_space_mean(intensity, weights).item()),
        "mean_event_count": float(weighted_spatial_mean(event_count, weights).item()),
        "mean_event_duration": float(weighted_event_duration_mean(duration_at_starts, weights).item()),
        "max_event_duration": float(max_duration.item()),
    }


def collect_annual_stats(input_dir, years, shapefile, use_land_mask=True, bounds=None):
    records = []
    land_mask = None

    for year in years:
        path = heatwave_file(input_dir, year)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing heatwave file: {path}")

        needed = ["heatwave_day", "heatwave_event_id", "heatwave_duration", "tw_daily_max", "threshold_90p"]
        with xr.open_dataset(path) as ds_in:
            if "heatwave_threshold" in ds_in:
                needed.append("heatwave_threshold")
            ds = ds_in[needed].load()
        if bounds is not None:
            ds = select_bbox(ds, bounds)

        if use_land_mask and land_mask is None:
            land_mask = build_land_mask(ds, shapefile)

        stats = year_stats(ds, land_mask=land_mask if use_land_mask else None)
        stats["year"] = year
        records.append(stats)
        print(
            f"{year}: days={stats['mean_heatwave_days']:.2f}, "
            f"intensity={stats['mean_heatwave_intensity']:.2f}, "
            f"events={stats['mean_event_count']:.2f}, "
            f"duration={stats['mean_event_duration']:.2f}, "
            f"max={stats['max_event_duration']:.0f}"
        )

    return pd.DataFrame.from_records(records).set_index("year")


def p_label(p_value):
    if not np.isfinite(p_value):
        return "p = NA"
    if p_value < 0.001:
        return "p < 0.001"
    return f"p = {p_value:.3f}"


def plot_series_with_trend(ax, years, values, ylabel, color):
    years = np.asarray(years, dtype=float)
    values = np.asarray(values, dtype=float)
    ax.plot(years, values, marker="o", lw=1.3, color=color, markersize=3.5)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))


def plot_annual_stats(df, threshold, domain_tag="europe"):
    fig, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH, 2.8), constrained_layout=True, sharex=True)
    panels = [
        ("mean_heatwave_days", "Days yr$^{-1}$", BLUE),
        ("mean_heatwave_intensity", "°C", BLUE),
    ]

    years = df.index.to_numpy()
    for ax, (column, ylabel, color) in zip(np.atleast_1d(axes).flat, panels):
        plot_series_with_trend(ax, years, df[column].to_numpy(dtype=float), ylabel, color)
        ax.set_xlim(years.min() - 0.5, years.max() + 0.5)
        ax.set_xticks(years[::2])
        ax.set_xticks(years, minor=True)
        ax.tick_params(axis="x", rotation=45)
        ax.set_xlabel("Year")
    label_panels(fig, np.atleast_1d(axes).flat, "ab")

    if domain_tag == "cg_domain" and threshold == "90p+20":

        save_figure(fig, 'wetbulb_heatwave_cg_domain_annual_timeseries', dpi=1200, bbox_inches="tight")
    plt.close(fig)
    return figure_path('wetbulb_heatwave_cg_domain_annual_timeseries')


def main():
    args = parse_args()
    if args.end_year < args.start_year:
        raise ValueError("--end-year must be greater than or equal to --start-year")

    if not args.no_csv:
        os.makedirs(args.table_dir, exist_ok=True)
    input_dir = input_dir_from_threshold(args)
    years = range(args.start_year, args.end_year + 1)
    bounds = None
    domain_tag = "europe"
    if args.cg_domain:
        bounds = cg_intersection_bounds(input_dir, args.cg_dir, args.start_year, args.end_year)
        bounds = override_bounds(bounds, args)
        domain_tag = "cg_domain"
        print(
            f"Using heatwave domain: lat {bounds[0]:.4f}-{bounds[1]:.4f}, lon {bounds[2]:.4f}-{bounds[3]:.4f}",
            flush=True,
        )
    df = collect_annual_stats(
        input_dir,
        years,
        args.shapefile,
        use_land_mask=not args.no_land_mask,
        bounds=bounds,
    )

    if not args.no_csv:
        csv_output = os.path.join(
            args.table_dir,
            f"wetbulb_heatwave_{domain_tag}_annual_timeseries_AprSep_{args.threshold.replace('+', 'plus')}.csv",
        )
        df.to_csv(csv_output, float_format="%.4f")
        print(f"Saved {csv_output}")

    plot_annual_stats(df, args.threshold, domain_tag=domain_tag)


if __name__ == "__main__":
    main()
