import argparse
import os
import struct
from copy import copy
from pathlib import Path as FsPath
import sys

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
from matplotlib.path import Path
import numpy as np
from scipy import stats as scipy_stats
import xarray as xr
import warnings
warnings.filterwarnings('ignore')


PROJECT_DIR = FsPath(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from common.paths import COUNTRIES_SHAPEFILE, HEATWAVE_FIGURES_DIR, WETBULB_HEATWAVES_DIR

DEFAULT_PROCESSED_DIR = str(WETBULB_HEATWAVES_DIR)
DEFAULT_FIG_DIR = str(HEATWAVE_FIGURES_DIR / "historical_spatial")
TEN_M_COUNTRIES_SHAPEFILE = (
    PROJECT_DIR
    / "data"
    / "shapefiles"
    / "ne_10m_admin_0_countries"
    / "ne_10m_admin_0_countries.shp"
)
DEFAULT_SHAPEFILE = str(TEN_M_COUNTRIES_SHAPEFILE if TEN_M_COUNTRIES_SHAPEFILE.exists() else COUNTRIES_SHAPEFILE)
from common.config import MANUSCRIPT_FIGURES_DIR
from common.style import FULL_WIDTH, apply_style, label_panels

apply_style()
DEFAULT_CG_DIR = str(PROJECT_DIR / "data" / "Euclid" / "monthly_nc_cg")
THRESHOLD_DIRS = {
    "90p": "90p",
    "90p+20": "90p20",
    "90p+22": "90p22",
    "90p+23": "90p23",
    "90p+24": "90p24",
}
DEFAULT_SEASON_START_MONTH = 4
DEFAULT_SEASON_END_MONTH = 9
PANEL_ORDER = [
    "Mean annual heatwave days",
    "Mean intensity",
    "Theil-Sen slope of heatwave days",
]
DIFFERENCE_COLORS = [
    (0.0, "#b9ddea"),
    (0.35, "#edf5f7"),
    (0.50, "#fbfbfb"),
    (0.65, "#fff0cf"),
    (0.82, "#f6c17c"),
    (1.0, "#d9793d"),
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Plot 2006-2025 historical spatial wet-bulb heatwave statistics over Europe: "
            "mean annual days, mean annual event count, duration, intensity, and frequency."
        )
    )
    parser.add_argument(
        "--threshold",
        choices=sorted(THRESHOLD_DIRS),
        default="90p",
        help="Heatwave result folder to use. 90p+XX maps to processed/wetbulb/heatwaves/90pXX.",
    )
    parser.add_argument("--start-year", type=int, default=2006)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--processed-dir", default=DEFAULT_PROCESSED_DIR)
    parser.add_argument(
        "--input-dir",
        default=None,
        help="Override heatwave input directory. If omitted, it is inferred from --threshold.",
    )
    parser.add_argument("--fig-dir", default=DEFAULT_FIG_DIR)
    parser.add_argument("--shapefile", default=DEFAULT_SHAPEFILE)
    # Default to the paper's actual study region (Summary all.md Section 2),
    # not the full, wider ERA5 download extent -- pass --lat-min/-max/lon-min/-max
    # explicitly to override.
    parser.add_argument("--lat-min", type=float, default=36.0)
    parser.add_argument("--lat-max", type=float, default=72.0)
    parser.add_argument("--lon-min", type=float, default=-10.0)
    parser.add_argument("--lon-max", type=float, default=33.0)
    parser.add_argument(
        "--cg-domain",
        action="store_true",
        help="Calculate statistics over the ERA5-EUCLID CG latitude-longitude intersection.",
    )
    parser.add_argument("--cg-dir", default=DEFAULT_CG_DIR, help="EUCLID CG monthly NetCDF directory.")
    parser.add_argument(
        "--season-start-month",
        type=int,
        default=DEFAULT_SEASON_START_MONTH,
        help="First month to include.",
    )
    parser.add_argument(
        "--season-end-month",
        type=int,
        default=DEFAULT_SEASON_END_MONTH,
        help="Last month to include.",
    )
    parser.add_argument("--no-land-mask", action="store_true", help="Use all grid cells, including ocean.")
    parser.add_argument("--no-boundaries", action="store_true", help="Do not draw country boundaries.")
    return parser.parse_args()


def input_dir_from_threshold(args):
    if args.input_dir is not None:
        return args.input_dir
    return os.path.join(args.processed_dir, THRESHOLD_DIRS[args.threshold])


def heatwave_file(input_dir, year):
    return os.path.join(input_dir, f"wetbulb_heatwave_{year}.nc")


def year_files(input_dir, start_year, end_year):
    files = []
    for year in range(start_year, end_year + 1):
        path = heatwave_file(input_dir, year)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing heatwave file: {path}")
        files.append(path)
    return files


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


def select_region(ds, args):
    if args.lat_min is not None or args.lat_max is not None:
        lat_min = float(ds.latitude.min()) if args.lat_min is None else args.lat_min
        lat_max = float(ds.latitude.max()) if args.lat_max is None else args.lat_max
        lat0, lat1 = sorted((lat_min, lat_max))
        ds = ds.sel(latitude=slice(lat1, lat0))

    if args.lon_min is not None or args.lon_max is not None:
        lon_min = float(ds.longitude.min()) if args.lon_min is None else args.lon_min
        lon_max = float(ds.longitude.max()) if args.lon_max is None else args.lon_max
        lon0, lon1 = sorted((lon_min, lon_max))
        ds = ds.sel(longitude=slice(lon0, lon1))

    return ds


def select_season(ds, start_month, end_month):
    if start_month is None and end_month is None:
        return ds
    if start_month is None or end_month is None:
        raise ValueError("--season-start-month and --season-end-month must be used together.")
    if not (1 <= start_month <= 12 and 1 <= end_month <= 12):
        raise ValueError("Season months must be between 1 and 12.")

    if start_month <= end_month:
        months = range(start_month, end_month + 1)
        return ds.sel(time=ds.time.dt.month.isin(months))
    return ds.sel(time=(ds.time.dt.month >= start_month) | (ds.time.dt.month <= end_month))


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


def draw_boundaries(ax, shapefile, xlim, ylim):
    if not shapefile or not os.path.exists(shapefile):
        return

    lon_min, lon_max = xlim
    lat_min, lat_max = ylim
    for coords in iter_shapefile_polygon_parts(shapefile):
        if (
            coords[:, 0].max() < lon_min
            or coords[:, 0].min() > lon_max
            or coords[:, 1].max() < lat_min
            or coords[:, 1].min() > lat_max
        ):
            continue
        ax.plot(coords[:, 0], coords[:, 1], color="#555555", lw=0.35)


def threshold_for_intensity(ds):
    if "heatwave_threshold" in ds:
        return ds["heatwave_threshold"]
    return ds["threshold_90p"]


def build_historical_stats(ds):
    heatwave_day = ds["heatwave_day"].fillna(False).astype(bool)
    intensity = (ds["tw_daily_max"] - threshold_for_intensity(ds)).where(heatwave_day)

    annual_days = heatwave_day.groupby("time.year").sum("time")
    heatwave_day_slope = theil_sen_slope(annual_days, annual_days["year"])
    heatwave_day_mk_p = mann_kendall_p_values(annual_days)

    stats = {
        "Mean annual heatwave days": annual_days.mean("year").where(annual_days.mean("year") > 0),
        "Mean intensity": intensity.mean("time", skipna=True),
        "Theil-Sen slope of heatwave days": heatwave_day_slope.where(annual_days.any("year")),
        "MK significant heatwave days": (heatwave_day_mk_p < 0.05).where(annual_days.any("year")),
    }
    return {name: data.rename(name.lower().replace(" ", "_")) for name, data in stats.items()}


def theil_sen_slope(annual: xr.DataArray, years: xr.DataArray) -> xr.DataArray:
    year_values = years.values.astype(float)
    values = annual.transpose("year", "latitude", "longitude").values.astype(np.float32)
    slopes = []
    for first in range(len(year_values) - 1):
        for second in range(first + 1, len(year_values)):
            slopes.append((values[second] - values[first]) / (year_values[second] - year_values[first]))
    slope = np.nanmedian(np.stack(slopes, axis=0), axis=0)
    return xr.DataArray(
        slope.astype(np.float32),
        coords={"latitude": annual.latitude, "longitude": annual.longitude},
        dims=("latitude", "longitude"),
    )


def mann_kendall_p_values(annual: xr.DataArray) -> xr.DataArray:
    values = annual.transpose("year", "latitude", "longitude").values.astype(np.float32)
    valid = np.isfinite(values)
    n = valid.sum(axis=0).astype(np.float32)
    filled = np.where(valid, values, np.nan)
    s = np.zeros(values.shape[1:], dtype=np.float32)
    for first in range(values.shape[0] - 1):
        s += np.nansum(np.sign(filled[first + 1 :] - filled[first]), axis=0).astype(np.float32)

    var_s = n * (n - 1.0) * (2.0 * n + 5.0) / 18.0
    with np.errstate(invalid="ignore", divide="ignore"):
        z = np.where(s > 0, (s - 1.0) / np.sqrt(var_s), np.where(s < 0, (s + 1.0) / np.sqrt(var_s), 0.0))
        p = 2.0 * scipy_stats.norm.sf(np.abs(z))
    p[(n < 2) | (var_s <= 0.0)] = np.nan
    return xr.DataArray(
        p.astype(np.float32),
        coords={"latitude": annual.latitude, "longitude": annual.longitude},
        dims=("latitude", "longitude"),
    )


def mask_stats(stats, land_mask):
    return {name: data.where(land_mask) for name, data in stats.items()}


def plot_limits(title, data):
    values = np.asarray(data.values)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0, 1
    if "Theil-Sen slope" in title:
        vmax = float(np.nanpercentile(np.abs(values), 98.0))
        return -vmax, vmax
    return 0, float(values.max())


def colorbar_label(title):
    labels = {
        "Mean annual heatwave days": "Days yr$^{-1}$",
        "Mean intensity": "°C",
        "Theil-Sen slope of heatwave days": "Days yr$^{-2}$",
    }
    return labels[title]


def add_significance_stippling(ax, data, sig):
    if sig is None:
        return
    base_mask = (sig.values == 1) & np.isfinite(data.values)
    if not base_mask.any():
        return
    for mask, color in (
        (base_mask & (data.values < 0), "#0b4f8a"),
        (base_mask & (data.values > 0), "#9c4f0f"),
    ):
        y_idx, x_idx = np.nonzero(mask)
        if len(y_idx) == 0:
            continue
        ax.scatter(
            data.longitude.values[x_idx],
            data.latitude.values[y_idx],
            s=0.4,
            marker=".",
            color=color,
            alpha=0.75,
            linewidths=0,
            zorder=5,
        )


def figure_suffix(args):
    if args.season_start_month is None:
        season = "fullyear"
    else:
        season = f"{args.season_start_month:02d}_{args.season_end_month:02d}"
    domain = "cg_domain_" if args.cg_domain else ""
    return f"{domain}{args.threshold.replace('+', 'plus')}_{season}"


def plot_stats(stats, args, shapefile=None):
    # Same layout as Fig 1: maps at equal aspect with the colour bar underneath.
    fig, axes = plt.subplots(1, 3, figsize=(FULL_WIDTH, 3.2), constrained_layout=True)

    for ax, title in zip(np.atleast_1d(axes).flat, PANEL_ORDER):
        data = stats[title]
        vmin, vmax = plot_limits(title, data)
        xlim = (float(data.longitude.min()), float(data.longitude.max()))
        ylim = (float(data.latitude.min()), float(data.latitude.max()))
        if "Theil-Sen slope" in title:
            cmap = mcolors.LinearSegmentedColormap.from_list("difference_soft_continuous", DIFFERENCE_COLORS)
        else:
            cmap = copy(plt.get_cmap("YlOrRd"))
        cmap.set_bad("white")
        mesh = ax.pcolormesh(
            data.longitude,
            data.latitude,
            data,
            shading="auto",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal")
        ax.xaxis.set_major_locator(mticker.MultipleLocator(10))
        ax.yaxis.set_major_locator(mticker.MultipleLocator(10))
        draw_boundaries(ax, shapefile, xlim, ylim)
        if "Theil-Sen slope" in title:
            add_significance_stippling(ax, data, stats.get("MK significant heatwave days"))
        cbar = fig.colorbar(mesh, ax=ax, orientation="horizontal", fraction=0.05, pad=0.04)
        cbar.set_label(colorbar_label(title))
    label_panels(fig, np.atleast_1d(axes).flat, "abc")

    output = os.path.join(args.fig_dir, f"wetbulb_heatwave_historical_spatial_{figure_suffix(args)}.png")
    fig.savefig(output, dpi=600, bbox_inches="tight")
    if args.threshold == "90p+20" and args.season_start_month == 4 and args.season_end_month == 9:
        MANUSCRIPT_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(MANUSCRIPT_FIGURES_DIR / "18_wetbulb_heatwave_historical_spatial.png", dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output}")


def main():
    args = parse_args()
    if args.end_year < args.start_year:
        raise ValueError("--end-year must be greater than or equal to --start-year")

    os.makedirs(args.fig_dir, exist_ok=True)
    input_dir = input_dir_from_threshold(args)
    files = year_files(input_dir, args.start_year, args.end_year)
    needed = [
        "tw_daily_max",
        "threshold_90p",
        "heatwave_day",
        "heatwave_duration",
    ]

    sample = xr.open_dataset(files[0])
    if "heatwave_threshold" in sample:
        needed.append("heatwave_threshold")
    sample.close()

    ds = xr.open_mfdataset(
        files,
        combine="by_coords",
        data_vars="minimal",
        coords="minimal",
        compat="override",
        chunks={"time": -1},
    )[needed]
    ds = select_region(select_season(ds, args.season_start_month, args.season_end_month), args)
    if args.cg_domain:
        bounds = cg_intersection_bounds(input_dir, args.cg_dir, args.start_year, args.end_year)
        print(
            f"Using ERA5-CG intersection: lat {bounds[0]:.4f}-{bounds[1]:.4f}, lon {bounds[2]:.4f}-{bounds[3]:.4f}",
            flush=True,
        )
        ds = select_bbox(ds, bounds)

    land_mask = None
    if not args.no_land_mask:
        land_mask = build_land_mask(ds, args.shapefile)
        ds = ds.where(land_mask)

    stats = build_historical_stats(ds)
    if land_mask is not None:
        stats = mask_stats(stats, land_mask)

    shapefile = None if args.no_boundaries else args.shapefile
    plot_stats(stats, args, shapefile=shapefile)


if __name__ == "__main__":
    main()
