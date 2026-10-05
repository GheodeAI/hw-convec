#!/usr/bin/env python3
"""Plot EUCLID CG occurrence and intensity spatial statistics."""

from __future__ import annotations

import argparse
from copy import copy
from pathlib import Path
import struct
import sys

import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 10, "legend.fontsize": 8,
    "xtick.labelsize": 8, "ytick.labelsize": 8})
import matplotlib.colors as mcolors
from matplotlib.path import Path as MplPath
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats


PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from common.paths import COMPOUND_DATA_DIR, COMPOUND_FIGURES_DIR, COUNTRIES_SHAPEFILE
from common.output import compact_stem, ensure_dir, figure_stem


DEFAULT_CG_DIR = PROJECT_DIR / "data" / "Euclid" / "monthly_nc_cg"
TEN_M_COUNTRIES_SHAPEFILE = (
    PROJECT_DIR
    / "data"
    / "shapefiles"
    / "ne_10m_admin_0_countries"
    / "ne_10m_admin_0_countries.shp"
)
DEFAULT_SHAPEFILE = TEN_M_COUNTRIES_SHAPEFILE if TEN_M_COUNTRIES_SHAPEFILE.exists() else COUNTRIES_SHAPEFILE
from common.config import MANUSCRIPT_FIGURES_DIR


def panel_letter(ax, c):
    ax.text(-.14, 1.08, c, transform=ax.transAxes, fontsize=11, fontweight="bold", va="top", ha="left")
DIFFERENCE_COLORS = [
    (0.0, "#b9ddea"),
    (0.35, "#edf5f7"),
    (0.50, "#fbfbfb"),
    (0.65, "#fff0cf"),
    (0.82, "#f6c17c"),
    (1.0, "#d9793d"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot mean annual CG days and mean CG-day flash intensity from EUCLID CG monthly NetCDF files."
        )
    )
    parser.add_argument("--start-year", type=int, default=2006)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--months", nargs="+", type=int, default=[4, 5, 6, 7, 8, 9])
    parser.add_argument("--cg-dir", default=str(DEFAULT_CG_DIR))
    parser.add_argument("--figures-dir", default=str(COMPOUND_FIGURES_DIR / "cg_spatial_stats"))
    parser.add_argument("--data-dir", default=str(COMPOUND_DATA_DIR / "cg_spatial_stats"))
    parser.add_argument("--shapefile", default=str(DEFAULT_SHAPEFILE))
    return parser.parse_args()


def monthly_path(cg_dir: Path, year: int, month: int) -> Path:
    return cg_dir / str(year) / f"euclid_flash_cg_{year}_{month:02d}.nc"


def first_existing_file(cg_dir: Path, years: list[int], months: list[int]) -> Path:
    for year in years:
        for month in months:
            path = monthly_path(cg_dir, year, month)
            if path.exists():
                return path
    raise FileNotFoundError(f"No EUCLID CG files found in {cg_dir}")


def area_weights(latitudes: xr.DataArray) -> xr.DataArray:
    weights = np.cos(np.deg2rad(latitudes))
    return weights / weights.mean()


def weighted_spatial_mean(data: xr.DataArray, weights: xr.DataArray) -> float:
    return float(data.weighted(weights).mean(("latitude", "longitude"), skipna=True).item())


def theil_sen_slope(annual: xr.DataArray, years: list[int]) -> xr.DataArray:
    year_values = np.asarray(years, dtype=float)
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
        p = 2.0 * stats.norm.sf(np.abs(z))
    p[(n < 2) | (var_s <= 0.0)] = np.nan
    return xr.DataArray(
        p.astype(np.float32),
        coords={"latitude": annual.latitude, "longitude": annual.longitude},
        dims=("latitude", "longitude"),
    )


def calculate_stats(
    cg_dir: Path,
    years: list[int],
    months: list[int],
    shapefile: Path | None,
) -> tuple[xr.Dataset, pd.DataFrame]:
    with xr.open_dataset(first_existing_file(cg_dir, years, months)) as sample:
        latitudes = sample.latitude.load()
        longitudes = sample.longitude.load()
        shape = (sample.sizes["latitude"], sample.sizes["longitude"])
    land_mask = build_land_mask(latitudes, longitudes, shapefile) if shapefile is not None else xr.ones_like(
        xr.DataArray(np.zeros(shape, dtype=bool), coords={"latitude": latitudes, "longitude": longitudes}, dims=("latitude", "longitude")),
        dtype=bool,
    )

    annual_cg_days = []
    annual_records = []
    cg_day_flash_sum = np.zeros(shape, dtype=np.float64)
    cg_day_count = np.zeros(shape, dtype=np.int32)
    weights = area_weights(latitudes)

    for year in years:
        year_cg_days = np.zeros(shape, dtype=np.int16)
        year_cg_day_flash_sum = np.zeros(shape, dtype=np.float64)
        year_cg_day_count = np.zeros(shape, dtype=np.int32)
        for month in months:
            path = monthly_path(cg_dir, year, month)
            if not path.exists():
                raise FileNotFoundError(f"Missing CG file: {path}")
            with xr.open_dataset(path) as ds:
                daily_flash = ds["flash_count"].resample(valid_time="1D").sum().load()
            daily_values = daily_flash.values.astype(np.float32)
            cg_day = daily_values > 0
            year_cg_days += cg_day.sum(axis=0).astype(np.int16)
            month_flash_sum = np.where(cg_day, daily_values, 0.0).sum(axis=0)
            month_day_count = cg_day.sum(axis=0).astype(np.int32)
            year_cg_day_flash_sum += month_flash_sum
            year_cg_day_count += month_day_count
            cg_day_flash_sum += month_flash_sum
            cg_day_count += month_day_count
        year_days_da = xr.DataArray(
            year_cg_days,
            coords={"latitude": latitudes, "longitude": longitudes},
            dims=("latitude", "longitude"),
        ).where(land_mask)
        with np.errstate(invalid="ignore", divide="ignore"):
            year_intensity = np.where(year_cg_day_count > 0, year_cg_day_flash_sum / year_cg_day_count, np.nan)
        year_intensity_da = xr.DataArray(
            year_intensity.astype(np.float32),
            coords={"latitude": latitudes, "longitude": longitudes},
            dims=("latitude", "longitude"),
        ).where(land_mask)
        annual_cg_days.append(
            year_days_da
        )
        annual_records.append(
            {
                "year": year,
                "mean_cg_days": weighted_spatial_mean(year_days_da, weights),
                "mean_cg_day_flash_intensity": weighted_spatial_mean(year_intensity_da, weights),
            }
        )
        print(
            f"Processed {year}: days={annual_records[-1]['mean_cg_days']:.2f}, "
            f"intensity={annual_records[-1]['mean_cg_day_flash_intensity']:.2f}",
            flush=True,
        )

    annual_cg_days_da = xr.concat(annual_cg_days, dim="year").assign_coords(year=years)
    mean_annual_cg_days = annual_cg_days_da.mean("year")
    cg_day_slope = theil_sen_slope(annual_cg_days_da, years).where(land_mask)
    cg_day_mk_p = mann_kendall_p_values(annual_cg_days_da)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_cg_day_intensity = cg_day_flash_sum / cg_day_count
    mean_cg_day_intensity_da = xr.DataArray(
        np.where(cg_day_count > 0, mean_cg_day_intensity, np.nan).astype(np.float32),
        coords={"latitude": latitudes, "longitude": longitudes},
        dims=("latitude", "longitude"),
    ).where(land_mask)
    cg_day_count_da = xr.DataArray(
        cg_day_count.astype(np.float32),
        coords={"latitude": latitudes, "longitude": longitudes},
        dims=("latitude", "longitude"),
    ).where(land_mask)

    return xr.Dataset(
        {
            "mean_annual_cg_days": mean_annual_cg_days.astype("float32"),
            "mean_cg_day_flash_intensity": mean_cg_day_intensity_da,
            "theil_sen_slope_cg_days": cg_day_slope.astype("float32"),
            "mk_significant_cg_days": (cg_day_mk_p < 0.05).where(land_mask),
            "cg_day_count": cg_day_count_da,
            "land_mask": land_mask,
        },
        coords={"latitude": latitudes, "longitude": longitudes},
        attrs={
            "description": "EUCLID CG spatial statistics.",
            "mean_annual_cg_days": "Annual number of days with at least one CG flash, averaged over years.",
            "mean_cg_day_flash_intensity": "Mean daily flash count on days with at least one CG flash.",
            "months": ",".join(f"{month:02d}" for month in months),
            "n_years": len(years),
            "land_mask": str(shapefile) if shapefile is not None else "none",
        },
    ), pd.DataFrame.from_records(annual_records)


def iter_shapefile_polygon_parts(path: Path):
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


def build_land_mask(latitudes: xr.DataArray, longitudes: xr.DataArray, shapefile: Path) -> xr.DataArray:
    if not shapefile.exists():
        raise FileNotFoundError(f"Missing shapefile for land mask: {shapefile}")

    lon = longitudes.values
    lat = latitudes.values
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
        is_land |= MplPath(coords).contains_points(points, radius=1.0e-9)

    return xr.DataArray(
        is_land.reshape(lat.size, lon.size),
        coords={"latitude": latitudes, "longitude": longitudes},
        dims=("latitude", "longitude"),
    )


def draw_boundaries(ax: plt.Axes, shapefile: Path | None, xlim: tuple[float, float], ylim: tuple[float, float]) -> None:
    if shapefile is None or not shapefile.exists():
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
        ax.plot(coords[:, 0], coords[:, 1], color="0.15", lw=0.45, alpha=0.8)


def finite_vmax(data: xr.DataArray) -> float:
    values = data.values
    finite = values[np.isfinite(values) & (values > 0)]
    if finite.size == 0:
        return 1.0
    return float(np.nanpercentile(finite, 99.5))


def signed_limits(data: xr.DataArray) -> tuple[float, float]:
    values = data.values
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return -1.0, 1.0
    vmax = float(np.nanpercentile(np.abs(finite), 98.0))
    if vmax <= 0.0:
        vmax = 1.0
    return -vmax, vmax


def add_significance_stippling(ax: plt.Axes, ds: xr.Dataset, field: xr.DataArray) -> None:
    if "mk_significant_cg_days" not in ds:
        return
    base_mask = (ds["mk_significant_cg_days"].values == 1) & np.isfinite(field.values)
    if not base_mask.any():
        return
    for mask, color in (
        (base_mask & (field.values < 0), "#0b4f8a"),
        (base_mask & (field.values > 0), "#9c4f0f"),
    ):
        y_idx, x_idx = np.nonzero(mask)
        if len(y_idx) == 0:
            continue
        ax.scatter(
            ds.longitude.values[x_idx],
            ds.latitude.values[y_idx],
            s=2.0,
            marker=".",
            color=color,
            alpha=0.75,
            linewidths=0,
            zorder=5,
        )


def plot_stats(ds: xr.Dataset, output_path: Path, start_year: int, end_year: int, shapefile: Path | None) -> None:
    panels = [
        ("mean_annual_cg_days", "Mean annual CG days", "days yr$^{-1}$", "YlOrRd"),
        ("mean_cg_day_flash_intensity", "Mean CG-day intensity", "flashes day$^{-1}$", "YlOrRd"),
        ("theil_sen_slope_cg_days", "Theil-Sen slope of CG days", "days yr$^{-2}$", "RdBu_r"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), constrained_layout=True)
    for ax, (name, title, label, cmap_name), c in zip(axes, panels, "abc"):
        data = ds[name].where(ds[name] > 0) if cmap_name != "RdBu_r" else ds[name]
        if cmap_name == "RdBu_r":
            cmap = mcolors.LinearSegmentedColormap.from_list("difference_soft_continuous", DIFFERENCE_COLORS)
        else:
            cmap = copy(plt.get_cmap(cmap_name))
        cmap.set_bad("white")
        if cmap_name == "RdBu_r":
            vmin, vmax = signed_limits(data)
        else:
            vmin, vmax = 0, finite_vmax(data)
        mesh = ax.pcolormesh(
            ds.longitude,
            ds.latitude,
            data,
            shading="auto",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_xlim(float(ds.longitude.min()), float(ds.longitude.max()))
        ax.set_ylim(float(ds.latitude.min()), float(ds.latitude.max()))
        ax.grid(alpha=0.2)
        draw_boundaries(ax, shapefile, ax.get_xlim(), ax.get_ylim())
        if cmap_name == "RdBu_r":
            add_significance_stippling(ax, ds, data)
        panel_letter(ax, c)
        cbar = fig.colorbar(mesh, ax=ax, shrink=0.88, fraction=0.025, aspect=36)
        cbar.set_label(label)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=600)
    MANUSCRIPT_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(MANUSCRIPT_FIGURES_DIR / "17_euclid_cg_mean_days_intensity.png", dpi=600)
    plt.close(fig)


def p_label(p_value: float) -> str:
    if not np.isfinite(p_value):
        return "p = NA"
    if p_value < 0.001:
        return "p < 0.001"
    return f"p = {p_value:.3f}"


def plot_series_with_trend(ax: plt.Axes, years: np.ndarray, values: np.ndarray, ylabel: str, color: str) -> None:
    years = np.asarray(years, dtype=float)
    values = np.asarray(values, dtype=float)
    ax.plot(years, values, color=color, marker="o", linewidth=2.0, markersize=4.5)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))


def plot_annual_timeseries(df: pd.DataFrame, output_path: Path, start_year: int, end_year: int) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True, sharex=True)
    years = df["year"].to_numpy(dtype=float)
    panels = [
        ("mean_cg_days", "Days yr$^{-1}$", "tab:blue"),
        ("mean_cg_day_flash_intensity", "flashes day$^{-1}$", "tab:orange"),
    ]
    for ax, (column, ylabel, color), c in zip(axes, panels, "ab"):
        plot_series_with_trend(ax, years, df[column].to_numpy(dtype=float), ylabel, color)
        ax.set_xlim(years.min() - 0.5, years.max() + 0.5)
        ax.set_xticks(years)
        ax.tick_params(axis="x", rotation=45)
        ax.set_xlabel("Year")
        panel_letter(ax, c)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=1200, facecolor="white")
    MANUSCRIPT_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(MANUSCRIPT_FIGURES_DIR / "16_euclid_cg_annual_timeseries.png", dpi=1200, facecolor="white")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    years = list(range(args.start_year, args.end_year + 1))
    months = sorted(set(args.months))
    shapefile = Path(args.shapefile) if args.shapefile else None
    ds, annual = calculate_stats(Path(args.cg_dir), years, months, shapefile)
    stem = compact_stem(f"euclid_cg_mean_days_intensity_{args.start_year}_{args.end_year}")
    annual_stem = compact_stem(f"euclid_cg_annual_timeseries_{args.start_year}_{args.end_year}")
    data_dir = ensure_dir(args.data_dir)
    figures_dir = ensure_dir(args.figures_dir)
    nc_path = data_dir / f"{stem}.nc"
    fig_path = figures_dir / f"{figure_stem(stem)}.png"
    csv_path = data_dir / f"{annual_stem}.csv"
    annual_fig_path = figures_dir / f"{figure_stem(annual_stem)}.png"
    ds.to_netcdf(nc_path)
    annual.to_csv(csv_path, index=False, float_format="%.4f", lineterminator="\n")
    plot_stats(ds, fig_path, args.start_year, args.end_year, shapefile)
    plot_annual_timeseries(annual, annual_fig_path, args.start_year, args.end_year)
    print(f"Wrote {nc_path}", flush=True)
    print(f"Wrote {csv_path}", flush=True)
    print(f"Wrote {fig_path}", flush=True)
    print(f"Wrote {annual_fig_path}", flush=True)


if __name__ == "__main__":
    main()
