#!/usr/bin/env python3
"""Plot compound event location density maps for ESWD and EUCLID."""

from __future__ import annotations

import argparse
from copy import copy
from pathlib import Path
import re
import struct
import sys

import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 10, "legend.fontsize": 8,
    "xtick.labelsize": 8, "ytick.labelsize": 8})
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.path import Path as MplPath
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import PowerNorm
from scipy import stats
from scipy.ndimage import gaussian_filter


PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from common.paths import (
    COMPOUND_ESWD_DIR,
    COMPOUND_EUCLID_DIR,
    COMPOUND_DATA_DIR,
    COMPOUND_FIGURES_DIR,
    COUNTRIES_SHAPEFILE,
    WETBULB_HEATWAVES_DIR,
)


EVENT_ID_PATTERN = re.compile(r"_lat(?P<lat>-?\d+(?:\.\d+)?)_lon(?P<lon>-?\d+(?:\.\d+)?)_event")
DEFAULT_METHOD = "90p20"
TEN_M_COUNTRIES_SHAPEFILE = (
    PROJECT_DIR
    / "data"
    / "shapefiles"
    / "ne_10m_admin_0_countries"
    / "ne_10m_admin_0_countries.shp"
)
DEFAULT_SHAPEFILE = TEN_M_COUNTRIES_SHAPEFILE if TEN_M_COUNTRIES_SHAPEFILE.exists() else COUNTRIES_SHAPEFILE
WHITE_TO_RED_CMAP = LinearSegmentedColormap.from_list(
    "white_to_red",
    ["#ffffff", "#fee5d9", "#fcae91", "#fb6a4a", "#de2d26", "#a50f15"],
)
DIFFERENCE_COLORS = [
    (0.0, "#b9ddea"),
    (0.35, "#edf5f7"),
    (0.50, "#fbfbfb"),
    (0.65, "#fff0cf"),
    (0.82, "#f6c17c"),
    (1.0, "#d9793d"),
]
from common.config import MANUSCRIPT_FIGURES_DIR


def panel_letter(ax, c):
    ax.text(-.14, 1.08, c, transform=ax.transAxes, fontsize=11, fontweight="bold", va="top", ha="left")


SOURCE_CONFIG = {
    "eswd": {
        "label": "ESWD",
        "csv_prefix": "compound_eswd_heatwave_convection_events",
        "netcdf_prefix": "eswd_compound_event_location_density",
        "event_id_column": "heatwave_event_id",
    },
    "euclid": {
        "label": "EUCLID",
        "csv_prefix": "compound_euclid_heatwave_lightning_events",
        "netcdf_prefix": "euclid_compound_event_location_density",
        "event_id_column": "heatwave_event_id",
    },
    "euclid_cg": {
        "label": "EUCLID CG",
        "csv_prefix": "compound_cg_heatwave_lightning_events",
        "netcdf_prefix": "cg_compound_event_location_density",
        "event_id_column": "heatwave_event_id",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot annualized compound event location density maps from ESWD and EUCLID "
            "grid-cell compound-event tables."
        )
    )
    parser.add_argument("--method", default=DEFAULT_METHOD)
    parser.add_argument("--start-year", type=int, default=2006)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--eswd-dir", default=str(COMPOUND_ESWD_DIR))
    parser.add_argument("--euclid-dir", default=str(COMPOUND_EUCLID_DIR))
    parser.add_argument("--euclid-cg-dir", default=str(PROJECT_DIR / "processed" / "compound" / "euclid_cg"))
    parser.add_argument("--euclid-cg-grid-dir", default=str(PROJECT_DIR / "data" / "Euclid" / "monthly_nc_cg"))
    parser.add_argument(
        "--cg-smooth-sigma",
        type=float,
        default=3.0,
        help="Gaussian smoothing sigma in CG grid cells for euclid_cg density maps. Use 0 for raw grid-cell counts.",
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        choices=sorted(SOURCE_CONFIG),
        default=["eswd", "euclid"],
        help="One or more compound-event sources to plot.",
    )
    parser.add_argument(
        "--figures-dir",
        default=str(COMPOUND_FIGURES_DIR / "location_density"),
    )
    parser.add_argument(
        "--data-dir",
        default=str(COMPOUND_DATA_DIR / "location_density"),
    )
    parser.add_argument(
        "--shapefile",
        default=str(DEFAULT_SHAPEFILE),
        help="Country boundary shapefile drawn over the density map.",
    )
    parser.add_argument(
        "--aggregate-euclid-cg-grid",
        action="store_true",
        help="Aggregate EUCLID CG events onto a regular lat/lon grid before mapping.",
    )
    parser.add_argument("--aggregate-lat-min", type=float, default=36.0)
    parser.add_argument("--aggregate-lat-max", type=float, default=72.0)
    parser.add_argument("--aggregate-lon-min", type=float, default=-10.0)
    parser.add_argument("--aggregate-lon-max", type=float, default=33.0)
    parser.add_argument("--aggregate-resolution", type=float, default=0.25)
    return parser.parse_args()


def heatwave_path(method_dir: Path, year: int) -> Path:
    return method_dir / f"wetbulb_heatwave_{year}.nc"


def open_sample_grid(method: str, start_year: int) -> tuple[np.ndarray, np.ndarray]:
    path = heatwave_path(WETBULB_HEATWAVES_DIR / method, start_year)
    if not path.exists():
        raise FileNotFoundError(f"Missing heatwave file: {path}")
    with xr.open_dataset(path) as ds:
        return ds["latitude"].values, ds["longitude"].values


def open_cg_grid(cg_grid_dir: Path, start_year: int, end_year: int) -> tuple[np.ndarray, np.ndarray]:
    for year in range(start_year, end_year + 1):
        files = sorted((cg_grid_dir / str(year)).glob("euclid_flash_cg_*.nc"))
        if not files:
            continue
        with xr.open_dataset(files[0]) as ds:
            return ds["latitude"].values, ds["longitude"].values
    raise FileNotFoundError(f"No EUCLID CG monthly NetCDF files found in {cg_grid_dir}")


def regular_grid(lat_min: float, lat_max: float, lon_min: float, lon_max: float, resolution: float) -> tuple[np.ndarray, np.ndarray]:
    latitudes = np.arange(lat_min, lat_max + resolution * 0.5, resolution, dtype=np.float64)
    longitudes = np.arange(lon_min, lon_max + resolution * 0.5, resolution, dtype=np.float64)
    return latitudes, longitudes


def grid_tag(resolution: float) -> str:
    return f"{resolution:g}deg".replace(".", "p")


def nearest_index(values: np.ndarray, targets: np.ndarray) -> np.ndarray:
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


def parse_event_coordinates(event_ids: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    lats: list[float] = []
    lons: list[float] = []
    for event_id in event_ids.astype(str):
        match = EVENT_ID_PATTERN.search(event_id)
        if match is None:
            raise ValueError(f"Could not parse latitude/longitude from event id: {event_id}")
        lats.append(float(match.group("lat")))
        lons.append(float(match.group("lon")))
    return np.asarray(lats, dtype=np.float64), np.asarray(lons, dtype=np.float64)


def build_land_mask(latitudes: np.ndarray, longitudes: np.ndarray, shapefile: Path | None) -> np.ndarray:
    if shapefile is None or not shapefile.exists():
        return np.ones((len(latitudes), len(longitudes)), dtype=bool)

    lon2d, lat2d = np.meshgrid(longitudes, latitudes)
    points = np.column_stack([lon2d.ravel(), lat2d.ravel()])
    is_land = np.zeros(points.shape[0], dtype=bool)

    lon_min, lon_max = float(np.nanmin(longitudes)), float(np.nanmax(longitudes))
    lat_min, lat_max = float(np.nanmin(latitudes)), float(np.nanmax(latitudes))
    for coords in iter_shapefile_polygon_parts(shapefile):
        if (
            coords[:, 0].max() < lon_min
            or coords[:, 0].min() > lon_max
            or coords[:, 1].max() < lat_min
            or coords[:, 1].min() > lat_max
        ):
            continue
        is_land |= MplPath(coords).contains_points(points, radius=1.0e-9)

    return is_land.reshape(len(latitudes), len(longitudes))


def theil_sen_slope(values: np.ndarray, years: np.ndarray) -> np.ndarray:
    slopes = []
    for first in range(len(years) - 1):
        for second in range(first + 1, len(years)):
            slopes.append((values[second] - values[first]) / (years[second] - years[first]))
    return np.nanmedian(np.stack(slopes, axis=0), axis=0).astype(np.float32)


def mann_kendall_p_values(values: np.ndarray) -> np.ndarray:
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
    return p.astype(np.float32)


def load_density(
    csv_path: Path,
    event_id_column: str,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    n_years: int,
    start_year: int,
    end_year: int,
    shapefile: Path | None,
    smooth_sigma: float = 0.0,
    crop_to_events: bool = True,
) -> xr.Dataset:
    df = pd.read_csv(csv_path, usecols=[event_id_column, "start_time", "duration"])
    event_lats, event_lons = parse_event_coordinates(df[event_id_column])
    in_domain = (
        (event_lats >= float(np.nanmin(latitudes)))
        & (event_lats <= float(np.nanmax(latitudes)))
        & (event_lons >= float(np.nanmin(longitudes)))
        & (event_lons <= float(np.nanmax(longitudes)))
    )
    df = df.loc[in_domain].reset_index(drop=True)
    event_lats = event_lats[in_domain]
    event_lons = event_lons[in_domain]
    if len(df) == 0:
        raise ValueError(f"No events from {csv_path} fall inside the requested grid domain.")
    lat_idx = nearest_index(latitudes, event_lats).astype(np.int64)
    lon_idx = nearest_index(longitudes, event_lons).astype(np.int64)
    years = np.arange(start_year, end_year + 1, dtype=np.int32)
    event_years = pd.to_datetime(df["start_time"], utc=True).dt.year.to_numpy(dtype=np.int32)
    valid_year = (event_years >= start_year) & (event_years <= end_year)
    year_idx = event_years[valid_year] - start_year
    durations = pd.to_numeric(df["duration"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)

    counts = np.zeros((len(latitudes), len(longitudes)), dtype=np.int32)
    np.add.at(counts, (lat_idx, lon_idx), 1)
    annual_counts = np.zeros((len(years), len(latitudes), len(longitudes)), dtype=np.float32)
    np.add.at(annual_counts, (year_idx, lat_idx[valid_year], lon_idx[valid_year]), 1.0)
    annual_days = np.zeros((len(years), len(latitudes), len(longitudes)), dtype=np.float32)
    np.add.at(annual_days, (year_idx, lat_idx[valid_year], lon_idx[valid_year]), durations[valid_year])
    land_mask = build_land_mask(latitudes, longitudes, shapefile)
    counts = np.where(land_mask, counts, 0).astype(np.int32)
    annual_counts = np.where(land_mask[None, :, :], annual_counts, np.nan).astype(np.float32)
    annual_days = np.where(land_mask[None, :, :], annual_days, np.nan).astype(np.float32)
    annualized = np.where(land_mask, counts.astype(np.float32) / float(n_years), np.nan)
    mean_annual_days = np.where(land_mask, np.nansum(annual_days, axis=0) / float(n_years), np.nan)
    smoothed = None
    if smooth_sigma > 0.0:
        smoothed_raw = gaussian_filter(counts.astype(np.float32), sigma=smooth_sigma, mode="nearest") / float(n_years)
        smoothed = np.where(land_mask, smoothed_raw, np.nan)
    slope = np.where(land_mask, theil_sen_slope(annual_counts, years.astype(float)), np.nan)
    mk_p = np.where(land_mask, mann_kendall_p_values(annual_counts), np.nan)
    mk_significant = np.isfinite(slope) & (mk_p < 0.05)

    data_vars = {
        "compound_event_count": (("latitude", "longitude"), counts),
        "annual_compound_event_count": (("year", "latitude", "longitude"), annual_counts),
        "annual_compound_event_days": (("year", "latitude", "longitude"), annual_days),
        "mean_annual_compound_event_days": (("latitude", "longitude"), mean_annual_days.astype(np.float32)),
        "compound_event_density_per_year": (("latitude", "longitude"), annualized),
        "theil_sen_slope_compound_events": (("latitude", "longitude"), slope.astype(np.float32)),
        "compound_event_mk_p_value": (("latitude", "longitude"), mk_p.astype(np.float32)),
        "compound_event_mk_significant": (("latitude", "longitude"), mk_significant),
        "land_mask": (("latitude", "longitude"), land_mask),
    }
    if smoothed is not None:
        data_vars["compound_event_density_per_year_smoothed"] = (("latitude", "longitude"), smoothed.astype(np.float32))

    ds = xr.Dataset(
        data_vars,
        coords={"year": years, "latitude": latitudes, "longitude": longitudes},
        attrs={
            "description": "Grid-cell compound event location density.",
            "n_years": n_years,
            "smooth_sigma_gridcells": smooth_sigma,
            "trend": "Theil-Sen slope of annual compound-event counts by grid cell; MK p-values use the Mann-Kendall test.",
            "trend_units": "events yr^-1 per grid cell",
            "event_latitude_min": float(np.nanmin(event_lats)),
            "event_latitude_max": float(np.nanmax(event_lats)),
            "event_longitude_min": float(np.nanmin(event_lons)),
            "event_longitude_max": float(np.nanmax(event_lons)),
        },
    )
    if crop_to_events:
        return crop_density_to_event_extent(ds, lat_idx, lon_idx)
    return ds


def crop_density_to_event_extent(ds: xr.Dataset, lat_idx: np.ndarray, lon_idx: np.ndarray) -> xr.Dataset:
    lat_start = int(lat_idx.min())
    lat_stop = int(lat_idx.max()) + 1
    lon_start = int(lon_idx.min())
    lon_stop = int(lon_idx.max()) + 1
    return ds.isel(latitude=slice(lat_start, lat_stop), longitude=slice(lon_start, lon_stop))


def panel_limits(values: np.ndarray) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 1.0
    positive = finite[finite > 0.0]
    reference = positive if positive.size else finite
    vmax = float(np.nanpercentile(reference, 99.5))
    if vmax <= 0.0 or not np.isfinite(vmax):
        vmax = float(np.nanmax(reference)) or 1.0
    return 0.0, vmax


def signed_limits(values: np.ndarray) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    finite = finite[finite != 0.0]
    if finite.size == 0:
        return -1.0, 1.0
    vmax = float(np.nanpercentile(np.abs(finite), 99.0))
    if vmax <= 0.0 or not np.isfinite(vmax):
        vmax = float(np.nanmax(np.abs(finite))) or 1.0
    return -vmax, vmax


def plot_density_field(ax: plt.Axes, field: xr.DataArray, cmap: LinearSegmentedColormap, vmax: float):
    data = field.where(field > 0.0)
    if "land_mask" in field.coords:
        data = data.where(field.coords["land_mask"])
    lons = field["longitude"].values
    lats = field["latitude"].values
    extent = [float(lons.min()), float(lons.max()), float(lats.min()), float(lats.max())]
    return ax.imshow(
        data.values,
        extent=extent,
        origin="lower" if float(lats[0]) < float(lats[-1]) else "upper",
        interpolation="bilinear",
        cmap=cmap,
        norm=PowerNorm(gamma=0.85, vmin=0.0, vmax=vmax),
        aspect="auto",
    )


def add_significance_stippling(ax: plt.Axes, ds: xr.Dataset, field: xr.DataArray) -> None:
    if "compound_event_mk_significant" not in ds:
        return
    land = ds.land_mask.values.astype(bool) if "land_mask" in ds else True
    base_mask = (ds.compound_event_mk_significant.values == 1) & np.isfinite(field.values) & land
    if not base_mask.any():
        return
    for mask, color in (
        (base_mask & (field.values < 0.0), "#0b4f8a"),
        (base_mask & (field.values > 0.0), "#9c4f0f"),
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


def plot_density_maps(
    datasets: list[tuple[str, xr.Dataset]],
    output_path: Path,
    method: str,
    start_year: int,
    end_year: int,
    shapefile: Path | None,
) -> None:
    n_panels = len(datasets)
    if n_panels == 1:
        fig, ax = plt.subplots(1, 1, figsize=(7.6, 5.8), constrained_layout=True)
        axes = [ax]
    else:
        fig, axes = plt.subplots(1, n_panels, figsize=(7.0 * n_panels, 5.5), constrained_layout=True)
        axes = np.atleast_1d(axes).tolist()

    for (label, ds), ax, c in zip(datasets, axes, "abcdefgh"):
        field_name = (
            "compound_event_density_per_year_smoothed"
            if "compound_event_density_per_year_smoothed" in ds
            else "compound_event_density_per_year"
        )
        field = ds[field_name]
        vmin, vmax = panel_limits(field.values)
        mesh = plot_density_field(ax, field, WHITE_TO_RED_CMAP, vmax)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_xlim(float(ds["longitude"].min()), float(ds["longitude"].max()))
        ax.set_ylim(float(ds["latitude"].min()), float(ds["latitude"].max()))
        draw_boundaries(ax, shapefile, ax.get_xlim(), ax.get_ylim())
        if n_panels > 1:
            panel_letter(ax, c)
        cb = fig.colorbar(mesh, ax=ax, shrink=0.88)
        cb.set_label("events yr$^{-1}$ gridcell$^{-1}$")

    fig.savefig(output_path, dpi=600)
    if len(datasets) == 1 and datasets[0][0] == "EUCLID CG":
        MANUSCRIPT_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(MANUSCRIPT_FIGURES_DIR / "15_cg_compound_event_location_density.png", dpi=600)
    plt.close(fig)


def plot_trend_maps(
    datasets: list[tuple[str, xr.Dataset]],
    output_path: Path,
    start_year: int,
    end_year: int,
    shapefile: Path | None,
) -> None:
    n_panels = len(datasets)
    if n_panels == 1:
        fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.8), constrained_layout=True, facecolor="white")
        axes_by_dataset = [np.atleast_1d(axes).tolist()]
    else:
        fig, axes = plt.subplots(
            n_panels,
            2,
            figsize=(13.8, 5.2 * n_panels),
            constrained_layout=True,
            facecolor="white",
        )
        axes_by_dataset = np.atleast_2d(axes).tolist()

    days_cmap = copy(plt.get_cmap("YlOrRd"))
    days_cmap.set_bad("white")
    trend_cmap = LinearSegmentedColormap.from_list("difference_soft_continuous", DIFFERENCE_COLORS)
    trend_cmap.set_bad("white")
    for (label, ds), axes_pair in zip(datasets, axes_by_dataset):
        days_ax, trend_ax = axes_pair
        for ax in axes_pair:
            ax.set_facecolor("white")

        days_field = ds.mean_annual_compound_event_days.where(ds.land_mask)
        days_data = days_field.where(days_field > 0.0)
        days_mesh = days_ax.pcolormesh(
            ds.longitude,
            ds.latitude,
            days_data,
            shading="auto",
            cmap=days_cmap,
            vmin=0.0,
            vmax=panel_limits(days_data.values)[1],
        )
        days_ax.set_title(f"{label} mean annual compound-event days")
        days_ax.set_xlabel("Longitude")
        days_ax.set_ylabel("Latitude")
        days_ax.set_xlim(float(ds["longitude"].min()), float(ds["longitude"].max()))
        days_ax.set_ylim(float(ds["latitude"].min()), float(ds["latitude"].max()))
        days_ax.grid(alpha=0.2)
        draw_boundaries(days_ax, shapefile, days_ax.get_xlim(), days_ax.get_ylim())
        days_cb = fig.colorbar(days_mesh, ax=days_ax, shrink=0.88, fraction=0.025, aspect=36)
        days_cb.set_label("days yr$^{-1}$ per grid cell")

        field = ds.theil_sen_slope_compound_events.where(ds.land_mask)
        plot_field = field.where(field != 0.0)
        vmin, vmax = signed_limits(field.values)
        norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
        mesh = trend_ax.pcolormesh(
            ds.longitude,
            ds.latitude,
            plot_field,
            shading="auto",
            cmap=trend_cmap,
            norm=norm,
        )
        add_significance_stippling(trend_ax, ds, field)
        trend_ax.set_title(f"{label} compound-event count trend")
        trend_ax.set_xlabel("Longitude")
        trend_ax.set_ylabel("Latitude")
        trend_ax.set_xlim(float(ds["longitude"].min()), float(ds["longitude"].max()))
        trend_ax.set_ylim(float(ds["latitude"].min()), float(ds["latitude"].max()))
        trend_ax.grid(alpha=0.2)
        draw_boundaries(trend_ax, shapefile, trend_ax.get_xlim(), trend_ax.get_ylim())
        cb = fig.colorbar(mesh, ax=trend_ax, shrink=0.88, fraction=0.025, aspect=36, extend="both")
        cb.set_label("events yr$^{-1}$ per grid cell")

    fig.suptitle(f"Compound Event Count Spatial Trend (Apr-Sep, {start_year}-{end_year})", fontsize=16)
    fig.savefig(output_path, dpi=220, facecolor="white")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    start_year = args.start_year
    end_year = args.end_year
    n_years = end_year - start_year + 1
    era5_latitudes, era5_longitudes = open_sample_grid(args.method, start_year)
    figures_dir = Path(args.figures_dir)
    data_dir = Path(args.data_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    datasets: list[tuple[str, xr.Dataset]] = []
    sources = list(dict.fromkeys(args.sources))
    shapefile = Path(args.shapefile) if args.shapefile else None
    aggregate_tag = grid_tag(args.aggregate_resolution) if args.aggregate_euclid_cg_grid else ""
    source_dirs = {
        "eswd": Path(args.eswd_dir),
        "euclid": Path(args.euclid_dir),
        "euclid_cg": Path(args.euclid_cg_dir),
    }
    cg_latitudes = None
    cg_longitudes = None

    for source in sources:
        config = SOURCE_CONFIG[source]
        csv_path = source_dirs[source] / f"{config['csv_prefix']}_{args.method}_{start_year}_{end_year}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing {config['label']} compound event table: {csv_path}")
        if source == "euclid_cg":
            if args.aggregate_euclid_cg_grid:
                latitudes, longitudes = regular_grid(
                    args.aggregate_lat_min,
                    args.aggregate_lat_max,
                    args.aggregate_lon_min,
                    args.aggregate_lon_max,
                    args.aggregate_resolution,
                )
            elif cg_latitudes is None or cg_longitudes is None:
                cg_latitudes, cg_longitudes = open_cg_grid(Path(args.euclid_cg_grid_dir), start_year, end_year)
                latitudes, longitudes = cg_latitudes, cg_longitudes
            else:
                latitudes, longitudes = cg_latitudes, cg_longitudes
        else:
            latitudes, longitudes = era5_latitudes, era5_longitudes
        density = load_density(
            csv_path,
            config["event_id_column"],
            latitudes,
            longitudes,
            n_years,
            start_year,
            end_year,
            shapefile,
            smooth_sigma=args.cg_smooth_sigma if source == "euclid_cg" else 0.0,
            crop_to_events=not (source == "euclid_cg" and args.aggregate_euclid_cg_grid),
        )
        output_prefix = config["netcdf_prefix"]
        if source == "euclid_cg" and args.aggregate_euclid_cg_grid:
            output_prefix = f"{output_prefix}_{aggregate_tag}"
            density.attrs.update(
                {
                    "aggregation_grid": aggregate_tag,
                    "aggregation_latitude_min": args.aggregate_lat_min,
                    "aggregation_latitude_max": args.aggregate_lat_max,
                    "aggregation_longitude_min": args.aggregate_lon_min,
                    "aggregation_longitude_max": args.aggregate_lon_max,
                    "aggregation_resolution_deg": args.aggregate_resolution,
                }
            )
        nc_path = data_dir / f"{output_prefix}_{args.method}_{start_year}_{end_year}.nc"
        density.to_netcdf(nc_path)
        plot_label = config["label"]
        if source == "euclid_cg" and args.aggregate_euclid_cg_grid:
            plot_label = f"{plot_label} {aggregate_tag}"
        datasets.append((plot_label, density))
        print(f"Wrote {nc_path}", flush=True)

    source_tag = "_".join(sources)
    if args.aggregate_euclid_cg_grid and sources == ["euclid_cg"]:
        source_tag = f"{source_tag}_{aggregate_tag}"
    if sources == ["eswd", "euclid"]:
        figure_path = figures_dir / f"compound_event_location_density_{args.method}_{start_year}_{end_year}.png"
        trend_path = figures_dir / f"compound_event_count_spatial_trend_{args.method}_{start_year}_{end_year}.png"
    else:
        figure_path = figures_dir / f"{source_tag}_compound_event_location_density_{args.method}_{start_year}_{end_year}.png"
        trend_path = figures_dir / f"{source_tag}_compound_event_count_spatial_trend_{args.method}_{start_year}_{end_year}.png"
    plot_density_maps(datasets, figure_path, args.method, start_year, end_year, shapefile)
    plot_trend_maps(datasets, trend_path, start_year, end_year, shapefile)
    print(f"Wrote {figure_path}", flush=True)
    print(f"Wrote {trend_path}", flush=True)


if __name__ == "__main__":
    main()
