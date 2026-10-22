#!/usr/bin/env python3
"""Compare the Stull (2011) and Romps (2026) wet-bulb heatwave definitions at
the primary 90p+20 threshold, to establish whether pressure-related
differences affect heatwave-day classification, particularly over elevated
regions.

Reads the existing Stull-based heatwave_day masks
(processed/wetbulb/heatwaves/90p20) and the Romps-based ones built with
--wetbulb-method romps (processed/wetbulb/heatwaves/90p20_romps), both for the
same 2006-2025 period, and reports per-grid-cell disagreement rate against
surface elevation (from ERA5's invariant surface geopotential).

This does NOT rerun the downstream compound-event/regression pipeline -- it
only asks whether the two definitions actually disagree enough, and where, to
be worth that much larger follow-up effort.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import shapefile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Reds' own low end is a pale pink, not pure white, which is easy to confuse
# with the grey no-data background; force 0 to render as exact white instead.
WHITE_REDS = LinearSegmentedColormap.from_list("white_reds", ["#ffffff", "#fc9272", "#a50f15"])

def panel_letter(ax, c):
    ax.text(-.14, 1.08, c, transform=ax.transAxes, fontsize=11, fontweight="bold", va="top", ha="left")


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from common.analysis_inputs import input_path
from common.figures import save_figure
from common.config import RUN_DIR, WETBULB_COMPARISON_DIR
from common.style import FULL_WIDTH, apply_style, label_panels

apply_style()

DEFAULT_STULL_DIR = REPO / "processed/wetbulb/heatwaves/90p20"
DEFAULT_ROMPS_DIR = REPO / "processed/wetbulb/heatwaves/90p20_romps"
DEFAULT_ERA5_DIR = REPO / "data/Era5"
DEFAULT_OUTPUT_DIR = WETBULB_COMPARISON_DIR
COUNTRIES_SHAPEFILE = REPO / "data/shapefiles/ne_10m_admin_0_countries/ne_10m_admin_0_countries.shp"
DEFAULT_LAND_MASK_SOURCE = input_path(RUN_DIR, "grids.npz")
YEARS = range(2006, 2026)
WARM_SEASON = list(range(4, 10))
SEASON_TAG = "4-9"
GRAVITY = 9.80665
# Study region: 36N-72N, 10W-33E. The ERA5
# download domain is wider (out to 66.5E), so this comparison must be
# restricted to the actual study region, not the full download extent.
STUDY_LON = (-10, 33)
STUDY_LAT = (36, 72)


def load_land_mask(latitude: np.ndarray, longitude: np.ndarray, source: Path = DEFAULT_LAND_MASK_SOURCE) -> np.ndarray:
    """Reuse the era5_land mask already computed for the primary 90p20 land
    analysis, instead of re-deriving it from the country shapefile. Also
    restricts to the study region (STUDY_LON/STUDY_LAT), not the full,
    wider ERA5 download domain."""
    with np.load(source) as grid:
        if not (np.array_equal(grid["latitude"], latitude) and np.array_equal(grid["longitude"], longitude)):
            raise ValueError(
                f"{source} grid does not match this comparison's ERA5 grid -- "
                "pass --land-mask-source pointing at a grids.npz built on the same domain."
            )
        land = grid["era5_land"].astype(bool)
    lat2d, lon2d = np.meshgrid(latitude, longitude, indexing="ij")
    in_region = (
        (lon2d >= STUDY_LON[0]) & (lon2d <= STUDY_LON[1])
        & (lat2d >= STUDY_LAT[0]) & (lat2d <= STUDY_LAT[1])
    )
    return land & in_region


def draw_country_borders(ax, lon_min=-10, lon_max=33, lat_min=36, lat_max=72) -> None:
    with shapefile.Reader(str(COUNTRIES_SHAPEFILE)) as sf:
        shapes = sf.shapes()
    for shp in shapes:
        x0, y0, x1, y1 = shp.bbox
        if x1 < lon_min or x0 > lon_max or y1 < lat_min or y0 > lat_max:
            continue
        pts = np.asarray(shp.points)
        parts = list(shp.parts) + [len(pts)]
        for a, b in zip(parts[:-1], parts[1:]):
            ax.plot(pts[a:b, 0], pts[a:b, 1], c="#555555", lw=.35)


def load_elevation(era5_dir: Path, latitude, longitude) -> np.ndarray:
    # Surface geopotential is a time-invariant field; any single instant file has it.
    sample = era5_dir / "2006/instant/2006_01.nc"
    with xr.open_dataset(sample) as ds:
        z = ds["z"]
        if "time" in z.dims:
            z = z.isel(time=0)
        elif "valid_time" in z.dims:
            z = z.isel(valid_time=0)
        z = z.sel(latitude=latitude, longitude=longitude)
        return (z.values / GRAVITY).astype(np.float32)


def summarize(scope: str, both_days: np.ndarray, neither_days: np.ndarray,
              stull_only_days: np.ndarray, romps_only_days: np.ndarray) -> dict:
    both, neither = int(both_days.sum()), int(neither_days.sum())
    stull_only, romps_only = int(stull_only_days.sum()), int(romps_only_days.sum())
    total = both + neither + stull_only + romps_only
    return dict(
        scope=scope, total_gridcell_days=total, both_heatwave=both, neither_heatwave=neither,
        stull_only=stull_only, romps_only=romps_only,
        agreement_rate=(both + neither) / total, disagreement_rate=(stull_only + romps_only) / total,
        stull_heatwave_days=both + stull_only, romps_heatwave_days=both + romps_only,
        stull_only_share_of_stull=stull_only / (both + stull_only) if (both + stull_only) else np.nan,
        romps_only_share_of_romps=romps_only / (both + romps_only) if (both + romps_only) else np.nan,
    )


def write_agreement_summaries(output_dir: Path, land: np.ndarray, both_days: np.ndarray,
                               neither_days: np.ndarray, stull_only_days: np.ndarray,
                               romps_only_days: np.ndarray) -> None:
    # Full domain includes ocean and the wider ERA5 download extent beyond the
    # study region; the manuscript-facing headline number needs the land +
    # study-region scope, so both are reported side by side for transparency.
    summary = pd.DataFrame([
        summarize("land_study_region", both_days[land], neither_days[land],
                   stull_only_days[land], romps_only_days[land]),
        summarize("full_domain", both_days, neither_days, stull_only_days, romps_only_days),
    ])
    summary.to_csv(output_dir / "agreement_summary.csv", index=False)
    print(summary.to_string(index=False), flush=True)


def run(stull_dir: Path, romps_dir: Path, era5_dir: Path, output_dir: Path, force: bool = False,
        land_mask_source: Path = DEFAULT_LAND_MASK_SOURCE) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    spatial_path = output_dir / "spatial_disagreement.nc"
    if not force and spatial_path.exists():
        # The 20-year grid-cell-day comparison is the expensive part; replotting
        # alone (e.g. to add country borders or the land mask) should not redo it.
        print(f"Reusing existing {spatial_path.name} (pass --force to recompute)", flush=True)
        with xr.open_dataset(spatial_path) as spatial:
            if "both_days" not in spatial:
                raise SystemExit(
                    f"{spatial_path} predates the per-cell day counts needed for the "
                    "land-restricted agreement summary -- rerun once with --force."
                )
            if spatial.attrs.get("months") != SEASON_TAG:
                # An older cache counted every day of the year; silently reusing it
                # would report a year-round agreement next to April-September results.
                raise SystemExit(
                    f"{spatial_path} was computed for months={spatial.attrs.get('months', 'all')}, "
                    f"not {SEASON_TAG} -- rerun once with --force."
                )
            latitude = spatial.latitude.values
            longitude = spatial.longitude.values
            disagreement_rate = spatial.disagreement_rate.values
            elevation = spatial.elevation_m.values
            both_days = spatial.both_days.values
            neither_days = spatial.neither_days.values
            stull_only_days = spatial.stull_only_days.values
            romps_only_days = spatial.romps_only_days.values
        land = load_land_mask(latitude, longitude, land_mask_source)
        write_agreement_summaries(output_dir, land, both_days, neither_days, stull_only_days, romps_only_days)
        plot(output_dir, latitude, longitude, disagreement_rate, elevation, land)
        return

    both_days = neither_days = stull_only_days = romps_only_days = None
    total_days = 0
    latitude = longitude = None

    for year in YEARS:
        stull_path = stull_dir / f"wetbulb_heatwave_{year}.nc"
        romps_path = romps_dir / f"wetbulb_heatwave_{year}.nc"
        if not stull_path.exists() or not romps_path.exists():
            raise FileNotFoundError(f"Missing {year}: stull={stull_path.exists()}, romps={romps_path.exists()}")
        with xr.open_dataset(stull_path) as ds_s, xr.open_dataset(romps_path) as ds_r:
            # Same April-September season as every other analysis in the paper.
            hw_s = ds_s["heatwave_day"].sel(time=ds_s.time.dt.month.isin(WARM_SEASON))
            hw_r = ds_r["heatwave_day"].sel(time=ds_r.time.dt.month.isin(WARM_SEASON))
            if not np.array_equal(hw_s.time.values, hw_r.time.values):
                raise ValueError(f"{year}: Stull and Romps warm-season dates differ")
            s = hw_s.values.astype(bool)
            r = hw_r.values.astype(bool)
            if s.shape != r.shape:
                raise ValueError(f"{year}: shape mismatch stull={s.shape} romps={r.shape}")
            if latitude is None:
                latitude = ds_s["latitude"].values
                longitude = ds_s["longitude"].values
            year_both = (s & r).sum(axis=0)
            year_neither = (~s & ~r).sum(axis=0)
            year_stull_only = (s & ~r).sum(axis=0)
            year_romps_only = (~s & r).sum(axis=0)
            total_days += s.shape[0]
            if both_days is None:
                both_days, neither_days = year_both, year_neither
                stull_only_days, romps_only_days = year_stull_only, year_romps_only
            else:
                both_days = both_days + year_both
                neither_days = neither_days + year_neither
                stull_only_days = stull_only_days + year_stull_only
                romps_only_days = romps_only_days + year_romps_only
        print(f"{year}: stull_only={int(year_stull_only.sum())}, romps_only={int(year_romps_only.sum())}", flush=True)

    disagreement_rate = (stull_only_days + romps_only_days) / total_days
    elevation = load_elevation(era5_dir, latitude, longitude)

    spatial = xr.Dataset(
        dict(disagreement_rate=(("latitude", "longitude"), disagreement_rate),
             elevation_m=(("latitude", "longitude"), elevation),
             both_days=(("latitude", "longitude"), both_days),
             neither_days=(("latitude", "longitude"), neither_days),
             stull_only_days=(("latitude", "longitude"), stull_only_days),
             romps_only_days=(("latitude", "longitude"), romps_only_days)),
        coords=dict(latitude=latitude, longitude=longitude),
        attrs=dict(months=SEASON_TAG, days_per_cell=total_days),
    )
    spatial.to_netcdf(spatial_path)

    land = load_land_mask(latitude, longitude, land_mask_source)
    write_agreement_summaries(output_dir, land, both_days, neither_days, stull_only_days, romps_only_days)
    plot(output_dir, latitude, longitude, disagreement_rate, elevation, land)
    print(f"Wrote outputs to {output_dir}", flush=True)


def plot(output_dir: Path, latitude: np.ndarray, longitude: np.ndarray,
         disagreement_rate: np.ndarray, elevation: np.ndarray, land: np.ndarray) -> None:
    disagreement_rate_land = np.where(land, disagreement_rate, np.nan)
    flat_elev = elevation[land]
    flat_rate = disagreement_rate[land] * 100
    bands = [(-500, 0), (0, 500), (500, 1000), (1000, 1500), (1500, 2000), (2000, 6000)]
    rows = []
    for lo, hi in bands:
        sel = (flat_elev >= lo) & (flat_elev < hi)
        if sel.any():
            rows.append(dict(elevation_lower_m=lo, elevation_upper_m=hi, n_cells=int(sel.sum()),
                              mean_disagreement_percent=float(flat_rate[sel].mean()),
                              max_disagreement_percent=float(flat_rate[sel].max())))
    band_table = pd.DataFrame(rows)
    band_table.to_csv(output_dir / "disagreement_by_elevation_band.csv", index=False)
    print(band_table.to_string(index=False), flush=True)

    # The figure height is chosen so the equal-aspect map fills its slot: the
    # map, its colour bar (attached to the map, not the slot) and panel b then
    # share one height.
    fig, (left, right) = plt.subplots(1, 2, figsize=(FULL_WIDTH, 3.0))
    left.set_facecolor("#ffffff")  # ocean/outside-study-region NaN cells show through as white
    mesh = left.pcolormesh(longitude, latitude, disagreement_rate_land * 100, cmap=WHITE_REDS, shading="auto",
                           rasterized=True)
    draw_country_borders(left)
    left.set(xlabel="Longitude", ylabel="Latitude", xlim=STUDY_LON, ylim=STUDY_LAT)
    left.set_aspect("equal")
    left.xaxis.set_major_locator(MultipleLocator(10))
    left.yaxis.set_major_locator(MultipleLocator(10))
    cax = make_axes_locatable(left).append_axes("right", size="4%", pad=0.08)
    fig.colorbar(mesh, cax=cax).set_label("Disagreement (%)")

    # Same red as the map, so both panels read as the same quantity.
    right.scatter(flat_elev, flat_rate, s=3, alpha=.25, color="#a50f15", linewidth=0, rasterized=True)
    right.set(xlabel="Surface elevation (m)", ylabel="Disagreement (%)")
    right.grid(axis="y", alpha=.2)
    fig.tight_layout(w_pad=3)
    label_panels(fig, [left, right], "ab")

    save_figure(fig, 'disagreement_vs_elevation', directory=output_dir, dpi=600, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stull-dir", type=Path, default=DEFAULT_STULL_DIR)
    parser.add_argument("--romps-dir", type=Path, default=DEFAULT_ROMPS_DIR)
    parser.add_argument("--era5-dir", type=Path, default=DEFAULT_ERA5_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--force", action="store_true", help="Redo the 20-year comparison even if spatial_disagreement.nc already exists.")
    parser.add_argument("--land-mask-source", type=Path, default=DEFAULT_LAND_MASK_SOURCE,
                         help="grids.npz (from rebuild_compound_land.py) to reuse the era5_land mask from.")
    args = parser.parse_args()
    run(args.stull_dir, args.romps_dir, args.era5_dir, args.output_dir, force=args.force,
        land_mask_source=args.land_mask_source)
