#!/usr/bin/env python3
"""Fit the warm-season CG-occurrence model with cell-month and year effects.

The daily binary observations are losslessly collapsed to
cell-month x year x heatwave-status binomial counts for estimation.  The
cell-month effects are absorbed algebraically rather than expanded into a
large dummy matrix.  Cluster scores are then evaluated on the original daily
grid-cell observations to obtain cell + date two-way clustered standard
errors.
"""

from __future__ import annotations

import argparse
import calendar
import json
import math
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import shapefile
import shapely
import xarray as xr
from scipy.special import expit
from scipy.stats import norm
from shapely.geometry import shape
from shapely.strtree import STRtree


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from common.config import RUN_DIR
from common.fe_logit import FitResult, fit_absorbed_logit, residualize_exposure
from common.geo import nearest_indices
DEFAULT_HW_DIR = REPO / "processed" / "wetbulb" / "heatwaves" / "90p20"
DEFAULT_CG_DIR = REPO / "data" / "Euclid" / "monthly_nc_cg"
DEFAULT_SHAPEFILE = (
    REPO
    / "data"
    / "shapefiles"
    / "ne_10m_admin_0_countries"
    / "ne_10m_admin_0_countries.shp"
)
DEFAULT_OUTPUT_DIR = RUN_DIR / "tables" / "warmseason_fe"


@dataclass(frozen=True)
class MonthBlock:
    year: int
    month: int
    year_index: int
    month_index: int
    start: int
    stop: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-year", type=int, default=2006)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--months", type=int, nargs="+", default=[4, 5, 6, 7, 8, 9])
    parser.add_argument("--method", default="90p20")
    parser.add_argument("--heatwave-dir", type=Path, default=DEFAULT_HW_DIR)
    parser.add_argument("--cg-dir", type=Path, default=DEFAULT_CG_DIR)
    parser.add_argument("--land-shapefile", type=Path, default=DEFAULT_SHAPEFILE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("/tmp"),
        help="Directory for two temporary one-byte-per-observation arrays.",
    )
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--tolerance", type=float, default=1e-9)
    parser.add_argument(
        "--no-small-sample-correction",
        action="store_true",
        help="Use asymptotic CGM meat without CR1 finite-sample multipliers.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.start_year > args.end_year:
        raise ValueError("start-year must not exceed end-year")
    months = sorted(set(args.months))
    if months != args.months:
        raise ValueError("months must be unique and sorted")
    if not months or months[0] < 1 or months[-1] > 12:
        raise ValueError("months must be between 1 and 12")
    for path, label in [
        (args.heatwave_dir, "heatwave directory"),
        (args.cg_dir, "CG directory"),
        (args.land_shapefile, "land shapefile"),
    ]:
        if not path.exists():
            raise FileNotFoundError(f"Missing {label}: {path}")


def build_month_blocks(years: list[int], months: list[int]) -> tuple[list[MonthBlock], pd.DatetimeIndex]:
    blocks: list[MonthBlock] = []
    all_dates: list[pd.DatetimeIndex] = []
    cursor = 0
    for year_index, year in enumerate(years):
        for month_index, month in enumerate(months):
            ndays = calendar.monthrange(year, month)[1]
            dates = pd.date_range(f"{year}-{month:02d}-01", periods=ndays, freq="D")
            blocks.append(
                MonthBlock(
                    year=year,
                    month=month,
                    year_index=year_index,
                    month_index=month_index,
                    start=cursor,
                    stop=cursor + ndays,
                )
            )
            all_dates.append(dates)
            cursor += ndays
    return blocks, pd.DatetimeIndex(np.concatenate([dates.values for dates in all_dates]))


def land_domain_mask(
    cg_latitude: np.ndarray,
    cg_longitude: np.ndarray,
    hw_latitude: np.ndarray,
    hw_longitude: np.ndarray,
    shapefile_path: Path,
) -> np.ndarray:
    lat2d, lon2d = np.meshgrid(cg_latitude, cg_longitude, indexing="ij")
    in_hw_domain = (
        (lat2d >= np.min(hw_latitude))
        & (lat2d <= np.max(hw_latitude))
        & (lon2d >= np.min(hw_longitude))
        & (lon2d <= np.max(hw_longitude))
    )

    reader = shapefile.Reader(str(shapefile_path))
    geometries = [shape(item.__geo_interface__) for item in reader.shapes()]
    points = shapely.points(lon2d.ravel(), lat2d.ravel())
    pairs = STRtree(geometries).query(points, predicate="intersects")
    on_land = np.zeros(points.size, dtype=bool)
    on_land[np.asarray(pairs[0], dtype=np.int64)] = True
    return in_hw_domain & on_land.reshape(lat2d.shape)


def read_daily_cg(path: Path, expected_dates: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with xr.open_dataset(path) as ds:
        times = pd.DatetimeIndex(ds["valid_time"].values)
        latitude = ds["latitude"].values
        longitude = ds["longitude"].values
        values = np.asarray(ds["flash_count"].values)

    normalized = times.normalize()
    unique_days, counts = np.unique(normalized.values, return_counts=True)
    if not np.all(counts == 24):
        raise ValueError(f"Expected 24 hourly CG fields per day in {path}; got {counts}")
    if not np.array_equal(unique_days, expected_dates.values):
        raise ValueError(f"CG dates do not match the expected calendar in {path}")
    daily = (values > 0).reshape(expected_dates.size, 24, latitude.size, longitude.size).any(axis=1)
    return daily, latitude, longitude


def safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator != 0 else float("nan")


def write_outputs(
    args: argparse.Namespace,
    years: list[int],
    dates: pd.DatetimeIndex,
    counts: np.ndarray,
    successes: np.ndarray,
    both_exposure: np.ndarray,
    outcome_varies: np.ndarray,
    retained: np.ndarray,
    retained_counts: np.ndarray,
    retained_successes: np.ndarray,
    fit: FitResult,
    cluster_se: float,
    variance_components: dict[str, float],
    n_land_cells: int,
    n_cell_clusters: int,
    small_sample_correction: bool,
) -> list[Path]:
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    month_tag = f"m{args.months[0]:02d}-{args.months[-1]:02d}"
    stem = f"cg_hw_fe_logit_{args.method}_{args.start_year}-{args.end_year}_{month_tag}"
    summary_path = output_dir / f"{stem}_summary.csv"
    attrition_path = output_dir / f"{stem}_attrition.csv"
    year_path = output_dir / f"{stem}_year_effects.csv"
    metadata_path = output_dir / f"{stem}_metadata.json"

    all_n = counts.sum(axis=(0, 1))
    all_s = successes.sum(axis=(0, 1))
    retained_n = retained_counts.sum(axis=(0, 1))
    retained_s = retained_successes.sum(axis=(0, 1))
    raw_nonhw = safe_ratio(float(all_s[0]), float(all_n[0]))
    raw_hw = safe_ratio(float(all_s[1]), float(all_n[1]))
    raw_nonhw_retained = safe_ratio(float(retained_s[0]), float(retained_n[0]))
    raw_hw_retained = safe_ratio(float(retained_s[1]), float(retained_n[1]))

    n_by_group_year = retained_counts.sum(axis=2)
    gamma = np.r_[0.0, fit.year_effects]
    base_eta = fit.alpha[:, None] + gamma[None, :]
    adjusted_nonhw = float(np.sum(n_by_group_year * expit(base_eta)) / np.sum(n_by_group_year))
    adjusted_hw = float(
        np.sum(n_by_group_year * expit(base_eta + fit.beta)) / np.sum(n_by_group_year)
    )

    z975 = float(norm.ppf(0.975))
    beta_lower = fit.beta - z975 * cluster_se
    beta_upper = fit.beta + z975 * cluster_se
    z_value = fit.beta / cluster_se
    p_value = float(2.0 * norm.sf(abs(z_value)))
    total_records = int(counts.sum())
    retained_records = int(retained_counts.sum())
    total_hw = int(all_n[1])
    retained_hw = int(retained_n[1])
    n_strata = counts.shape[0]
    n_retained_strata = int(retained.sum())

    summary = pd.DataFrame(
        [
            {
                "method": args.method,
                "start_year": args.start_year,
                "end_year": args.end_year,
                "months": ",".join(map(str, args.months)),
                "total_sample_records": total_records,
                "retained_sample_records": retained_records,
                "retained_record_fraction": retained_records / total_records,
                "total_heatwave_records": total_hw,
                "retained_heatwave_records": retained_hw,
                "heatwave_record_retention_rate": retained_hw / total_hw,
                "land_cells": n_land_cells,
                "initial_cell_month_strata": n_strata,
                "retained_cell_month_strata": n_retained_strata,
                "cell_clusters": n_cell_clusters,
                "date_clusters": dates.size,
                "unadjusted_p_cg_nonheatwave_full_sample": raw_nonhw,
                "unadjusted_p_cg_heatwave_full_sample": raw_hw,
                "unadjusted_probability_difference_full_sample": raw_hw - raw_nonhw,
                "unadjusted_occurrence_ratio_full_sample": safe_ratio(raw_hw, raw_nonhw),
                "unadjusted_p_cg_nonheatwave_retained_sample": raw_nonhw_retained,
                "unadjusted_p_cg_heatwave_retained_sample": raw_hw_retained,
                "unadjusted_occurrence_ratio_retained_sample": safe_ratio(
                    raw_hw_retained, raw_nonhw_retained
                ),
                "adjusted_p_cg_nonheatwave": adjusted_nonhw,
                "adjusted_p_cg_heatwave": adjusted_hw,
                "adjusted_probability_difference": adjusted_hw - adjusted_nonhw,
                "adjusted_occurrence_ratio": safe_ratio(adjusted_hw, adjusted_nonhw),
                "heatwave_logit_coefficient": fit.beta,
                "heatwave_coefficient_two_way_cluster_se": cluster_se,
                "heatwave_coefficient_ci95_lower": beta_lower,
                "heatwave_coefficient_ci95_upper": beta_upper,
                "heatwave_odds_ratio": math.exp(fit.beta),
                "heatwave_odds_ratio_ci95_lower": math.exp(beta_lower),
                "heatwave_odds_ratio_ci95_upper": math.exp(beta_upper),
                "heatwave_coefficient_wald_z": z_value,
                "heatwave_coefficient_wald_p_two_sided": p_value,
                "log_likelihood": fit.log_likelihood,
                "converged": fit.converged,
                "iterations": fit.iterations,
                "cluster_small_sample_correction": small_sample_correction,
                **variance_components,
            }
        ]
    )
    summary.to_csv(summary_path, index=False)

    group_records = counts.sum(axis=(1, 2)).astype(np.int64)
    group_hw = counts[:, :, 1].sum(axis=1).astype(np.int64)
    group_nonhw = counts[:, :, 0].sum(axis=1).astype(np.int64)
    after_exposure = both_exposure
    after_outcome = both_exposure & outcome_varies

    def stage_row(
        stage: str,
        before: np.ndarray,
        after: np.ndarray,
        rule: str,
    ) -> dict[str, object]:
        dropped = before & ~after
        return {
            "stage": stage,
            "rule": rule,
            "strata_before": int(before.sum()),
            "strata_dropped_at_stage": int(dropped.sum()),
            "strata_retained": int(after.sum()),
            "records_before": int(group_records[before].sum()),
            "records_dropped_at_stage": int(group_records[dropped].sum()),
            "records_retained": int(group_records[after].sum()),
            "heatwave_records_before": int(group_hw[before].sum()),
            "heatwave_records_dropped_at_stage": int(group_hw[dropped].sum()),
            "heatwave_records_retained": int(group_hw[after].sum()),
            "nonheatwave_records_before": int(group_nonhw[before].sum()),
            "nonheatwave_records_dropped_at_stage": int(group_nonhw[dropped].sum()),
            "nonheatwave_records_retained": int(group_nonhw[after].sum()),
        }

    initial = np.ones(counts.shape[0], dtype=bool)
    attrition = pd.DataFrame(
        [
            stage_row("initial_land_cell_month_sample", initial, initial, "none"),
            stage_row(
                "exposure_informative",
                initial,
                after_exposure,
                "within cell-month, retain both heatwave and nonheatwave days",
            ),
            stage_row(
                "outcome_informative_final",
                after_exposure,
                after_outcome,
                "within cell-month, retain variation in daily CG occurrence",
            ),
        ]
    )
    attrition.to_csv(attrition_path, index=False)

    pd.DataFrame(
        {
            "year": years,
            "year_logit_effect_relative_to_first_year": np.r_[0.0, fit.year_effects],
            "year_odds_ratio_relative_to_first_year": np.exp(np.r_[0.0, fit.year_effects]),
        }
    ).to_csv(year_path, index=False)

    metadata = {
        "model": "logit[P(CG_cell,date=1)] = alpha_cell_x_month + gamma_year + beta*HW_cell,date",
        "outcome": "1 if at least one hourly EUCLID CG flash occurs in the CG grid cell that UTC day",
        "exposure": "ERA5 90p+20 wet-bulb heatwave_day mapped by nearest grid centre to the CG grid",
        "spatial_sample": "CG grid-cell centres on Natural Earth 10m country land polygons and inside the ERA5 heatwave domain",
        "estimation": "Grouped-binomial unconditional fixed-effect logit; grouping by cell-month x year x heatwave status is lossless for this design",
        "fixed_effects": ["CG grid cell x calendar month", "calendar year"],
        "filter_order": [
            "retain cell-month strata containing heatwave and nonheatwave days",
            "then retain strata containing both CG=0 and CG=1 outcomes",
        ],
        "adjusted_probability": "Average marginal prediction over the retained sample's cell-month-year distribution, setting HW to 0 or 1 for every retained observation",
        "standard_error": "Cameron-Gelbach-Miller cell + UTC-date - cell-date two-way cluster sandwich for beta after weighted projection of HW on all nuisance effects",
        "small_sample_correction": (
            "CR1: each cluster-meat component multiplied by G/(G-1)*(N-1)/(N-K), K includes retained cell-month and year effects plus beta"
            if small_sample_correction
            else "none"
        ),
        "heatwave_source": str(args.heatwave_dir),
        "cg_source": str(args.cg_dir),
        "land_shapefile": str(args.land_shapefile),
        "variance_components": variance_components,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
    return [summary_path, attrition_path, year_path, metadata_path]


def main() -> None:
    args = parse_args()
    validate_args(args)
    years = list(range(args.start_year, args.end_year + 1))
    blocks, dates = build_month_blocks(years, args.months)

    first_cg_path = args.cg_dir / str(args.start_year) / f"euclid_flash_cg_{args.start_year}_{args.months[0]:02d}.nc"
    first_hw_path = args.heatwave_dir / f"wetbulb_heatwave_{args.start_year}.nc"
    with xr.open_dataset(first_cg_path) as ds:
        cg_latitude = ds["latitude"].values
        cg_longitude = ds["longitude"].values
    with xr.open_dataset(first_hw_path) as ds:
        hw_latitude = ds["latitude"].values
        hw_longitude = ds["longitude"].values

    spatial_mask = land_domain_mask(
        cg_latitude,
        cg_longitude,
        hw_latitude,
        hw_longitude,
        args.land_shapefile,
    )
    flat_land = np.flatnonzero(spatial_mask.ravel())
    land_lat_index, land_lon_index = np.unravel_index(flat_land, spatial_mask.shape)
    land_latitude = cg_latitude[land_lat_index]
    land_longitude = cg_longitude[land_lon_index]
    hw_lat_index = nearest_indices(hw_latitude, land_latitude)
    hw_lon_index = nearest_indices(hw_longitude, land_longitude)
    n_cells = flat_land.size
    n_months = len(args.months)
    n_years = len(years)
    n_strata = n_cells * n_months
    print(
        f"Analysis grid: {n_cells:,} land cells, {n_strata:,} cell-month strata, "
        f"{dates.size:,} dates, {n_cells * dates.size:,} daily records"
    )

    counts = np.zeros((n_strata, n_years, 2), dtype=np.uint16)
    successes = np.zeros_like(counts)
    cell_index = np.arange(n_cells, dtype=np.int64)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="cg_hw_fe_", dir=args.cache_dir) as cache_name:
        cache_dir = Path(cache_name)
        cg_cache = np.memmap(
            cache_dir / "cg_daily.bin",
            mode="w+",
            dtype=np.uint8,
            shape=(dates.size, n_cells),
        )
        hw_cache = np.memmap(
            cache_dir / "hw_daily.bin",
            mode="w+",
            dtype=np.uint8,
            shape=(dates.size, n_cells),
        )

        open_hw_year = None
        hw_ds = None
        try:
            for block in blocks:
                expected_dates = dates[block.start : block.stop]
                cg_path = (
                    args.cg_dir
                    / str(block.year)
                    / f"euclid_flash_cg_{block.year}_{block.month:02d}.nc"
                )
                if not cg_path.exists():
                    raise FileNotFoundError(f"Missing CG input: {cg_path}")
                daily_cg_full, current_latitude, current_longitude = read_daily_cg(
                    cg_path, expected_dates
                )
                if not np.array_equal(current_latitude, cg_latitude) or not np.array_equal(
                    current_longitude, cg_longitude
                ):
                    raise ValueError(f"CG grid changed in {cg_path}")
                daily_cg = daily_cg_full.reshape(block.stop - block.start, -1)[:, flat_land]
                del daily_cg_full

                if block.year != open_hw_year:
                    if hw_ds is not None:
                        hw_ds.close()
                    hw_path = args.heatwave_dir / f"wetbulb_heatwave_{block.year}.nc"
                    if not hw_path.exists():
                        raise FileNotFoundError(f"Missing heatwave input: {hw_path}")
                    hw_ds = xr.open_dataset(hw_path)
                    open_hw_year = block.year
                month_hw = hw_ds["heatwave_day"].sel(
                    time=slice(expected_dates[0], expected_dates[-1])
                )
                if month_hw.sizes["time"] != expected_dates.size:
                    raise ValueError(f"Heatwave dates do not match for {block.year}-{block.month:02d}")
                daily_hw = np.asarray(month_hw.values[:, hw_lat_index, hw_lon_index], dtype=bool)

                cg_cache[block.start : block.stop] = daily_cg
                hw_cache[block.start : block.stop] = daily_hw
                group_index = cell_index * n_months + block.month_index
                hw_count = daily_hw.sum(axis=0, dtype=np.uint16)
                counts[group_index, block.year_index, 1] = hw_count
                counts[group_index, block.year_index, 0] = expected_dates.size - hw_count
                successes[group_index, block.year_index, 1] = np.sum(
                    daily_cg & daily_hw, axis=0, dtype=np.uint16
                )
                successes[group_index, block.year_index, 0] = np.sum(
                    daily_cg & ~daily_hw, axis=0, dtype=np.uint16
                )
                print(
                    f"Prepared {block.year}-{block.month:02d}: "
                    f"HW={int(hw_count.sum()):,}, CG={int(daily_cg.sum()):,}"
                )
        finally:
            if hw_ds is not None:
                hw_ds.close()
        cg_cache.flush()
        hw_cache.flush()

        group_n = counts.sum(axis=(1, 2), dtype=np.int64)
        group_hw_n = counts[:, :, 1].sum(axis=1, dtype=np.int64)
        group_success = successes.sum(axis=(1, 2), dtype=np.int64)
        both_exposure = (group_hw_n > 0) & (group_hw_n < group_n)
        outcome_varies = (group_success > 0) & (group_success < group_n)
        retained = both_exposure & outcome_varies
        print(
            f"Retained {int(retained.sum()):,}/{n_strata:,} strata and "
            f"{int(group_n[retained].sum()):,}/{int(group_n.sum()):,} records"
        )
        if not np.any(retained):
            raise RuntimeError("No fixed-effect strata survived both filters")

        retained_counts = counts[retained]
        retained_successes = successes[retained]
        nonzero = retained_counts > 0
        group, year, exposure = np.nonzero(nonzero)
        totals = retained_counts[nonzero].astype(np.float64)
        grouped_successes = retained_successes[nonzero].astype(np.float64)
        exposure = exposure.astype(np.float64)
        group = group.astype(np.int64)
        year = year.astype(np.int64)
        fit = fit_absorbed_logit(
            group,
            year,
            exposure,
            totals,
            grouped_successes,
            int(retained.sum()),
            n_years,
            args.max_iter,
            args.tolerance,
        )
        if not fit.converged:
            raise RuntimeError(f"Logit failed to converge in {args.max_iter} iterations")

        _, group_projection, year_projection, bread = residualize_exposure(
            group,
            year,
            exposure,
            fit.weights,
            int(retained.sum()),
            n_years,
        )
        original_to_fit = np.full(n_strata, -1, dtype=np.int64)
        original_to_fit[retained] = np.arange(int(retained.sum()), dtype=np.int64)
        cell_score = np.zeros(n_cells, dtype=np.float64)
        date_score = np.zeros(dates.size, dtype=np.float64)
        intersection_meat = 0.0
        gamma = np.r_[0.0, fit.year_effects]
        projection_gamma = np.r_[0.0, year_projection]

        for block in blocks:
            original_group = cell_index * n_months + block.month_index
            keep = retained[original_group]
            fit_group = original_to_fit[original_group[keep]]
            base_eta = fit.alpha[fit_group] + gamma[block.year_index]
            group_rx_base = -group_projection[fit_group] - projection_gamma[block.year_index]
            hw = np.asarray(hw_cache[block.start : block.stop, keep], dtype=np.float64)
            cg = np.asarray(cg_cache[block.start : block.stop, keep], dtype=np.float64)
            mu = expit(base_eta[None, :] + fit.beta * hw)
            rx = group_rx_base[None, :] + hw
            score = rx * (cg - mu)
            kept_cells = cell_index[keep]
            cell_score[kept_cells] += score.sum(axis=0)
            date_score[block.start : block.stop] += score.sum(axis=1)
            intersection_meat += float(np.sum(score * score))

        active_cell = np.any(retained.reshape(n_cells, n_months), axis=1)
        active_date = np.ones(dates.size, dtype=bool)
        cell_meat_raw = float(np.sum(cell_score[active_cell] ** 2))
        date_meat_raw = float(np.sum(date_score[active_date] ** 2))
        n_observations = int(retained_counts.sum())
        n_parameters = int(retained.sum()) + (n_years - 1) + 1
        n_cell_clusters = int(active_cell.sum())
        n_date_clusters = int(active_date.sum())

        if args.no_small_sample_correction:
            common = cell_factor = date_factor = intersection_factor = 1.0
        else:
            common = (n_observations - 1) / (n_observations - n_parameters)
            cell_factor = n_cell_clusters / (n_cell_clusters - 1) * common
            date_factor = n_date_clusters / (n_date_clusters - 1) * common
            intersection_factor = n_observations / (n_observations - 1) * common
        cell_component = cell_factor * cell_meat_raw / (bread * bread)
        date_component = date_factor * date_meat_raw / (bread * bread)
        intersection_component = intersection_factor * intersection_meat / (bread * bread)
        cluster_variance = cell_component + date_component - intersection_component
        if cluster_variance <= 0 or not np.isfinite(cluster_variance):
            raise RuntimeError(
                "Non-positive two-way cluster variance: "
                f"cell={cell_component}, date={date_component}, "
                f"intersection={intersection_component}"
            )
        cluster_se = math.sqrt(cluster_variance)
        variance_components = {
            "two_way_cluster_variance": cluster_variance,
            "cell_cluster_variance_component": cell_component,
            "date_cluster_variance_component": date_component,
            "cell_date_intersection_variance_component_subtracted": intersection_component,
            "cluster_common_n_over_n_minus_k_factor": common,
        }

        paths = write_outputs(
            args,
            years,
            dates,
            counts,
            successes,
            both_exposure,
            outcome_varies,
            retained,
            retained_counts,
            retained_successes,
            fit,
            cluster_se,
            variance_components,
            n_cells,
            n_cell_clusters,
            not args.no_small_sample_correction,
        )
        print(f"beta={fit.beta:.8f}, two-way cluster SE={cluster_se:.8f}")
        for path in paths:
            print(f"Wrote {path}")


if __name__ == "__main__":
    main()
