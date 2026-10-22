#!/usr/bin/env python3
"""Extend the absolute wet-bulb threshold sweep to the matched environmental
differences (q850, TCWV, DPD2m, qv850, qu850, Z500), on the land-restricted
pipeline.

Requires run_absolute_threshold_sweep.py to have already completed the `rebuild`
stage for every threshold (reuses that output; does not redo it).

Matching design (the manuscript's primary method): exact start-month matching,
five covariates (centre latitude, centre longitude, duration, intensity, log
peak daily land area), and start year controlled via year fixed effects in a
pair + year-FE regression on the matched differences, clustered by matched
pair -- the "area_only" scenario combined with fit_area_matched_year_fe.py.

Per threshold, in order (each stage skipped if its output already exists):
  1. rebuild_compound_land.py coverage
  2. rebuild_compound_land.py match                 (only to satisfy an unused
                                                       read in step 3; not the
                                                       matching used for results)
  3. check_land_area_year_balance.py           -> events_with_peak_daily_land_area.csv
  4. prepare_confirmed_land.py --method <label>
  5. analyze_land_environment.py --climatology-dir <shared>
       The 2006-2025 calendar-day/3h climatology does not depend on the
       absolute threshold (it depends only on the calendar), so it is computed
       once in a shared directory and reused across all thresholds -- this is
       the single biggest avoidable cost in this sweep.
  6. compare_land_matching_ablation.py         -> matching/comparison/area_only_pairs.csv
                                                    (exact-month, 5-covariate match)
  7. fit_area_matched_year_fe.py               -> environment_pair_year_FE.csv
                                                    (pair + year-FE regression, pair-
                                                    clustered CR1 SE -- the actual number
                                                    quoted in the manuscript)

Finally collects each threshold's year-FE-adjusted difference + 95% CI +
BH-corrected p-value per variable into one comparison table.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.analysis_inputs import validate_run_inputs
from common.config import SHARED_CLIMATOLOGY_DIR, SWEEP_RUN_ROOT as DEFAULT_LAND_RUN_ROOT, SWEEP_SUMMARY_DIR
from common.sweep import REPO, SCRIPTS, DEFAULT_THRESHOLDS, method_label, run

DEFAULT_SUMMARY_PATH = SWEEP_SUMMARY_DIR / "environment_threshold_sensitivity_2006_2025.csv"


def run_dir_for(threshold: float) -> Path:
    return DEFAULT_LAND_RUN_ROOT / f"{method_label(threshold)}_2006-2025"


def step_coverage(run_dir: Path, threshold: float, dry_run: bool) -> None:
    label = method_label(threshold)
    if (run_dir / "checks/cg_coverage.csv").exists():
        print(f"[{label}] coverage already done, skipping")
        return
    run([sys.executable, str(SCRIPTS / "pipeline/rebuild_compound_land.py"), "coverage", "--out", str(run_dir)], dry_run)


def step_match(run_dir: Path, threshold: float, dry_run: bool) -> None:
    label = method_label(threshold)
    if (run_dir / "matching/baseline/match_pairs.csv").exists():
        print(f"[{label}] match already done, skipping")
        return
    run([sys.executable, str(SCRIPTS / "pipeline/rebuild_compound_land.py"), "match", "--out", str(run_dir)], dry_run)


def step_area_balance(run_dir: Path, threshold: float, dry_run: bool) -> Path:
    label = method_label(threshold)
    out_dir = run_dir / "matching/diagnostics"
    target = out_dir / "tables/events_with_peak_daily_land_area.csv"
    if target.exists():
        print(f"[{label}] area/year balance already done, skipping")
        return target
    run([sys.executable, str(SCRIPTS / "pipeline/check_land_area_year_balance.py"),
         "--source", str(run_dir), "--output", str(out_dir)], dry_run)
    return target


def step_prepare_confirmed(run_dir: Path, threshold: float, dry_run: bool) -> None:
    label = method_label(threshold)
    if (run_dir / "first_cg/data/confirmed_anchors.parquet").exists():
        print(f"[{label}] confirmed anchors already done, skipping")
        return
    run([sys.executable, str(SCRIPTS / "pipeline/prepare_confirmed_land.py"),
         "--out", str(run_dir), "--method", label], dry_run)


def step_analyze_environment(run_dir: Path, threshold: float, dry_run: bool) -> None:
    label = method_label(threshold)
    if (run_dir / "environment/tables/event_environment_anomaly.csv").exists():
        print(f"[{label}] event environment anomalies already done, skipping")
        return
    SHARED_CLIMATOLOGY_DIR.mkdir(parents=True, exist_ok=True)
    run([sys.executable, str(SCRIPTS / "pipeline/analyze_land_environment.py"),
         "--out", str(run_dir), "--climatology-dir", str(SHARED_CLIMATOLOGY_DIR)], dry_run)


def step_matching_ablation(run_dir: Path, threshold: float, dry_run: bool) -> Path:
    label = method_label(threshold)
    out_dir = run_dir / "matching/comparison"
    target = out_dir / "area_only_pairs.csv"
    if target.exists():
        print(f"[{label}] matching ablation already done, skipping")
        return target
    run([sys.executable, str(SCRIPTS / "pipeline/compare_land_matching_ablation.py"),
         "--source", str(run_dir), "--output", str(out_dir)], dry_run)
    return target


def step_year_fe(run_dir: Path, threshold: float, dry_run: bool) -> Path:
    label = method_label(threshold)
    out_dir = run_dir / "matching/final"
    target = out_dir / "tables/environment_pair_year_FE.csv"
    if target.exists():
        print(f"[{label}] pair+year-FE regression already done, skipping")
        return target
    run([sys.executable, str(SCRIPTS / "pipeline/fit_area_matched_year_fe.py"),
         "--source", str(run_dir), "--output", str(out_dir)], dry_run)
    return target


def collect(thresholds: list[float], dry_run: bool) -> None:
    if dry_run:
        print("(dry run: skipping collection -- nothing was actually produced)")
        return
    frames = []
    for threshold in thresholds:
        label = method_label(threshold)
        path = run_dir_for(threshold) / "matching/final/tables" / "environment_pair_year_FE.csv"
        if not path.exists():
            print(f"[{label}] missing {path}, skipping in summary")
            continue
        tab = pd.read_csv(path)
        tab.insert(0, "threshold_degC", threshold)
        tab.insert(1, "method", label)
        frames.append(tab)
    if not frames:
        print("No complete threshold results found; nothing to summarize yet.")
        return
    summary = pd.concat(frames, ignore_index=True).sort_values(["variable", "threshold_degC"])
    DEFAULT_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(DEFAULT_SUMMARY_PATH, index=False)
    print(f"Wrote {DEFAULT_SUMMARY_PATH}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--thresholds", type=float, nargs="+", default=DEFAULT_THRESHOLDS)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--skip-through",
        choices=["coverage", "match", "area-balance", "prepare-confirmed", "environment", "ablation", "year-fe"],
        default=None,
        help="Skip stages up to and including this one for every threshold (for resuming manually).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    thresholds = sorted(args.thresholds)
    order = ["coverage", "match", "area-balance", "prepare-confirmed", "environment", "ablation", "year-fe"]
    start = order.index(args.skip_through) + 1 if args.skip_through else 0

    for threshold in thresholds:
        run_dir = run_dir_for(threshold)
        if not (run_dir / "checks/rebuild_complete.json").exists():
            raise SystemExit(
                f"{run_dir} has no completed 'rebuild' stage -- run "
                f"run_absolute_threshold_sweep.py for {method_label(threshold)} first."
            )
        validate_run_inputs(run_dir)
        label = method_label(threshold)
        print(f"\n=== {label} ({threshold:g} degC) ===", flush=True)
        if start <= order.index("coverage"):
            step_coverage(run_dir, threshold, args.dry_run)
        if start <= order.index("match"):
            step_match(run_dir, threshold, args.dry_run)
        if start <= order.index("area-balance"):
            step_area_balance(run_dir, threshold, args.dry_run)
        if start <= order.index("prepare-confirmed"):
            step_prepare_confirmed(run_dir, threshold, args.dry_run)
        if start <= order.index("environment"):
            step_analyze_environment(run_dir, threshold, args.dry_run)
        if start <= order.index("ablation"):
            step_matching_ablation(run_dir, threshold, args.dry_run)
        if start <= order.index("year-fe"):
            step_year_fe(run_dir, threshold, args.dry_run)

    print("\n=== Collecting comparison table ===", flush=True)
    collect(thresholds, args.dry_run)


if __name__ == "__main__":
    main()
