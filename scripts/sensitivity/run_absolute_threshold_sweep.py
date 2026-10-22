#!/usr/bin/env python3
"""Orchestrate the land-restricted CG-heatwave analysis across a grid of absolute
wet-bulb thresholds (18-22 degC) as a sensitivity check.

This does NOT reimplement any statistics. It drives, in order, the same scripts
already used for the primary 90p+20 land analysis, once per threshold:

  1. scripts/pipeline/build_wetbulb_heatwaves.py   (batched across all thresholds)
  2. scripts/pipeline/build_regional_compound_events_euclid.py
       -> produces the unmasked "old" catalogue that rebuild_compound_land.py's
          rebuild stage checks its land-masked connectivity against.
  3. scripts/pipeline/rebuild_compound_land.py rebuild --method <label>
       -> land-restricted regional events + daily grid-cell arrays.
  4. scripts/pipeline/fit_land_intensity.py
       -> adjusted intensity-quintile CG probability, Q5-Q1 effect (regional-event
          cluster inference).
  5. scripts/pipeline/fit_cg_heatwave_fixed_effects.py
       -> point estimate of the adjusted heatwave-vs-non-heatwave occurrence ratio
          (cell-month + year FE). Written under <run_dir>/occurrence/tables/,
          because step 6 depends on that exact location.
  6. scripts/pipeline/report_warmseason_fe_margins.py
       -> the actual delta-method 95% CI for the occurrence ratio (and probability
          difference), propagating uncertainty through all fixed effects, not just
          beta. This is the CI quoted in the manuscript, and it re-fits the model from rebuild_compound_land.py's cached daily
          arrays to get the full joint covariance -- it is not just a formatting
          step, so it must run after both stage 3 and stage 5.

Finally it collects the occurrence ratio, the Q5-Q1 effect, and sample/event
counts from every threshold into one comparison table, showing whether the
relationship fades gradually, has an identifiable transition, or mainly
reflects smaller samples at the highest thresholds.

Every stage is skipped if its output already exists, so the sweep is safe to
re-run/resume after a partial failure. Use --dry-run to print the exact
subprocess commands without running anything.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.analysis_inputs import validate_run_inputs
from common.config import CG as DEFAULT_EUCLID_CG_DIR, HW as DEFAULT_HEATWAVE_BASE, SWEEP_RUN_ROOT, SWEEP_SUMMARY_DIR, heatwave_dir
from common.sweep import REPO, SCRIPTS, DEFAULT_THRESHOLDS, method_label, run

HEATWAVE_START_YEAR = 2006
HEATWAVE_END_YEAR = 2025
DEFAULT_OLD_CATALOGUE_DIR = REPO / "processed/compound/euclid_cg"
DEFAULT_LAND_RUN_ROOT = SWEEP_RUN_ROOT
DEFAULT_SUMMARY_DIR = SWEEP_SUMMARY_DIR


def heatwave_dir_for(threshold: float) -> Path:
    return heatwave_dir(method_label(threshold))


def heatwave_files_complete(threshold: float) -> bool:
    # Checks every year's file, not just the first -- a single early file existing
    # (e.g. from an interrupted prior run) must not be mistaken for "done".
    directory = heatwave_dir_for(threshold)
    return all(
        (directory / f"wetbulb_heatwave_{year}.nc").exists()
        for year in range(HEATWAVE_START_YEAR, HEATWAVE_END_YEAR + 1)
    )


def step_build_heatwave_files(thresholds: list[float], dry_run: bool) -> None:
    missing = [t for t in thresholds if not heatwave_files_complete(t)]
    if not missing:
        print("Heatwave files already exist for all thresholds, skipping build_wetbulb_heatwaves.py")
        return
    cmd = [
        sys.executable,
        str(SCRIPTS / "pipeline/build_wetbulb_heatwaves.py"),
        "--output-dir", str(DEFAULT_HEATWAVE_BASE),
        "--absolute-tw-thresholds", *[repr(t) for t in missing],
    ]
    run(cmd, dry_run)


def step_old_catalogue(threshold: float, dry_run: bool) -> Path:
    label = method_label(threshold)
    out_path = DEFAULT_OLD_CATALOGUE_DIR / f"regional_cg_heatwave_lightning_events_{label}_2006_2025.csv"
    if out_path.exists():
        print(f"[{label}] old unmasked catalogue already exists, skipping build_regional_compound_events_euclid.py")
        return out_path
    cmd = [
        sys.executable,
        str(SCRIPTS / "pipeline/build_regional_compound_events_euclid.py"),
        "--euclid-dir", str(DEFAULT_EUCLID_CG_DIR),
        "--euclid-pattern", "euclid_flash_cg_*.nc",
        "--output-dir", str(DEFAULT_OLD_CATALOGUE_DIR),
        "--output-prefix", "cg",
        "--tables-dir", str(DEFAULT_OLD_CATALOGUE_DIR / "tables"),
        "--method", label, str(heatwave_dir_for(threshold)),
    ]
    run(cmd, dry_run)
    return out_path


def step_rebuild_land(threshold: float, dry_run: bool) -> Path:
    label = method_label(threshold)
    run_dir = DEFAULT_LAND_RUN_ROOT / f"{label}_2006-2025"
    marker = run_dir / "checks/rebuild_complete.json"
    if marker.exists():
        validate_run_inputs(run_dir)
        print(f"[{label}] land rebuild already complete, skipping rebuild_compound_land.py rebuild")
        return run_dir
    cmd = [
        sys.executable,
        str(SCRIPTS / "pipeline/rebuild_compound_land.py"),
        "rebuild",
        "--out", str(run_dir),
        "--method", label,
    ]
    run(cmd, dry_run)
    return run_dir


def step_fit_intensity(run_dir: Path, threshold: float, dry_run: bool) -> Path:
    label = method_label(threshold)
    diagnostics = run_dir / "intensity/tables/quintile_diagnostics.json"
    if diagnostics.exists():
        print(f"[{label}] quintile diagnostics already exist, skipping fit_land_intensity.py")
        return diagnostics
    cmd = [
        sys.executable,
        str(SCRIPTS / "pipeline/fit_land_intensity.py"),
        "--out", str(run_dir),
    ]
    run(cmd, dry_run)
    return diagnostics


def step_fit_fixed_effects(run_dir: Path, threshold: float, dry_run: bool) -> Path:
    # report_warmseason_fe_margins.py (next stage) hardcodes this exact relative
    # location -- source/'occurrence/tables'/*summary.csv -- so the FE output
    # must land there, not in a separate tree, for the two stages to line up.
    label = method_label(threshold)
    output_dir = run_dir / "occurrence/tables"
    summary_path = output_dir / f"cg_hw_fe_logit_{label}_2006-2025_m04-09_summary.csv"
    if summary_path.exists():
        print(f"[{label}] FE summary already exists, skipping fit_cg_heatwave_fixed_effects.py")
        return summary_path
    cmd = [
        sys.executable,
        str(SCRIPTS / "pipeline/fit_cg_heatwave_fixed_effects.py"),
        "--method", label,
        "--heatwave-dir", str(heatwave_dir_for(threshold)),
        "--output-dir", str(output_dir),
    ]
    run(cmd, dry_run)
    return summary_path


def step_report_margins(run_dir: Path, threshold: float, dry_run: bool) -> Path:
    # Recomputes the same beta from rebuild_compound_land.py's cached daily arrays
    # as a consistency check, then produces the full delta-method CI for the
    # adjusted occurrence ratio / probability difference -- not just beta's CI.
    label = method_label(threshold)
    out_dir = run_dir / "occurrence"
    effects_path = out_dir / "tables/adjusted_effects_95CI.csv"
    if effects_path.exists():
        print(f"[{label}] adjusted_effects_95CI.csv already exists, skipping report_warmseason_fe_margins.py")
        return effects_path
    cmd = [
        sys.executable,
        str(SCRIPTS / "pipeline/report_warmseason_fe_margins.py"),
        "--source", str(run_dir),
    ]
    run(cmd, dry_run)
    return effects_path


def collect_and_plot(thresholds: list[float], dry_run: bool) -> None:
    if dry_run:
        print("(dry run: skipping collection/plot -- nothing was actually produced)")
        return
    rows = []
    for threshold in thresholds:
        label = method_label(threshold)
        run_dir = DEFAULT_LAND_RUN_ROOT / f"{label}_2006-2025"
        effects_path = run_dir / "occurrence/tables/adjusted_effects_95CI.csv"
        diagnostics_path = run_dir / "intensity/tables/quintile_diagnostics.json"
        if not effects_path.exists() or not diagnostics_path.exists():
            print(f"[{label}] missing outputs, skipping in summary (effects={effects_path.exists()}, "
                  f"quintile={diagnostics_path.exists()})")
            continue
        effects = pd.read_csv(effects_path).set_index("metric")
        ratio = effects.loc["occurrence_ratio"]
        diag = json.loads(diagnostics_path.read_text())
        events = pd.read_csv(run_dir / "events/regional_events.csv")
        # Two different populations are reported side by side on purpose:
        # n_*_land_events counts every land regional event, while n_*_eligible
        # applies the April-September start and complete-CG-coverage
        # restrictions that define the event-level matching pool elsewhere in
        # the paper. Quoting one where the other is meant is the easiest way to
        # make the manuscript look internally inconsistent.
        eligible = events[events.start_month.between(4, 9) & (events.cg_coverage_fraction == 1)]
        rows.append(
            dict(
                threshold_degC=threshold,
                method=label,
                n_land_regional_events=len(events),
                n_compound_land_events=int(events.has_lightning.sum()),
                n_eligible_events=len(eligible),
                n_compound_events_eligible=int(eligible.has_lightning.sum()),
                adjusted_occurrence_ratio=ratio["estimate"],
                adjusted_occurrence_ratio_ci95_lower=ratio["ci95_lower"],
                adjusted_occurrence_ratio_ci95_upper=ratio["ci95_upper"],
                n_intensity_quintile_records=diag["retained_n"],
                n_intensity_quintile_event_clusters=diag["regional_event_clusters"],
                adjusted_q5_minus_q1_pp=diag["q5_minus_q1"] * 100.0,
                adjusted_q5_minus_q1_ci95_lower_pp=diag["q5_minus_q1_ci95_lower"] * 100.0,
                adjusted_q5_minus_q1_ci95_upper_pp=diag["q5_minus_q1_ci95_upper"] * 100.0,
                ordered_quintile_p=diag["ordered_p"],
            )
        )
    if not rows:
        print("No complete threshold results found; nothing to summarize yet.")
        return
    summary = pd.DataFrame(rows).sort_values("threshold_degC")
    DEFAULT_SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = DEFAULT_SUMMARY_DIR / "absolute_threshold_sensitivity_2006_2025.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Wrote {summary_path}")



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--thresholds", type=float, nargs="+", default=DEFAULT_THRESHOLDS)
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    parser.add_argument(
        "--skip-through",
        choices=["heatwave", "old-catalogue", "rebuild", "intensity", "fixed-effects", "margins"],
        default=None,
        help="Skip stages up to and including this one for every threshold (for resuming manually).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    thresholds = sorted(args.thresholds)
    order = ["heatwave", "old-catalogue", "rebuild", "intensity", "fixed-effects", "margins"]
    start = order.index(args.skip_through) + 1 if args.skip_through else 0

    if start <= order.index("heatwave"):
        step_build_heatwave_files(thresholds, args.dry_run)

    for threshold in thresholds:
        label = method_label(threshold)
        print(f"\n=== {label} ({threshold:g} degC) ===", flush=True)
        if start <= order.index("old-catalogue"):
            step_old_catalogue(threshold, args.dry_run)
        if start <= order.index("rebuild"):
            run_dir = step_rebuild_land(threshold, args.dry_run)
        else:
            run_dir = DEFAULT_LAND_RUN_ROOT / f"{label}_2006-2025"
        if start <= order.index("intensity"):
            step_fit_intensity(run_dir, threshold, args.dry_run)
        if start <= order.index("fixed-effects"):
            step_fit_fixed_effects(run_dir, threshold, args.dry_run)
        if start <= order.index("margins"):
            step_report_margins(run_dir, threshold, args.dry_run)

    print("\n=== Collecting comparison table and figure ===", flush=True)
    collect_and_plot(thresholds, args.dry_run)


if __name__ == "__main__":
    main()
