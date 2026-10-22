#!/usr/bin/env python3
"""Re-estimate the matched environmental contrast using only the part of each
event that precedes its first CG.

The existing event means average the whole event, so for compound events they
include the convective period itself; a difference could then reflect the
convection rather than the environment that preceded it. Here the matching is
left untouched -- the same pairs, covariates and caliper -- and only the
outcome changes:

  compound event   averaged over [onset, t_c), t_c = the event's earliest CG
                    anywhere in its footprint (regional first CG)
  control event    averaged over [onset, onset + (t_c - onset of its partner)),
                    truncated at its own end, so both members of a pair cover
                    the same elapsed time since onset
  footprint        the cells belonging to the event during that window only,
                    not the whole-event union

Pairs whose compound member has its first CG within a minimum window of onset
are dropped, because an average over a few hours is dominated by a single ERA5
time step. Both a 12 h and a 24 h minimum are reported, and for each of those
pair sets the whole-event average is refitted as well, so the three columns
differ only in the truncation.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats, ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.config import SHARED_CLIMATOLOGY_DIR
from common.cache import register_climatology
from common.layout import sensitivity_dir
from common.analysis_inputs import input_path
from common.config import YEARS
from common.environment import load_fields, VARS, MATCHV
from common.stats import bh, fit_difference

YEAR_DUMMIES = np.arange(2007, 2026)


def event_windows(out: Path, pairs: pd.DataFrame) -> pd.DataFrame:
    """One row per event taking part in a pair, with its truncation time."""
    e = pd.read_csv(out / "events/regional_events.csv",
                    parse_dates=["start_time", "end_time", "first_cg_time"]).set_index("event_id")
    rows = []
    for pair in pairs.itertuples():
        compound = e.loc[pair.compound_event_id]
        control = e.loc[pair.control_event_id]
        window = compound.first_cg_time - compound.start_time
        rows.append(dict(match_set_id=pair.match_set_id, role="compound",
                         event_id=pair.compound_event_id, year=int(compound.year),
                         start_time=compound.start_time, end_time=compound.end_time,
                         cutoff=compound.first_cg_time, window_hours=window.total_seconds() / 3600))
        # Same elapsed time since onset, but never past the control's own end.
        control_cutoff = min(control.start_time + window, control.end_time)
        rows.append(dict(match_set_id=pair.match_set_id, role="control",
                         event_id=pair.control_event_id, year=int(control.year),
                         start_time=control.start_time, end_time=control.end_time,
                         cutoff=control_cutoff,
                         window_hours=(control_cutoff - control.start_time).total_seconds() / 3600))
    return pd.DataFrame(rows)


def truncated_footprints(out: Path, windows: pd.DataFrame) -> dict[int, np.ndarray]:
    """Cells belonging to each event during its truncated window only."""
    footprints: dict[int, np.ndarray] = {}
    for year in YEARS:
        subset = windows[windows.year == year]
        if subset.empty:
            continue
        with np.load(input_path(out, f"labels_{year}.npz")) as d:
            labels = d["labels"]
            offset = int(d["offset"])
        nx = labels.shape[2]
        year_start = pd.Timestamp(year, 1, 1)
        for row in subset.itertuples():
            label_id = row.event_id - offset
            first_day = int((row.start_time - year_start).days)
            # Include every day whose 00-24 UTC interval overlaps [start, cutoff).
            last_day = int(np.ceil((row.cutoff - year_start).total_seconds() / 86400)) - 1
            last_day = max(last_day, first_day)
            block = labels[first_day:last_day + 1]
            _, rr, cc = np.nonzero(block == label_id)
            footprints[row.event_id] = np.unique(rr * nx + cc)
        print(f"Footprints for {year}: {len(footprints)} events so far", flush=True)
    return footprints


def extract(out: Path, windows: pd.DataFrame, footprints: dict[int, np.ndarray],
            climatology_dir: Path) -> pd.DataFrame:
    index = {eid: i for i, eid in enumerate(windows.event_id)}
    sums = {v: np.zeros(len(windows)) for v in MATCHV}
    counts = {v: np.zeros(len(windows), np.int64) for v in MATCHV}

    months = set()
    for row in windows.itertuples():
        months.update(pd.date_range(row.start_time.normalize(), row.cutoff.normalize()).month)
    for month in sorted(months):
        clim_path = climatology_dir / f"month_{month:02d}.npz"
        if not clim_path.exists():
            raise FileNotFoundError(f"Missing cached climatology {clim_path}")
        with np.load(clim_path) as d:
            clim = {v: d[v] for v in VARS}
        for year in YEARS:
            active = windows[(windows.year == year)
                             & (windows.start_time.dt.month <= month)
                             & (windows.cutoff.dt.month >= month)]
            if active.empty:
                continue
            dates, fields = load_fields(year, month)
            slots = (dates.day.values - 1) * 8 + dates.hour.values // 3
            for v in MATCHV:
                anomaly = fields[v] - clim[v][slots]
                flat = anomaly.reshape(len(dates), -1)
                for row in active.itertuples():
                    start = dates.searchsorted(row.start_time)
                    stop = dates.searchsorted(row.cutoff)  # left-closed, right-open
                    if stop <= start:
                        continue
                    values = flat[start:stop, footprints[row.event_id]]
                    position = index[row.event_id]
                    sums[v][position] += np.nansum(values)
                    counts[v][position] += int(np.isfinite(values).sum())
            print(f"  extracted {year}-{month:02d} ({len(active)} events)", flush=True)

    result = windows[["match_set_id", "role", "event_id", "year", "window_hours"]].copy()
    for v in MATCHV:
        result[v] = np.divide(sums[v], counts[v], out=np.full(len(windows), np.nan),
                              where=counts[v] > 0)
        result[v + "_n"] = counts[v]
    return result


def matched_environment_scale(out: Path) -> np.ndarray:
    """The denominator of the main matched-environment result: the pre-match pooled standard deviation
    of the whole-event means over the full eligible sample. Fixed once here so
    the whole-event, 12 h and 24 h results are all expressed in the same units
    and differ only by the truncation."""
    env = pd.read_csv(out / "environment/tables/event_environment_anomaly.csv")
    scales = []
    for v in MATCHV:
        compound = env.loc[env.has_lightning, v].dropna()
        control = env.loc[~env.has_lightning, v].dropna()
        scales.append(np.sqrt((compound.var(ddof=1) + control.var(ddof=1)) / 2))
    return np.array(scales)


def fit(values: pd.DataFrame, pairs_kept: pd.DataFrame, label: str, scale: np.ndarray) -> pd.DataFrame:
    wide = values.pivot(index="match_set_id", columns="role", values=MATCHV)
    wide = wide.loc[pairs_kept.match_set_id]
    y = np.column_stack([wide[(v, "compound")] - wide[(v, "control")] for v in MATCHV])
    good = np.isfinite(y).all(axis=1)
    y = y[good]
    kept = pairs_kept.loc[good.nonzero()[0]] if good.sum() < len(pairs_kept) else pairs_kept

    years = values.pivot(index="match_set_id", columns="role", values="year").loc[kept.match_set_id]
    z = ((years["compound"].to_numpy()[:, None] == YEAR_DUMMIES).astype(float)
         - (years["control"].to_numpy()[:, None] == YEAR_DUMMIES).astype(float))
    design = np.column_stack([np.ones(len(kept)), z])
    beta, covariance, _, correction = fit_difference(design, y)
    se = np.sqrt(covariance[:, 0, 0])
    critical = stats.t.ppf(.975, len(kept) - 1)

    table = pd.DataFrame(dict(
        scheme=label, variable=MATCHV, n_pairs=len(kept), difference=beta[0], cluster_se=se,
        ci95_lower=beta[0] - critical * se, ci95_upper=beta[0] + critical * se,
        standardized_difference=beta[0] / scale,
        standardized_ci95_lower=(beta[0] - critical * se) / scale,
        standardized_ci95_upper=(beta[0] + critical * se) / scale,
        p=2 * stats.t.sf(abs(beta[0] / se), len(kept) - 1), cr1_correction=correction))
    table["p_bh"] = bh(table.p.to_numpy())
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, default=None,
                        help="Defaults to <out>/matching/comparison/area_only_pairs.csv")
    parser.add_argument("--climatology-dir", type=Path, default=SHARED_CLIMATOLOGY_DIR)
    parser.add_argument("--output", type=Path, default=None,
                        help="Defaults to outputs/sensitivity/pre_cg_environment for the primary run")
    args = parser.parse_args()

    register_climatology(args.climatology_dir)
    pairs_path = args.pairs or args.out / "matching/comparison/area_only_pairs.csv"
    output = args.output or sensitivity_dir(args.out) / "pre_cg_environment"
    output.mkdir(parents=True, exist_ok=True)

    pairs = pd.read_csv(pairs_path)
    windows = event_windows(args.out, pairs)
    windows.to_csv(output / "event_windows.csv", index=False)

    compound_windows = windows[windows.role == "compound"].set_index("match_set_id").window_hours
    print(f"\nCompound truncation windows (hours): median={compound_windows.median():.2f}, "
          f"IQR=[{compound_windows.quantile(.25):.2f}, {compound_windows.quantile(.75):.2f}], "
          f"n={len(compound_windows)}")

    footprints = truncated_footprints(args.out, windows)
    values = extract(args.out, windows, footprints, args.climatology_dir)
    values.to_csv(output / "truncated_event_environment.csv", index=False)

    full = pd.read_csv(args.out / "environment/tables/event_environment_anomaly.csv").set_index("event_id")
    full_values = windows[["match_set_id", "role", "event_id", "year"]].copy()
    for v in MATCHV:
        full_values[v] = full.loc[full_values.event_id, v].to_numpy()

    scale = matched_environment_scale(args.out)
    print("\nFixed standardisation denominator (main matched-environment pooled SD): "
          + ", ".join(f"{v}={s:.6g}" for v, s in zip(MATCHV, scale)))

    tables, summary = [], []
    for minimum in (12, 24):
        keep_ids = compound_windows[compound_windows >= minimum].index
        kept = pairs[pairs.match_set_id.isin(keep_ids)].reset_index(drop=True)
        window_subset = compound_windows.loc[kept.match_set_id]
        summary.append(dict(minimum_window_hours=minimum, pairs=len(kept),
                            window_median_hours=window_subset.median(),
                            window_q25_hours=window_subset.quantile(.25),
                            window_q75_hours=window_subset.quantile(.75)))
        tables.append(fit(values, kept, f"pre-CG, >={minimum} h", scale))
        tables.append(fit(full_values, kept, f"whole event, same >={minimum} h pairs", scale))

    result = pd.concat(tables, ignore_index=True)
    result.to_csv(output / "pre_cg_environment_effects.csv", index=False)
    pd.DataFrame(summary).to_csv(output / "window_summary.csv", index=False)
    (output / "method.json").write_text(json.dumps(dict(
        truncation="compound: [onset, first CG); control: same elapsed time since onset, capped at its own end",
        footprint="cells belonging to the event during the truncated window only",
        matching="unchanged; pairs reused from " + str(pairs_path),
        model="pair + start-year fixed effects on matched differences, CR1 pair-clustered, BH across six variables",
        standardisation="one fixed denominator per variable for every scheme: the pre-match pooled SD of the "
                        "whole-event means over the full eligible sample, as in the main matched-environment result",
        climatology="cached 2006-2025 calendar-day/3-hour means, identical to the whole-event analysis"),
        indent=2) + "\n")

    print("\n" + pd.DataFrame(summary).to_string(index=False))
    print("\n" + result[["scheme", "variable", "n_pairs", "difference", "standardized_difference",
                          "standardized_ci95_lower", "standardized_ci95_upper", "p_bh"]].to_string(index=False))
    print(f"\nWrote {output}")


if __name__ == "__main__":
    main()
