#!/usr/bin/env python3
"""One-off summary stats for the first-CG lifecycle timing and composite
sample sizes, requested alongside the existing 09/10/11 first-CG figures.

Lifecycle phase uses the same definition already used for figure
11_confirmed_first_CG_timing.png (plot_confirmed_land.py:timing()):
phase = lag_hours_from_heatwave_onset / (24 * grid-cell event duration in days),
computed on the grid-cell "compound" population in
data/confirmed_compound_timing.parquet (only fully CG-covered, has_lightning
qualifying regional events, per prepare_confirmed_land.py's 'qualified' filter).
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

VARIABLES = ["tcwv", "cape", "cin", "omega700"]


def run(out: Path) -> None:
    timing = pd.read_parquet(out / "data/confirmed_compound_timing.parquet")
    phase = timing.lag_hours / (24 * timing.duration)
    lag_days = timing.lag_hours / 24

    print(f"n grid-cell compound first-CG onsets (lifecycle sample) = {len(timing)}")
    print(f"lifecycle phase: median={phase.median():.4f}, IQR=[{phase.quantile(.25):.4f}, {phase.quantile(.75):.4f}]")
    print(f"median lag from heatwave onset to first CG = {lag_days.median():.4f} days "
          f"({lag_days.median()*24:.2f} hours)")

    names = np.select([phase < 1/3, phase < 2/3, phase < 1], ["first_third", "middle_third", "last_third"],
                       default="post_24h")
    counts = pd.Series(names).value_counts().reindex(["first_third", "middle_third", "last_third", "post_24h"],
                                                       fill_value=0)
    thirds = pd.DataFrame({"n": counts, "percentage": counts / len(timing) * 100})
    print(thirds.to_string())

    anchors = pd.read_parquet(out / "data/confirmed_anchors.parquet")
    compound = anchors[anchors.kind == "compound"]
    nonhw = anchors[anchors.kind == "nonhw_onset"]
    print(f"\ncompound composite: {len(compound)} grid-cell CG onsets, "
          f"from {compound.event_id.nunique()} distinct regional events")
    print(f"non-heatwave composite: {len(nonhw)} independent onsets "
          f"(from {nonhw.cell.nunique()} distinct grid cells)")

    values = pd.read_parquet(out / "data/first_cg_event_values.parquet")
    lag_rows = []
    for kind in ["compound", "nonhw_onset"]:
        sub = values[values.kind == kind]
        for var in VARIABLES:
            col = f"{var}_climatological_anomaly"
            per_hour = sub.groupby("relative_hour")[col].apply(lambda s: int(s.notna().sum()))
            for hour, n in per_hour.items():
                lag_rows.append(dict(kind=kind, variable=var, relative_hour=hour, n_effective=n))
    lag_table = pd.DataFrame(lag_rows)
    print("\nEffective sample size (non-missing rows) by relative hour and variable:")
    print(lag_table.pivot_table(index="relative_hour", columns=["kind", "variable"], values="n_effective")
          .to_string())
    consistent = lag_table.groupby("kind").n_effective.nunique()
    print(f"\nDistinct n_effective values per kind (1 = fully consistent across all hours/variables):")
    print(consistent.to_string())

    lag_table.to_csv(out / "tables/first_cg_lifecycle_effective_n.csv", index=False)
    print(f"\nWrote {out / 'tables/first_cg_lifecycle_effective_n.csv'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(args.out)
