#!/usr/bin/env python3
"""How soon after heatwave onset does the first CG occur?

Reports the distribution of the lag from the regional event's start (00 UTC of
its first heatwave day) to its first CG flash, for the compound events used in
the event-level and lifecycle analyses. Answers how large the "convection right
at onset" group is, which the early/middle/late thirds table cannot show.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def describe(name: str, lag_hours: pd.Series) -> None:
    n = len(lag_hours)
    print(f"\n=== {name} (n={n}) ===")
    for hours in (3, 6, 12, 24, 48, 72):
        share = (lag_hours < hours).mean() * 100
        print(f"  first CG within {hours:>3d} h of onset: {int((lag_hours < hours).sum()):>5d}  ({share:5.2f}%)")
    print("  quantiles of lag (hours): " + ", ".join(
        f"p{q}={np.percentile(lag_hours, q):.1f}" for q in (10, 25, 50, 75, 90)))
    print(f"  median lag = {lag_hours.median()/24:.2f} days")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    e = pd.read_csv(args.out / "tables/regional_events.csv",
                    parse_dates=["first_cg_time", "start_time"])
    covered = e[e.has_lightning & (e.cg_coverage_fraction == 1)].copy()
    covered["lag_hours"] = (covered.first_cg_time - covered.start_time).dt.total_seconds() / 3600

    describe("Event-level pool: start month Apr-Sep",
             covered.loc[covered.start_month.between(4, 9), "lag_hours"])
    describe("Lifecycle sample: first CG in Apr-Sep",
             covered.loc[covered.first_cg_time.dt.month.between(4, 9), "lag_hours"])

    same_day = covered.first_cg_time.dt.normalize() == covered.start_time.dt.normalize()
    print(f"\nFirst CG falls on the event's first calendar day: "
          f"{int(same_day.sum())} of {len(covered)} ({same_day.mean()*100:.2f}%)")


if __name__ == "__main__":
    main()
