#!/usr/bin/env python3
"""Trend tests behind the two opening Results statements: the domain-mean
wet-bulb heatwave frequency (Supplementary Fig. S2) and the domain-mean number
of CG days (Supplementary Fig. S4), April-September 2006-2025.

Both series are tested the same way so the Methods can cite one procedure:
Mann-Kendall (Kendall's tau of value against year), the Theil-Sen slope with
its 95% interval, and OLS for reference.
"""
from __future__ import annotations
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))
from common.paths import COMPOUND_DATA_DIR, HEATWAVE_TABLES_DIR

SERIES = [
    ("Heatwave days per year (S2a)",
     HEATWAVE_TABLES_DIR / "timeseries/wetbulb_heatwave_cg_domain_annual_timeseries_AprSep_90pplus20.csv",
     "year", "mean_heatwave_days"),
    ("Heatwave intensity (S2b)",
     HEATWAVE_TABLES_DIR / "timeseries/wetbulb_heatwave_cg_domain_annual_timeseries_AprSep_90pplus20.csv",
     "year", "mean_heatwave_intensity"),
    ("CG days per year (S4a)",
     COMPOUND_DATA_DIR / "cg_spatial_stats/euclid_cg_annual_2006-2025.csv",
     "year", "mean_cg_days"),
    ("CG-day flash intensity (S4b)",
     COMPOUND_DATA_DIR / "cg_spatial_stats/euclid_cg_annual_2006-2025.csv",
     "year", "mean_cg_day_flash_intensity"),
]


def main() -> None:
    for label, path, xcol, ycol in SERIES:
        df = pd.read_csv(path)
        if xcol not in df:
            df = df.reset_index()
        x = df[xcol].to_numpy(float)
        y = df[ycol].to_numpy(float)
        tau, mk_p = stats.kendalltau(x, y)
        ts = stats.theilslopes(y, x, alpha=.95)
        ols = stats.linregress(x, y)
        print(f"\n=== {label} ===  ({path.name}, n={len(y)}, {int(x.min())}-{int(x.max())})")
        print(f"  mean = {y.mean():.3f}, SD = {y.std(ddof=1):.3f}")
        print(f"  Mann-Kendall: tau = {tau:.3f}, p = {mk_p:.4f}")
        print(f"  Theil-Sen slope = {ts.slope:.4f} per year "
              f"(95% CI {ts.low_slope:.4f} to {ts.high_slope:.4f}); "
              f"change over record = {ts.slope * (x.max() - x.min()):.3f}")
        print(f"  OLS slope = {ols.slope:.4f} per year, p = {ols.pvalue:.4f}, r = {ols.rvalue:.3f}")


if __name__ == "__main__":
    main()
