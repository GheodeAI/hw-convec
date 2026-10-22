#!/usr/bin/env python3
"""Sensitivity of the lightning-occurrence results to the absolute wet-bulb
threshold (18-22 degC). Two separate figures:
  - adjusted P(CG|HW)/P(CG|non-HW) vs absolute threshold
  - adjusted Q5-Q1 CG-probability effect vs absolute threshold, with points
    whose 95% CI includes zero drawn hollow instead of solid.

Uses the shared journal style (common/style.py).
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.figures import figure_path, save_figure
from common.config import SWEEP_SUMMARY_DIR
from common.style import BLUE, MID_WIDTH, apply_style

apply_style()

DEFAULT_SUMMARY = SWEEP_SUMMARY_DIR / "absolute_threshold_sensitivity_2006_2025.csv"
GREEN, PURPLE = BLUE, BLUE


def plot(summary_path: Path) -> None:
    df = pd.read_csv(summary_path).sort_values("threshold_degC")
    x = df.threshold_degC.to_numpy()

    fig, ax = plt.subplots(figsize=(MID_WIDTH, 3.2))
    ax.fill_between(x, df.adjusted_occurrence_ratio_ci95_lower, df.adjusted_occurrence_ratio_ci95_upper,
                     color=GREEN, alpha=.2, linewidth=0, label="95% CI")
    ax.plot(x, df.adjusted_occurrence_ratio, "o-", color=GREEN, lw=1.3, ms=3.5)
    ax.axhline(1.0, color="grey", lw=.9, ls="--", label="P(CG|HW) = P(CG|non-HW)")
    ax.set(xlabel="Absolute wet-bulb threshold (°C)",
           ylabel="Adjusted occurrence ratio", xticks=x)
    ax.grid(axis="y", alpha=.2)
    fig.tight_layout()
    ratio_path = figure_path('occurrence_ratio_threshold_sensitivity')
    save_figure(fig, 'occurrence_ratio_threshold_sensitivity', dpi=1200)
    plt.close(fig)

    significant = (df.adjusted_q5_minus_q1_ci95_lower_pp > 0) | (df.adjusted_q5_minus_q1_ci95_upper_pp < 0)
    fig, ax = plt.subplots(figsize=(MID_WIDTH, 3.2))
    ax.fill_between(x, df.adjusted_q5_minus_q1_ci95_lower_pp, df.adjusted_q5_minus_q1_ci95_upper_pp,
                     color=PURPLE, alpha=.2, linewidth=0, label="95% CI")
    ax.plot(x, df.adjusted_q5_minus_q1_pp, "-", color=PURPLE, lw=1.3, zorder=3)
    ax.scatter(x[significant], df.adjusted_q5_minus_q1_pp[significant], s=14, color=PURPLE,
               zorder=3, label="95% CI excludes zero")
    ax.scatter(x[~significant], df.adjusted_q5_minus_q1_pp[~significant], s=14, facecolors="white",
               edgecolors=PURPLE, linewidths=1, zorder=3, label="95% CI includes zero")
    ax.axhline(0.0, color="grey", lw=.9, ls="--")
    ax.set(xlabel="Absolute wet-bulb threshold (°C)",
           ylabel="Adjusted Q5 − Q1 difference (pp)", xticks=x)
    ax.grid(axis="y", alpha=.2)
    fig.tight_layout()
    effect_path = figure_path('intensity_effect_threshold_sensitivity')
    save_figure(fig, 'intensity_effect_threshold_sensitivity', dpi=1200)
    plt.close(fig)
    print(ratio_path)
    print(effect_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-csv", type=Path, default=DEFAULT_SUMMARY)
    args = parser.parse_args()
    plot(args.summary_csv)
