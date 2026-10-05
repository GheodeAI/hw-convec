#!/usr/bin/env python3
"""Manuscript figure 14: absolute wet-bulb threshold sensitivity sweep
(reviewer-requested "sequence of absolute thresholds"; Section 8.1 candidate).

One figure, two stacked subplots:
  (a) adjusted P(CG|HW)/P(CG|non-HW) vs absolute threshold
  (b) adjusted Q5-Q1 CG-probability effect vs absolute threshold, with points
      whose 95% CI includes zero drawn hollow instead of solid, so
      non-significance is visible without reading the error bar.

Uses the shared journal style (common/style.py).
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.config import MANUSCRIPT_FIGURES_DIR, SWEEP_SUMMARY_DIR
from common.style import apply_style

apply_style()

DEFAULT_SUMMARY = SWEEP_SUMMARY_DIR / "absolute_threshold_sensitivity_2006_2025.csv"
DEFAULT_OUTPUT_DIR = MANUSCRIPT_FIGURES_DIR
GREEN, PURPLE = "#1665a3", "#1665a3"


def plot(summary_path: Path, output_dir: Path) -> None:
    df = pd.read_csv(summary_path).sort_values("threshold_degC")
    x = df.threshold_degC.to_numpy()
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    ax.fill_between(x, df.adjusted_occurrence_ratio_ci95_lower, df.adjusted_occurrence_ratio_ci95_upper,
                     color=GREEN, alpha=.2, linewidth=0, label="95% CI")
    ax.plot(x, df.adjusted_occurrence_ratio, "o-", color=GREEN, lw=1.8, ms=5)
    ax.axhline(1.0, color="grey", lw=1, ls="--", label="P(CG|HW) = P(CG|non-HW)")
    ax.set(xlabel="Absolute wet-bulb threshold (°C)",
           ylabel="Adjusted occurrence ratio, P(CG|HW) / P(CG|non-HW)", xticks=x)
    ax.grid(axis="y", alpha=.2)
    fig.tight_layout()
    ratio_path = output_dir / "14_occurrence_ratio_threshold_sensitivity.png"
    fig.savefig(ratio_path, dpi=1200)
    plt.close(fig)

    significant = (df.adjusted_q5_minus_q1_ci95_lower_pp > 0) | (df.adjusted_q5_minus_q1_ci95_upper_pp < 0)
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    ax.fill_between(x, df.adjusted_q5_minus_q1_ci95_lower_pp, df.adjusted_q5_minus_q1_ci95_upper_pp,
                     color=PURPLE, alpha=.2, linewidth=0, label="95% CI")
    ax.plot(x, df.adjusted_q5_minus_q1_pp, "-", color=PURPLE, lw=1.8, zorder=3)
    ax.scatter(x[significant], df.adjusted_q5_minus_q1_pp[significant], s=20, color=PURPLE,
               zorder=3, label="95% CI excludes zero")
    ax.scatter(x[~significant], df.adjusted_q5_minus_q1_pp[~significant], s=20, facecolors="white",
               edgecolors=PURPLE, linewidths=1.2, zorder=3, label="95% CI includes zero")
    ax.axhline(0.0, color="grey", lw=1, ls="--")
    ax.set(xlabel="Absolute wet-bulb threshold (°C)",
           ylabel="Adjusted Q5 - Q1 CG probability (pp)", xticks=x)
    ax.grid(axis="y", alpha=.2)
    fig.tight_layout()
    effect_path = output_dir / "14_intensity_effect_threshold_sensitivity.png"
    fig.savefig(effect_path, dpi=1200)
    plt.close(fig)
    print(ratio_path)
    print(effect_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-csv", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    plot(args.summary_csv, args.output_dir)
