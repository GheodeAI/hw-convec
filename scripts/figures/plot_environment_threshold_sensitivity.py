#!/usr/bin/env python3
"""Figure for the matched environmental-difference threshold sensitivity
(pair + year-FE regression; the corrected, land-restricted version of Section
8.1's second table). One 2x3 grid, one panel per variable.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.config import MANUSCRIPT_FIGURES_DIR, SWEEP_SUMMARY_DIR
from common.environment import ORDER, LABELS, UNITS
from common.style import BLUE as COLOR, apply_style, panel_letter

apply_style()

DEFAULT_SUMMARY = SWEEP_SUMMARY_DIR / "environment_threshold_sensitivity_2006_2025.csv"
DEFAULT_OUTPUT = SWEEP_SUMMARY_DIR / "environment_threshold_sensitivity.png"


def plot(summary_path: Path, output_path: Path) -> None:
    df = pd.read_csv(summary_path)
    fig, axs = plt.subplots(2, 3, figsize=(14, 8))
    for ax, v, c in zip(axs.flat, ORDER, "abcdef"):
        sub = df[df.variable == v].sort_values("threshold_degC")
        x = sub.threshold_degC.to_numpy()
        significant = (sub.standardized_ci95_lower > 0) | (sub.standardized_ci95_upper < 0)
        ax.fill_between(x, sub.standardized_ci95_lower, sub.standardized_ci95_upper,
                        color=COLOR, alpha=.15, linewidth=0)
        ax.plot(x, sub.standardized_difference, "-", color=COLOR, lw=1.8, zorder=3)
        ax.scatter(x[significant], sub.standardized_difference[significant], s=20, color=COLOR, zorder=3)
        ax.scatter(x[~significant], sub.standardized_difference[~significant], s=20, facecolors="white",
                   edgecolors=COLOR, linewidths=1.2, zorder=3)
        ax.axhline(0.0, color="grey", lw=1, ls="--")
        ax.set(xlabel="Absolute wet-bulb threshold (°C)", ylabel="Standardized difference")
        ax.grid(axis="y", alpha=.2)
        panel_letter(ax, c)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=1200)
    MANUSCRIPT_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(MANUSCRIPT_FIGURES_DIR / "21_environment_threshold_sensitivity.png", dpi=1200)
    plt.close(fig)
    print(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-csv", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    plot(args.summary_csv, args.output)
