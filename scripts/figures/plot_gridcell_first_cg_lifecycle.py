#!/usr/bin/env python3
"""Timing of the grid-cell first CG within the local heatwave spell.

a  lifecycle phase = hours from local onset / (24 x local duration). Values
   above one fall in the 24 h after the local spell ends; because that window
   is a different fraction of each spell, it is drawn as a separate shaded
   segment measured in hours after the end rather than stretched in phase.
b  lag from local onset to first CG, in days.
Same population as the compound composites (first_cg/data/confirmed_compound_timing.parquet).
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.figures import figure_path, save_figure
from common.style import BLUE, FULL_WIDTH, apply_style, label_panels

apply_style()

POST_WIDTH = .25          # width on the x axis given to the 24 h post-event segment
MAIN_BINS = 30
POST_BINS = 8             # 3 h each


def plot(out: Path) -> None:
    t = pd.read_parquet(out / "first_cg/data/confirmed_compound_timing.parquet")
    phase = (t.lag_hours / (24 * t.duration)).to_numpy()
    hours_after_end = (t.lag_hours - 24 * t.duration).to_numpy()
    n = len(t)
    post = phase >= 1

    fig, (ax_a, ax_b) = plt.subplots(2, 1, figsize=(FULL_WIDTH, 5.4))

    main_counts, main_edges = np.histogram(phase[~post], bins=MAIN_BINS, range=(0, 1))
    post_counts, post_edges = np.histogram(hours_after_end[post], bins=POST_BINS, range=(0, 24))
    post_x = 1 + post_edges / 24 * POST_WIDTH
    ax_a.axvspan(1, 1 + POST_WIDTH, color="#e6e6e6", zorder=0)
    ax_a.bar(main_edges[:-1], main_counts / n * 100, width=np.diff(main_edges), align="edge",
             color=BLUE, edgecolor="white", linewidth=.4, zorder=2)
    ax_a.bar(post_x[:-1], post_counts / n * 100, width=np.diff(post_x), align="edge",
             color=BLUE, edgecolor="white", linewidth=.4, zorder=2)
    for x in (1 / 3, 2 / 3):
        ax_a.axvline(x, color="grey", lw=.9, ls=":", zorder=3)
    ax_a.axvline(1, color="black", lw=.8, zorder=3)
    ax_a.axvline(np.median(phase), color="black", lw=1, ls="--", zorder=4)
    ax_a.set_xlim(0, 1 + POST_WIDTH)
    ax_a.set_xticks([0, 1 / 3, 2 / 3, 1, 1 + POST_WIDTH / 2, 1 + POST_WIDTH],
                    ["0", "1/3", "2/3", "1", "+12 h", "+24 h"])
    ax_a.set(xlabel="Lifecycle phase of first CG (local heatwave onset = 0, end = 1)",
             ylabel="First-CG occurrences (%)")
    ax_a.grid(axis="y", alpha=.2)

    lag_days = t.lag_hours.to_numpy() / 24
    upper = np.ceil(np.percentile(lag_days, 99))
    counts, edges = np.histogram(lag_days, bins=np.arange(0, upper + .5, .5))
    ax_b.bar(edges[:-1], counts / n * 100, width=.5, align="edge",
             color=BLUE, edgecolor="white", linewidth=.4, zorder=2)
    ax_b.axvline(np.median(lag_days), color="black", lw=1, ls="--", zorder=3)
    ax_b.set_xlim(0, upper)
    ax_b.set(xlabel="Lag from local heatwave onset to first CG (days)", ylabel="First-CG occurrences (%)")
    ax_b.grid(axis="y", alpha=.2)

    fig.tight_layout(h_pad=1.6)
    label_panels(fig, [ax_a, ax_b], "ab", align=True)
    target = figure_path('gridcell_first_cg_lifecycle', out)
    save_figure(fig, 'gridcell_first_cg_lifecycle', run_dir=out, dpi=1200, bbox_inches="tight")
    plt.close(fig)
    print(f"n = {n}, median phase = {np.median(phase):.3f}, median lag = {np.median(lag_days):.2f} days, "
          f"post-event share = {post.mean()*100:.1f}%, lag axis cut at p99 = {upper:.0f} days "
          f"({(lag_days > upper).mean()*100:.2f}% beyond)")
    print(target)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    plot(parser.parse_args().out)
