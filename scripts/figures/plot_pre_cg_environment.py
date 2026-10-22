#!/usr/bin/env python3
"""Matched environmental differences over the whole event versus only the
period preceding the first CG.

One panel per minimum window (12 h, 24 h), because the two windows retain
different pairs: within a panel the whole-event square and the pre-CG circle
share the same pairs, so the connector isolates the truncation. A hollow
marker means a BH-adjusted p >= 0.05.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.layout import sensitivity_dir
from common.figures import figure_path, save_figure
from common.environment import ORDER
from common.style import MID_WIDTH, apply_style, label_panels

apply_style()

TICKS = [r"$q_{850}$", "TCWV", r"$DPD_{2m}$", r"$qv_{850}$", r"$qu_{850}$", r"$Z_{500}$"]
CONNECTOR = "#b0b0b0"
# Whole-event points reuse the blue of the main matched-environment figure, so
# the same quantity reads as the same colour; the pre-CG estimate gets a much
# darker blue so the two are separable in greyscale print as well.
WHOLE_BLUE, PRE_CG_BLUE = "#4e79a7", "#0b3d66"
PANELS = [("whole event, same >=12 h pairs", "pre-CG, >=12 h", "a"),
          ("whole event, same >=24 h pairs", "pre-CG, >=24 h", "b")]


def plot(effects_path: Path, output: Path, *, run_dir: Path | None = None) -> None:
    df = pd.read_csv(effects_path)
    fig, axs = plt.subplots(2, 1, figsize=(MID_WIDTH, 5.6), sharex=True)
    offset = .16
    for ax, (whole_scheme, cut_scheme, _) in zip(axs, PANELS):
        whole = df[df.scheme == whole_scheme].set_index("variable").loc[ORDER]
        cut = df[df.scheme == cut_scheme].set_index("variable").loc[ORDER]
        y = range(len(ORDER))

        # Thin connector, not an arrow: the shift is small for five of the six
        # variables, and what the reader needs to judge is how far the two
        # intervals overlap, not the direction of a near-zero move.
        for yi, v in zip(y, ORDER):
            ax.plot([whole.loc[v, "standardized_difference"], cut.loc[v, "standardized_difference"]],
                    [yi - offset, yi + offset], color=CONNECTOR, lw=.9, ls="--", zorder=2)
        # Hollow means one thing in both series: BH-adjusted p >= 0.05. Applying
        # it only to the truncated estimates would imply the whole-event qu850
        # point was significant, which it is not.
        for table, shift, marker, colour, size in [(whole, -offset, "s", WHOLE_BLUE, 4),
                                                    (cut, offset, "o", PRE_CG_BLUE, 4.5)]:
            significant = table.p_bh < .05
            for mask, face in [(significant, colour), (~significant, "white")]:
                if not mask.any():
                    continue
                sub = table[mask]
                rows = [ORDER.index(v) + shift for v in sub.index]
                ax.errorbar(sub.standardized_difference, rows,
                            xerr=[sub.standardized_difference - sub.standardized_ci95_lower,
                                  sub.standardized_ci95_upper - sub.standardized_difference],
                            fmt=marker, color=colour, markerfacecolor=face, markersize=size,
                            capsize=2.5, linewidth=1, zorder=3 if marker == "s" else 4)

        ax.axvline(0, color="black", lw=.8)
        ax.set_yticks(list(y), TICKS)
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=.2)
    axs[1].set_xlabel("Standardized compound − non-compound difference")
    fig.tight_layout(h_pad=1.6)
    label_panels(fig, axs, "ab", align=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, 'pre_cg_environment', run_dir=run_dir, dpi=1200, bbox_inches="tight")
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    plot(sensitivity_dir(args.out) / "pre_cg_environment/pre_cg_environment_effects.csv",
         figure_path('pre_cg_environment', args.out), run_dir=args.out)
