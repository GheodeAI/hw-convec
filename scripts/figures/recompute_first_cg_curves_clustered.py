#!/usr/bin/env python3
"""Recompute the first-CG environmental-evolution curves (09_first_CG_compound
/ 10_compound_vs_nonHW_CG_onsets) with cluster-robust uncertainty, per Jurg and
Elena's suggestion: "estimating the uncertainty by clustering or resampling at
the regional-event level, rather than treating the individual grid-cell events
as independent. The same approach would strengthen the comparison between CG
events during and outside heatwaves."

The original plot_confirmed_land.py:curves() treats every grid-cell first-CG
"id" as an independent observation for both the SEM/IQR summary and the
adjacent-3h-change significance test (stats.sem / stats.ttest_1samp on the raw
rows) -- but many grid-cell first-CG occurrences belong to the same regional
heatwave event, so they are spatially/temporally correlated, not independent.

Cluster definition:
  - kind == 'compound'    -> cluster on event_id (the parent regional event;
                              already present in the data, no recomputation of
                              the underlying values needed)
  - kind == 'nonhw_onset' -> no regional event exists outside a heatwave, so
                              cluster on grid cell instead (repeated onsets at
                              the same cell are not independent either)

Method: cluster bootstrap (resample clusters with replacement, pool all rows
from the resampled clusters, 1000 replicates) for both the per-hour mean CI
and the adjacent-hour change CI/p-value, then BH-correct across the 9 adjacent
comparisons per (kind, variable) exactly as before.

Writes NEW files alongside the originals -- confirmed_curve_summary.csv and
confirmed_adjacent_changes.csv, and figures 09/10, are left untouched so the
two versions can be compared directly.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.stats import bh
from common.style import BLUE, ORANGE, FULL_WIDTH, apply_style, label_panels

apply_style()

VARIABLES = ["tcwv", "cape", "cin", "omega700"]
LABELS = {"tcwv": "Total column water vapour", "cape": "CAPE", "cin": "CIN", "omega700": "700 hPa omega"}
UNITS = {"tcwv": r"kg m$^{-2}$", "cape": r"J kg$^{-1}$", "cin": r"J kg$^{-1}$", "omega700": r"Pa s$^{-1}$"}
N_BOOT = 1000
SEED = 20260916


def cluster_totals(values: np.ndarray, clusters: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-cluster sum and row count. Pooling the rows of a set of resampled
    clusters and taking their mean is exactly sum(sums)/sum(counts), so the
    bootstrap never has to touch the individual rows again -- with millions of
    non-heatwave onsets that is the difference between hours and seconds."""
    _, inverse = np.unique(clusters, return_inverse=True)
    sums = np.bincount(inverse, weights=values)
    counts = np.bincount(inverse).astype(float)
    return sums, counts


def bootstrap_pooled_means(sums: np.ndarray, counts: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n = len(sums)
    boot = np.empty(N_BOOT)
    for k in range(N_BOOT):
        picked = rng.integers(n, size=n)
        boot[k] = sums[picked].sum() / counts[picked].sum()
    return boot


def cluster_bootstrap_mean_ci(values: np.ndarray, clusters: np.ndarray, rng: np.random.Generator) -> tuple[float, float, float]:
    """Point estimate is the plain sample mean; CI comes from resampling
    unique clusters with replacement and pooling their rows each time."""
    if len(np.unique(clusters)) < 2:
        return float(np.mean(values)), np.nan, np.nan
    boot = bootstrap_pooled_means(*cluster_totals(values, clusters), rng)
    return float(np.mean(values)), float(np.quantile(boot, .025)), float(np.quantile(boot, .975))


def cluster_bootstrap_diff_p(delta: np.ndarray, clusters: np.ndarray, rng: np.random.Generator) -> tuple[float, float, float, float]:
    """Same idea for a paired adjacent-hour difference; p-value from a normal
    approximation using the bootstrap standard error (matches the two-sided
    Wald-CI convention used elsewhere in this codebase, e.g. fit_land_intensity.py)."""
    if len(np.unique(clusters)) < 2:
        return float(np.mean(delta)), np.nan, np.nan, np.nan
    boot = bootstrap_pooled_means(*cluster_totals(delta, clusters), rng)
    mean = float(np.mean(delta))
    se = float(np.std(boot, ddof=1))
    p = float(2 * stats.norm.sf(abs(mean / se))) if se > 0 else np.nan
    return mean, np.quantile(boot, .025), np.quantile(boot, .975), p


def cluster_bootstrap_two_sample_diff(
    x1: np.ndarray, c1: np.ndarray, x2: np.ndarray, c2: np.ndarray, rng: np.random.Generator
) -> tuple[float, float, float, float]:
    """Direct event-level (cluster-level) comparison between the two groups at
    one hour: independent two-sample cluster bootstrap for mean(x1)-mean(x2)
    -- group 1 (compound) resampled by event_id, group 2 (nonhw_onset) by
    cell, each independently of the other."""
    if len(np.unique(c1)) < 2 or len(np.unique(c2)) < 2:
        return float(x1.mean() - x2.mean()), np.nan, np.nan, np.nan
    boot = (bootstrap_pooled_means(*cluster_totals(x1, c1), rng)
            - bootstrap_pooled_means(*cluster_totals(x2, c2), rng))
    mean_diff = float(x1.mean() - x2.mean())
    se = float(np.std(boot, ddof=1))
    p = float(2 * stats.norm.sf(abs(mean_diff / se))) if se > 0 else np.nan
    return mean_diff, float(np.quantile(boot, .025)), float(np.quantile(boot, .975)), p


def cluster_column(kind: str) -> str:
    return "event_id" if kind == "compound" else "cell"


def compute_between_group(out: Path, rng: np.random.Generator) -> pd.DataFrame:
    d = pd.read_parquet(out / "data/first_cg_event_values.parquet")
    rows = []
    for v in VARIABLES:
        col = f"{v}_climatological_anomaly"
        compound = d[d.kind == "compound"]
        control = d[d.kind == "nonhw_onset"]
        for h in range(-12, 13, 3):
            g1 = compound[compound.relative_hour == h].dropna(subset=[col])
            g2 = control[control.relative_hour == h].dropna(subset=[col])
            diff, lo, hi, p = cluster_bootstrap_two_sample_diff(
                g1[col].to_numpy(), g1["event_id"].to_numpy(),
                g2[col].to_numpy(), g2["cell"].to_numpy(), rng,
            )
            rows.append(dict(variable=v, hour=h, n_compound=len(g1), n_compound_events=g1.event_id.nunique(),
                              n_control=len(g2), n_control_cells=g2.cell.nunique(),
                              difference=diff, ci95_lower=lo, ci95_upper=hi, p=p))
    tab = pd.DataFrame(rows)
    tab["p_bh"] = tab.groupby("variable").p.transform(lambda x: bh(x.to_numpy()))
    return tab


def compute(out: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = pd.read_parquet(out / "data/first_cg_event_values.parquet")
    rng = np.random.default_rng(SEED)
    summary_rows, change_rows = [], []
    for kind in ["compound", "nonhw_onset"]:
        a = d[d.kind == kind]
        cluster_col = cluster_column(kind)
        for v in VARIABLES:
            col = f"{v}_climatological_anomaly"
            for h, g in a.groupby("relative_hour"):
                g = g.dropna(subset=[col])
                x = g[col].to_numpy()
                clusters = g[cluster_col].to_numpy()
                n_clusters = len(np.unique(clusters))
                mean, lo, hi = cluster_bootstrap_mean_ci(x, clusters, rng)
                summary_rows.append(dict(kind=kind, variable=v, hour=h, n=len(x), n_clusters=n_clusters,
                                          mean=mean, median=np.median(x),
                                          q25=np.quantile(x, .25), q75=np.quantile(x, .75),
                                          ci95_lower=lo, ci95_upper=hi))
            wide_value = a.pivot(index="id", columns="relative_hour", values=col)
            wide_cluster = a.drop_duplicates("id").set_index("id")[cluster_col]
            for prev, h in zip(range(-12, 12, 3), range(-9, 13, 3)):
                delta_series = (wide_value[h] - wide_value[prev]).dropna()
                clusters = wide_cluster.loc[delta_series.index].to_numpy()
                mean, lo, hi, p = cluster_bootstrap_diff_p(delta_series.to_numpy(), clusters, rng)
                change_rows.append(dict(kind=kind, variable=v, hour=h, n=len(delta_series),
                                         n_clusters=len(np.unique(clusters)),
                                         difference=mean, ci95_lower=lo, ci95_upper=hi, p=p))
    summary = pd.DataFrame(summary_rows)
    changes = pd.DataFrame(change_rows)
    changes["p_bh"] = changes.groupby(["kind", "variable"]).p.transform(lambda x: bh(x.to_numpy()))
    return summary, changes


def plot(out: Path, summary: pd.DataFrame, changes: pd.DataFrame) -> None:
    fig, axs = plt.subplots(2, 2, figsize=(12.1, 7.15))
    for ax, v in zip(axs.flat, VARIABLES):
        a = summary[(summary.kind == "compound") & (summary.variable == v)].sort_values("hour")
        ax.fill_between(a.hour, a.q25, a.q75, color="#a5d0e5", alpha=.6, label="IQR")
        ax.fill_between(a.hour, a.ci95_lower, a.ci95_upper, color=BLUE, alpha=.25, linewidth=0,
                         label="95% event-cluster bootstrap CI of the mean")
        ax.plot(a.hour, a["mean"], "o-", c=BLUE, lw=2, label="Mean")
        ax.plot(a.hour, a["median"], "--", c="grey", label="Median")
        sig = changes[(changes.kind == "compound") & (changes.variable == v) & (changes.p_bh < .05)].hour
        ss = a.set_index("hour").loc[sig]
        ax.scatter(ss.index, ss["mean"], marker="*", s=90, c="#a62a55", zorder=5,
                   label="Adjacent change: BH q < 0.05 (event-cluster bootstrap)")
        ax.axvline(0, c="black", lw=.7); ax.axhline(0, c="black", lw=.7)
        ax.set(title=LABELS[v], xlabel="Hours from first CG (nearest 3-hour reference)", ylabel=UNITS[v])
        ax.set_xticks(range(-12, 13, 3)); ax.grid(alpha=.2)
    axs[0, 0].legend(ncol=1, fontsize=8)
    fig.suptitle("Land grid-cell first CG within qualifying regional compound events: anomalies\n"
                 "(event-cluster bootstrap uncertainty, not per grid-cell-event)")
    fig.tight_layout()
    fig.savefig(out / "figures/09_first_CG_compound_event_cluster.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, axs = plt.subplots(2, 2, figsize=(12.1, 7.15))
    for ax, v in zip(axs.flat, VARIABLES):
        for kind, col, label in [("compound", BLUE, "compound-event first CG"),
                                  ("nonhw_onset", ORANGE, "non-heatwave CG onset")]:
            a = summary[(summary.kind == kind) & (summary.variable == v)].sort_values("hour")
            ax.fill_between(a.hour, a.ci95_lower, a.ci95_upper, color=col, alpha=.2, linewidth=0)
            ax.plot(a.hour, a["mean"], "o-", c=col, lw=2, label=label)
        ax.axvline(0, c="black", lw=.7); ax.axhline(0, c="black", lw=.7)
        ax.set(title=LABELS[v], xlabel="Hours from CG reference (nearest 3-hour slot)", ylabel=UNITS[v])
        ax.set_xticks(range(-12, 13, 3)); ax.grid(alpha=.2)
    axs[0, 0].legend(fontsize=9)
    fig.suptitle("Land compound first CG vs non-heatwave CG onsets: Apr-Sep centres, 2006-2025\n"
                 "(shaded = 95% cluster bootstrap CI: event-level for compound, cell-level for non-heatwave onsets)")
    fig.tight_layout()
    fig.savefig(out / "figures/10_compound_vs_nonHW_CG_onsets_event_cluster.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_merged(out: Path, summary: pd.DataFrame, changes: pd.DataFrame) -> None:
    """09+10 combined: one panel per variable, compound (full detail: IQR, CI,
    significance stars) overlaid with the non-heatwave onset comparison (CI
    only) -- the 'compound' curve is otherwise duplicated across two figures.
    The direct compound-vs-non-heatwave comparison itself lives in its own
    figure (11_first_CG_compound_vs_nonHW_difference.png), not marked here."""
    fig, axs = plt.subplots(2, 2, figsize=(FULL_WIDTH, 4.8))
    for i, (ax, v, c) in enumerate(zip(axs.flat, VARIABLES, "abcd")):
        a = summary[(summary.kind == "compound") & (summary.variable == v)].sort_values("hour")
        ax.fill_between(a.hour, a.q25, a.q75, color="#a5d0e5", alpha=.5, linewidth=0, label="Compound IQR")
        ax.fill_between(a.hour, a.ci95_lower, a.ci95_upper, color=BLUE, alpha=.3, linewidth=0,
                         label="95% event/cell-cluster bootstrap CI")
        ax.plot(a.hour, a["mean"], "o-", c=BLUE, lw=1.3, ms=3.5, label="Compound-event first CG")
        sig = changes[(changes.kind == "compound") & (changes.variable == v) & (changes.p_bh < .05)].hour
        ss = a.set_index("hour").loc[sig]
        ax.scatter(ss.index, ss["mean"], marker="*", s=36, c="#a62a55", zorder=5,
                   label="Adjacent change: BH q < 0.05")

        b = summary[(summary.kind == "nonhw_onset") & (summary.variable == v)].sort_values("hour")
        ax.fill_between(b.hour, b.ci95_lower, b.ci95_upper, color=ORANGE, alpha=.25, linewidth=0)
        ax.plot(b.hour, b["mean"], "^--", c=ORANGE, lw=1.3, ms=3.5, label="Non-heatwave CG onset")

        ax.axvline(0, c="black", lw=.7); ax.axhline(0, c="grey", lw=.9, ls="--")
        ax.set_ylabel(UNITS[v])
        ax.set_xticks(range(-12, 13, 3)); ax.grid(alpha=.2)
    fig.supxlabel("Hours from CG reference (nearest 3-hour slot)", fontsize=plt.rcParams["axes.labelsize"])
    fig.tight_layout(h_pad=1.6)
    label_panels(fig, axs.flat, "abcd")
    fig.savefig(out / "figures/09_10_first_CG_merged_event_cluster.png", dpi=1200, bbox_inches="tight")
    plt.close(fig)


def plot_between_group(out: Path, between_group: pd.DataFrame) -> None:
    """Standalone figure for the direct event-level compound-vs-non-heatwave
    comparison: the difference curve itself, with its own cluster-bootstrap CI
    and BH-significance stars, rather than reading it off two overlaid curves."""
    fig, axs = plt.subplots(2, 2, figsize=(FULL_WIDTH, 4.8))
    for i, (ax, v, c) in enumerate(zip(axs.flat, VARIABLES, "abcd")):
        a = between_group[between_group.variable == v].sort_values("hour")
        ax.fill_between(a.hour, a.ci95_lower, a.ci95_upper, color=BLUE, alpha=.2, linewidth=0,
                         label="95% two-sample cluster bootstrap CI")
        ax.plot(a.hour, a.difference, "o-", c=BLUE, lw=1.3, ms=3.5, label="Compound - non-heatwave onset")
        sig = a[a.p_bh < .05]
        ax.scatter(sig.hour, sig.difference, marker="*", s=36, c="#a62a55", zorder=5, label="BH q < 0.05")
        ax.axvline(0, c="black", lw=.7); ax.axhline(0, c="grey", lw=.9, ls="--")
        ax.set_ylabel(f"Difference ({UNITS[v]})")
        ax.set_xticks(range(-12, 13, 3)); ax.grid(alpha=.2)
    fig.supxlabel("Hours from CG reference (nearest 3-hour slot)", fontsize=plt.rcParams["axes.labelsize"])
    fig.tight_layout(h_pad=1.6)
    label_panels(fig, axs.flat, "abcd")
    fig.savefig(out / "figures/first_CG_compound_vs_nonHW_difference_event_cluster.png", dpi=1200, bbox_inches="tight")
    plt.close(fig)


def run(out: Path, force: bool = False) -> None:
    summary_path = out / "tables/confirmed_curve_summary_event_cluster.csv"
    changes_path = out / "tables/confirmed_adjacent_changes_event_cluster.csv"
    between_path = out / "tables/first_cg_compound_vs_nonHW_difference.csv"
    rng = np.random.default_rng(SEED)
    if not force and summary_path.exists() and changes_path.exists():
        # The 1000x cluster bootstrap is the expensive part; replotting alone
        # (e.g. to move a legend) should not redo it. Pass --force to redo it
        # (e.g. after changing N_BOOT or the clustering logic itself).
        print(f"Reusing existing {summary_path.name}/{changes_path.name} (pass --force to recompute)")
        summary = pd.read_csv(summary_path)
        changes = pd.read_csv(changes_path)
    else:
        summary, changes = compute(out)
        summary.to_csv(summary_path, index=False)
        changes.to_csv(changes_path, index=False)
    if not force and between_path.exists():
        print(f"Reusing existing {between_path.name} (pass --force to recompute)")
        between_group = pd.read_csv(between_path)
    else:
        between_group = compute_between_group(out, rng)
        between_group.to_csv(between_path, index=False)
    plot(out, summary, changes)
    plot_merged(out, summary, changes)
    plot_between_group(out, between_group)
    print(f"Wrote {summary_path}")
    print(f"Wrote {changes_path}")
    print(f"Wrote {between_path}")
    print(f"Wrote {out / 'figures/09_first_CG_compound_event_cluster.png'}")
    print(f"Wrote {out / 'figures/10_compound_vs_nonHW_CG_onsets_event_cluster.png'}")
    print(f"Wrote {out / 'figures/09_10_first_CG_merged_event_cluster.png'}")
    print(f"Wrote {out / 'figures/first_CG_compound_vs_nonHW_difference_event_cluster.png'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--force", action="store_true", help="Redo the cluster bootstrap even if the CSVs already exist.")
    args = parser.parse_args()
    run(args.out, force=args.force)
