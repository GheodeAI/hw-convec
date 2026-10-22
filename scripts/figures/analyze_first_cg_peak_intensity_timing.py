#!/usr/bin/env python3
"""Timing of the first CG relative to heatwave onset and maximum intensity.

"The heatwave" here is the regional event, so this is computed at the
regional-event level only (not per grid cell -- a grid cell's own local
intensity peak isn't a meaningful analog for "the heatwave's mature phase").
For every qualifying regional event, finds the day of peak wet-bulb excess
(mean Tw_daily_max - threshold across the event's whole footprint that day)
and reports the lag from that day to the event's first CG: negative = CG
occurs before the event reaches its most intense day (still intensifying),
positive = after the peak (declining, "breaking down").
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
from scipy import ndimage
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.analysis_inputs import input_path
from common.figures import figure_path, save_figure
from common.config import YEARS, HW
from common.style import BLUE, ORANGE, FULL_WIDTH, apply_style

apply_style()

GREY = "#7f7f7f"


def event_peak_days(out: Path) -> dict[int, tuple[int, int]]:
    """event_id -> (year, peak_day_index): the day (day-of-year index into
    that year's heatwave file) with the highest mean excess across the
    event's whole footprint."""
    event_peak: dict[int, tuple[int, int]] = {}
    for year in YEARS:
        with np.load(input_path(out, f"labels_{year}.npz")) as d:
            labels = d["labels"]
            offset = int(d["offset"])
        with xr.open_dataset(HW / f"wetbulb_heatwave_{year}.nc") as ds:
            excess = (ds.tw_daily_max - ds.heatwave_threshold).values
        for label_id, slices in enumerate(ndimage.find_objects(labels), 1):
            if slices is None:
                continue
            t, r, c = np.nonzero(labels[slices] == label_id)
            t = t + slices[0].start
            r = r + slices[1].start
            c = c + slices[2].start
            day_mean = pd.Series(excess[t, r, c]).groupby(t).mean()
            event_peak[offset + label_id] = (year, int(day_mean.idxmax()))
        print(f"Peak days computed for {year} ({len(event_peak)} events so far)", flush=True)
    return event_peak


def peak_timestamp(year: int, day_index: int) -> pd.Timestamp:
    # Midday, not midnight. The heatwave fields are daily, so the peak is only
    # resolved to a day; anchoring it at the day's centre makes the +-12 h
    # "near peak" window coincide exactly with the peak day. Anchored at 00
    # UTC instead, that window ran from noon of the previous day to noon of the
    # peak day, so a first CG on the peak day's afternoon counted as "after the
    # peak", which inflated that share.
    return pd.Timestamp(year, 1, 1) + pd.Timedelta(days=day_index) + pd.Timedelta(hours=12)


def run(out: Path) -> None:
    event_peak = event_peak_days(out)

    e = pd.read_csv(out / "events/regional_events.csv", parse_dates=["first_cg_time", "start_time"])
    e = e[e.has_lightning & (e.cg_coverage_fraction == 1) & e.first_cg_time.dt.month.between(4, 9)].copy()
    e["peak_time"] = [peak_timestamp(*event_peak[eid]) if eid in event_peak else pd.NaT for eid in e.event_id]
    e = e.dropna(subset=["peak_time"]).copy()
    e["lag_hours_from_peak"] = (e.first_cg_time - e.peak_time).dt.total_seconds() / 3600
    # Both first-CG timing and peak-intensity timing expressed as a fraction of
    # the event's own duration since onset -- 0 = onset, 1 = the end of the
    # post-heatwave 24h window during which a compound CG can still count --
    # so the two are on a genuine, shared [0,1) scale (first CG can land in
    # that extra day; peak intensity never can, since it is defined only on
    # the event's own heatwave days).
    window_days = e.duration + 1
    e["cg_phase"] = (e.first_cg_time - e.start_time).dt.total_seconds() / 3600 / 24 / window_days
    e["peak_phase"] = (e.peak_time - e.start_time).dt.total_seconds() / 3600 / 24 / window_days

    lag = e.lag_hours_from_peak.to_numpy()
    rows = []
    for name, n in [
        ("before peak (>12h)", int((lag < -12).sum())),
        ("near peak (±12h)", int(((lag >= -12) & (lag <= 12)).sum())),
        ("after peak (>12h)", int((lag > 12).sum())),
    ]:
        rows.append(dict(phase=name, n=n, total=len(e), percentage=n / len(e) * 100))
    tab = pd.DataFrame(rows)

    # Statistics always use the true, unjittered/unclipped values.
    later = (e.cg_phase > e.peak_phase).mean() * 100
    earlier = (e.cg_phase < e.peak_phase).mean() * 100
    n_over = int((e.cg_phase > 1).sum())
    if n_over:
        print(f"Note: {n_over} events have cg_phase slightly > 1 even under the +1-day window; clipped for display.", flush=True)

    # Plotted at their true positions. peak_phase is quantised -- a 3-day event
    # only allows 0/4, 1/4, 2/4, 3/4 once the +1-day window is included -- so
    # the points form vertical stripes, but that is real structure (event
    # durations are small integers), and the earlier jitter was the same width
    # as the +-12 h "near peak" band, smearing exactly the distinction this
    # figure exists to show.
    peak_phase_plot = e.peak_phase.clip(0, 1)
    cg_phase_display = e.cg_phase.clip(upper=1.0)

    lims = (0, 1)
    fig = plt.figure(figsize=(FULL_WIDTH, FULL_WIDTH))
    gs = fig.add_gridspec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4], wspace=.04, hspace=.04)
    ax_main = fig.add_subplot(gs[1, 0])
    ax_top = fig.add_subplot(gs[0, 0], sharex=ax_main)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)

    # Color by position in the *displayed* (jittered) coordinates, not the true
    # ones -- otherwise a point can be colored "before" from its true values
    # but jitter nudges its plotted x far enough to visually land above the
    # diagonal, which looks like a contradiction. The text annotations below
    # still report percentages from the true, unjittered comparison.
    # Three-way split on the true lag in hours, matching the reported table.
    # A strict before/after comparison left the events whose first CG falls
    # exactly at the peak day's 00 UTC in neither class; a +-12 h "near peak"
    # band removes those ties and the three shares sum to 100%.
    px = peak_phase_plot.to_numpy()
    py = cg_phase_display.to_numpy()
    classes = [
        (lag < -12, BLUE, "o", ""),
        (np.abs(lag) <= 12, GREY, "s", "..."),
        (lag > 12, ORANGE, "^", "///"),
    ]
    for sel, color, marker, _ in classes:
        ax_main.scatter(px[sel], py[sel], s=8, alpha=.35, color=color, marker=marker, linewidth=0)
    ax_main.plot(lims, lims, "k--", lw=1, label="First CG at same phase as peak intensity")
    ax_main.set(xlabel="Peak intensity: fraction of event duration since onset",
                ylabel="First CG: fraction of event duration since onset", xlim=lims, ylim=lims)
    ax_main.grid(alpha=.2)

    ax_top.hist(px, bins=40, range=lims, color="#888888", alpha=.7)
    ax_right.hist([py[sel] for sel, *_ in classes], bins=40, range=lims,
                  color=[c for _, c, _, _ in classes], hatch=[h for *_, h in classes],
                  alpha=.8, stacked=True, orientation="horizontal")
    for ax in (ax_top, ax_right):
        ax.tick_params(labelbottom=False, labelleft=False, length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
    ax_top.set_ylabel("N events")
    ax_right.set_xlabel("N events")

    save_figure(fig, 'first_CG_onset_and_peak_intensity_joint', run_dir=out, dpi=1200, bbox_inches="tight")
    plt.close(fig)

    e.to_csv(out / "first_cg/tables/confirmed_first_cg_peak_intensity_timing_events.csv", index=False)
    tab.to_csv(out / "first_cg/tables/confirmed_first_cg_peak_intensity_timing.csv", index=False)
    print(tab.to_string(index=False), flush=True)
    tied = 100 - later - earlier
    print(f"CG after peak: {later:.1f}%, CG before peak: {earlier:.1f}%, "
          f"exactly at the peak anchor (peak day 12 UTC): {tied:.1f}%", flush=True)
    print(f"Wrote {figure_path('first_CG_onset_and_peak_intensity_joint', out)}")
    print(f"Wrote {out / 'first_cg/tables/confirmed_first_cg_peak_intensity_timing.csv'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(args.out)
