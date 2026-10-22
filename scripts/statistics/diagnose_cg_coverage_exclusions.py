#!/usr/bin/env python3
"""Why does the complete-CG-coverage requirement exclude 46% of land regional
heatwave events?

Regional events are delineated on the full ERA5 download domain, which reaches
66.5 E, while the EUCLID CG grid stops at 32.94 E. Any event whose footprint
extends east of the CG grid therefore has coverage below one. This script
separates that cause from the remaining edge effects.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

CG_LON_MAX = 32.94
CG_LON_MIN = -11.0
CG_LAT_MIN, CG_LAT_MAX = 35.0, 71.99


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    e = pd.read_csv(args.out / "events/regional_events.csv")
    complete = e.cg_coverage_fraction == 1
    print(f"total land regional events       : {len(e)}")
    print(f"  complete CG coverage           : {int(complete.sum())}")
    print(f"  incomplete                     : {int((~complete).sum())}")

    excluded = e[~complete]
    beyond_east = excluded.max_longitude > CG_LON_MAX
    beyond_west = excluded.min_longitude < CG_LON_MIN
    beyond_lat = (excluded.max_latitude > CG_LAT_MAX) | (excluded.min_latitude < CG_LAT_MIN)
    outside = beyond_east | beyond_west | beyond_lat

    print(f"\nAmong the {len(excluded)} excluded events:")
    print(f"  footprint reaches east of {CG_LON_MAX} E : {int(beyond_east.sum())}"
          f"  ({beyond_east.mean()*100:.1f}%)")
    print(f"  reaches west of {CG_LON_MIN} E           : {int(beyond_west.sum())}")
    print(f"  reaches beyond the CG latitude range     : {int(beyond_lat.sum())}")
    print(f"  entirely inside the CG bounding box      : {int((~outside).sum())}"
          f"  ({(~outside).mean()*100:.1f}%)")

    inside = excluded[~outside]
    if len(inside):
        print("\n  For those entirely inside the box, coverage fraction quantiles:")
        for q in (5, 25, 50, 75, 95):
            print(f"    p{q:<3d} = {np.percentile(inside.cg_coverage_fraction, q):.3f}")
        print(f"    median footprint cells = {inside.n_gridcells.median():.0f}")

    print("\nCentre longitude by group (quantiles):")
    for label, group in [("complete", e[complete]), ("incomplete", excluded)]:
        qs = np.percentile(group.center_longitude, [5, 25, 50, 75, 95])
        print(f"  {label:<10s} " + ", ".join(f"p{q}={v:.1f}" for q, v in zip((5, 25, 50, 75, 95), qs)))

    east_events = e[e.max_longitude > CG_LON_MAX]
    print(f"\nEvents whose footprint reaches east of {CG_LON_MAX} E: {len(east_events)}"
          f" ({len(east_events)/len(e)*100:.1f}% of all events)")
    print(f"  of which excluded: {int((east_events.cg_coverage_fraction < 1).sum())}")


if __name__ == "__main__":
    main()
