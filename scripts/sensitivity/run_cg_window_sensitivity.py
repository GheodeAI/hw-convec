#!/usr/bin/env python3
"""Sensitivity of the matched environmental contrast to the CG window used to
classify compound events.

Main definition: a regional event is compound when CG occurs in at least 10%
of its land grid cells during the event or in the 24 h after it ends. Here only
CG during the event counts. Events are reclassified, rematched with the main
design (exact start month; five covariates; Mahalanobis caliper 0.75; global
greedy 1:1) and the pair + start-year FE differences refitted. Standardised
differences use the main analysis's pre-match pooled SD (the Fig 7
denominator) for both definitions, so the two are directly comparable.

The main classification and Fig 7 results are recomputed by the same code and
checked against the published tables before the sensitivity is reported.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage, sparse, stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.layout import sensitivity_dir
from common.analysis_inputs import input_path
from common.config import RUN_DIR, YEARS
from common.environment import ORDER
from common.matching import COV, match
from common.stats import bh, fit_difference

OUT = sensitivity_dir(RUN_DIR) / 'cg_window_sensitivity'
YEAR_DUMMIES = np.arange(2007, 2026)


def classify(events: pd.DataFrame) -> pd.DataFrame:
    """Compound status with and without the 24 h after the event, on the event's
    land footprint, with the main rule (CG cells >= ceil(10% of cells))."""
    rows = []
    for year in YEARS:
        sub = events[events.year == year]
        if sub.empty:
            continue
        with np.load(input_path(RUN_DIR, f'labels_{year}.npz')) as d:
            lab, offset = d['labels'], int(d['offset'])
        nx = lab.shape[2]
        hour = sparse.load_npz(input_path(RUN_DIR, f'cg_hourly_era5_{year}.npz')).tocsc()
        objects = ndimage.find_objects(lab)
        for event_id in sub.index:
            label = event_id - offset
            s = objects[label - 1]
            _, r, c = np.nonzero(lab[s] == label)
            foot = np.unique((r + s[1].start) * nx + c + s[2].start)
            start, last = s[0].start, s[0].stop - 1
            required = max(1, int(np.ceil(.1 * len(foot))))
            row = dict(event_id=event_id, n_cells=len(foot), required_cells=required)
            for name, end_day in [('with_post24h', last + 2), ('during_only', last + 1)]:
                hits = hour[start * 24:min(end_day * 24, hour.shape[0]), foot].tocoo()
                n_hit = len(np.unique(hits.col))
                row[f'cg_cells_{name}'] = n_hit
                row[f'compound_{name}'] = n_hit >= required
            rows.append(row)
        print('classified', year, flush=True)
    return pd.DataFrame(rows).set_index('event_id')


def contrast(e: pd.DataFrame, env: pd.DataFrame, scale: np.ndarray, label: str):
    pairs, _, _ = match(e, 'exact_month', base_cov=COV)
    a, b = e.loc[pairs.compound_event_id], e.loc[pairs.control_event_id]
    assert np.array_equal(a.start_month, b.start_month)
    c, n = e[e.has_lightning], e[~e.has_lightning]
    balance = {v: float((a[v].mean() - b[v].mean()) / np.sqrt((c[v].var() + n[v].var()) / 2)) for v in COV}
    y = env.loc[a.index, ORDER].to_numpy() - env.loc[b.index, ORDER].to_numpy()
    z = ((a.year.to_numpy()[:, None] == YEAR_DUMMIES).astype(float)
         - (b.year.to_numpy()[:, None] == YEAR_DUMMIES).astype(float))
    beta, covariance, _, _ = fit_difference(np.column_stack([np.ones(len(pairs)), z]), y)
    se = np.sqrt(covariance[:, 0, 0])
    g = len(pairs)
    crit = stats.t.ppf(.975, g - 1)
    tab = pd.DataFrame(dict(scheme=label, variable=ORDER, n_pairs=g, difference=beta[0], cluster_se=se,
                            ci95_lower=beta[0] - crit * se, ci95_upper=beta[0] + crit * se,
                            p=2 * stats.t.sf(abs(beta[0] / se), g - 1)))
    tab['p_bh'] = bh(tab.p.to_numpy())
    for col in ['difference', 'ci95_lower', 'ci95_upper']:
        tab['standardized_' + col] = tab[col] / scale
    counts = dict(compound_available=len(c), noncompound_available=len(n), matched_pairs=g,
                  compound_retention=g / len(c), max_abs_smd_after=max(abs(v) for v in balance.values()))
    return tab, pairs, counts, balance


def main() -> None:
    OUT.mkdir(exist_ok=True)
    events = pd.read_csv(RUN_DIR / 'matching/diagnostics/tables/events_with_peak_daily_land_area.csv')
    events = events[events.start_month.between(4, 9) & (events.cg_coverage_fraction == 1)].set_index('event_id')
    env = pd.read_csv(RUN_DIR / 'environment/tables/event_environment_anomaly.csv').set_index('event_id')
    assert np.isfinite(env.loc[events.index, ORDER]).all().all()

    cls = classify(events)
    assert (cls.compound_with_post24h == events.has_lightning).all(), 'main classification not reproduced'
    assert (cls.n_cells == events.n_gridcells).all()
    cls.to_csv(OUT / 'event_classification.csv')
    cross = pd.crosstab(cls.compound_with_post24h.rename('main (with 24 h after)'),
                        cls.compound_during_only.rename('during event only'))
    print('\nClassification of eligible events:\n' + cross.to_string())

    main_c, main_n = events[events.has_lightning], events[~events.has_lightning]
    scale = np.sqrt((env.loc[main_c.index, ORDER].var() + env.loc[main_n.index, ORDER].var()).to_numpy() / 2)

    results, summary = [], {}
    for label, column in [('main: during + 24 h after', 'compound_with_post24h'),
                          ('during event only', 'compound_during_only')]:
        e = events.copy()
        e['has_lightning'] = cls[column]
        tab, pairs, counts, balance = contrast(e, env, scale, label)
        pairs.to_csv(OUT / f"match_pairs_{column}.csv", index=False)
        results.append(tab)
        summary[label] = dict(counts, balance_smd_after=balance)
        print(f"\n{label}: {counts['compound_available']} compound / {counts['noncompound_available']} "
              f"non-compound, {counts['matched_pairs']} pairs, max |SMD| after {counts['max_abs_smd_after']:.3f}")

    reference = pd.read_csv(RUN_DIR / 'matching/final/tables/environment_pair_year_FE.csv').set_index('variable').loc[ORDER]
    main = results[0].set_index('variable')
    assert np.allclose(main[['difference', 'cluster_se', 'p_bh']], reference[['difference', 'cluster_se', 'p_bh']],
                       rtol=1e-8, atol=1e-12), 'main Fig 7 results not reproduced'
    assert np.allclose(main.standardized_difference, reference.standardized_difference, rtol=1e-8)
    print('\nMain classification, pairs and Fig 7 estimates reproduced exactly.')

    table = pd.concat(results, ignore_index=True)
    table.to_csv(OUT / 'environment_pair_year_FE_by_cg_window.csv', index=False)
    a, b = results[0].set_index('variable'), results[1].set_index('variable')
    side = pd.DataFrame({
        'main_std_diff': a.standardized_difference,
        'main_ci': [f'{lo:.3f} to {hi:.3f}' for lo, hi in zip(a.standardized_ci95_lower, a.standardized_ci95_upper)],
        'main_p_bh': a.p_bh,
        'during_only_std_diff': b.standardized_difference,
        'during_only_ci': [f'{lo:.3f} to {hi:.3f}' for lo, hi in zip(b.standardized_ci95_lower, b.standardized_ci95_upper)],
        'during_only_p_bh': b.p_bh,
        'same_sign': np.sign(a.difference) == np.sign(b.difference),
        'same_significance': (a.p_bh < .05) == (b.p_bh < .05),
    })
    side.to_csv(OUT / 'comparison.csv')
    (OUT / 'summary.json').write_text(json.dumps(dict(
        reclassification={f'{k[0]}->{k[1]}': int(v) for k, v in cross.stack().items()}, **summary,
        standardisation='main pre-match pooled SD of whole-event means (Fig 7 denominator), both schemes'),
        indent=2, default=float) + '\n')
    pd.set_option('display.width', 200)
    print('\nStandardised compound - non-compound differences (pair + year FE, BH across six variables):')
    print(side.round(3).to_string())
    print(f'\nWrote {OUT}')


if __name__ == '__main__':
    main()
