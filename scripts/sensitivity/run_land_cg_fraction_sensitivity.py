"""Land CG-fraction sensitivity using the primary area-match/pair+year FE method."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.config import RUN_DIR
from common.environment import ORDER
from common.matching import COV, match
from common.stats import bh, fit_difference

DEFAULT_SOURCE = RUN_DIR
LABELS = ['q850', 'TCWV', 'DPD2m', 'qv850', 'qu850', 'Z500']


def run(source, output):
    output.mkdir(parents=True, exist_ok=True)
    paths = [source / 'area_year_diagnostics_v1/events_with_peak_daily_land_area.csv',
             source / 'tables/event_environment_anomaly.csv']
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    events = pd.read_csv(paths[0]).set_index('event_id', verify_integrity=True)
    events = events[events.start_month.between(4, 9) & (events.cg_coverage_fraction == 1)].copy()
    env = pd.read_csv(paths[1]).set_index('event_id', verify_integrity=True)
    assert np.isfinite(env.loc[events.index, ORDER]).all().all()
    fraction = events.lightning_event_gridcell_fraction
    assert fraction.between(0, 1).all()
    all_results, compact, counts, balances = [], [], [], []
    for threshold, label in [(0, 'Any CG versus no CG'), (.1, '≥10% versus <10%'), (.2, '≥20% versus <20%')]:
        e = events.copy()
        e['has_lightning'] = fraction.gt(0) if threshold == 0 else fraction.ge(threshold)
        pairs, cov, _ = match(e, 'exact_month', base_cov=COV)
        pairs.to_csv(output / f'match_pairs_{int(threshold*100):02d}pct.csv', index=False)
        a, b = e.loc[pairs.compound_event_id], e.loc[pairs.control_event_id]
        assert np.array_equal(a.start_month, b.start_month)
        c, n = e[e.has_lightning], e[~e.has_lightning]
        for v in cov:
            sd = np.sqrt((c[v].var() + n[v].var()) / 2)
            balances.append(dict(threshold=threshold, variable=v,
                                 smd_before=(c[v].mean()-n[v].mean())/sd,
                                 smd_after=(a[v].mean()-b[v].mean())/sd))
        assert max(abs(r['smd_after']) for r in balances[-len(cov):]) < .1
        y = env.loc[a.index, ORDER].to_numpy() - env.loc[b.index, ORDER].to_numpy()
        years = np.arange(2007, 2026)
        z = (a.year.to_numpy()[:, None] == years).astype(float) - (b.year.to_numpy()[:, None] == years).astype(float)
        design = np.column_stack([np.ones(len(pairs)), z])
        beta, covariance, _, correction = fit_difference(design, y)
        se = np.sqrt(covariance[:, 0, 0])
        critical = stats.t.ppf(.975, len(pairs)-1)
        result = pd.DataFrame(dict(variable=ORDER, difference=beta[0], cluster_se=se,
                                   ci95_lower=beta[0]-critical*se, ci95_upper=beta[0]+critical*se,
                                   p=2*stats.t.sf(abs(beta[0]/se), len(pairs)-1)))
        result['p_bh'] = bh(result.p)
        result['threshold'] = threshold
        result['n_pairs'] = len(pairs)
        result['cr1_correction'] = correction
        # Reproduce the existing primary classification whenever it matches.
        if np.array_equal(e.has_lightning, events.has_lightning):
            reference = pd.read_csv(source / 'area_match_pair_year_FE_v1/environment_pair_year_FE.csv').set_index('variable').loc[ORDER]
            assert np.allclose(result[['difference', 'cluster_se', 'p_bh']], reference[['difference', 'cluster_se', 'p_bh']], rtol=1e-8, atol=1e-12)
            print(f'{label}: reproduced existing primary results', flush=True)
        all_results.append(result)
        compact.append([label] + [f"{'↑' if r.difference > 0 else '↓' if r.difference < 0 else '='} ({r.p_bh:.3f})" for r in result.itertuples()])
        counts.append(dict(threshold=threshold, classification=label, compound_available=len(c), noncompound_available=len(n), matched_pairs=len(pairs)))
        print(label, len(c), len(n), len(pairs), flush=True)
    pd.concat(all_results, ignore_index=True).to_csv(output / 'environment_comparison.csv', index=False)
    pd.DataFrame(counts).to_csv(output / 'event_counts.csv', index=False)
    pd.DataFrame(balances).to_csv(output / 'matching_balance.csv', index=False)
    pd.DataFrame(compact, columns=['CG-fraction classification']+LABELS).to_csv(output / 'compact_table.csv', index=False)
    lines = ['| CG-fraction classification | ' + ' | '.join(LABELS) + ' |', '| ' + ' | '.join(['---']*7) + ' |']
    lines += ['| ' + ' | '.join(row) + ' |' for row in compact]
    notes = ('Land-only regional events, 90p20, 2006–2025, April–September start month, complete CG coverage. '
             'CG fraction is the fraction of event land grid cells with CG. Each threshold is independently rematched: '
             'exact start month; latitude, longitude, duration, intensity and log peak daily land area; '
             'Mahalanobis caliper 0.75; global distance-order greedy 1:1 matching without replacement. '
             'Arrows show compound minus control effects adjusted for pair and start-year fixed effects. '
             'Parentheses contain BH-adjusted p values across six variables separately within each threshold. '
             'CR1 standard errors cluster by pair, counting pair fixed effects in the finite-sample correction; t(G−1) inference. '
             'Environment anomalies reuse the land event footprint and calendar-day/3-hour climatology.\n')
    (output / 'compact_table.md').write_text('\n'.join(lines)+'\n\n'+notes)
    assert all(hashlib.sha256(p.read_bytes()).hexdigest() == hashes[str(p)] for p in paths)
    (output / 'provenance.json').write_text(json.dumps(dict(inputs_sha256=hashes, method=notes), indent=2)+'\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--output', type=Path, default=DEFAULT_SOURCE / 'cg_fraction_sensitivity_v1')
    args = parser.parse_args()
    run(args.source, args.output)
