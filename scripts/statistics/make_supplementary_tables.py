#!/usr/bin/env python3
"""Write the numbers of the two supplementary tables as CSV:

  table_S1_retention.csv            stratum and record retention in the two fixed-effects models
  table_S2_compound_definition.csv  matched environmental differences under four compound-event rules

Every number is read from the outputs of fit_cg_heatwave_fixed_effects.py,
fit_land_intensity.py, run_land_cg_fraction_sensitivity.py and
run_cg_window_sensitivity.py; the breakdown of heatwave days left out of the
quintile sample is recomputed from the daily grid-cell arrays and checked
against both models' sample sizes.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.layout import sensitivity_dir
from common.analysis_inputs import input_path
from common.config import RUN_DIR, YEARS
from common.environment import ORDER
from common.paths import OUTPUTS_DIR

TARGET = OUTPUTS_DIR / 'manuscript/tables'
NAMES = {'q850': 'q850', 'tcwv': 'TCWV', 't2m-d2m': 'DPD2m', 'qv850': 'qv850', 'qu850': 'qu850', 'z500': 'Z500'}


def quintile_exclusions() -> tuple[int, int, int]:
    """April-September heatwave grid-cell days, and those left out of the
    quintile sample: non-positive wet-bulb excess (gap days), and positive
    excess whose assigned ERA5 cell is not in a land regional event."""
    total = gap = off_land = 0
    for year in YEARS:
        with np.load(input_path(RUN_DIR, f'daily_{year}.npz')) as d:
            month = pd.DatetimeIndex(d['dates']).month
            warm = (month >= 4) & (month <= 9)
            hw = d['hw'][warm].astype(bool)
            positive = d['excess'][warm] > 0
            total += int(hw.sum())
            gap += int((hw & ~positive).sum())
            off_land += int((hw & positive & (d['event'][warm] == 0)).sum())
    return total, gap, off_land


def retention_table() -> pd.DataFrame:
    a = pd.read_csv(RUN_DIR / 'occurrence/tables/cg_hw_fe_logit_90p20_2006-2025_m04-09_attrition.csv').set_index('stage')
    q = json.loads((RUN_DIR / 'intensity/tables/quintile_diagnostics.json').read_text())
    initial, exposure, outcome = (a.loc[s] for s in
                                  ['initial_land_cell_month_sample', 'exposure_informative', 'outcome_informative_final'])
    hw_days = int(initial.heatwave_records_before)
    total, gap, off_land = quintile_exclusions()
    assert total == hw_days, f'daily arrays give {total:,} heatwave days, FE model {hw_days:,}'
    assert total - gap - off_land == q['original_n'], 'quintile sample not reproduced'

    all_days = 'All-days model: heatwave versus non-heatwave days'
    quintile = 'Heatwave-only model: intensity quintiles'
    rows = [
        (all_days, 'Original sample', initial.strata_before, initial.records_before),
        (all_days, 'Removed, no variation in heatwave status', exposure.strata_dropped_at_stage, exposure.records_dropped_at_stage),
        (all_days, 'Removed, no variation in CG outcome', outcome.strata_dropped_at_stage, outcome.records_dropped_at_stage),
        (all_days, 'Retained estimation sample', outcome.strata_retained, outcome.records_retained),
        (quintile, 'Heatwave grid-cell days in the domain', None, hw_days),
        (quintile, 'Excluded before stratum filter: non-positive wet-bulb excess (gap days)', None, gap),
        (quintile, 'Excluded before stratum filter: assigned ERA5 cell not in a land regional event', None, off_land),
        (quintile, 'Original sample', q['original_strata'], q['original_n']),
        (quintile, 'Removed, no variation in CG outcome', q['original_strata'] - q['retained_strata'], q['dropped_n']),
        (quintile, 'Retained estimation sample', q['retained_strata'], q['retained_n']),
    ]
    table = pd.DataFrame(rows, columns=['model', 'stage', 'strata', 'grid_cell_days'])
    return table.astype({'strata': 'Int64', 'grid_cell_days': 'int64'})


def compound_definition_table() -> pd.DataFrame:
    fractions = pd.read_csv(sensitivity_dir(RUN_DIR) / 'cg_fraction_sensitivity/environment_comparison.csv')
    window = pd.read_csv(sensitivity_dir(RUN_DIR) / 'cg_window_sensitivity/environment_pair_year_FE_by_cg_window.csv')
    rules = [('Any CG versus no CG', 'event + 24 h after', fractions[fractions.threshold == 0]),
             ('>=10% versus <10%', 'event + 24 h after', fractions[fractions.threshold == .1]),
             ('>=20% versus <20%', 'event + 24 h after', fractions[fractions.threshold == .2]),
             ('>=10% versus <10%', 'event only', window[window.scheme == 'during event only'])]
    rows = []
    for rule, cg_window, tab in rules:
        tab = tab.set_index('variable').loc[ORDER]
        for variable, r in tab.iterrows():
            rows.append(dict(classification_rule=rule, cg_window=cg_window, matched_pairs=int(r.n_pairs),
                             variable=NAMES[variable], difference=r.difference,
                             direction='positive' if r.difference > 0 else 'negative', p_bh=r.p_bh))
    return pd.DataFrame(rows)


def main() -> None:
    TARGET.mkdir(parents=True, exist_ok=True)
    pd.set_option('display.width', 200)
    for name, table in [('table_S1_retention.csv', retention_table()),
                        ('table_S2_compound_definition.csv', compound_definition_table())]:
        table.to_csv(TARGET / name, index=False)
        print(f'\n{name}\n{table.to_string(index=False)}')
    print(f'\nWrote {TARGET}')


if __name__ == '__main__':
    main()
