#!/usr/bin/env python3
"""Resume the main analysis or robustness checks; --force archives results first."""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.config import RUN_DIR, SHARED_CLIMATOLOGY_DIR, SWEEP_DIR, METHOD
from common.paths import PROJECT_DIR, OUTPUTS_DIR
from common.layout import sensitivity_dir
from common.analysis_inputs import validate_run_inputs
from common.cache import metadata_path, validate_cache, register_climatology
from common.execution import archive_paths, command, recipe_signature, run_stage, stage_complete


def script(relative, *args):
    return [sys.executable, str(PROJECT_DIR / 'scripts' / relative), *map(str, args)]


def main_stages(run, climatology):
    def stage(name, program, args, products, cleanup):
        return (name, script('pipeline/' + program, *args),
                [run / p for p in products], [run / p for p in cleanup])
    out = ['--out', run]
    source = ['--source', run]
    raw_prefix = f'occurrence/tables/cg_hw_fe_logit_{METHOD}_2006-2025_m04-09_'
    fe_files = [raw_prefix + suffix for suffix in ['summary.csv', 'attrition.csv', 'year_effects.csv', 'metadata.json']]
    report_files = ['occurrence/tables/' + name for name in [
        'adjusted_effects_95CI.csv', 'unadjusted_probabilities.csv', 'sequential_attrition.csv',
        'joint_exclusion_diagnostics.csv', 'cell_month_sample_diagnostics.csv', 'original_model_summary.csv']]
    report_files += ['occurrence/data/margin_cluster_covariance.npz', 'occurrence/metadata.json', 'occurrence/README.md']
    monthly = [f'first_cg/data/first_cg_values_m{m:02d}.parquet' for m in range(3, 11)]
    env_monthly = [f'environment/data/{stem}_m{m:02d}.npz'
                   for stem in ['event_environment', 'spatial_environment'] for m in range(3, 11)]
    return [
        stage('rebuild', 'rebuild_compound_land.py', ['rebuild', *out, '--method', METHOD],
              ['checks/rebuild_complete.json', 'events/regional_events.csv', 'checks/land_rebuild_audit.csv', 'checks/unmasked_reproduction.csv'], []),
        stage('coverage', 'rebuild_compound_land.py', ['coverage', *out], ['checks/cg_coverage.csv'], ['checks/cg_coverage.csv']),
        stage('probability', 'rebuild_compound_land.py', ['probability', *out],
              ['occurrence/tables/occurrence_probability.csv', 'occurrence/tables/occurrence_by_year.csv'],
              ['occurrence/tables/occurrence_probability.csv', 'occurrence/tables/occurrence_by_year.csv']),
        stage('baseline', 'rebuild_compound_land.py', ['match', *out],
              ['matching/baseline/match_pairs.csv', 'matching/baseline/matching_summary.json'], ['matching/baseline']),
        stage('area_balance', 'check_land_area_year_balance.py', [*source, '--output', run/'matching/diagnostics'],
              ['matching/diagnostics/tables/events_with_peak_daily_land_area.csv', 'matching/diagnostics/tables/area_smd.csv', 'matching/diagnostics/tables/year_indicator_smd.csv'], ['matching/diagnostics']),
        stage('anchors', 'prepare_confirmed_land.py', [*out, '--method', METHOD],
              ['first_cg/data/confirmed_anchors.parquet', 'first_cg/confirmed_definition.json', 'first_cg/tables/confirmed_anchor_counts.csv'], ['first_cg']),
        stage('environment', 'analyze_land_environment.py', [*out, '--climatology-dir', climatology],
              ['environment/data/environment_complete.json', 'environment/tables/event_environment_anomaly.csv', *monthly, *env_monthly], ['environment', *monthly]),
        stage('intensity', 'fit_land_intensity.py', out,
              ['intensity/tables/quintile_diagnostics.json', 'intensity/tables/quintile_adjusted.csv', 'intensity/data/quintile_margin_coefficient_covariance.npy', 'intensity/data/intensity_sample.parquet'], ['intensity']),
        stage('fixed_effects', 'fit_cg_heatwave_fixed_effects.py', ['--output-dir', run/'occurrence/tables'], fe_files, fe_files),
        stage('margins', 'report_warmseason_fe_margins.py', source,
              ['occurrence/tables/adjusted_effects_95CI.csv', 'occurrence/data/margin_cluster_covariance.npz', 'occurrence/tables/unadjusted_probabilities.csv'], report_files),
        stage('comparison', 'compare_land_matching_ablation.py', [*source, '--output', run/'matching/comparison'],
              ['matching/comparison/area_only_pairs.csv', 'matching/comparison/retention_comparison.csv'], ['matching/comparison']),
        stage('final_matching', 'fit_area_matched_year_fe.py', [*source, '--output', run/'matching/final'],
              ['matching/final/tables/environment_pair_year_FE.csv', 'matching/final/data/regression_arrays.npz', 'matching/final/validation.json'], ['matching/final']),
    ]


def state_path(run):
    return run / 'checks/workflow.json'


def save_state(run, state, dry_run):
    if not dry_run:
        target = state_path(run); target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix('.tmp')
        temporary.write_text(json.dumps(state, indent=2) + '\n'); temporary.replace(target)


def associated_checks(run):
    return [sensitivity_dir(run) / name for name in
            ['cg_fraction_sensitivity', 'cg_window_sensitivity', 'pre_cg_environment', 'nonhw_control']]


def run_main(run, climatology, *, force=False, dry_run=False):
    stages = main_stages(run, climatology)
    # Validate existing caches before archiving any result, including force previews.
    for method in ['shared', METHOD]:
        if metadata_path(method).exists(): validate_cache(method)
    if (Path(climatology)/'metadata.json').exists(): register_climatology(climatology)
    state = json.loads(state_path(run).read_text()) if state_path(run).exists() else {}
    if not force:
        for name, cmd, _, _ in stages:
            if name in state and state[name]['recipe'] != recipe_signature(cmd):
                raise ValueError(f'{name}: analysis code/settings changed. Use --force to archive and rerun.')
        if (run/'checks/analysis_inputs.json').exists() and stage_complete(stages[0][2]):
            validate_run_inputs(run)
    # A failed rebuild invalidates the whole run; later failures invalidate all later stages.
    first_missing = next((i for i, (_, _, products, _) in enumerate(stages) if not stage_complete(products)), None)
    if force or (first_missing == 0 and run.exists()):
        paths = [run, *associated_checks(run)]
        if run.resolve() == (OUTPUTS_DIR/'main_analysis').resolve(): paths += [OUTPUTS_DIR/'manuscript/tables']
        archive_paths(paths, dry_run=dry_run)
        state = {}; first_missing = 0
    elif first_missing is not None:
        cleanup = [p for _, _, _, paths in stages[first_missing:] for p in paths]
        cleanup += [run/'occurrence/figures', run/'first_cg/figures', *associated_checks(run)]
        if run.resolve() == (OUTPUTS_DIR/'main_analysis').resolve(): cleanup += [OUTPUTS_DIR/'manuscript/tables']
        archive_paths(cleanup, dry_run=dry_run)
        for name, _, _, _ in stages[first_missing:]: state.pop(name, None)
    if first_missing is not None and 0 < first_missing <= next((i for i, stage in enumerate(stages) if stage[0] == 'anchors'), -1):
        # prepare_confirmed_land updates catalogue centres in place. Restore the
        # original area centres when resuming an earlier stage, matching a fresh run.
        catalogue = run/'events/regional_events.csv'
        if catalogue.exists() and not dry_run:
            import pandas as pd
            events = pd.read_csv(catalogue)
            previous = ['area_center_latitude_previous', 'area_center_longitude_previous']
            if all(column in events for column in previous):
                for column, old in zip(['center_latitude', 'center_longitude'], previous):
                    events[column] = events[old]
                events.to_csv(catalogue, index=False)
    for i, (name, cmd, products, _) in enumerate(stages):
        run_stage(name, cmd, products, state, dry_run=dry_run,
                  rerun=first_missing is not None and i >= first_missing)
        save_state(run, state, dry_run)


def run_sensitivity(run, climatology, *, force=False, dry_run=False):
    validate_run_inputs(run)
    if not dry_run or (Path(climatology)/'metadata.json').exists():
        register_climatology(climatology)
    directory = sensitivity_dir(run)
    if force:
        archive_paths([SWEEP_DIR, *associated_checks(run), *([OUTPUTS_DIR/'manuscript/tables']
                       if run.resolve() == (OUTPUTS_DIR/'main_analysis').resolve() else [])], dry_run=dry_run)
    state = json.loads(state_path(run).read_text()) if state_path(run).exists() else {}
    pre = directory/'pre_cg_environment'
    frac = directory/'cg_fraction_sensitivity'
    window = directory/'cg_window_sensitivity'
    stages = [
        ('pre_cg', script('pipeline/compute_pre_cg_environment.py', '--out', run, '--climatology-dir', climatology),
         [pre/'pre_cg_environment_effects.csv', pre/'truncated_event_environment.csv', pre/'event_windows.csv', pre/'window_summary.csv']),
        ('pre_cg_figure', script('figures/plot_pre_cg_environment.py', '--out', run), [pre/'pre_cg_environment.png']),
        ('cg_fraction', script('sensitivity/run_land_cg_fraction_sensitivity.py', '--source', run, '--output', frac),
         [frac/'environment_comparison.csv', frac/'provenance.json', frac/'event_counts.csv']),
        ('cg_window', script('sensitivity/run_cg_window_sensitivity.py'),
         [window/'environment_pair_year_FE_by_cg_window.csv', window/'summary.json', window/'event_classification.csv']),
    ]
    if force:
        for name, _, _ in stages: state.pop(name, None)
    else:
        for name, cmd, _ in stages:
            if name in state and state[name]['recipe'] != recipe_signature(cmd):
                raise ValueError(f'{name}: code/settings changed; use --force to archive and rerun.')
    pre_changed = False
    for name, cmd, products in stages:
        if not stage_complete(products):
            partial = {'pre_cg': pre, 'pre_cg_figure': pre/'pre_cg_environment.png',
                       'cg_fraction': frac, 'cg_window': window}[name]
            archive_paths([partial], dry_run=dry_run)
        changed = run_stage(name, cmd, products, state, dry_run=dry_run,
                            rerun=force or (name == 'pre_cg_figure' and pre_changed))
        if name == 'pre_cg': pre_changed = changed
        save_state(run, state, dry_run)
    # These two orchestrators already resume every individual threshold stage.
    for program in ['run_absolute_threshold_sweep.py', 'run_environment_sensitivity_sweep.py']:
        cmd = script('sensitivity/' + program)
        if dry_run: cmd += ['--dry-run']
        command(cmd, dry_run=dry_run)
    for program in ['plot_threshold_sensitivity.py', 'plot_environment_threshold_sensitivity.py']:
        command(script('figures/' + program), dry_run=dry_run)
    command(script('figures/check_figures.py', '--group', 'sensitivity'), dry_run=dry_run)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['main', 'sensitivity'])
    parser.add_argument('--run', type=Path, default=RUN_DIR)
    parser.add_argument('--climatology-dir', type=Path, default=Path(os.environ.get('CLIM', SHARED_CLIMATOLOGY_DIR)))
    parser.add_argument('--force', action='store_true', help='Archive existing analysis outputs and recompute; reuse compatible preprocessing.')
    parser.add_argument('--dry-run', action='store_true', help='Show archive actions and commands without writing files.')
    args = parser.parse_args()
    # Child scripts that use RUN_DIR must select the same run as --run.
    run = args.run.absolute()
    os.environ['RUN'] = str(run)
    function = run_main if args.stage == 'main' else run_sensitivity
    function(run, args.climatology_dir.absolute(), force=args.force, dry_run=args.dry_run)


if __name__ == '__main__':
    main()
