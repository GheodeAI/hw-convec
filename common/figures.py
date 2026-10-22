"""Descriptive figure names and destinations by analysis subject."""
from pathlib import Path

from common.config import RUN_DIR, SWEEP_FIGURES_DIR, WETBULB_COMPARISON_DIR
from common.paths import CG_DIR, HEATWAVE_DIR
from common.layout import sensitivity_dir

# Paper labels are metadata only; they are never included in the filenames.
FIGURES = {
    'Fig01': 'occurrence_maps',
    'Fig02': 'probability',
    'Fig03': 'intensity_FE',
    'Fig04': 'environment_matching',
    'Fig05': 'first_CG_onset_and_peak_intensity_joint',
    'Fig06': 'first_CG_merged_event_cluster',
    'Fig07': 'first_CG_compound_vs_nonHW_difference_event_cluster',
    'FigS01': 'wetbulb_heatwave_historical_spatial',
    'FigS02': 'wetbulb_heatwave_cg_domain_annual_timeseries',
    'FigS03': 'euclid_cg_mean_days_intensity',
    'FigS04': 'euclid_cg_annual_timeseries',
    'FigS05': 'cg_compound_event_location_density',
    'FigS06': 'intensity_enhancement',
    'FigS07': 'occurrence_ratio_threshold_sensitivity',
    'FigS08': 'intensity_effect_threshold_sensitivity',
    'FigS09': 'disagreement_vs_elevation',
    'FigS10': 'environment_means',
    'FigS11': 'event_centered_environment',
    'FigS12': 'descriptive_balance',
    'FigS13': 'match_balance',
    'FigS14': 'environment_threshold_sensitivity',
    'FigS15': 'pre_cg_environment',
    'FigS16': 'gridcell_first_cg_lifecycle',
}

# Supporting figures live with their supporting data rather than the main run.
SUPPORTING_FIGURES = {
    'wetbulb_heatwave_historical_spatial': HEATWAVE_DIR,
    'wetbulb_heatwave_cg_domain_annual_timeseries': HEATWAVE_DIR,
    'euclid_cg_mean_days_intensity': CG_DIR,
    'euclid_cg_annual_timeseries': CG_DIR,
}
SENSITIVITY_FIGURE_DIRS = {
    'occurrence_ratio_threshold_sensitivity': SWEEP_FIGURES_DIR,
    'intensity_effect_threshold_sensitivity': SWEEP_FIGURES_DIR,
    'environment_threshold_sensitivity': SWEEP_FIGURES_DIR,
    'disagreement_vs_elevation': WETBULB_COMPARISON_DIR,
}
# Paths relative to the selected primary analysis run.
RUN_FIGURE_SUBDIRS = {
    'cg_compound_event_location_density': 'compound_density',
    'cg_compound_event_count_spatial_trend': 'compound_density',
    'occurrence_maps': 'occurrence/figures',
    'probability': 'occurrence/figures',
    'intensity_FE': 'intensity/figures',
    'intensity_enhancement': 'intensity/figures',
    'environment_matching': 'matching/final/figures',
    'match_balance': 'matching/final/figures',
    'descriptive_balance': 'matching/diagnostics/figures',
    'environment_means': 'environment/figures',
    'event_centered_environment': 'environment/figures',
    'first_CG_onset_and_peak_intensity_joint': 'first_cg/figures',
    'first_CG_merged_event_cluster': 'first_cg/figures',
    'first_CG_compound_vs_nonHW_difference_event_cluster': 'first_cg/figures',
    'gridcell_first_cg_lifecycle': 'first_cg/figures',
}

EXTRA_FIGURES = {'cg_compound_event_count_spatial_trend'}

PAPER_NUMBERS = {stem: number for number, stem in FIGURES.items()}
# Figures produced by the sensitivity stage; S9 is drawn in the figures stage.
SENSITIVITY_FIGURES = {'FigS07', 'FigS08', 'FigS14', 'FigS15'}


def figure_path(stem: str, run_dir: Path | None = None) -> Path:
    """Return the supporting, sensitivity, or selected main-run PNG path."""
    if stem in SUPPORTING_FIGURES:
        return SUPPORTING_FIGURES[stem] / f'{stem}.png'
    if stem in SENSITIVITY_FIGURE_DIRS:
        return SENSITIVITY_FIGURE_DIRS[stem] / f'{stem}.png'
    if stem == 'pre_cg_environment':
        run = Path(run_dir if run_dir is not None else RUN_DIR)
        return sensitivity_dir(run) / 'pre_cg_environment' / f'{stem}.png'
    if stem not in RUN_FIGURE_SUBDIRS:
        raise ValueError(f'Unknown figure: {stem}')
    run = Path(run_dir if run_dir is not None else RUN_DIR)
    return run / RUN_FIGURE_SUBDIRS[stem] / f'{stem}.png'


def save_figure(fig, stem: str, *, run_dir: Path | None = None,
                directory: Path | None = None, **kwargs) -> Path:
    """Save a figure once in its assigned category, without numeric prefixes."""
    target = figure_path(stem, run_dir)
    if directory is not None:
        target = Path(directory) / target.name
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, **kwargs)
    print(f'Wrote {target}')
    return target
