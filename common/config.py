"""Study-wide constants shared by the pipeline, figure and sensitivity scripts."""
from common.paths import DATA_DIR, OUTPUTS_DIR, PROJECT_DIR, WETBULB_HEATWAVES_DIR

ROOT = PROJECT_DIR
YEARS = list(range(2006, 2026))
WARM_SEASON_MONTHS = list(range(4, 10))

# Primary heatwave definition: local 90th percentile plus a 20 degC floor.
METHOD = '90p20'
HW = WETBULB_HEATWAVES_DIR / METHOD
CG = DATA_DIR / 'Euclid' / 'monthly_nc_cg'

# Final land-only run used in the manuscript. Plotting scripts write their
# working figures to MANUSCRIPT_FIGURES_DIR; collect_manuscript_figures.py then
# copies them, renumbered, to outputs/manuscript/figures.
RUN_DIR = OUTPUTS_DIR / 'run_90p20'
MANUSCRIPT_FIGURES_DIR = RUN_DIR / 'figures'
MANUSCRIPT_DIR = OUTPUTS_DIR / 'manuscript'

# Sensitivity analyses.
SENSITIVITY_DIR = OUTPUTS_DIR / 'sensitivity'
SWEEP_RUN_ROOT = SENSITIVITY_DIR / 'threshold_sweep'
SWEEP_SUMMARY_DIR = SENSITIVITY_DIR / 'threshold_tables'
# Calendar-day/3-hour 2006-2025 climatology; independent of the heatwave
# threshold, so every run shares it.
SHARED_CLIMATOLOGY_DIR = SWEEP_RUN_ROOT / 'shared_climatology'


def heatwave_dir(method=METHOD):
    """Directory holding wetbulb_heatwave_<year>.nc for a 90p+<threshold> method."""
    return WETBULB_HEATWAVES_DIR / method
