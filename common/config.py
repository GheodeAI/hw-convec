"""Study-wide constants shared by the pipeline, figure and sensitivity scripts."""
from common.paths import DATA_DIR, MAIN_ANALYSIS_DIR, OUTPUTS_DIR, PROJECT_DIR, PROCESSED_DIR, WETBULB_HEATWAVES_DIR

ROOT = PROJECT_DIR
YEARS = list(range(2006, 2026))
WARM_SEASON_MONTHS = list(range(4, 10))

# Primary heatwave definition: local 90th percentile plus a 20 degC floor.
METHOD = '90p20'
HW = WETBULB_HEATWAVES_DIR / METHOD
CG = DATA_DIR / 'Euclid' / 'monthly_nc_cg'

# Data and statistical results for the final land-only run.
RUN_DIR = MAIN_ANALYSIS_DIR
# Tables prepared for the manuscript are kept separately from analysis figures.
MANUSCRIPT_DIR = OUTPUTS_DIR / 'manuscript'

# Sensitivity analyses.
SENSITIVITY_DIR = OUTPUTS_DIR / 'sensitivity'
SWEEP_DIR = SENSITIVITY_DIR / 'threshold_sweep'
SWEEP_RUN_ROOT = SWEEP_DIR / 'runs'
SWEEP_SUMMARY_DIR = SWEEP_DIR / 'tables'
SWEEP_FIGURES_DIR = SWEEP_DIR / 'figures'
WETBULB_COMPARISON_DIR = SENSITIVITY_DIR / 'wetbulb_method_comparison'
# Calendar-day/3-hour 2006-2025 climatology; independent of the heatwave
# threshold, so every run shares it.
SHARED_CLIMATOLOGY_DIR = PROCESSED_DIR / 'climatology'


def heatwave_dir(method=METHOD):
    """Directory holding wetbulb_heatwave_<year>.nc for a 90p+<threshold> method."""
    return WETBULB_HEATWAVES_DIR / method
