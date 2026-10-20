#!/usr/bin/env bash
# Reproduce the paper from raw data, in order.
#
#   bash scripts/run_all.sh data         # ERA5 3-hourly, wet-bulb thresholds/heatwaves, EUCLID NetCDF, event catalogues
#   bash scripts/run_all.sh main         # land-only regional events, FE models, matching, environment (into $RUN)
#   bash scripts/run_all.sh figures      # all main and supplementary figures from $RUN
#   bash scripts/run_all.sh sensitivity  # absolute-threshold sweeps, CG-fraction rule, CG window (Fig S7, S8, S14)
#   bash scripts/run_all.sh numbers      # manuscript statistics; Tables S1, S2 as CSV
#   bash scripts/run_all.sh all
#
# Several steps refuse to overwrite an existing output directory, so a full
# rerun should go to a fresh RUN, e.g.  RUN=outputs/run_90p20_rerun bash scripts/run_all.sh main
# Run inside the conda environment from environment.yml.
set -euo pipefail
cd "$(dirname "$0")/.."

RUN=${RUN:-outputs/run_90p20}
CLIM=${CLIM:-outputs/sensitivity/threshold_sweep/shared_climatology}
P=scripts/pipeline
F=scripts/figures
S=scripts/sensitivity
D=scripts/statistics
py() { echo "+ python $*"; python "$@"; }

data() {
    py $P/build_era5_3hourly.py
    py $P/build_wetbulb_threshold.py
    py $P/build_wetbulb_heatwaves.py                      # 90p20, Stull (primary)
    py $P/build_wetbulb_threshold.py --wetbulb-method romps
    py $P/build_wetbulb_heatwaves.py --wetbulb-method romps   # 90p20_romps, Fig S9 only
    py $P/convert_euclid_dat_to_monthly_nc.py             # reads data/Euclid/original/Euclid_<year>.tar
    py $P/derive_euclid_cg_monthly_nc.py                  # CG-only monthly files in data/Euclid/monthly_nc_cg
    # Unmasked regional catalogue: rebuild_compound_land.py checks its connectivity against it.
    py $P/build_regional_compound_events_euclid.py \
        --euclid-dir data/Euclid/monthly_nc_cg --euclid-pattern 'euclid_flash_cg_*.nc' \
        --output-dir processed/compound/euclid_cg --output-prefix cg \
        --tables-dir processed/compound/euclid_cg/tables \
        --method 90p20 processed/wetbulb/heatwaves/90p20
    # Grid-cell CG compound events for Fig S5; writes
    # processed/compound/euclid_cg/compound_cg_heatwave_lightning_events_90p20_2006_2025.csv.
    py $P/build_gridcell_compound_events_euclid.py \
        --euclid-dir data/Euclid/monthly_nc_cg --euclid-pattern 'euclid_flash_cg_*.nc' \
        --output-dir processed/compound/euclid_cg --output-prefix cg \
        --method 90p20 processed/wetbulb/heatwaves/90p20
}

main() {
    py $P/rebuild_compound_land.py rebuild --out "$RUN" --method 90p20
    py $P/rebuild_compound_land.py coverage --out "$RUN"
    py $P/rebuild_compound_land.py probability --out "$RUN"
    py $P/rebuild_compound_land.py match --out "$RUN"
    py $P/check_land_area_year_balance.py --source "$RUN" --output "$RUN/area_year_diagnostics_v1"
    py $P/prepare_confirmed_land.py --out "$RUN" --method 90p20
    py $P/analyze_land_environment.py --out "$RUN" --climatology-dir "$CLIM"
    py $P/fit_land_intensity.py --out "$RUN"
    py $P/fit_cg_heatwave_fixed_effects.py --output-dir "$RUN/tables/warmseason_fe"
    py $P/report_warmseason_fe_margins.py --source "$RUN"
    py $P/compare_land_matching_ablation.py --source "$RUN" --output "$RUN/matching_ablation_v1"
    py $P/fit_area_matched_year_fe.py --source "$RUN" --output "$RUN/area_match_pair_year_FE_v1"
    py $P/compute_pre_cg_environment.py --out "$RUN" --climatology-dir "$CLIM"
}

figures() {
    py $F/plot_confirmed_land.py --out "$RUN"                     # Fig 1, 3, 4, S6, S10-S13 (+ timing data for S16)
    py $F/plot_standalone_heatwave_probability.py --source "$RUN" # Fig 2
    py $F/analyze_first_cg_peak_intensity_timing.py --out "$RUN"  # Fig 5
    py $F/recompute_first_cg_curves_clustered.py --out "$RUN"     # Fig 6, 7
    py $F/plot_heatwave_europe_historical_spatial.py              # Fig S1
    py $F/plot_heatwave_europe_annual_timeseries.py               # Fig S2
    py $F/plot_cg_spatial_stats.py                                # Fig S3, S4
    py $F/plot_compound_event_location_density.py --sources euclid_cg  # Fig S5
    py $F/compare_wetbulb_methods.py                              # Fig S9
    py $F/plot_pre_cg_environment.py --out "$RUN"                 # Fig S15
    py $F/plot_gridcell_first_cg_lifecycle.py --out "$RUN"        # Fig S16
    py $F/collect_manuscript_figures.py --source "$RUN/figures"   # -> outputs/manuscript/figures/Fig*.png
}

sensitivity() {
    py $S/run_absolute_threshold_sweep.py
    py $S/run_environment_sensitivity_sweep.py
    py $S/run_land_cg_fraction_sensitivity.py --source "$RUN" --output "$RUN/cg_fraction_sensitivity_v1"  # Table S2
    py $S/run_cg_window_sensitivity.py                            # compound defined without the 24 h after
    py $F/plot_threshold_sensitivity.py                           # Fig S7, S8
    py $F/plot_environment_threshold_sensitivity.py               # Fig S14
}

numbers() {
    py $D/test_domain_mean_trends.py                              # trend tests (S2, S4 series)
    py $D/diagnose_cg_coverage_exclusions.py --out "$RUN"         # study-domain event counts
    py $D/summarize_first_cg_lifecycle_stats.py --out "$RUN"      # lifecycle phase / composite sizes
    py $D/make_supplementary_tables.py                            # Tables S1, S2 as CSV -> outputs/manuscript/tables/
}

case "${1:-}" in
    data|main|figures|sensitivity|numbers) "$1" ;;
    all) data; main; figures; sensitivity; numbers ;;
    *) sed -n '2,13p' "$0"; exit 1 ;;
esac
