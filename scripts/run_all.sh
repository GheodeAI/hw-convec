#!/usr/bin/env bash
# Reproduce the paper from raw data, in order.
#
#   bash scripts/run_all.sh data         # ERA5 3-hourly, wet-bulb thresholds/heatwaves, EUCLID NetCDF, event catalogues
#   bash scripts/run_all.sh main         # land-only regional events, FE models, matching, environment (into $RUN)
#   bash scripts/run_all.sh figures      # 19 paper figures in assigned analysis directories; sensitivity makes the other 4
#   bash scripts/run_all.sh sensitivity  # absolute-threshold sweeps, CG-fraction rule, CG window (Fig S7, S8, S14, S15)
#   bash scripts/run_all.sh numbers      # manuscript statistics; Tables S1, S2 as CSV
#   bash scripts/run_all.sh all
#
# main and sensitivity resume completed stages by default. Pass --force to
# archive their previous results and rerun; --dry-run previews commands.
# A custom RUN keeps a separate primary analysis, e.g. RUN=outputs/main_analysis_rerun.
# Run inside the conda environment from environment.yml.
set -euo pipefail
cd "$(dirname "$0")/.."

export RUN=${RUN:-outputs/main_analysis}
CLIM=${CLIM:-processed/climatology}
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
    py $P/run_analysis.py main --run "$RUN" --climatology-dir "$CLIM" "$@"
}

figures() {
    mkdir -p outputs/supporting/{cg,heatwaves}
    py $F/plot_confirmed_land.py --out "$RUN"                     # Fig 1, 3, 4, S6, S10-S13 (+ timing data for S16)
    py $F/plot_standalone_heatwave_probability.py --source "$RUN" # Fig 2
    py $F/analyze_first_cg_peak_intensity_timing.py --out "$RUN"  # Fig 5
    py $F/recompute_first_cg_curves_clustered.py --out "$RUN"     # Fig 6, 7
    py $F/plot_heatwave_europe_historical_spatial.py --threshold 90p+20   # Fig S1
    py $F/plot_heatwave_europe_annual_timeseries.py --threshold 90p+20 --cg-domain  # Fig S2
    py $F/plot_cg_spatial_stats.py                                # Fig S3, S4
    py $F/plot_compound_event_location_density.py --sources euclid_cg  # Fig S5
    py $F/compare_wetbulb_methods.py                              # Fig S9
    py $F/plot_gridcell_first_cg_lifecycle.py --out "$RUN"        # Fig S16
    py $F/check_figures.py --group figures          # validate all assigned figure locations
}

sensitivity() {
    py $P/run_analysis.py sensitivity --run "$RUN" --climatology-dir "$CLIM" "$@"
}

numbers() {
    py $D/test_domain_mean_trends.py                              # trend tests (S2, S4 series)
    py $D/diagnose_cg_coverage_exclusions.py --out "$RUN"         # study-domain event counts
    py $D/summarize_first_cg_lifecycle_stats.py --out "$RUN"      # lifecycle phase / composite sizes
    py $D/make_supplementary_tables.py                            # Tables S1, S2 as CSV -> outputs/manuscript/tables/
}

case "${1:-}" in
    main|sensitivity) stage=$1; shift; "$stage" "$@" ;;
    data|figures|numbers) "$1" ;;
    all) data; main; figures; sensitivity; numbers; py $F/check_figures.py ;;
    *) sed -n '2,13p' "$0"; exit 1 ;;
esac
