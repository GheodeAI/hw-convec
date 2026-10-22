# Generated files and their locations

The default primary run is `outputs/main_analysis/`. `RUN` selects another run. Paths are defined in `common/paths.py`, `common/config.py`, `common/layout.py` and `common/figures.py`.

## Raw inputs and preprocessing

Raw observations are in `data/`. Reusable products are in `processed/`:

```text
processed/
├── era5_3hourly/
├── wetbulb/
│   ├── thresholds/
│   └── heatwaves/<method>/
├── compound/
├── climatology/                    # month_03.npz through month_10.npz
└── analysis_inputs/
    ├── shared/
    │   ├── grids.npz               # Coordinates, land masks and CG-to-ERA5 mapping
    │   ├── cg_hourly_era5_<year>.npz  # Sparse hourly CG counts on the ERA5 grid
    │   ├── cg_daily_<year>.npz     # Reusable daily CG observations
    │   └── metadata.json           # Cache metadata (see below)
    ├── 90p20/
    │   ├── daily_<year>.npz
    │   └── labels_<year>.npz
    └── <other methods>/
```

The input arrays cover 2006–2025 under the current grid and processing rules. Daily files combine CG observations with definition-dependent heatwave flags, excess, event IDs and dates; labels contain connected-event IDs, offsets and dates. Runs using the same definition share these arrays. Each run records its method in `checks/analysis_inputs.json` and its own rebuild completion in `checks/rebuild_complete.json`. Sharing input arrays does not imply that a run's catalogue is complete.

`processed/climatology/` contains the 2006–2025 calendar-day/3-hour environment climatology, including baseline-year counts. It is shared by the main environment analysis, pre-CG check and threshold sweep. Standalone environment scripts default to this directory; `--climatology-dir` overrides it. `CLIM` controls the main and pre-CG commands in `run_all.sh`.

Cache metadata records the study years, domain, wet-bulb method, source-file inventory (sizes and modification times), recipe fingerprints and generator version; incompatible inputs or recipes stop reuse. Downstream scripts resolve cache paths through `common/analysis_inputs.py`. Changing the study period, grid or preprocessing rules requires regenerating the affected caches: select a new `ANALYSIS_INPUTS_ROOT` (and a new `CLIM` for environment inputs), then run the main analysis with `--force` or a fresh `RUN`; the old cache stays available.

First-CG environmental values are read directly from the eight monthly `first_cg_values_m*.parquet` files, projecting only the needed columns; no merged copy is written.

## Main analysis

The directories below are relative to `outputs/main_analysis/`. The nine threshold runs under `outputs/sensitivity/threshold_sweep/runs/<method>_2006-2025/` use the same internal layout.

| Directory | Contents |
| --- | --- |
| `events/` | `regional_events.csv`, the shared land regional-event catalogue |
| `checks/` | Method registration, rebuild completion, land-rebuild audit, unmasked-catalogue reproduction, CG coverage checks and `workflow.json` stage records |
| `occurrence/tables/` | `occurrence_probability.csv`, `occurrence_by_year.csv`, raw fixed-effect model summaries, attrition and year effects, `adjusted_effects_95CI.csv`, `unadjusted_probabilities.csv`, and sample diagnostics |
| `occurrence/figures/` | `occurrence_maps.png`, `probability.png` |
| `occurrence/data/` | `occurrence_maps.nc`, `margin_cluster_covariance.npz` |
| `intensity/tables/` | Quintile summaries, coefficients, enhancement statistics, `quintile_convergence.csv` and `quintile_stratum_coverage.csv` |
| `intensity/figures/` | `intensity_FE.png`, `intensity_enhancement.png` |
| `intensity/data/` | `intensity_sample.parquet`, `quintile_cluster_bootstrap.npy`, `quintile_margin_coefficient_covariance.npy` |
| `environment/tables/` | Event environment anomalies and environmental group comparisons |
| `environment/figures/` | `environment_means.png`, `event_centered_environment.png` |
| `environment/data/` | Monthly `event_environment_m*.npz` and `spatial_environment_m*.npz`, `event_centered_environment.nc`, `environment_complete.json` |
| `first_cg/tables/` | Timing, peak-intensity timing, curve summaries, adjacent changes, group differences, effective sample sizes and `confirmed_anchor_counts.csv` |
| `first_cg/figures/` | `first_CG_onset_and_peak_intensity_joint.png`, `first_CG_merged_event_cluster.png`, `first_CG_compound_vs_nonHW_difference_event_cluster.png`, `gridcell_first_cg_lifecycle.png` |
| `first_cg/data/` | `confirmed_anchors.parquet`, monthly `first_cg_values_m*.parquet`, `confirmed_compound_timing.parquet` |
| `matching/baseline/` | Original `match_pairs.csv`, `matched_events.csv`, `match_balance.csv`, `matching_summary.json` |
| `matching/final/tables/` | Final matched pairs, balance, environmental effects, regression coefficients and retention statistics |
| `matching/final/figures/` | `environment_matching.png`, `match_balance.png` |
| `matching/final/data/` | `regression_arrays.npz` |
| `matching/diagnostics/tables/` | Peak daily event areas, area/year balance and descriptive year distributions |
| `matching/diagnostics/figures/` | `descriptive_balance.png` |
| `matching/comparison/` | Matching-design comparison, including `area_only_pairs.csv`, scenario covariances and retention summaries |
| `compound_density/` | `cg_compound_event_location_density.png`, `cg_compound_event_count_spatial_trend.png`, and `cg_compound_event_location_density_90p20_2006_2025.nc` |

First-CG sample definitions are recorded in `first_cg/confirmed_definition.json`. Analysis metadata and READMEs remain at their analysis root. Statistical diagnostics stay with their associated tables. There is no run-wide `main_analysis/data/`, `tables/` or `figures/` directory. Analysis-specific caches remain with each run: first-CG control definitions and sampled populations must not be merged just because two runs use `90p20`.

The compound-density NetCDF contains event counts and days, annual and smoothed density, trends, significance and the land mask. Its `--data-dir` option relocates both the NetCDF and its two PNGs. The density PNG is Figure S5; the event-days/trend PNG is additional to the 23 paper figures.

The standalone regional-event builder defaults summary tables to `$RUN/events/`; `run_all.sh data` directs preliminary catalogue tables to `processed/compound/euclid_cg/tables/`.

## Supporting results

`outputs/supporting/` contains only `cg/` and `heatwaves/`, with all files directly inside each subject directory:

```text
outputs/supporting/
├── cg/
│   ├── euclid_cg_mean_days_int_2006-2025.nc
│   ├── euclid_cg_annual_2006-2025.csv
│   ├── euclid_cg_mean_days_intensity.png
│   └── euclid_cg_annual_timeseries.png
└── heatwaves/
    ├── wetbulb_heatwave_historical_spatial.png
    ├── wetbulb_heatwave_cg_domain_annual_timeseries.png
    └── wetbulb_heatwave_cg_domain_annual_timeseries_AprSep_90pplus20.csv
```

The CG NetCDF contains CG-day frequency, flash intensity, trends, significance and the land mask. The CSV contains annual domain means. The heatwave CSV contains annual heatwave days, intensity, event counts and duration in the CG observation domain. Their four PNGs are Figures S1–S4. The CG script's `--data-dir` and `--tables-dir` both default to `supporting/cg/`; heatwave tables default to `supporting/heatwaves/`. The domain-trend script reads these CSVs and prints tests.

## Sensitivity and robustness

| Directory under `outputs/sensitivity/` | Contents |
| --- | --- |
| `threshold_sweep/figures/` | `occurrence_ratio_threshold_sensitivity.png`, `intensity_effect_threshold_sensitivity.png`, `environment_threshold_sensitivity.png` |
| `threshold_sweep/tables/` | `absolute_threshold_sensitivity_2006_2025.csv`, `environment_threshold_sensitivity_2006_2025.csv` |
| `threshold_sweep/runs/<method>_2006-2025/` | Individual threshold analyses using the main-analysis layout above |
| `cg_fraction_sensitivity/` | Compound-definition fraction comparisons, matched-pair outputs and provenance |
| `cg_window_sensitivity/` | Comparisons with and without the post-event 24-hour CG window |
| `pre_cg_environment/` | `pre_cg_environment.png`, `pre_cg_environment_effects.csv`, event-window tables and extraction arrays |
| `wetbulb_method_comparison/` | `disagreement_vs_elevation.png`, `spatial_disagreement.nc`, `agreement_summary.csv`, `disagreement_by_elevation_band.csv` |

The pre-CG and wet-bulb comparison directories store figures with their other files, without a `figures/` subdirectory. The default workflow runs the pre-CG computation and Figure S15 in the `sensitivity` stage. The `figures` stage produces 19 paper figures; `sensitivity` produces the remaining four. `scripts/figures/check_figures.py` checks all 23 paper figures and the additional density plot without copying them.

For custom `RUN` directories, CG-fraction, CG-window and pre-CG results use `<RUN>/sensitivity/`, preserving their association with that run. The default `outputs/main_analysis/` uses `outputs/sensitivity/`. Shared threshold sweeps, supporting results and wet-bulb comparison retain their project-wide locations.

## Manuscript tables and archives

`outputs/manuscript/tables/` contains supplementary Tables S1–S2 as CSV.

`archive/` holds results moved aside by `--force` reruns, and `logs/` records those moves; current code does not read from either. The `data/`, `processed/`, `outputs/`, `archive/` and `logs/` directories are excluded from Git: a repository clone includes code and documentation, not the generated files listed here.

## Resume and force modes

`bash scripts/run_all.sh main` resumes complete main-analysis stages; incomplete steps invalidate and archive their downstream results before recomputation. `sensitivity` reuses completed pre-CG and CG-definition checks and lets the threshold orchestrators resume their individual stages. Stage records are stored in the selected run's `checks/workflow.json`.

Use `bash scripts/run_all.sh main --dry-run` to preview actions, or add `--force` to `main` or `sensitivity` to archive that stage's previous results and recompute the statistics. Force mode retains compatible preprocessing and climatology. It does not silently reuse changed input recipes. Main-analysis recomputation also archives its associated robustness checks and the default run's manuscript tables; run `figures`, `sensitivity` and `numbers` afterwards to regenerate the corresponding products. Sensitivity force mode archives threshold runs, their aggregate tables/figures, associated robustness checks and the default run's manuscript tables. It preserves the independent wet-bulb comparison produced by the figures stage. `--force --dry-run` previews those moves without writing files.

Archives from reruns are timestamped under `archive/reruns/`, with moves recorded in `logs/reruns.tsv`. Partially written NPZ caches are published with atomic renames so interrupted writes are not mistaken for completed inputs. Compatible grid and hourly/daily CG caches are reused during new analyses.
