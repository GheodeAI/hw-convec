# hw-convec

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23560242.svg)](https://doi.org/10.5281/zenodo.23560242)

Analysis code for **“Cloud-to-ground lightning is enhanced during European wet-bulb heatwaves”** by Yulin Zou, Jorge Pérez-Aracil, Jürg Luterbacher, Elena Xoplaki and Sancho Salcedo-Sanz (in preparation).

The code identifies wet-bulb heatwaves over European land (April–September, 2006–2025), estimates how they change cloud-to-ground (CG) lightning occurrence, compares the atmospheric environments of heatwaves with and without lightning, and follows conditions around the first CG flash. Data and results are not included.

## Installation

```bash
git clone https://github.com/GheodeAI/hw-convec.git
cd hw-convec
conda env create -f environment.yml   # Python 3.11
conda activate hw-convec
```

## Data

| Dataset | Coverage | Location |
| --- | --- | --- |
| ERA5 single-level instantaneous fields | 1991–2025 | `data/Era5/<year>/instant/` |
| ERA5 accumulated and pressure-level fields | 2006–2025 | `data/Era5/<year>/{accum,pressure}/` |
| EUCLID hourly CG lightning | 2006–2025 | `data/Euclid/original/Euclid_<year>.tar` |
| Natural Earth 1:10m admin-0 countries | — | `data/shapefiles/ne_10m_admin_0_countries/` |

EUCLID data must be obtained from the provider. ERA5 is downloaded from the [Copernicus Climate Data Store](https://cds.climate.copernicus.eu) (credentials in `~/.cdsapirc`):

```bash
python scripts/download/download_era5.py --start-year 1991 --end-year 2005 --groups instant
python scripts/download/download_era5.py --start-year 2006 --end-year 2025
```

## Running the analysis

```bash
bash scripts/run_all.sh data          # ERA5 fields, wet-bulb heatwaves, EUCLID processing, event catalogues
bash scripts/run_all.sh main          # regional events, occurrence models, matching, environments
bash scripts/run_all.sh figures       # 7 main and 12 supplementary figures
bash scripts/run_all.sh sensitivity   # robustness checks and the remaining 4 supplementary figures
bash scripts/run_all.sh numbers       # statistics quoted in the text, Tables S1–S2
```

`bash scripts/run_all.sh all` runs every stage. Completed stages are reused; `--dry-run` previews and `--force` recomputes, and `RUN=<dir>` keeps a separate analysis run (see [docs/outputs.md](docs/outputs.md)). Each script lists its options with `--help`.

## Outputs

```text
outputs/
├── main_analysis/        # occurrence/, intensity/, environment/, first_cg/, matching/,
│                         # compound_density/, events/, checks/
├── supporting/           # cg/, heatwaves/ (Figures S1–S4)
├── sensitivity/          # threshold_sweep/, cg_fraction_sensitivity/, cg_window_sensitivity/,
│                         # pre_cg_environment/, wetbulb_method_comparison/
└── manuscript/tables/    # Tables S1–S2
```

Each analysis keeps its own `tables/`, `figures/` and `data/`. [common/figures.py](common/figures.py) maps paper figure numbers to files, and [docs/outputs.md](docs/outputs.md) describes every output and cache.

## Methods in brief

- **Wet-bulb heatwaves:** daily maximum wet-bulb temperature (Stull, from three-hourly ERA5) above both the local 1991–2020 calendar-day 90th percentile (31-day window) and 20°C for at least three days; spells one day apart are merged and connected in space and time into land regional events.
- **Compound events:** at least 10% of an event's land grid cells record CG lightning during the heatwave or the following 24 h; events with incomplete CG coverage are excluded.
- **Lightning occurrence:** logit models with cell × month and year fixed effects and two-way (cell, date) clustered inference; heatwave-intensity quintiles with event-clustered inference.
- **Matched environments:** 1:1 matching on exact start month and a Mahalanobis caliper of 0.75 (centre, duration, intensity, log peak land area), with pair and start-year fixed effects.
- **Robustness:** absolute thresholds of 18–22°C, the CG-fraction rule, excluding the 24 h after the event, environments before the first CG, and Stull versus Romps wet-bulb temperature.

## Code organization

| Directory | Contents |
| --- | --- |
| `common/` | Configuration and paths, data readers, statistics, matching, figure style |
| `scripts/download/` | ERA5 download |
| `scripts/pipeline/` | Preprocessing and main analysis |
| `scripts/figures/` | Paper figures |
| `scripts/sensitivity/` | Robustness checks |
| `scripts/statistics/` | Statistics and supplementary tables |

Tests run on small synthetic inputs (`python -m unittest discover -s tests`); `python scripts/figures/check_figures.py` checks that all 23 paper figures were produced.

## Citation and contact

> Zou, Y., Pérez-Aracil, J., Luterbacher, J., Xoplaki, E., & Salcedo-Sanz, S. (2026). *hw-convec: code for “Cloud-to-ground lightning is enhanced during European wet-bulb heatwaves”*. Zenodo. https://doi.org/10.5281/zenodo.23560242

Code under the [MIT License](LICENSE); input datasets retain their providers' terms. Questions: open an issue or contact Yulin Zou ([yulin.zou@uah.es](mailto:yulin.zou@uah.es)).
