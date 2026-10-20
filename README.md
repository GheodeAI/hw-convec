# hw-convec

Code for **"Cloud-to-ground lightning is enhanced during European wet-bulb heatwaves"**
by Yulin Zou, Jorge Pérez-Aracil, Jürg Luterbacher, Elena Xoplaki and Sancho Salcedo-Sanz (in preparation).

The analysis asks how often cloud-to-ground (CG) lightning occurs during wet-bulb heatwaves over
European land, how that depends on heatwave intensity, and which atmospheric conditions separate
heatwaves that produce CG lightning from those that do not.

## Approach

- **Heatwaves.** Daily maximum wet-bulb temperature (Stull 2011) from 3-hourly ERA5. A grid cell is
  in a heatwave when Tw,max exceeds both its 1991–2020 calendar-day 90th percentile (31-day window)
  and 20 °C for at least 3 days; spells separated by one day are merged. Grid-cell spells are joined
  into regional events by space–time connectivity over land.
- **Lightning.** EUCLID hourly CG flash counts on a 10 km grid, 2006–2025, April–September.
  A regional heatwave is *compound* when at least 10% of its land cells record CG during the event
  or the following 24 h.
- **Occurrence.** Logit with cell × month and year fixed effects, two-way (cell, date) clustered
  errors, delta-method adjusted probabilities; heatwave-intensity quintiles with event-clustered inference.
- **Environment.** Compound events are matched 1:1 to non-compound events (exact start month;
  Mahalanobis distance on centre, duration, intensity and peak land area; caliper 0.75). Environmental
  anomalies (q850, TCWV, 2 m dewpoint depression, moisture fluxes, Z500) are compared with
  pair + year fixed effects; time evolution around the first CG uses cluster bootstraps.
- **Sensitivity.** Absolute wet-bulb thresholds 18–22 °C, CG-fraction rule, compound classification
  counting only CG during the event (without the 24 h after), environment before the first CG only,
  and Stull versus Romps wet-bulb formulations.

## Repository layout

```
common/                shared code: paths and study constants (config.py, paths.py), statistics,
                       matching, fixed-effect logit, ERA5 readers, land mask, figure style
scripts/
  download/            ERA5 download from the Copernicus Climate Data Store
  pipeline/            raw data -> heatwaves -> events -> models -> matched environment
  figures/             one script per paper figure (figure numbers noted in run_all.sh)
  sensitivity/         threshold sweeps, CG-fraction rule and CG time window of the compound
                       classification (Table S2)
  statistics/          numbers quoted in the text (trend tests, event counts, lifecycle timing)
                       and Tables S1, S2 as CSV
  run_all.sh           the full run, in order
```

Data and outputs are not tracked; the scripts create them under `data/`, `processed/` and `outputs/`.

## Data

| Source | Use | Location expected by the code |
|---|---|---|
| ERA5 single and pressure levels, 3-hourly ([Copernicus CDS](https://cds.climate.copernicus.eu)) | wet-bulb temperature, environmental variables | `data/Era5/<year>/{instant,pressure}/` via `scripts/download/download_era5.py` |
| EUCLID lightning, hourly CG on a 10 km grid | lightning | `data/Euclid/original/Euclid_<year>.tar` |
| Natural Earth 1:10m admin-0 countries | land mask, maps | `data/shapefiles/ne_10m_admin_0_countries/` |

EUCLID data are licensed and are **not** included; they must be obtained from EUCLID directly.
ERA5 downloads need a CDS account with credentials in `~/.cdsapirc`.

## Requirements

Python 3.11. The conda environment used for the paper is in `environment.yml`:

```bash
conda env create -f environment.yml
conda activate hw-convec
```

## Reproducing the paper

```bash
bash scripts/run_all.sh data          # downloads and preprocessing (long; needs the raw data above)
bash scripts/run_all.sh main          # events, fixed-effect models, matching, environment
bash scripts/run_all.sh figures       # all figures; copies Fig01–Fig07, FigS01–FigS16 to outputs/manuscript/figures/
bash scripts/run_all.sh sensitivity   # threshold sweeps, CG-fraction rule, CG window; Fig S7, S8, S14
bash scripts/run_all.sh numbers       # statistics quoted in the text; Tables S1, S2 to outputs/manuscript/tables/
```

The main run is written to `outputs/run_90p20/`. Several steps refuse to overwrite an existing
output directory; for a fresh rerun set another target, e.g. `RUN=outputs/run_rerun bash scripts/run_all.sh main`.
The mapping from working figure names to paper figure numbers is in
`scripts/figures/collect_manuscript_figures.py`.

## Contact

Yulin Zou (yulin.zou@uah.es), or open an issue in this repository.
