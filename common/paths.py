from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
ERA5_DIR = DATA_DIR / "Era5"
PROCESSED_DIR = PROJECT_DIR / "processed"
OUTPUTS_DIR = PROJECT_DIR / "outputs"
# Intermediate figures/tables/data behind the supplementary figures; the main
# run, sensitivity runs and manuscript set are in common/config.py.
SUPPORTING_DIR = OUTPUTS_DIR / "supporting"
OUTPUT_DATA_DIR = SUPPORTING_DIR / "data"
FIGURES_DIR = SUPPORTING_DIR / "figures"
TABLES_DIR = SUPPORTING_DIR / "tables"

HEATWAVE_FIGURES_DIR = FIGURES_DIR / "heatwaves"
HEATWAVE_TABLES_DIR = TABLES_DIR / "heatwaves"
COMPOUND_FIGURES_DIR = FIGURES_DIR / "compound"
COMPOUND_TABLES_DIR = TABLES_DIR / "compound"
COMPOUND_DATA_DIR = OUTPUT_DATA_DIR / "compound"
COMPOUND_HEATWAVE_TABLES_DIR = COMPOUND_TABLES_DIR / "characteristics"
COMPOUND_ENV_TABLES_DIR = COMPOUND_TABLES_DIR / "compound_noncompound_variables"
COMPOUND_ENV_DATA_DIR = COMPOUND_DATA_DIR / "compound_noncompound_variables"
COMPOUND_ENV_FIGURES_DIR = COMPOUND_FIGURES_DIR / "compound_noncompound_variables"
SHAPEFILE_DIR = DATA_DIR / "shapefiles"
COUNTRIES_10M_SHAPEFILE = (
    SHAPEFILE_DIR
    / "ne_10m_admin_0_countries"
    / "ne_10m_admin_0_countries.shp"
)
COUNTRIES_110M_SHAPEFILE = (
    SHAPEFILE_DIR
    / "ne_110m_admin_0_countries"
    / "ne_110m_admin_0_countries.shp"
)
COUNTRIES_SHAPEFILE = COUNTRIES_10M_SHAPEFILE if COUNTRIES_10M_SHAPEFILE.exists() else COUNTRIES_110M_SHAPEFILE

ERA5_3HOURLY_DIR = PROCESSED_DIR / "era5_3hourly"

WETBULB_DIR = PROCESSED_DIR / "wetbulb"
WETBULB_THRESHOLDS_DIR = WETBULB_DIR / "thresholds"
WETBULB_HEATWAVES_DIR = WETBULB_DIR / "heatwaves"

COMPOUND_DIR = PROCESSED_DIR / "compound"
COMPOUND_EUCLID_DIR = COMPOUND_DIR / "euclid"
