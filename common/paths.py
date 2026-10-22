import os
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
ERA5_DIR = DATA_DIR / "Era5"
PROCESSED_DIR = PROJECT_DIR / "processed"
ANALYSIS_INPUTS_DIR = Path(os.environ.get("ANALYSIS_INPUTS_ROOT", str(PROCESSED_DIR / "analysis_inputs")))
if not ANALYSIS_INPUTS_DIR.is_absolute():
    ANALYSIS_INPUTS_DIR = PROJECT_DIR / ANALYSIS_INPUTS_DIR
SHARED_ANALYSIS_INPUTS_DIR = ANALYSIS_INPUTS_DIR / "shared"
OUTPUTS_DIR = PROJECT_DIR / "outputs"
# Select the primary analysis once; supporting categories are independent of RUN.
MAIN_ANALYSIS_DIR = Path(os.environ.get("RUN", str(OUTPUTS_DIR / "main_analysis")))
if not MAIN_ANALYSIS_DIR.is_absolute():
    MAIN_ANALYSIS_DIR = PROJECT_DIR / MAIN_ANALYSIS_DIR
DENSITY_DIR = MAIN_ANALYSIS_DIR / "compound_density"
SUPPORTING_DIR = OUTPUTS_DIR / "supporting"
CG_DIR = SUPPORTING_DIR / "cg"
HEATWAVE_DIR = SUPPORTING_DIR / "heatwaves"

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
