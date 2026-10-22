"""Locate reusable analysis inputs independently of result directories."""
import json
import re
from pathlib import Path

from common.paths import ANALYSIS_INPUTS_DIR, SHARED_ANALYSIS_INPUTS_DIR
from common.cache import initialize_cache, validate_cache, code_version


def register_inputs(run, method):
    """Record a run's heatwave definition, including custom RUN directories."""
    if not re.fullmatch(r"90p\d+(?:p\d+)?", method):
        raise ValueError(f"Invalid heatwave method: {method!r}")
    run = Path(run)
    checks = run / "checks"
    checks.mkdir(parents=True, exist_ok=True)
    manifest = checks / "analysis_inputs.json"
    if manifest.exists() and json.loads(manifest.read_text())["method"] != method:
        raise ValueError(f"{run} is already registered with a different heatwave method")
    initialize_cache('shared')
    initialize_cache(method)
    (ANALYSIS_INPUTS_DIR / method).mkdir(parents=True, exist_ok=True)
    SHARED_ANALYSIS_INPUTS_DIR.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps({"method": method, "cache_schema": 1, "registered_with": code_version()}, indent=2) + "\n")


def input_path(run, name):
    """Resolve shared grids/CG or definition-specific daily/event inputs."""
    if name == "grids.npz" or re.fullmatch(r"cg_hourly_era5_\d{4}\.npz", name):
        validate_cache('shared')
        return SHARED_ANALYSIS_INPUTS_DIR / name
    if not re.fullmatch(r"(?:daily|labels)_\d{4}\.npz", name):
        raise ValueError(f"Unknown analysis input: {name!r}")
    manifest = Path(run) / "checks/analysis_inputs.json"
    if not manifest.exists():
        raise FileNotFoundError(f"Missing {manifest}; run the rebuild stage first")
    method = json.loads(manifest.read_text())["method"]
    if not re.fullmatch(r"90p\d+(?:p\d+)?", method):
        raise ValueError(f"Invalid heatwave method in {manifest}")
    validate_cache('shared')
    validate_cache(method)
    return ANALYSIS_INPUTS_DIR / method / name


def validate_run_inputs(run):
    """Require complete arrays before reusing a completed run."""
    from common.config import YEARS
    for year in YEARS:
        for stem in ['daily', 'labels', 'cg_hourly_era5']:
            path = input_path(run, f'{stem}_{year}.npz')
            if not path.is_file():
                raise FileNotFoundError(f'Incomplete preprocessing cache: {path}')
    if not input_path(run, 'grids.npz').is_file():
        raise FileNotFoundError('Missing grids.npz')
