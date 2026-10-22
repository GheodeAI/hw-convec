"""Validate the scientific inputs used by reusable land-analysis caches."""
import hashlib
import json
import subprocess
from functools import lru_cache
from pathlib import Path

from common.config import CG, YEARS, WARM_SEASON_MONTHS, heatwave_dir
from common.paths import ANALYSIS_INPUTS_DIR, SHARED_ANALYSIS_INPUTS_DIR, PROJECT_DIR, COUNTRIES_10M_SHAPEFILE

SCHEMA_VERSION = 1


def code_version():
    files = ['scripts/pipeline/rebuild_compound_land.py', 'common/geo.py', 'common/config.py']
    digest = hashlib.sha256()
    for name in files:
        digest.update(name.encode()); digest.update((PROJECT_DIR / name).read_bytes())
    commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=PROJECT_DIR,
                            capture_output=True, text=True, check=False).stdout.strip() or None
    return {'git_commit': commit, 'recipe_source_sha256': digest.hexdigest()}


def file_inventory(paths):
    result = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f'Missing preprocessing source: {path}')
        stat = path.stat()
        result.append({'path': str(path.relative_to(PROJECT_DIR)), 'bytes': stat.st_size,
                       'mtime_ns': stat.st_mtime_ns})
    return result


@lru_cache(maxsize=32)
def grid_description(method):
    import numpy as np
    import xarray as xr
    with xr.open_dataset(heatwave_dir(method) / f'wetbulb_heatwave_{YEARS[0]}.nc') as data:
        lat, lon = data.latitude.values, data.longitude.values
    return {'latitude': [float(lat.min()), float(lat.max())],
            'longitude': [float(lon.min()), float(lon.max())],
            'shape': [len(lat), len(lon)],
            'coordinates_sha256': hashlib.sha256(np.asarray(lat).tobytes() + np.asarray(lon).tobytes()).hexdigest()}


@lru_cache(maxsize=32)
def expected_recipe(method):
    if method == 'shared':
        sources = [CG / str(y) / f'euclid_flash_cg_{y}_{m:02d}.nc'
                   for y in YEARS for m in range(1, 13)]
        sources += sorted(COUNTRIES_10M_SHAPEFILE.parent.glob(COUNTRIES_10M_SHAPEFILE.stem + '.*'))
        # This study uses the same ERA5 grid for every threshold.
        sources += [heatwave_dir('90p20') / f'wetbulb_heatwave_{YEARS[0]}.nc']
    else:
        sources = [heatwave_dir(method) / f'wetbulb_heatwave_{y}.nc' for y in YEARS]
    return {'schema_version': SCHEMA_VERSION, 'method': method,
            'years': YEARS, 'warm_season_months': WARM_SEASON_MONTHS,
            'wetbulb_method': 'stull', 'domain': grid_description('90p20' if method == 'shared' else method),
            'event_connectivity': '8 neighbours within day; same cell on adjacent days; land only',
            'recipe_source_sha256': code_version()['recipe_source_sha256'],
            'sources': file_inventory(sources)}


def metadata_path(method):
    return (SHARED_ANALYSIS_INPUTS_DIR if method == 'shared' else ANALYSIS_INPUTS_DIR / method) / 'metadata.json'


def write_metadata(method, *, origin):
    path = metadata_path(method)
    path.parent.mkdir(parents=True, exist_ok=True)
    value = {'recipe': expected_recipe(method), 'origin': origin,
             'registered_with': code_version(),
             'generated_with': code_version() if origin == 'generated' else None}
    path.write_text(json.dumps(value, indent=2) + '\n')
    return value


@lru_cache(maxsize=32)
def validate_cache(method):
    path = metadata_path(method)
    if not path.exists():
        raise FileNotFoundError(f'Missing cache metadata: {path}. Run the rebuild stage first.')
    value = json.loads(path.read_text())
    if value['recipe'] != expected_recipe(method):
        raise ValueError(f'Cache inputs/settings changed: {path}. Select a fresh ANALYSIS_INPUTS_ROOT '
                         'and rebuild the analysis; --force reuses only compatible preprocessing.')
    return value


def initialize_cache(method):
    path = metadata_path(method)
    if path.exists():
        return validate_cache(method)
    directory = path.parent
    if directory.exists() and any(directory.glob('*.npz')):
        raise ValueError(f'Existing arrays have no cache metadata: {directory}')
    value = write_metadata(method, origin='generated')
    validate_cache.cache_clear()
    return value


def save_array_cache(path, **arrays):
    """Publish a complete NPZ atomically so interrupted writes are never reused."""
    import os
    import uuid
    import numpy as np
    path = Path(path)
    temporary = path.with_name(f'.{path.stem}.{os.getpid()}.{uuid.uuid4().hex}.npz')
    try:
        np.savez_compressed(temporary, **arrays)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def save_sparse_cache(path, matrix):
    import os
    import uuid
    from scipy import sparse
    path = Path(path)
    temporary = path.with_name(f'.{path.stem}.{os.getpid()}.{uuid.uuid4().hex}.npz')
    try:
        sparse.save_npz(temporary, matrix)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


@lru_cache(maxsize=1)
def expected_climatology_recipe():
    sources = [PROJECT_DIR / f'data/Era5/{year}/{group}/{year}_{month:02d}.nc'
               for year in YEARS for month in range(3, 11) for group in ['instant', 'pressure']]
    sources += [PROJECT_DIR / f'processed/era5_3hourly/single_level/{year}.nc' for year in YEARS]
    return {'schema_version': SCHEMA_VERSION, 'years': YEARS,
            'calendar': 'month/day/3-hour UTC, leap-year template, includes target year',
            'environment_reader_sha256': hashlib.sha256((PROJECT_DIR/'common/environment.py').read_bytes()).hexdigest(),
            'sources': file_inventory(sources)}


def register_climatology(directory, *, adopt=False):
    directory = Path(directory)
    path = directory / 'metadata.json'
    expected = expected_climatology_recipe()
    if path.exists():
        value = json.loads(path.read_text())
        if value['recipe'] != expected:
            raise ValueError(f'Climatology inputs/settings changed: {path}. Select a fresh CLIM directory.')
        return
    if directory.exists() and any(directory.glob('month_*.npz')) and not adopt:
        raise ValueError(f'Existing climatology has no metadata: {directory}')
    directory.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({'recipe': expected, 'origin': 'adopted_existing' if adopt else 'generated',
                    'registered_with': code_version(), 'generated_with': None if adopt else code_version()}, indent=2) + '\n')
