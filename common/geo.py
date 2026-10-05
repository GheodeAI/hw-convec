"""Grid lookup and land-mask helpers."""
import numpy as np
import shapefile
import shapely
from shapely.geometry import shape
from shapely.strtree import STRtree

from common.paths import COUNTRIES_10M_SHAPEFILE


def nearest_indices(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Indices of nearest values in a monotonic one-dimensional coordinate."""
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if source.size < 2:
        return np.zeros(target.size, dtype=np.int32)
    if source[0] > source[-1]:
        return source.size - 1 - nearest_indices(source[::-1], target)
    right = np.searchsorted(source, target, side="left")
    right = np.clip(right, 0, source.size - 1)
    left = np.clip(right - 1, 0, source.size - 1)
    choose_left = np.abs(target - source[left]) <= np.abs(source[right] - target)
    return np.where(choose_left, left, right).astype(np.int32)


def land_at(lat, lon, shp=COUNTRIES_10M_SHAPEFILE):
    """True where a point (cell centre) intersects a Natural Earth country polygon."""
    with shapefile.Reader(str(shp)) as reader:
        tree = STRtree([shape(s.__geo_interface__) for s in reader.shapes()])
    pts = shapely.points(np.asarray(lon).ravel(), np.asarray(lat).ravel())
    mask = np.zeros(len(pts), bool)
    mask[tree.query(pts, predicate='intersects')[0]] = True
    return mask.reshape(np.shape(lat))
