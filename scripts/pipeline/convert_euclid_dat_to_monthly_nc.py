#!/usr/bin/env python3
"""Convert hourly EUCLID ASCII grids to monthly NetCDF files."""

from __future__ import annotations

import argparse
import io
import re
import tarfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import xarray as xr


FILENAME_RE = re.compile(
    r"EUCLID_Flash_CG_IC_1h_10km_(\d{8})_(\d{6})_(\d{8})_(\d{6})\.dat$"
)
HEADER_RE = re.compile(r"(LatMin|LatMax|LatStep|LonMin|LonMax|LonStep)=([0-9.\-]+)")
DAY_MEMBER_RE = re.compile(r"Euclid_(\d{8})\.tar\.gz$")
REPO = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert extracted EUCLID hourly .dat files to monthly NetCDF files."
    )
    parser.add_argument(
        "--input-dir",
        default=str(REPO / "data/Euclid/extracted"),
        help="Directory containing yearly folders with hourly .dat files (--source extracted only).",
    )
    parser.add_argument(
        "--tar-dir",
        default=str(REPO / "data/Euclid/original"),
        help="Directory containing Euclid_<year>.tar archives (--source tar, the default).",
    )
    parser.add_argument(
        "--source",
        choices=("tar", "extracted"),
        default="tar",
        help="Read hourly grids straight from the yearly tar archives (default), or from"
             " pre-extracted .dat files under --input-dir.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPO / "data/Euclid/monthly_nc"),
        help="Directory where yearly folders of monthly NetCDF files will be written.",
    )
    parser.add_argument("--start-year", type=int, default=2006)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument(
        "--months",
        nargs="+",
        type=int,
        default=None,
        help="Optional month numbers to convert, for example --months 7. Default: all months.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing monthly NetCDF files.",
    )
    return parser.parse_args()


def read_header(path: Path) -> list[str]:
    with path.open() as f:
        return [next(f).strip() for _ in range(4)]


def coords_from_header(header: list[str], shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    values = {key: float(value) for key, value in HEADER_RE.findall(" ".join(header))}
    nlat, nlon = shape
    latitudes = values["LatMin"] + np.arange(nlat, dtype=np.float64) * values["LatStep"]
    longitudes = values["LonMin"] + np.arange(nlon, dtype=np.float64) * values["LonStep"]
    return latitudes, longitudes


def load_grid_rows(path_or_buffer) -> np.ndarray:
    # The raw EUCLID ASCII grid stores rows north-to-south (row 0 = LatMax),
    # but coords_from_header() labels row 0 as LatMin -- flip here so the data
    # rows actually match that ascending LatMin->LatMax label array. Confirmed
    # empirically: real thunderstorm land/sea contrast (median land vs sea
    # flash count) only appears once this flip is applied.
    return np.loadtxt(path_or_buffer, skiprows=4, dtype=np.int16)[::-1]


def start_time_from_name(path: Path) -> np.datetime64:
    match = FILENAME_RE.match(path.name)
    if match is None:
        raise ValueError(f"Unexpected EUCLID filename: {path.name}")
    start_date, start_time, _, _ = match.groups()
    stamp = f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:]}T{start_time[:2]}:{start_time[2:4]}:{start_time[4:]}"
    return np.datetime64(stamp)


def month_key(path: Path) -> str:
    match = FILENAME_RE.match(path.name)
    if match is None:
        raise ValueError(f"Unexpected EUCLID filename: {path.name}")
    return match.group(1)[:6]


def start_time_from_basename(basename: str) -> np.datetime64:
    match = FILENAME_RE.match(basename)
    if match is None:
        raise ValueError(f"Unexpected EUCLID filename: {basename}")
    start_date, start_time, _, _ = match.groups()
    stamp = f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:]}T{start_time[:2]}:{start_time[2:4]}:{start_time[4:]}"
    return np.datetime64(stamp)


def load_month_from_tar(outer: tarfile.TarFile, day_members: list[str]) -> xr.Dataset:
    """Read one month of hourly grids straight out of the yearly tar archive,
    without extracting anything to disk."""
    grids: list[np.ndarray] = []
    times: list[np.datetime64] = []
    header: list[str] | None = None
    for day_member in day_members:
        day_blob = outer.extractfile(day_member).read()
        with tarfile.open(fileobj=io.BytesIO(day_blob), mode="r:gz") as daily:
            for name in sorted(daily.getnames()):
                basename = Path(name).name
                if FILENAME_RE.match(basename) is None:
                    continue
                raw = daily.extractfile(name).read()
                if header is None:
                    header = [line.decode("utf-8").strip() for line in raw.splitlines()[:4]]
                grids.append(load_grid_rows(io.BytesIO(raw)))
                times.append(start_time_from_basename(basename))
    if not grids:
        raise ValueError(f"No hourly CG+IC grids found in {len(day_members)} day archives")
    return build_month_dataset(np.stack(grids), np.array(times, dtype="datetime64[ns]"), header)


def load_month(files: list[Path]) -> xr.Dataset:
    header = read_header(files[0])
    data = np.stack([load_grid_rows(path) for path in files])
    times = np.array([start_time_from_name(path) for path in files], dtype="datetime64[ns]")
    return build_month_dataset(data, times, header)


def build_month_dataset(data: np.ndarray, times: np.ndarray, header: list[str]) -> xr.Dataset:
    latitudes, longitudes = coords_from_header(header, data.shape[1:])
    order = np.argsort(times)
    data = data[order].astype(np.int16)
    times = times[order]

    ds = xr.Dataset(
        data_vars={
            "flash_count": (
                ("valid_time", "latitude", "longitude"),
                data,
                {
                    "long_name": "EUCLID hourly CG+IC flash count",
                    "description": "Combined cloud-to-ground and intra-cloud flash count per grid cell.",
                    "units": "count h-1",
                },
            )
        },
        coords={
            "valid_time": times,
            "latitude": ("latitude", latitudes, {"units": "degrees_north"}),
            "longitude": ("longitude", longitudes, {"units": "degrees_east"}),
        },
        attrs={
            "source": "EUCLID hourly ASCII grids",
            "grid": "1h 10km EUCLID grid from file headers",
            "lightning_type": "CG+IC combined",
            "note": "Use occurrence rather than counts when consistency of pre-2017 IC flash grouping is a concern.",
            "sample_header": " | ".join(header),
        },
    )
    return ds


def write_dataset(ds: xr.Dataset, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    encoding = {
        "flash_count": {
            "zlib": True,
            "complevel": 4,
            "shuffle": True,
            "dtype": "int16",
            "chunksizes": (24, len(ds.latitude), len(ds.longitude)),
        }
    }
    ds.to_netcdf(tmp_path, engine="netcdf4", encoding=encoding)
    tmp_path.replace(output_path)
    ds.close()


def month_output(output_dir: Path, yyyymm: str) -> Path:
    return output_dir / yyyymm[:4] / f"euclid_flash_cg_ic_{yyyymm[:4]}_{yyyymm[4:]}.nc"


def convert_year_from_tar(tar_dir: Path, year: int, output_dir: Path, overwrite: bool,
                           months: list[int] | None = None) -> None:
    year_tar = tar_dir / f"Euclid_{year}.tar"
    if not year_tar.exists():
        print(f"{year}: missing {year_tar}", flush=True)
        return
    wanted = {f"{month:02d}" for month in months} if months else None
    with tarfile.open(year_tar) as outer:
        by_month: dict[str, list[str]] = defaultdict(list)
        for name in outer.getnames():
            match = DAY_MEMBER_RE.search(name)
            if match is None or not match.group(1).startswith(str(year)):
                continue
            if wanted is not None and match.group(1)[4:6] not in wanted:
                continue
            by_month[match.group(1)[:6]].append(name)
        if not by_month:
            print(f"{year}: no day archives inside {year_tar}", flush=True)
            return
        for yyyymm in sorted(by_month):
            out = month_output(output_dir, yyyymm)
            if out.exists() and not overwrite:
                print(f"skip {out}", flush=True)
                continue
            day_members = sorted(by_month[yyyymm])
            print(f"writing {out} from {len(day_members)} day archives", flush=True)
            write_dataset(load_month_from_tar(outer, day_members), out)


def convert_year_from_extracted(input_dir: Path, year: int, output_dir: Path, overwrite: bool,
                                 months: list[int] | None = None) -> None:
    year_dir = input_dir / str(year)
    files = sorted(year_dir.glob("*.dat"))
    if not files:
        print(f"{year}: no .dat files found in {year_dir}", flush=True)
        return

    wanted = {f"{month:02d}" for month in months} if months else None
    by_month: dict[str, list[Path]] = defaultdict(list)
    for path in files:
        yyyymm = month_key(path)
        if wanted is not None and yyyymm[4:6] not in wanted:
            continue
        by_month[yyyymm].append(path)

    for yyyymm in sorted(by_month):
        out = month_output(output_dir, yyyymm)
        if out.exists() and not overwrite:
            print(f"skip {out}", flush=True)
            continue
        print(f"writing {out} from {len(by_month[yyyymm])} hourly files", flush=True)
        write_dataset(load_month(sorted(by_month[yyyymm])), out)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)

    for year in range(args.start_year, args.end_year + 1):
        if args.source == "tar":
            convert_year_from_tar(Path(args.tar_dir), year, output_dir, args.overwrite, args.months)
        else:
            convert_year_from_extracted(Path(args.input_dir), year, output_dir, args.overwrite, args.months)


if __name__ == "__main__":
    main()
