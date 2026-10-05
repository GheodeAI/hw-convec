#!/usr/bin/env python3
"""Derive EUCLID CC-only and CG-only monthly NetCDF files."""

from __future__ import annotations

import argparse
import io
import re
import tarfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import xarray as xr


HEADER_RE = re.compile(r"(LatMin|LatMax|LatStep|LonMin|LonMax|LonStep)=([0-9.\-]+)")
REPO = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create EUCLID CC-only monthly NetCDF files from original_CC and "
            "CG-only files as CG = combined(CG+IC/CC) - CC."
        )
    )
    parser.add_argument(
        "--combined-dir",
        default=str(REPO / "data/Euclid/monthly_nc"),
        help="Directory containing yearly combined monthly NetCDF files.",
    )
    parser.add_argument(
        "--cc-original-dir",
        default=str(REPO / "data/Euclid/original_CC"),
        help="Directory containing yearly Euclid_YYYY.tar archives for CC grids.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPO / "data/Euclid/monthly_nc_cg"),
        help="Directory where derived CG-only monthly NetCDF files will be written.",
    )
    parser.add_argument(
        "--cc-output-dir",
        default=str(REPO / "data/Euclid/monthly_nc_cc"),
        help="Directory where CC-only monthly NetCDF files will be written.",
    )
    parser.add_argument("--start-year", type=int, default=2006)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument(
        "--months",
        nargs="+",
        type=int,
        default=None,
        help="Optional month numbers to process, e.g. --months 6 7 8.",
    )
    parser.add_argument(
        "--clip-negative",
        action="store_true",
        help="Clip negative CG counts to zero after reporting them.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing monthly NetCDF files.",
    )
    parser.add_argument(
        "--products",
        nargs="+",
        choices=("cc", "cg"),
        default=["cc", "cg"],
        help="Products to write. Defaults to both cc and cg.",
    )
    return parser.parse_args()


def coords_from_header(header: list[str], shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    values = {key: float(value) for key, value in HEADER_RE.findall(" ".join(header))}
    nlat, nlon = shape
    latitudes = values["LatMin"] + np.arange(nlat, dtype=np.float64) * values["LatStep"]
    longitudes = values["LonMin"] + np.arange(nlon, dtype=np.float64) * values["LonStep"]
    return latitudes, longitudes


def cc_name_for_time(time: np.datetime64) -> tuple[str, str]:
    stamp = np.datetime_as_string(time.astype("datetime64[s]"), unit="s")
    date = stamp[:10].replace("-", "")
    hour = stamp[11:13]
    start = f"{date}_{hour}0000"
    end = f"{date}_{hour}5959"
    day_member = f"./data/Euclid_{date}.tar.gz"
    grid_member = f"EUCLID_Flash_CC_1h_10km_{start}_{end}.dat"
    return day_member, grid_member


def read_cc_month(year_tar: Path, times: np.ndarray, shape: tuple[int, int]) -> tuple[np.ndarray, str]:
    cc = np.empty((len(times), *shape), dtype=np.int16)
    header_text = ""
    by_day: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for idx, time in enumerate(times):
        day_member, grid_member = cc_name_for_time(time)
        by_day[day_member].append((idx, grid_member))

    with tarfile.open(year_tar) as outer:
        for day_member in sorted(by_day):
            day_file = outer.extractfile(day_member)
            if day_file is None:
                raise FileNotFoundError(f"Missing {day_member} in {year_tar}")
            day_blob = day_file.read()
            with tarfile.open(fileobj=io.BytesIO(day_blob), mode="r:gz") as daily:
                for idx, grid_member in by_day[day_member]:
                    grid_file = daily.extractfile(grid_member)
                    if grid_file is None:
                        raise FileNotFoundError(f"Missing {grid_member} in {day_member}")
                    grid_blob = grid_file.read()
                    if not header_text:
                        header = grid_blob.splitlines()[:4]
                        header_text = " | ".join(line.decode("utf-8") for line in header)
                        latitudes, longitudes = coords_from_header(
                            [line.decode("utf-8") for line in header],
                            shape,
                        )
                        del latitudes, longitudes
                    # Raw EUCLID grid rows are north-to-south on disk; flip so
                    # row 0 matches the ascending LatMin->LatMax coords the
                    # combined file (and this file's own output, via
                    # write_month()'s coords=ds.coords) uses -- otherwise the
                    # CG = combined - CC subtraction below would compare
                    # mismatched rows against the now-fixed combined data.
                    cc[idx] = np.loadtxt(io.BytesIO(grid_blob), skiprows=4, dtype=np.int16)[::-1]
    return cc, header_text


def write_month(
    combined_path: Path,
    cc_year_tar: Path,
    cg_output_path: Path | None,
    cc_output_path: Path | None,
    clip_negative: bool,
) -> None:
    with xr.open_dataset(combined_path) as ds:
        combined = ds["flash_count"].values.astype(np.int32)
        times = ds["valid_time"].values
        cc, cc_header = read_cc_month(cc_year_tar, times, combined.shape[1:])
        cc_i32 = cc.astype(np.int32)
        cg = combined - cc_i32

        negative_count = 0
        if cg_output_path is not None:
            negative_count = int((cg < 0).sum())
            negative_min = int(cg.min()) if negative_count else 0
            if negative_count and not clip_negative:
                raise ValueError(
                    f"{combined_path}: found {negative_count} negative CG cells "
                    f"(minimum {negative_min}); rerun with --clip-negative to clamp them to zero."
                )
            if negative_count:
                print(
                    f"{combined_path}: clipping {negative_count} negative CG cells "
                    f"(minimum {negative_min})",
                    flush=True,
                )
                cg = np.maximum(cg, 0)

        if cg_output_path is not None and cg.max() > np.iinfo(np.int16).max:
            raise OverflowError(f"{combined_path}: CG count exceeds int16 range")
        if cc_output_path is not None and cc_i32.max() > np.iinfo(np.int16).max:
            raise OverflowError(f"{combined_path}: CC count exceeds int16 range")

        products = []
        if cc_output_path is not None:
            products.append(
                (
                    cc_output_path,
                    cc_i32.astype(np.int16),
                    "CC only",
                    "EUCLID hourly CC-only flash count",
                    "Cloud-cloud/intra-cloud flash count from original_CC grids.",
                    {
                        "derivation": "flash_count read from original_CC EUCLID Flash_CC hourly grids",
                        "cc_source": str(cc_year_tar),
                        "cc_sample_header": cc_header,
                    },
                )
            )
        if cg_output_path is not None:
            products.append(
                (
                    cg_output_path,
                    cg.astype(np.int16),
                    "CG only",
                    "EUCLID hourly CG-only flash count",
                    "Cloud-to-ground flash count derived as combined CG+IC minus CC.",
                    {
                        "derivation": "flash_count = combined CG+IC flash_count - CC flash_count",
                        "combined_source": str(combined_path),
                        "cc_source": str(cc_year_tar),
                        "cc_sample_header": cc_header,
                        "negative_cells_clipped": negative_count if clip_negative else 0,
                    },
                )
            )

        out_datasets = []
        for output_path, data, lightning_type, long_name, description, extra_attrs in products:
            out_ds = xr.Dataset(
                data_vars={
                    "flash_count": (
                        ("valid_time", "latitude", "longitude"),
                        data,
                        {
                            "long_name": long_name,
                            "description": description,
                            "units": "count h-1",
                        },
                    )
                },
                coords=ds.coords,
                attrs={
                    **ds.attrs,
                    "lightning_type": lightning_type,
                    **extra_attrs,
                },
            )
            out_datasets.append((output_path, out_ds))

    for output_path, out_ds in out_datasets:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
        encoding = {
            "flash_count": {
                "zlib": True,
                "complevel": 4,
                "shuffle": True,
                "dtype": "int16",
                "chunksizes": (24, out_ds.sizes["latitude"], out_ds.sizes["longitude"]),
            }
        }
        out_ds.to_netcdf(tmp_path, engine="netcdf4", encoding=encoding)
        tmp_path.replace(output_path)
        out_ds.close()


def main() -> None:
    args = parse_args()
    combined_dir = Path(args.combined_dir)
    cc_original_dir = Path(args.cc_original_dir)
    cg_output_dir = Path(args.output_dir)
    cc_output_dir = Path(args.cc_output_dir)

    for year in range(args.start_year, args.end_year + 1):
        cc_year_tar = cc_original_dir / f"Euclid_{year}.tar"
        if not cc_year_tar.exists():
            print(f"{year}: missing {cc_year_tar}", flush=True)
            continue

        year_dir = combined_dir / str(year)
        paths = sorted(year_dir.glob(f"euclid_flash_cg_ic_{year}_*.nc"))
        if args.months is not None:
            months = {f"{month:02d}" for month in args.months}
            paths = [path for path in paths if path.stem.rsplit("_", 1)[-1] in months]
        if not paths:
            print(f"{year}: no combined monthly files found in {year_dir}", flush=True)
            continue

        for combined_path in paths:
            cg_name = combined_path.name.replace("euclid_flash_cg_ic_", "euclid_flash_cg_")
            cc_name = combined_path.name.replace("euclid_flash_cg_ic_", "euclid_flash_cc_")
            cg_output_path = cg_output_dir / str(year) / cg_name if "cg" in args.products else None
            cc_output_path = cc_output_dir / str(year) / cc_name if "cc" in args.products else None

            if cg_output_path is not None and cg_output_path.exists() and not args.overwrite:
                print(f"skip {cg_output_path}", flush=True)
                cg_output_path = None
            if cc_output_path is not None and cc_output_path.exists() and not args.overwrite:
                print(f"skip {cc_output_path}", flush=True)
                cc_output_path = None
            if cg_output_path is None and cc_output_path is None:
                continue

            targets = [str(path) for path in (cc_output_path, cg_output_path) if path is not None]
            print(f"writing {', '.join(targets)}", flush=True)
            write_month(combined_path, cc_year_tar, cg_output_path, cc_output_path, args.clip_negative)


if __name__ == "__main__":
    main()
