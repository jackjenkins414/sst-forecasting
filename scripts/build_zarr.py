"""Build (or rebuild) the processed Zarr store from raw OISST NetCDF files.

Usage
-----
    python scripts/build_zarr.py [options]

Options
-------
    --raw-dir      PATH   Directory containing oisst_v21_*.nc  [default: data/raw]
    --zarr-path    PATH   Output Zarr store                    [default: data/processed/oisst_coralsea.zarr]
    --overwrite           Delete existing store and rebuild
    --log-level    LEVEL  [default: INFO]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sst_forecasting.data.preprocess import build_zarr_store


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert raw OISST NetCDF files to a Zarr store.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--raw-dir",   default="data/raw", metavar="PATH")
    p.add_argument("--zarr-path", default="data/processed/oisst_coralsea.zarr", metavar="PATH")
    p.add_argument("--overwrite", action="store_true",
                   help="Remove existing Zarr store and rebuild from scratch.")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    zarr_path = build_zarr_store(
        raw_dir=args.raw_dir,
        zarr_path=args.zarr_path,
        overwrite=args.overwrite,
    )
    print(f"\nZarr store ready: {zarr_path}")


if __name__ == "__main__":
    main()
