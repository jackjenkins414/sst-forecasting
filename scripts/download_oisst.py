"""Download NOAA OISST v2.1 (Coral Sea crop) from CoastWatch ERDDAP.

Usage
-----
    python scripts/download_oisst.py [options]

Options
-------
    --output-dir   PATH   Where to save .nc files  [default: data/raw]
    --start-year   INT    First year to download    [default: 1981]
    --end-year     INT    Last year to download     [default: 2000]
    --lat-min      FLOAT  [default: -25.0]
    --lat-max      FLOAT  [default:  -5.0]
    --lon-min      FLOAT  [default: 140.0]
    --lon-max      FLOAT  [default: 170.0]
    --no-skip             Re-download existing files
    --log-level    LEVEL  Logging verbosity         [default: INFO]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Allow running as a script from the repo root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sst_forecasting.data.download import (
    DEFAULT_LAT_MAX,
    DEFAULT_LAT_MIN,
    DEFAULT_LON_MAX,
    DEFAULT_LON_MIN,
    download_oisst,
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download NOAA OISST v2.1 from CoastWatch ERDDAP.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--output-dir", default="data/raw", metavar="PATH")
    p.add_argument("--start-year", type=int, default=1981)
    p.add_argument("--end-year",   type=int, default=2000)
    p.add_argument("--lat-min",    type=float, default=DEFAULT_LAT_MIN)
    p.add_argument("--lat-max",    type=float, default=DEFAULT_LAT_MAX)
    p.add_argument("--lon-min",    type=float, default=DEFAULT_LON_MIN)
    p.add_argument("--lon-max",    type=float, default=DEFAULT_LON_MAX)
    p.add_argument("--no-skip",    action="store_true",
                   help="Re-download files that already exist.")
    p.add_argument("--log-level",  default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    paths = download_oisst(
        output_dir=args.output_dir,
        start_year=args.start_year,
        end_year=args.end_year,
        lat_min=args.lat_min,
        lat_max=args.lat_max,
        lon_min=args.lon_min,
        lon_max=args.lon_max,
        skip_existing=not args.no_skip,
    )
    print(f"\nDownloaded {len(paths)} files to {args.output_dir}")


if __name__ == "__main__":
    main()
