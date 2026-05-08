"""Download NOAA OISST v2.1 daily SST from CoastWatch ERDDAP.

Downloads are performed year-by-year to keep individual file sizes manageable
and to allow resuming partial downloads.  The default spatial crop is the
Coral Sea region defined in PLAN.md §4.1.

ERDDAP dataset
--------------
Server  : https://coastwatch.pfeg.noaa.gov/erddap/
Dataset : ncdcOisst21Agg_LonPM180
Variable: sst   (°C, fill=NaN for land)
Coords  : time, altitude (=0), latitude, longitude

Usage (as a library)
--------------------
>>> from sst_forecasting.data.download import download_oisst
>>> paths = download_oisst("data/raw", start_year=1981, end_year=2000)

Usage (CLI helper, see scripts/download_oisst.py)
-------------------------------------------------
$ python scripts/download_oisst.py --output-dir data/raw
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import requests
from tqdm import tqdm

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ERDDAP constants
# ---------------------------------------------------------------------------

ERDDAP_BASE = "https://coastwatch.pfeg.noaa.gov/erddap/griddap"
OISST_DATASET_ID = "ncdcOisst21Agg_LonPM180"
OISST_VARIABLE = "sst"

# OISST record starts 1981-09-01
OISST_RECORD_START = "1981-09-01"

# Default spatial crop: Coral Sea  [140°E–170°E] × [25°S–5°S]
DEFAULT_LAT_MIN: float = -25.0
DEFAULT_LAT_MAX: float = -5.0
DEFAULT_LON_MIN: float = 140.0
DEFAULT_LON_MAX: float = 170.0


# ---------------------------------------------------------------------------
# URL builder
# ---------------------------------------------------------------------------


def build_erddap_url(
    dataset_id: str,
    variable: str,
    start: str,
    end: str,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    zlev: float = 0.0,
) -> str:
    """Build an ERDDAP gridDAP NetCDF4 download URL for a spatio-temporal subset.

    Parameters
    ----------
    dataset_id : ERDDAP dataset identifier.
    variable   : Variable name (e.g. ``"sst"``).
    start, end : ISO date strings ``"YYYY-MM-DD"``; ERDDAP appends ``T12:00:00Z``.
    lat_min / lat_max / lon_min / lon_max : Bounding box in decimal degrees.
    zlev       : Altitude level (0.0 for OISST).

    Returns
    -------
    str : Complete download URL.
    """
    return (
        f"{ERDDAP_BASE}/{dataset_id}.nc"
        f"?{variable}"
        f"[({start}T12:00:00Z):1:({end}T12:00:00Z)]"
        f"[({zlev:.1f}):1:({zlev:.1f})]"
        f"[({lat_min:.2f}):1:({lat_max:.2f})]"
        f"[({lon_min:.2f}):1:({lon_max:.2f})]"
    )


# ---------------------------------------------------------------------------
# Downloader
# ---------------------------------------------------------------------------


def _download_file(url: str, dest: Path, retries: int = 3) -> None:
    """Stream-download *url* to *dest* with exponential-backoff retry.

    Writes to a ``.tmp`` file first, then renames atomically on success so
    partially-downloaded files are never mistaken for complete ones.

    Raises
    ------
    RuntimeError if all retry attempts fail.
    """
    for attempt in range(1, retries + 1):
        try:
            with requests.get(url, stream=True, timeout=300) as resp:
                resp.raise_for_status()
                total = int(resp.headers.get("content-length", 0))
                dest.parent.mkdir(parents=True, exist_ok=True)
                tmp = dest.with_suffix(".tmp")
                with open(tmp, "wb") as fh, tqdm(
                    total=total,
                    unit="B",
                    unit_scale=True,
                    desc=dest.name,
                    leave=False,
                ) as bar:
                    for chunk in resp.iter_content(chunk_size=1 << 20):
                        fh.write(chunk)
                        bar.update(len(chunk))
                tmp.rename(dest)
                log.debug("Saved %s", dest)
                return
        except (requests.RequestException, OSError) as exc:
            if attempt == retries:
                raise RuntimeError(
                    f"Download failed after {retries} attempts: {exc}"
                ) from exc
            wait = 2**attempt
            log.warning("Attempt %d/%d failed (%s); retrying in %ds…", attempt, retries, exc, wait)
            time.sleep(wait)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def download_oisst(
    output_dir: str | Path,
    *,
    start_year: int = 1981,
    end_year: int = 2000,
    lat_min: float = DEFAULT_LAT_MIN,
    lat_max: float = DEFAULT_LAT_MAX,
    lon_min: float = DEFAULT_LON_MIN,
    lon_max: float = DEFAULT_LON_MAX,
    variable: str = OISST_VARIABLE,
    dataset_id: str = OISST_DATASET_ID,
    skip_existing: bool = True,
    retries: int = 3,
) -> list[Path]:
    """Download NOAA OISST v2.1 year-by-year into *output_dir*.

    Each year is saved as ``oisst_v21_{year}.nc``.  If *skip_existing* is
    True (default) already-present files are not re-downloaded, making it safe
    to re-run after interruptions.

    Parameters
    ----------
    output_dir   : Directory to write NetCDF files.
    start_year   : First year to download (≥1981).
    end_year     : Last year to download (inclusive, ≤2000 for the benchmark).
    lat_min / lat_max / lon_min / lon_max : Spatial bounding box.
    variable     : ERDDAP variable name.
    dataset_id   : ERDDAP dataset identifier.
    skip_existing: Skip files that already exist on disk.
    retries      : Number of HTTP retry attempts per year.

    Returns
    -------
    list[Path] : Sorted list of paths to downloaded (or pre-existing) files.
    """
    if start_year < 1981:
        raise ValueError("OISST v2.1 starts 1981-09-01; start_year must be ≥ 1981.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths: list[Path] = []
    for year in range(start_year, end_year + 1):
        # OISST record starts 1981-09-01, not 1981-01-01
        start = OISST_RECORD_START if year == 1981 else f"{year}-01-01"
        end   = f"{year}-12-31"

        fname = output_dir / f"oisst_v21_{year}.nc"
        if skip_existing and fname.exists():
            log.info("%s: already exists, skipping.", fname.name)
            paths.append(fname)
            continue

        url = build_erddap_url(
            dataset_id, variable, start, end, lat_min, lat_max, lon_min, lon_max
        )
        log.info("Downloading %d → %s …", year, fname.name)
        log.debug("URL: %s", url)
        _download_file(url, fname, retries=retries)
        paths.append(fname)

    log.info("Done. %d files in %s.", len(paths), output_dir)
    return sorted(paths)
