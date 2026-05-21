import numpy as np


def create_ocean_mask(sst):
    """
    Create a mask of ocean points that are valid across all time steps.

    Parameters
    ----------
    sst:
        xarray DataArray with shape:
            time x lat x lon

    Returns
    -------
    ocean_mask:
        Boolean xarray DataArray with shape:
            lat x lon
    """

    return sst.notnull().all(dim="time")


def flatten_to_ocean_points(sst, ocean_mask):
    """
    Flatten SST maps and keep only valid ocean points.

    Parameters
    ----------
    sst:
        xarray DataArray with shape:
            time x lat x lon

    ocean_mask:
        Boolean xarray DataArray with shape:
            lat x lon

    Returns
    -------
    sst_ocean:
        xarray DataArray with shape:
            time x ocean_points
    """

    sst_flat = sst.stack(points=("lat", "lon"))
    ocean_mask_flat = ocean_mask.stack(points=("lat", "lon"))

    sst_ocean = sst_flat[:, ocean_mask_flat]

    return sst_ocean


def standardise_sst(sst_ocean):
    """
    Convert ocean-only SST data to NumPy and standardise it.

    Returns
    -------
    sst_scaled:
        Standardised SST values with shape:
            time x ocean_points

    sst_mean:
        Mean SST used for standardisation.

    sst_std:
        Standard deviation used for standardisation.
    """

    sst_values = sst_ocean.values.astype("float32")

    sst_mean = float(sst_values.mean())
    sst_std = float(sst_values.std())

    sst_scaled = (sst_values - sst_mean) / sst_std

    return sst_scaled, sst_mean, sst_std