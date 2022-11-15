import xarray as xr
from xarray.core.weighted import DataArrayWeighted
import numpy as np


def area_weighting(var: xr.DataArray) -> DataArrayWeighted:
    """
    Apply area weighting to the variable `var` using the `cosine` of latitude: $\cos (\phi)$.

    Args:
        var: Variable to weight e.g. `ds.t_surf` to weight the surface temperature, where
            `ds` is the dataset for the experiment which contains all variables.

    Returns:
        Area weighted version of `var`.
    """
    weights = np.cos(np.deg2rad(var.lat))
    weights.name = "weights"
    return var.weighted(weights)
