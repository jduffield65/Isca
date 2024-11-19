import numpy as np
import xarray as xr
from typing import Union


def get_pressure(ps: Union[np.ndarray, xr.DataArray], p0: float, hya: Union[np.ndarray, xr.DataArray],
                 hyb: Union[np.ndarray, xr.DataArray]) -> Union[np.ndarray, xr.DataArray]:
    """
    Calculates pressure at the hybrid levels, similat to
    [NCAR function](https://www.ncl.ucar.edu/Document/Functions/Built-in/pres_hybrid_ccm.shtml).

    Args:
        ps: `float [n_time x n_lat x n_lon]`</br>
            Array of surface pressures in units of *Pa*.</br>
            `PS` in CESM atmospheric output.
        p0: Surface reference pressure in *Pa*.</br>
            `P0` in CESM atmospheric output.
        hya: `float [n_levels]`</br>
            Hybrid A coefficients.</br>
            `hyam` in CESM atmospheric output.
        hyb: `float [n_levels]`</br>
            Hybrid B coefficients.</br>
            `hybm` in CESM atmospheric output.

    Returns:
        `float [n_time x n_levels x n_lat x n_lon]`
            Pressure at the hybrid levels in *Pa*.
    """
    return hya * p0 + hyb * ps
