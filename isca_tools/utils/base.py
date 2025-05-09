import xarray as xr
try:
    from xarray.core.weighted import DataArrayWeighted
except ModuleNotFoundError:
    from xarray.computation.weighted import DataArrayWeighted      # Version issue as to where DataArrayWeighted is
import numpy as np
from typing import Optional
import psutil
import os
import numbers


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


def print_ds_var_list(ds: xr.Dataset, phrase: Optional[str] = None) -> None:
    """
    Prints all variables in `ds` which contain `phrase` in the variable name or variable `long_name`.

    Args:
        ds: Dataset to investigate variables of.
        phrase: Key phrase to search for in variable info.

    """
    # All the exceptions to deal with case when var does not have a long_name
    var_list = list(ds.keys())
    if phrase is None:
        for var in var_list:
            try:
                print(f'{var}: {ds[var].long_name}')
            except AttributeError:
                print(f'{var}')
    else:
        for var in var_list:
            if phrase.lower() in var.lower():
                try:
                    print(f'{var}: {ds[var].long_name}')
                except AttributeError:
                    print(f'{var}')
                continue
            try:
                if phrase.lower() in ds[var].long_name.lower():
                    print(f'{var}: {ds[var].long_name}')
                    continue
            except AttributeError:
                continue
    return None


# Memory usage function
def get_memory_usage() -> float:
    """
    Get current process’s memory  in MB.

    Returns:
        mem_mb: Memory usage in MB
    """
    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / (1024 * 1024)
    return mem_mb

def len_safe(x) -> int:
    """
    Return length of `x` which can have multiple values, or just be a number.

    Args:
        x: Variable to return length of.

    Returns:
        Number of elements in `x`.
    """
    if isinstance(x, numbers.Number):
        return 1
    try:
        return len(x)
    except TypeError:
        raise TypeError(f"Unsupported type with no length: {type(x)}")