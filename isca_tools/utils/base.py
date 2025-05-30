import xarray as xr
try:
    from xarray.core.weighted import DataArrayWeighted
except ModuleNotFoundError:
    from xarray.computation.weighted import DataArrayWeighted      # Version issue as to where DataArrayWeighted is
import numpy as np
from typing import Optional, Union, List, Callable
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


def split_list_max_n(lst: Union[List, np.ndarray], n: int) -> List:
    """
    Split `lst` into balanced chunks with at most `n` elements each.

    Args:
        lst: List to split.
        n: Maximum number of elements in each chunk of `lst`.

    Returns:
        List of `n` chunks of `lst`
    """
    k = int(np.ceil(len(lst) / n))  # Number of chunks needed
    avg = int(np.ceil(len(lst) / k))
    return [lst[i * avg : (i + 1) * avg] for i in range(k)]


def parse_int_list(value: Optional[Union[str, int, List]], format_func: Callable = lambda x: str(x)) -> List:
    """
    Takes in a value or list of values e.g. `[1, 2, 3]` and converts it into a list of strings where
    each string has the format given by `format_func` e.g. `['1', '2', '3']` for the default case.

    If `value='x:y'`, will return all integers between `x` and `y` inclusive.

    Args:
        value: Variable to convert into list of strings
        format_func: How to format each integer within the string.

    Returns:
        List, where each integer in `value` is converted using `format_func`.
    """
    if isinstance(value, list):
        pass
    elif isinstance(value, int):
        value = [value]
    elif isinstance(value, str):
        value = value.strip()       # remove blank space
        if ':' in value:
            # If '1979:2023' returns all integers from 1979 to 2023
            start, end = map(int, value.split(':'))
            value = list(range(start, end + 1))
        else:
            value = [int(value)]
    else:
        raise ValueError(f"Unsupported format: {value}")
    return [format_func(i) for i in value]
