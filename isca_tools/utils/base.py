import xarray as xr

try:
    from xarray.core.weighted import DataArrayWeighted
except ModuleNotFoundError:
    from xarray.computation.weighted import DataArrayWeighted  # Version issue as to where DataArrayWeighted is
import numpy as np
from typing import Optional, Union, List, Callable, Tuple
import psutil
import os
import numbers
import time
import logging
import re
import warnings
from .constants import g


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

def mass_weighted_vertical_integral(var: xr.DataArray, pressure: xr.DataArray,
                                    lev_dim: str = 'lev', norm: bool=True) -> xr.DataArray:
    """
    Performs the mass-weighted vertical integral $\int \chi dp/g$ of a given variable $\chi$

    E.g. Neelin and Held 1987 equation 2.5.

    Args:
        var: `float [n_lev]`
            Variable to integrate along `lev_dim` dimension.
        pressure: `float [n_lev]`
            Pressure at each model level
        lev_dim: Name of model level dimension along which to integrate.
        norm: If `True`, will normalize by mass of column i.e. becomes a mass weighted vertical average,
            with the same units as `var`.

    Returns:
        Value of integral
    """
    dp = np.abs(pressure.differentiate(lev_dim))  # Pa
    weights = (dp / g)  # mass weights (kg/m²)
    var_int = (var * weights).sum('lev')  # var_units * kg/m²
    if norm:
        var_int = var_int / weights.sum(lev_dim)
    return var_int


def print_log(text: str, logger: Optional[logging.Logger] = None) -> None:
    """
    Quick function to add to log if log exists, otherwise print it.

    Args:
        text: Text to be printed.
        logger:

    Returns:

    """
    logger.info(text) if logger else print(text)


def run_func_loop(func: Callable, max_wait_time: int = 300, wait_interval: int = 20,
                  func_check: Optional[Callable] = None, logger: Optional[logging.Logger] = None):
    """
    Safe way to run a function, such that if hit error, will try again every `wait_interval` seconds up to a
    maximum of `max_wait_time` seconds.

    If `func_check` is given and returns `True` at any point, it will exit the loop without executing `func`.

    Most obvious usage is for creating a directory e.g. `os.makedirs`, especially to a server where connection
    cuts in and out.

    Args:
        func: Function to run. Must have no arguments.
        max_wait_time: Maximum number of seconds to try and run `func`.
        wait_interval: Interval in seconds to wait between running `func`.
        func_check: Function that returns a boolean. If it returns `True` at any point, the loop
            will exit the loop without executing `func`.
        logger: Logger to record information

    Returns:
        Whatever `func` returns.
    """
    i = 0
    j = 0
    success = False
    start_time = time.time()
    output = None
    while not success and (time.time() - start_time) < max_wait_time:
        if func_check is not None:
            if func_check():
                print_log("func_check passed so did not exectute func", logger)
                success = True
                break
        try:
            output = func()
            success = True
        except PermissionError as e:
            i += 1
            if i == 1:
                # Only print on first instance of error
                print_log(f'Permission Error: {e}', logger)
            time.sleep(wait_interval)
        except Exception as e:
            j += 1
            if j == 1:
                # Only print on first instance of error
                print_log(f'Unexpected Error: {e}', logger)
            time.sleep(wait_interval)
    if not success:
        raise ValueError(f"Making output directory - Failed to run function after {max_wait_time} seconds.")
    return output


# Memory usage function
def get_memory_usage() -> float:
    """
    Get current process’s memory in MB.

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
    return [lst[i * avg: (i + 1) * avg] for i in range(k)]


def parse_int_list(value: Union[str, int, List], format_func: Callable = lambda x: str(x),
                   all_values: Optional[List] = None) -> List:
    """
    Takes in a value or list of values e.g. `[1, 2, 3]` and converts it into a list of strings where
    each string has the format given by `format_func` e.g. `['1', '2', '3']` for the default case.

    There are three string options for `value`:
    * `value='x:y'`, will return all integers between `x` and `y` inclusive.
    * `value='firstX'` will return first X values of `all_values`.
    * `value='firstY'` will return first Y values of `all_values`.

    Args:
        value: Variable to convert into list of strings
        format_func: How to format each integer within the string.
        all_values: List of all possible integers, must be provided if `value='firstX'` or `value='firstY'`.

    Returns:
        List, where each integer in `value` is converted using `format_func`.
    """
    if isinstance(value, list):
        pass
    elif isinstance(value, int):
        value = [value]
    elif isinstance(value, str):
        value = value.strip()  # remove blank space
        # Can specify just first or last n years
        if re.search(r'^first(\d+)', value):
            if all_values is None:
                raise ValueError(f'With value={value}, must provide all_values')
            n_req = int(re.search(r'^first(\d+)', value).group(1))
            if n_req > len(all_values):
                warnings.warn(f"Requested {value} but there are only "
                              f"{len(all_values)} available:\n{all_values}")
            value = all_values[:n_req]
        elif re.search(r'^last(\d+)', value):
            if all_values is None:
                raise ValueError(f'With value={value}, must provide all_values')
            n_req = int(re.search(r'^last(\d+)', all_values).group(1))
            if n_req > len(all_values):
                warnings.warn(f"Requested {value} but there are only "
                              f"{len(all_values)} available:\n{all_values}")
            value = all_values[-n_req:]
        elif ':' in value:
            # If '1979:2023' returns all integers from 1979 to 2023
            start, end = map(int, value.split(':'))
            value = list(range(start, end + 1))
        else:
            value = [int(value)]
    else:
        raise ValueError(f"Unsupported format: {value}")
    return [format_func(i) for i in value]


def round_any(x: Union[float, np.ndarray], base: float, round_type: str = 'round') -> Union[float, np.ndarray]:
    """
    Rounds `x` to the nearest multiple of `base` with the rounding done according to `round_type`.

    Args:
        x: Number or array to round.
        base: Rounds `x` to nearest integer multiple of value of `base`.
        round_type: One of the following, indicating how to round `x` -

            - `'round'`
            - `'ceil'`
            - `'float'`

    Returns:
        Rounded version of `x`.

    Example:
        ```
        round_any(3, 5) = 5
        round_any(3, 5, 'floor') = 0
        ```
    """
    if round_type == 'round':
        return base * np.round(x / base)
    elif round_type == 'ceil':
        return base * np.ceil(x / base)
    elif round_type == 'floor':
        return base * np.floor(x / base)
    else:
        raise ValueError(f"round_type specified was {round_type} but it should be one of the following:\n"
                         f"round, ceil, floor")


def has_out_of_range(val: Union[List, Tuple, np.ndarray, float], min_range: float, max_range: float) -> bool:
    """
    Check if any number within `val` is outside the range between `min_range` and `max_range`.

    Args:
        val: Numbers to check
        min_range: Minimum allowed value.
        max_range: Maximum allowed value.

    Returns:
        True if there is a value outside the range between `min_range` and `max_range`.
    """
    # If it's a single number, make it a list
    vals = val if isinstance(val, (list, tuple, np.ndarray)) else [val]
    return any((x < min_range or x > max_range) for x in vals)


def top_n_peaks_ind(
        var: np.ndarray,
        n: int = 1,
        min_ind_spacing: int = 0,
) -> np.ndarray:
    """Return the indices of the N largest values of `var`, such that the indices of these values
     are ≥`min_ind_spacing` apart.

    Args:
        var: 1D array containing variable values. Assumed in an order
        n: Number of peaks to select.
        min_ind_spacing: Minimum index spacing between selected peaks.

    Returns:
        Indices of `n` peak values of `var`.
    """
    # Sort indices by descending value of var
    order = np.argsort(var)[::-1]
    selected_ind = []

    for i in order:
        # Check spacing constraint
        if all(abs(i - s) >= min_ind_spacing for s in selected_ind):
            selected_ind.append(i)
            if len(selected_ind) == n:
                break

    return np.array(selected_ind, dtype=int)


def dp_from_pressure(p: xr.DataArray, dim: str = "lev") -> xr.DataArray:
    """Compute layer pressure thickness Δp, preserving extra dims and coord order.

    Args:
        p: Pressure [Pa], with vertical dimension `dim` (n_lev).
            Can include other dims (e.g., time, lat, lon).
        dim: Name of the vertical coordinate.

    Returns:
        xr.DataArray: Pressure thickness Δp [Pa], same shape and coords as `p`.
    """

    def _dp_1d(p_1d: np.ndarray) -> np.ndarray:
        # ensure increasing order (bottom→top) for calculation
        reversed_flag = p_1d[0] < p_1d[-1]
        if reversed_flag:
            p_1d = p_1d[::-1]

        # edges & dp calculation
        p_edge_mid = 0.5 * (p_1d[:-1] + p_1d[1:])
        p_edge_bot = p_1d[0] + 0.5 * (p_1d[0] - p_1d[1])
        p_edge_top = p_1d[-1] - 0.5 * (p_1d[-2] - p_1d[-1])
        p_edges = np.concatenate([[p_edge_bot], p_edge_mid, [p_edge_top]])
        dp = p_edges[:-1] - p_edges[1:]
        dp = np.abs(dp)  # ensure positive

        # if we reversed order, un-reverse result to match original orientation
        if reversed_flag:
            dp = dp[::-1]
        return dp

    dp = xr.apply_ufunc(
        _dp_1d,
        p,
        input_core_dims=[[dim]],
        output_core_dims=[[dim]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
    )

    dp.name = "dp"
    dp.attrs.update({"long_name": "pressure thickness", "units": "Pa"})
    return dp


def weighted_RMS(
        var: Union[xr.DataArray, np.ndarray],
        weight: Optional[Union[xr.DataArray, np.ndarray]] = None,
        dim: Optional[Union[str, int, List[Union[str, int]]]] = None
) -> Union[xr.DataArray, np.ndarray]:
    """
    Compute (weighted) RMS of a DataArray or numpy array along specified dimension(s).

    Args:
        var: Variable to compute RMS for (shape [...]).
        weight: Weights (same shape as var along `dim`).
            If None, computes unweighted RMS.
        dim: Dimension(s) to reduce over.
            - For xarray: names of dimensions.
            - For numpy: integer axis or list of axes.

    Returns:
        rms: Same type as input, reduced along `dim`.
    """

    # --- Handle dim input uniformly ---
    if isinstance(dim, (str, int)):
        dims = [dim]
    elif dim is None:
        # all dims
        if isinstance(var, xr.DataArray):
            dims = list(var.dims)
        else:
            dims = list(range(var.ndim))
    else:
        dims = dim

    # --- xarray branch ---
    if isinstance(var, xr.DataArray):
        if weight is None:
            rms_sq = (var ** 2).mean(dim=dims)
        else:
            rms_sq = ((var ** 2) * weight).sum(dim=dims) / weight.sum(dim=dims)
        return np.sqrt(rms_sq)

    # --- numpy branch ---
    else:
        if weight is None:
            rms_sq = np.nanmean(var ** 2, axis=tuple(dims))
        else:
            rms_sq = np.nansum((var ** 2) * weight, axis=tuple(dims)) / np.nansum(weight, axis=tuple(dims))
        return np.sqrt(rms_sq)


def insert_to_array(x_values: Union[np.ndarray, xr.DataArray], y_values: Union[np.ndarray, xr.DataArray],
                    x_new: Union[np.ndarray, xr.DataArray, List, float], y_new: Union[np.ndarray, xr.DataArray, List, float]
                    ) -> tuple[Union[np.ndarray, xr.DataArray], Union[np.ndarray, xr.DataArray]]:
    """Insert multiple (x, y) pairs into arrays while preserving the sort order of x (ascending or descending).

    Works for both NumPy arrays and xarray.DataArray objects.

    Args:
        x_values: Array of x-values (must be sorted, ascending or descending).
        y_values: Array of corresponding y-values.
        x_new: New x-values to insert.
        y_new: Corresponding y-values to insert.

    Returns:
        x_updated: `x_values` with `x_new` inserted in correct location.
        y_updated: `y_values` with `y_new` inserted in correct location.
    """
    # Extract data if xarray
    x_is_xr = isinstance(x_values, xr.DataArray)
    y_is_xr = isinstance(y_values, xr.DataArray)

    x_data = x_values.data if x_is_xr else np.asarray(x_values)
    y_data = y_values.data if y_is_xr else np.asarray(y_values)
    x_new = np.atleast_1d(x_new)
    y_new = np.atleast_1d(y_new)

    # Determine if x_values are ascending or descending
    ascending = x_data[0] < x_data[-1]

    # If descending, temporarily flip for insertion logic
    if not ascending:
        x_data = x_data[::-1]
        y_data = y_data[::-1]
        x_new = -x_new
        x_data = -x_data

    # Sort new inputs by x_new
    sort_idx = np.argsort(x_new)
    x_new = x_new[sort_idx]
    y_new = y_new[sort_idx]

    # Find insertion indices and insert
    insert_indices = np.searchsorted(x_data, x_new)
    x_combined = np.insert(x_data, insert_indices, x_new)
    y_combined = np.insert(y_data, insert_indices, y_new)

    # Flip back if descending
    if not ascending:
        x_combined = -x_combined[::-1]
        y_combined = y_combined[::-1]

    # Wrap back into xarray if needed
    if x_is_xr or y_is_xr:
        # Try to preserve dimension and coordinate naming
        dim = x_values.dims[0] if x_is_xr else (y_values.dims[0] if y_is_xr else "dim_0")
        x_combined = xr.DataArray(x_combined, dims=[dim], name=x_values.name if x_is_xr else None)
        y_combined = xr.DataArray(y_combined, dims=[dim], name=y_values.name if y_is_xr else None)

        # Rebuild coordinates if x-values represent coordinate axis
        x_combined = x_combined.assign_coords({dim: x_combined})

    return x_combined, y_combined
