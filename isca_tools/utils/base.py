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
import time
import logging
import re
import warnings


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


def print_log(text: str, logger:Optional[logging.Logger] = None) -> None:
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
    return [lst[i * avg : (i + 1) * avg] for i in range(k)]


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
        value = value.strip()       # remove blank space
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