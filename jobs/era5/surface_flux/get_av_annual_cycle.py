import os
import logging
import sys
import f90nml
import pandas as pd
import xarray as xr
import inspect
from typing import Literal, List, Optional, Union
from isca_tools.era5.get_jasmin_era5 import Find_era5
from isca_tools.utils.base import get_memory_usage
# Set up logging configuration to output to console and don't output milliseconds, and stdout so saved to out file
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S',
                    stream=sys.stdout)

def create_var_per_job_nml(input_file_path: str, exist_ok: Optional[bool] = None) -> List:
    """
    A `nml` file is created for each `var` with name same as `input_file_path` but with variable
    as a suffix e.g. `input_nml` becomes `input_t.nml` with `input_info['script_info']['var']` set to
    `t`.

    Args:
        input_file_path: Path to `nml` file for experiment.
        exist_ok: If `True`, do not raise exception if any file to be created already exists.
            If `False`, will overwrite it. If `None` leaves the existing file unchanged.

    Returns:
        List of paths to nml files created e.g. `['/Users/.../input_t.nml', '/Users/.../input_q.nml']`
    """
    input_info = f90nml.read(input_file_path)
    vars_all = input_info['script_info']['var']
    out_file_names = []
    for i, var in enumerate(vars_all):
        out_file_names.append(input_file_path.replace('.nml', f'_{var}.nml'))
        if os.path.exists(out_file_names[-1]):
            if exist_ok is None:
                print(f'Variable {i+1}/{len(vars_all)} | {var}: Output nml file already exists. Leaving unchanged')
                continue

        input_info['script_info']['var'] = var
        # Set output directory to be in directory corresponding to the variable
        input_info['script_info']['out_path'] = os.path.join(input_info['script_info']['out_dir'], f"{var}.nc")
        input_info.write(out_file_names[-1], force=exist_ok)
        print(f'Variable {i+1}/{len(vars_all)} | {var}: Output nml file created')
    return out_file_names


def mean_over_years_by_day(ds: xr.Dataset, time_dim: str="time") -> xr.Dataset:
    """Compute a 365-day daily climatology by averaging each calendar day over years.

    This function is intended for daily data such as ERA5 that span multiple years.
    It drops February 29 before averaging, then groups by calendar day using
    month-day labels (for example, "01-01", "07-15"). This avoids the mismatch
    that occurs when grouping by day-of-year across leap and non-leap years.

    Args:
        ds: An xarray DataArray or Dataset with a time dimension.
        time_dim: Name of the time dimension. Defaults to "time".

    Returns:
        An xarray DataArray or Dataset containing a 365-day climatology with a
        synthetic non-leap-year time coordinate running from 2001-01-01 to
        2001-12-31.

    Raises:
        ValueError: If `time_dim` is not present in `ds`.
    """
    if time_dim not in ds.dims:
        raise ValueError(f"{time_dim!r} is not a dimension in the input object.")

    time = ds[time_dim]
    is_feb29 = (time.dt.month == 2) & (time.dt.day == 29)

    ds_no_leap = ds.where(~is_feb29, drop=True)
    day_key = ds_no_leap[time_dim].dt.strftime("%m-%d")

    climatology = ds_no_leap.groupby(day_key).mean(time_dim)
    climatology = climatology.rename({"strftime": time_dim})

    labels = [str(x) for x in climatology[time_dim].values]
    target_time = pd.DatetimeIndex(pd.to_datetime([f"2001-{x}" for x in labels]))

    climatology = climatology.assign_coords({time_dim: target_time})
    return climatology


def process_var(out_path: str, var: Union[List[str], str], year_start: int, year_end: int,
                stat: Literal['mean', 'min', 'max'] = 'mean', stat_freq: str = '1D', level: Optional[int] = None,
                lon_min: Optional[float] = None, lon_max: Optional[float] = None, lat_min: Optional[float] = None,
                lat_max: Optional[float] = None, smooth_n_days: int = 50, smooth_center: bool = True,
                smooth_min_periods: int = 10, model: Literal["oper", "enda"] = "oper", load_all_at_start: bool = False,
                exist_ok: Optional[bool] = None, complevel: int = 4, single_month: bool = False,
                logger: Optional[logging.Logger] = None):
    """Process ERA5 hourly data into a smoothed annual-mean daily climatology.

    This function loads an ERA5 variable from hourly data over a specified time
    range, aggregates it to a lower temporal frequency (typically daily) using a
    chosen statistic, applies a rolling mean in time, computes the mean seasonal
    cycle across years, and saves the result to a NetCDF file.

    The function can optionally subset the data in pressure level and geographic
    extent, control memory usage by delaying full loading, and skip or overwrite
    existing output files.

    Args:
        out_path: Path to the output NetCDF file.
        var: ERA5 variable name, or list-like variable selector understood by
            ``Find_era5()``.
        year_start: First year to include.
        year_end: Last year to include.
        stat: Statistic applied during temporal resampling. Must be one of
            ``"mean"``, ``"min"``, or ``"max"``.
        stat_freq: Resampling frequency passed to ``xarray.DataArray.resample``;
            typically ``"1D"`` for daily aggregation.
        level: Vertical level to select. If negative, it is interpreted relative
            to the model top using ``138 + level``.
        lon_min: Minimum longitude for spatial subsetting.
            If not provided, will consider the entire globe.
        lon_max: Maximum longitude for spatial subsetting.
            If not provided, will consider the entire globe.
        lat_min: Minimum latitude for spatial subsetting.
            If not provided, will consider the entire globe.
        lat_max: Maximum latitude for spatial subsetting.
            If not provided, will consider the entire globe.
        smooth_n_days: Rolling window size, in time steps after resampling, used
            for temporal smoothing.
        smooth_center: Whether the rolling window is centered.
        smooth_min_periods: Minimum number of valid samples required within the
            rolling window.
        model: ERA5 product to use. Must be either ``"oper"`` or ``"enda"``.
        load_all_at_start: If True, eagerly loads the hourly data before
            processing. If False, processing remains lazy until the final stage.
        exist_ok: Behaviour if ``out_path`` already exists. If None, skip
            processing and return. If True, overwrite. If False, raise an error.
        complevel: NetCDF compression level used when saving.
        single_month: If True, only processes January of ``year_start`` as a
            quick test.
        logger: Optional logger used for progress and memory-usage messages.

    Returns:
        None. The processed data are written to ``out_path``.

    Raises:
        ValueError: If only one longitude bound is provided.
        ValueError: If only one latitude bound is provided.
        ValueError: If ``out_path`` already exists and ``exist_ok`` is False.

    Notes:
        The processing pipeline is:

        1. Load hourly ERA5 data.
        2. Resample in time using ``stat`` at frequency ``stat_freq``.
        3. Rechunk along time to support rolling operations efficiently.
        4. Apply a rolling mean over time.
        5. Compute the mean annual cycle with ``mean_over_years_by_day``.
        6. Cast to ``float32`` and save as compressed NetCDF.

        If ``load_all_at_start`` is False, the function delays full realization
        of the computation until after the annual-mean smoothed field has been
        produced, which can reduce memory pressure.

    """
    if os.path.exists(out_path):
        if exist_ok is None:
            if logger is not None:
                logger.info(f'Output file already exists, skipping.')
            # If None, and file exist skip
            return None
        elif exist_ok:
            if logger is not None:
                logger.info(f'Output file already exists, will be overwritten')
        else:
            raise ValueError(f'Output file {out_path} already exists')
    if logger is not None:
        logger.info(f'{var} - Start | Memory used {get_memory_usage() / 1000:.1f}GB')

    if (level is not None) and (level < 0):
        level = 138 + level     # highest level is 137
    if (lon_min is None) and (lon_max is None):
        lon_range = None
    else:
        if lon_max is None:
            raise ValueError('lon_max is required')
        if lon_min is None:
            raise ValueError('lon_min is required')
        lon_range = slice(lon_min, lon_max)

    if (lat_min is None) and (lat_max is None):
        lat_range = None
    else:
        if lat_max is None:
            raise ValueError('lat_max is required')
        if lat_min is None:
            raise ValueError('lat_min is required')
        lat_range = slice(lat_min, lat_max)

    time_start = f"{year_start}-01-01"
    if single_month:
        time_end = f"{year_start}-02-01"  # hack for quick test
    else:
        time_end = f"{year_end + 1}-01-01"  # go up to the day just before 1st day of next year

    era5 = Find_era5()
    var_hourly = era5[var, time_start:time_end, level, lon_range, lat_range, model]
    if logger is not None:
        logger.info(f'Lazy loaded | Memory used {get_memory_usage() / 1000:.1f}GB')

    if load_all_at_start:
        var_hourly.load()
        if logger is not None:
            logger.info(f'Fully loaded | Memory used {get_memory_usage() / 1000:.1f}GB')

    # Process hourly data to convert to daily
    var_daily = getattr(var_hourly.resample(time=stat_freq), stat)(dim='time')
    if logger is not None:
        logger.info(f'Daily data obtained | Memory used {get_memory_usage() / 1000:.1f}GB')

    var_daily = var_daily.chunk({"time": min(var_daily.time.size, 2*smooth_n_days)})  # rechunk so clusters larger than smooth window
    if logger is not None:
        logger.info(f'Rechunked in time dimension | Memory used {get_memory_usage() / 1000:.1f}GB')

    # Obtain annual mean smoothed data
    var_av = var_daily.rolling(time=int(smooth_n_days), center=smooth_center, min_periods=smooth_min_periods).mean()
    if logger is not None:
        logger.info(f'Smoothed data | Memory used {get_memory_usage() / 1000:.1f}GB')
    var_av = mean_over_years_by_day(var_av)
    if logger is not None:
        logger.info(f'Obtained annual mean data | Memory used {get_memory_usage() / 1000:.1f}GB')
    if not load_all_at_start:
        var_av.load()
        if logger is not None:
            logger.info(f'Fully loaded processed data | Memory used {get_memory_usage() / 1000:.1f}GB')

    # Save data to file
    encoding = {var: {'zlib': True, 'complevel': complevel} for var in var_av.data_vars}
    var_av = var_av.astype('float32')  # save as float32 to reduce memory
    var_av.to_netcdf(out_path, encoding=encoding)
    if logger is not None:
        logger.info(f'Saved data to {out_path} | Memory used {get_memory_usage() / 1000:.1f}GB')

    return None


def main(input_file_path: str):
    # Run processing for all required years
    input_info = f90nml.read(input_file_path)
    script_info = input_info['script_info']
    func_arg_names = inspect.signature(process_var).parameters
    func_args = {k: v for k, v in script_info.items() if k in func_arg_names}
    logger = logging.getLogger()  # for printing to console time info
    process_var(**func_args, logger=logger)


if __name__ == '__main__':
    main(sys.argv[1])
