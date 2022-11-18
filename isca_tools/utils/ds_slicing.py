from netCDF4 import Dataset
from typing import List, Optional
import numpy as np
import xarray as xr


def annual_time_slice(ds: Dataset, include_months: Optional[List[int]] = None, include_days: Optional[List[int]] = None,
                      month_days: int = 30, year_months: int = 12, first_day: int = 1) -> Dataset:
    """
    Slices dataset `ds` so only contains data corresponding to given months or days for each year.

    Examples:
        `ds_summer = annual_time_slice(ds, [6, 7, 8])`</br>
            This return a dataset containing data corresponding to
            June, July and August in each year i.e. only northern hemisphere summer.

        `ds_day = annual_time_slice(ds, include_days = [36])`</br>
            This will return a dataset containing data corresponding to the 36th day of the year for each year
            of the simulation. The length of the time dimension will be the number of years of the simulation.

    Args:
        ds: Dataset for particular experiment.
        include_months: `int [n_months]`</br>
            Months to keep (1 refers to January).
        include_days: `int [n_days]`</br>
            Days to keep (1 refers to 1st January).
            If `include_months` is provided, this will be ignored.
        month_days: Number of days in each month used for the simulation.
            This depends on the `calendar` option in the `main_nml` namelist.
        year_months: Number of months in a year. I think this is always likely to be `12`.
        first_day: Day used in starting date for the simulation.
            It is equal to the third number in the `current_date` option in the `main_nml` namelist.
            `1` refers to January 1st.

    Returns:
        Dataset only including certain months/days for each year.

    """
    year_days = year_months * month_days    # number of days in a year
    # ceil to deal with daily output data when 1st day is 0.5, 2nd day is 1.5 etc
    ds_days = (first_day - 1 + np.ceil(ds.time)) % year_days  # day in a given year that each value in ds.time refers to
    if include_months is not None:
        include_days = [np.arange(1, month_days+1) + month_days * (month-1) for month in include_months]
        include_days = np.concatenate(include_days)
    elif include_days is None:
        raise ValueError("Either include_months or include_days need to be specified but both are None.")
    return ds.where(ds_days.isin(include_days), drop=True)


def lat_lon_slice(ds: Dataset, lat: np.ndarray, lon: np.ndarray) -> Dataset:
    """
    Returns dataset, `ds`, keeping only data at the coordinate indicated by `(lat[i], lon[i])` for all `i`.

    If `ds` contained `t_surf` then the returned dataset would contain `t_surf` as a function of the variables
    `time` and `location` with each value of `location` corresponding to a specific `(lat, lon)` combination.
    For the original `ds`, it would be a function of `time`, `lat` and `lon`.

    This is inspired from a
    [stack overflow post](https://stackoverflow.com/questions/72179103/xarray-select-the-data-at-specific-x-and-y-coordinates).

    Args:
        ds: Dataset for particular experiment.
        lat: `float [n_coords]`
            Latitude coordinates to keep.
        lon: `float [n_coords]`
            Longitude coordinates to keep.

    Returns:
        Dataset only including the desired coordinates.
    """
    # To get dataset at specific coordinates not all perutations, turn to xarray first
    lat_xr = xr.DataArray(lat, dims=['location'])
    lon_xr = xr.DataArray(lon, dims=['location'])
    return ds.sel(lat=lat_xr, lon=lon_xr, method="nearest")
