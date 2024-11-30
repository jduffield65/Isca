from typing import List, Optional, Union
import numpy as np
import xarray as xr
from xarray import Dataset, DataArray
from .base import area_weighting


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
    year_days = year_months * month_days  # number of days in a year
    # ceil to deal with daily output data when 1st day is 0.5, 2nd day is 1.5 etc
    ds_days = (first_day - 1 + np.ceil(ds.time)) % year_days  # day in a given year that each value in ds.time refers to
    ds_days[ds_days == 0] = year_days  # correction so last day of year has index 360 not 0
    ds_days_step = float(ds_days[1] - ds_days[0])
    if include_months is not None:
        include_days = [np.arange(1, month_days + 1) + month_days * (month - 1) for month in include_months]
        include_days = np.concatenate(include_days)
    elif include_days is None:
        raise ValueError("Either include_months or include_days need to be specified but both are None.")

    # account for ds data being monthly or daily
    include_days = include_days[(include_days - float(ds_days.min())) % ds_days_step == 0]

    if not np.isin(include_days, ds_days).all():
        raise ValueError("Not all months / days provided in include_months / include_days are valid")
    return ds.where(ds_days.isin(include_days), drop=True)


def annual_mean(ds: Dataset, n_year_days: int = 360, first_day: int = 1) -> Dataset:
    """
    Returns dataset `ds` with variables being the average over all years i.e. time dimension of `ds` will now be from
    0.5 to 359.5 if a year has 360 days.

    Args:
        ds: Dataset for particular experiment.
        n_year_days: Number of days in a year used for the simulation.
            This depends on the `calendar` option in the `main_nml` namelist.
        first_day: Day used in starting date for the simulation.
            It is equal to the third number in the `current_date` option in the `main_nml` namelist.
            `1` refers to January 1st.

    Returns:
        Dataset containing the annual average of each variable

    """
    ds_days = (first_day - 1 + ds.time) % n_year_days  # day in a given year that each value in ds.time refers to
    return ds.groupby(ds_days).mean(dim='time')


def anom_from_annual_mean(ds: Dataset, combine_lon: bool = False, n_year_days: int = 360,
                          first_day: int = 1) -> Dataset:
    """
    For each lat, lon and pressure; this computes the annual mean of each variable. It then subtracts it
    from the initial dataset, to give the anomaly relative to the annual mean value.

    Args:
        ds: Dataset for particular experiment.
        combine_lon: If `True` will be anomaly with respect to zonal annual mean, otherwise will just
            be with respect to annual mean.
        n_year_days: Number of days in a year used for the simulation.
            This depends on the `calendar` option in the `main_nml` namelist.
        first_day: Day used in starting date for the simulation.
            It is equal to the third number in the `current_date` option in the `main_nml` namelist.
            `1` refers to January 1st.

    Returns:
        Dataset with same `time` variable as `ds`, containing the annomaly relative to the
        annual average.
    """
    if combine_lon:
        ds_annual_mean = annual_mean(ds.mean(dim='lon'), n_year_days, first_day)
    else:
        ds_annual_mean = annual_mean(ds, n_year_days, first_day)
    ds_annual_mean = ds_annual_mean.rename({'time': 'day_of_year'})  # change coordinate to day of year
    # make it integer starting at 0
    ds_annual_mean = ds_annual_mean.assign_coords(day_of_year=(ds_annual_mean.day_of_year -
                                                               ds_annual_mean.day_of_year.min()).astype(int))
    ds['day_of_year'] = (ds.time % n_year_days - (ds.time % n_year_days).min()).astype(int)
    return ds.groupby('day_of_year') - ds_annual_mean


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


def area_weight_mean_lat(ds: Dataset) -> Dataset:
    """
    For all variables in `ds`, an area weighted mean is taken over all latitudes in the dataset.

    Args:
        ds: Dataset for particular experiment.

    Returns:
        Dataset containing averaged variables with no latitude dependence.
    """
    var_averaged = []
    for var in ds.keys():
        if 'lat' in list(ds[var].coords):
            ds[var] = area_weighting(ds[var]).mean(dim='lat')
            var_averaged += [var]
    print(f"Variables Averaged: {var_averaged}")
    return ds


def lat_lon_rolling(ds: Union[Dataset, DataArray], window_lat: int, window_lon: int) -> Union[Dataset, DataArray]:
    """
    This creates a rolling averaged version of the dataset or data-array in the spatial dimension.
    Returned data will have first `np.ceil((window_lat-1)/2)` and last `np.floor((window_lat-1)/2)`
    values as `nan` in latitude dimension.
    The averaging also does not take account of area weighting in latitude dimension.

    Args:
        ds: Dataset or DataArray to find rolling mean of.
        window_lat: Size of window for rolling average in latitude dimension [number of grid points]
        window_lon: Size of window for rolling average in longitude dimension [number of grid points].

    Returns:
        Rolling averaged dataset or DataArray.

    """
    ds_roll = ds.pad(lon=window_lon, mode='wrap')       # first pad in lon so wraps around when doing rolling mean
    ds_roll = ds_roll.rolling({'lon': window_lon, 'lat': window_lat}, center=True).mean()
    return ds_roll.isel(lon=slice(window_lon, -window_lon))     # remove the padded longitude values


def time_rolling(ds: Union[Dataset, DataArray], window_time: int, wrap: bool = True) -> Union[Dataset, DataArray]:
    """
    This creates a rolling averaged version of the dataset or data-array in the time dimension. Useful for when
    have annual average dataset.

    Args:
        ds: Dataset or DataArray to find rolling mean of.
        window_time: Size of window for rolling average in time dimension [number of time units e.g. days]
        wrap: If first time comes immediately after last time i.e. for annual mean data

    Returns:
        Rolling averaged dataset or DataArray.
    """
    if wrap:
        ds_roll = ds.pad(time=window_time, mode='wrap')  # first pad in time so wraps around when doing rolling mean
        ds_roll = ds_roll.rolling(time=window_time, center=True).mean()
        return ds_roll.isel(time=slice(window_time, -window_time))  # remove the padded time values
    else:
        return ds.rolling(time=window_time, center=True).mean()
