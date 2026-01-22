from typing import List, Optional, Union
import numpy as np
import xarray as xr
from xarray import Dataset, DataArray
from .base import area_weighting
import warnings


def annual_time_slice(ds: Dataset, include_months: Optional[List[int]] = None, include_days: Optional[List[int]] = None,
                      month_days: int = 30, year_months: int = 12, first_day: int = 1) -> Dataset:
    """
    Slices dataset `ds` so only contains data corresponding to given months or days for each year.

    Examples:
        `ds_summer = annual_time_slice(ds, [6, 7, 8])`</br>
            This return a dataset containing data corresponding to
            June, July, and August in each year i.e. only northern hemisphere summer.

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
    ds_days_step = float(ds_days[1] - ds_days[0])
    ds_days[ds_days == 0] = year_days  # correction so last day of year has index 360 not 0
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
    For each lat, lon, and pressure; this computes the annual mean of each variable. It then subtracts it
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


def lat_lon_coord_slice(ds: Dataset, lat: np.ndarray, lon: np.ndarray) -> Dataset:
    """
    Returns dataset, `ds`, keeping only data at the coordinate indicated by `(lat[i], lon[i])` for all `i`.

    If `ds` contained `t_surf` then the returned dataset would contain `t_surf` as a function of the variables
    `time` and `location` with each value of `location` corresponding to a specific `(lat, lon)` combination.
    For the original `ds`, it would be a function of `time`, `lat` and `lon`.

    This is inspired by a
    [stack overflow post](https://stackoverflow.com/questions/72179103/xarray-select-the-data-at-specific-x-and-y-coordinates).

    Args:
        ds: Dataset for a particular experiment.
        lat: `float [n_coords]`
            Latitude coordinates to keep.
        lon: `float [n_coords]`
            Longitude coordinates to keep.

    Returns:
        Dataset only including the desired coordinates.
    """
    # To get dataset at specific coordinates, not all permutations, turn to xarray first
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
    This creates a rolling-averaged version of the dataset or data-array in the time dimension. Useful for when
    you have an annual average dataset.

    Args:
        ds: Dataset or DataArray to find rolling mean of.
        window_time: Size of window for rolling average in time dimension [number of time units e.g. days]
        wrap: If the first time comes immediately after the last time i.e. for annual mean data

    Returns:
        Rolling averaged dataset or DataArray.
    """
    if wrap:
        ds_roll = ds.pad(time=window_time, mode='wrap')  # first pad in time so wraps around when doing rolling mean
        ds_roll = ds_roll.rolling(time=window_time, center=True).mean()
        return ds_roll.isel(time=slice(window_time, -window_time))  # remove the padded time values
    else:
        return ds.rolling(time=window_time, center=True).mean()


def lat_lon_range_slice(ds: Union[Dataset, DataArray], lat_min: Optional[float] = None,
                        lat_max: Optional[float] = None, lon_min: Optional[float] = None,
                        lon_max: Optional[float] = None):
    """

    Args:
        ds:
        lat_min:
        lat_max:
        lon_min:
        lon_max:

    Returns:

    """
    if (lon_min is None) and lon_max is None:
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

    if lat_range is not None:
        ds = ds.sel(lat=lat_range)
    if lon_range is not None:
        ds = ds.sel(lon=lon_range)
    return ds

def get_time_sample_indices(times_sample: xr.DataArray, times_all: xr.DataArray) -> xr.DataArray:
    """Return indices of `times_sample` in `times_all` for each coordinate.

    Args:
        times_sample: Times for each sample at each location to get index in `time_all` for (sample, lat, lon).
        times_all: All times in a given simulation (time).

    Returns:
        time_index: Indices of `times_sample` within `times_all`, with shape (sample, lat, lon),
                      filled with NaN where times are not found. Hence, output is float to include NaN.
    """

    # Broadcast ds1 times to full grid (sample, lat, lon)
    times_sample_val = times_sample.values  # shape (sample, lat, lon)

    # initialize indices with NaNs
    indices = np.full(times_sample_val.shape, np.nan)

    # Create lookup dict from times_all to indices
    time_to_index = {t: i for i, t in enumerate(times_all.time.values)}

    # Vectorized mapping (but must loop over unique times for efficiency)
    unique_times = np.unique(times_sample_val)
    for t in unique_times:
        if t in time_to_index:
            indices[times_sample_val == t] = time_to_index[t]
    # Return as DataArray
    return xr.DataArray(
        indices,
        dims=times_sample.dims,
        coords={dim: times_sample.coords[dim] for dim in times_sample.dims},
        name="time_index"
    )


def fold_coarsen(ds: xr.Dataset, k_lat: int, k_lon: int, coarsen_dim: str = 'sample') -> xr.Dataset:
    """
    Coarsen the latitude and longitude dimensions by grouping fine-grid cells
    into blocks and folding the sub-grid structure into the sample dimension.

    This transformation preserves the total number of data points by expanding
    the `coarsen_dim` dimension while reducing the spatial resolution. Variables that
    contain the dimensions ``("lat", "lon", coarsen_dim)`` are reshaped; variables
    that do not include all three of these dimensions are returned unchanged.
    Any additional dimensions (such as ``co2``) are left untouched and preserved
    in their original order.

    Latitude and longitude coordinates for the coarsened grid are computed as
    the mean of the original coordinates within each spatial block.

    Args:
        ds: Input dataset containing ``lat``, ``lon``, and
            ``coarsen_dim`` dimensions.
        k_lat: Coarsening factor along the latitude dimension. Must evenly
            divide the length of ``lat``.
        k_lon: Coarsening factor along the longitude dimension. Must evenly
            divide the length of ``lon``.
        coarsen_dim: Name of dimension to add extra data along.

    Returns:
        xr.Dataset: Dataset with coarsened ``lat`` and ``lon`` dimensions and an
            expanded ``coarsen_dim`` dimension. Variables lacking the full set of
            ``("lat", "lon", coarsen_dim)`` dimensions are passed through unchanged.

    Raises:
        ValueError: If ``lat`` or ``lon`` lengths are not divisible by their
            respective coarsening factors.
    """
    L = ds.sizes["lat"]
    M = ds.sizes["lon"]
    S = ds.sizes[coarsen_dim]

    if L % k_lat != 0 or M % k_lon != 0:
        raise ValueError("lat/lon sizes must be divisible by k_lat/k_lon")

    # ----- new coordinates -----
    lat_vals = ds["lat"].values
    lon_vals = ds["lon"].values
    new_lat = lat_vals.reshape(L // k_lat, k_lat).mean(axis=1)
    new_lon = lon_vals.reshape(M // k_lon, k_lon).mean(axis=1)
    new_sample = np.arange(S * k_lat * k_lon)

    out = {}

    for name, da in ds.data_vars.items():
        dims = da.dims

        # If variable does NOT have lat/lon/coarsen_dim -> leave unchanged
        if not set(("lat", "lon", coarsen_dim)).issubset(dims):
            warnings.warn(f"Cannot coarsen variable {name} as only has dimensions:\n{dims}")
            out[name] = da.copy()
            continue

        # Identify extra dims (e.g. 'co2')
        extra_dims = [d for d in dims if d not in ("lat", "lon", coarsen_dim)]

        # New order: keep extra dims first, then lat/lon/sample
        new_order = extra_dims + ["lat", "lon", coarsen_dim]
        da_aligned = da.transpose(*new_order)

        # Shape needed for reshape
        extra_shape = [da.sizes[d] for d in extra_dims]

        # Reshape:
        #   (extra..., lat//k_lat, k_lat, lon//k_lon, k_lon, sample)
        arr = da_aligned.values.reshape(
            *extra_shape,
            L // k_lat, k_lat,
            M // k_lon, k_lon,
            S
        )

        # Move k_lat,k_lon next to sample
        arr = arr.transpose(
            *range(len(extra_shape)),          # extra dims unchanged
            len(extra_shape) + 0,              # coarse lat
            len(extra_shape) + 2,              # coarse lon
            len(extra_shape) + 1,              # k_lat
            len(extra_shape) + 3,              # k_lon
            len(extra_shape) + 4               # sample
        )

        # Fold (k_lat, k_lon, sample) → sample
        arr = arr.reshape(
            *extra_shape,
            L // k_lat,
            M // k_lon,
            S * k_lat * k_lon
        )

        # Coordinates for output
        coords = {d: da.coords[d] for d in extra_dims}  # pass through extra dims
        coords["lat"] = new_lat
        coords["lon"] = new_lon
        coords[coarsen_dim] = new_sample

        out[name] = xr.DataArray(
            arr,
            dims=extra_dims + ["lat", "lon", coarsen_dim],
            coords=coords,
        )

    return xr.Dataset(out)
