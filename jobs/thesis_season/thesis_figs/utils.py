import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from typing import Union, Literal, Optional
from isca_tools.utils import numerical

from isca_tools.thesis.surface_energy_budget import get_temp_extrema_numerical
from isca_tools.utils import area_weighting, annual_mean
import isca_tools.utils.fourier as fourier
from isca_tools.utils.xarray import wrap_with_apply_ufunc

# Plotting info
width = {'one_col': 3.2, 'two_col': 5.5}  # width in inches
# Default parameters
ax_linewidth = plt.rcParams['axes.linewidth']
smooth_n_days = 50  # default smoothing window in days
resample = False    # Don't do resample in polyfit_phase as complicated


def get_annual_zonal_mean(ds, combine_abs_lat=False, lat_name='lat', smooth_n_days=smooth_n_days,
                          smooth_center=True):
    """Compute annual-mean zonal mean, optionally combining ±latitudes.

    This function:
    1) Computes the annual mean via `annual_mean(ds)`,
    2) Takes the zonal mean over longitude,
    3) Optionally averages fields at latitudes with the same absolute value
       (e.g., $(+30^\circ)$ and $(-30^\circ)$) using a groupby on $(|\mathrm{lat}|)$,
    4) Resets the time coordinate to start at 0 (integer years since the first).

    Args:
        ds: An xarray Dataset or DataArray with dimensions including `lon` and
            typically `time` and `lat`.
        combine_abs_lat: If True, combine values at +lat and -lat by averaging
            them together into a single latitude coordinate \(|\mathrm{lat}|\).
            The equator (0) remains unchanged. Defaults to False.
        lat_name: Name of the latitude dimension/coordinate. Defaults to 'lat'.
        smooth_n_days: Optional integer window length for time smoothing (in
            number of time steps, e.g. days). If None or <= 1, no smoothing.
        smooth_center: If True, use a centered window for smoothing.

    Returns:
        An xarray Dataset or DataArray containing the annual-mean zonal mean.
        If `combine_abs_lat` is True, the latitude coordinate will be nonnegative
        and sorted (e.g., 0, 30, 60, ...).

    Raises:
        ValueError: If `combine_abs_lat` is True but `lat_name` is not a
            dimension of the input after zonal averaging.
    """
    if smooth_n_days is not None and smooth_n_days > 1:
        ds = ds.rolling(time=int(smooth_n_days), center=smooth_center).mean()
    ds_av = annual_mean(ds).mean(dim='lon')

    if combine_abs_lat:
        if lat_name not in ds_av.dims:
            raise ValueError(f"Expected latitude dim '{lat_name}' in {ds_av.dims}")

        abs_lat = ds_av[lat_name].astype(float).copy()
        ds_av = (
            ds_av.assign_coords(abs_lat=abs_lat.abs())
            .groupby('abs_lat')
            .mean(dim=lat_name)
            .rename({'abs_lat': lat_name})
            .sortby(lat_name)
        )

    ds_av = ds_av.assign_coords(time=(ds_av.time - ds_av.time.min()).astype(int))
    for key in ds:
        # Get rid of time dimension of variables that dont have time dimension initially
        if 'time' not in ds[key].dims:
            ds_av[key] = ds_av[key].isel(time=0)
    return ds_av


get_fourier_fit_xr = wrap_with_apply_ufunc(fourier.get_fourier_fit, input_core_dims=[['time'], ['time']],
                                           output_core_dims=[['time'], ['harmonic'], ['harmonic']])

fourier_series_xr = wrap_with_apply_ufunc(fourier.fourier_series,
                                          input_core_dims=[['time'], ['harmonic'], ['harmonic']],
                                          output_core_dims=[['time']])

get_temp_extrema_numerical_xr = wrap_with_apply_ufunc(get_temp_extrema_numerical, input_core_dims=[['time'], ['time']],
                                                      output_core_dims=[[], [], [], []])


# Might need to do different version when include fourier coefs, as more outputs
def polyfit_phase_xr(x: np.ndarray, y: np.ndarray,
                     deg: int, time: Optional[np.ndarray] = None, time_start: Optional[float] = None,
                     time_end: Optional[float] = None,
                     deg_phase_calc: int = 10, resample: bool = resample,
                     include_phase: bool = True, fourier_harmonics: Optional[Union[int, np.ndarray]] = None,
                     integ_method: str = 'spline',
                     pad_coefs_phase: bool = False):
    polyfit_phase_wrap = wrap_with_apply_ufunc(numerical.polyfit_phase, input_core_dims=[['time'], ['time']],
                                             output_core_dims=[['deg']])
    var = polyfit_phase_wrap(x, y, deg=deg, time=time, time_start=time_start, time_end=time_end,
                             deg_phase_calc=deg_phase_calc, resample=resample, include_phase=include_phase,
                             fourier_harmonics=fourier_harmonics, integ_method=integ_method,
                             pad_coefs_phase=pad_coefs_phase)
    deg_full = xr.DataArray(['phase', 'cos', 'sin'] + np.arange(deg+1).tolist(), dims="deg", name="deg")
    if fourier_harmonics is None:
        var = var.assign_coords(deg=deg_full[[0, *np.arange(3, deg_full.size)]])
        # Also output the fourier cos and sin coefs but set to zero
        var = var.reindex(deg=deg_full, fill_value=0)
        return var
    else:
        # TODO: will need to modify this so returns in same format as above
        return var


def update_ds_extrema(ds: xr.Dataset, time: xr.DataArray, temp: xr.DataArray, fit_method: str):
    """Update extrema diagnostics for a given fitting method entry in `ds`.

    This function assumes `ds` stores results indexed by a leading `fit_method`
    dimension (e.g., different fitting approaches). It locates the index
    corresponding to `fit_method`, computes temperature extrema diagnostics from
    the provided time series (after removing the mean), and writes those values
    into the appropriate slice of `ds`.

    Args:
        ds: Dataset containing an ordered leading dimension called `fit_method`
            and variables `time_min`, `time_max`, `amp_min`, `amp_max` indexed
            along that dimension.
        time: Time coordinate (1D) corresponding to `temp`.
        temp: Temperature-like DataArray varying along `time`.
        fit_method: Label identifying which entry along `ds.fit_method` to
            update.

    Returns:
        The same Dataset instance with `time_min`, `time_max`, `amp_min`,
        and `amp_max` updated for the selected `fit_method`.

    Raises:
        ValueError: If the first dimension of `ds` is not `fit_method`.
        ValueError: If `fit_method` is not found in `ds.fit_method` (via the
            index lookup).
    """
    # TODO: use update_dim_slice here
    first_dim = next(iter(ds.dims))
    if first_dim != 'fit_method':
        raise ValueError(f'First dimension is "{first_dim}" but should be "fit_method"')
    ind = int(np.where(ds.fit_method == fit_method)[0])
    ds['time_min'].values[ind], ds['time_max'].values[ind], \
        ds['amp_min'].values[ind], ds['amp_max'].values[ind] = get_temp_extrema_numerical_xr(time, temp - temp.mean())
    return ds


def get_error(x: xr.DataArray, x_approx: xr.DataArray, kind: Literal['mean', 'median', 'max'] = "mean",
              norm: bool = False, dim: Union[str, list] = "time") -> xr.DataArray:
    """Compute an absolute-error summary between two DataArrays.

    Args:
        x: Reference DataArray.
        x_approx: Approximation DataArray (broadcastable to `x`).
        kind: Error reduction to apply over `dim`. One of {"mean", "median", "max"}.
            Defaults to "mean".
        norm: If True, normalize by 0.01*(max-min) of `x` over `dim`. Defaults to False.
        dim: Dimension name(s) to reduce over. Defaults to "time".

    Returns:
        DataArray of the reduced absolute error (and optionally normalized), with
        remaining dimensions preserved.

    Raises:
        ValueError: If `kind` is not one of {"mean", "median", "max"}.
    """
    err = np.abs(x - x_approx)

    if kind == "mean":
        out = err.mean(dim=dim)
    elif kind == "median":
        out = err.median(dim=dim)
    elif kind == "max":
        out = err.max(dim=dim)
    else:
        raise ValueError(f'Unknown kind="{kind}". Expected "mean", "median", or "max".')

    if norm:
        scale = 0.01 * (x.max(dim=dim) - x.min(dim=dim))
        out = out / scale

    return out
