from ..utils.constants import g, R
from ..utils.base import round_any
import numpy as np
import xarray as xr
from typing import Union
from geocat.comp.interpolation import interp_hybrid_to_pressure

def get_bulk_lapse_rate(temp1: xr.DataArray, temp2: xr.DataArray, p1: Union[xr.DataArray, float],
                        p2: Union[xr.DataArray, float]):
    """
    Compute the bulk environmental lapse rate, $\Gamma$, between pressure `p1` at environmental temperature `temp1`
    and `p2` at environmental temperature `temp2`:

    $$\Gamma = \\frac{g}{R}\ln\\left(\\frac{T_1}{T_2}\\right)/\ln\\left(\\frac{p_1}{p_2}\\right)$$

    This equation assumes hydrostatic equilibrium, ideal gas equation of state and that $\Gamma$ is constant
    between `p1` and `p2`.

    Args:
        temp1: Temperature at pressure `p1`. Units: *K*.
        temp2: Temperature at pressure `p2`. Units: *K*.
        p1: Pressure at environmental temperature `temp1`. Units: *Pa*.
        p2: Pressure at environmental temperature `temp2`. Units: *Pa*.

    Returns:
        Bulk environmental lapse rate, positive if `temp2<temp1` and `p2<p1`. Units are *K/m*.
    """
    return g/R * np.log(temp1/temp2) / np.log(p1/p2)


def reconstruct_temp(temp3: xr.DataArray, p1: Union[xr.DataArray, float], p2: Union[xr.DataArray, float],
                     p3: Union[xr.DataArray, float],
                     lapse_12: xr.DataArray, lapse_23: xr.DataArray):
    """
    The temperature, $T_1$, at $p_1$ can be reconstructed from the lapse rate, $\Gamma_{12}$, between $p_1$ and $p_2$;
    the lapse rate $\Gamma_{23}$, between $p_2$ and $p_3$; and the temperature at $p_3$, $T_3$:

    $$
    T_1 = T_{3}\\left((\\frac{p_2}{p_1})^{\Gamma_{23}-\Gamma_{12}}(\\frac{p_3}{p_1})^{-\Gamma_{23}}\\right)^{R/g}
    $$

    Args:
        temp3: Temperature at pressure `p3`. Units: *K*.
        p1: Pressure at level to reconstruct `temp1`. Units: *Pa*.
        p2: Pressure at environmental temperature `temp2`. Units: *Pa*.
        p3: Pressure at environmental temperature `temp3`. Units: *Pa*.
        lapse_12: Bulk environmental lapse rate between `p1` and `p2`. Units are *K/m*.
        lapse_23: Bulk environmental lapse rate between `p2` and `p3`. Units are *K/m*.

    Returns:
        temp1: Temperature at pressure `p1`. Units: *K*.
    """
    sigma_12 = p2 / p1     # if p1 is surface, this should be <1
    sigma_13 = p3 / p1
    return temp3 * (sigma_12**(lapse_23-lapse_12) * sigma_13**(-lapse_23))**(R/g)


def interp_var_at_pressure(var: Union[xr.DataArray, xr.Dataset, np.ndarray], p_desired: Union[xr.DataArray, np.ndarray],
                           p_surf: Union[xr.DataArray, np.ndarray],
                           hyam: xr.DataArray, hybm: xr.DataArray, p0: float,
                           plev_step: float = 1000, extrapolate: bool = False,
                           lev_dim: str = 'lev', var_name: str = 'new_var') -> xr.Dataset:
    """
    Function to get the value of variable `var` at the pressure `p_desired`, where `p_desired` is expected to
    be a different value at each lat and lon.

    Args:
        var: Variable to do interpolation of. Should have `lev` dimension as well as lat, lon and possibly time.
            If give Dataset, will do interpolation on all variables.
        p_desired: Desired pressure to find `var` at.
            Should have same dimension as `var` but no `lev`. Units: *Pa*.
        p_surf: Surface pressure.
            Should have same dimension as `var` but no `lev`. Units: *Pa*.
        hyam: Hybrid a coefficients. Should have dimension of `lev` only.
        hybm: Hybrid b coefficients. Should have dimension of `lev` only.
        p0: Reference pressure. Units: *Pa*.
        plev_step: Will find var at value closest to `p_desired` on pressure grid with this spacing,
            so sets accuracy of interpolation.
        extrapolate: If True, below ground extrapolation for variable will be done, otherwise will return nan.
        lev_dim: String that is the name of level dimension in input data.
        var_name: String that is the name of variable in input data. Only used if `var` is numpy array

    Returns:
        Dataset with `plev` indicating approximate value of `p_desired` used, as well as `var` interpolated
            to that pressure level.
    """
    plevs = np.arange(round_any(float(p_desired.min()), plev_step, 'floor'),
                      round_any(float(p_desired.max()), plev_step, 'ceil')+plev_step/2, plev_step)
    plevs_expand = xr.DataArray(plevs, dims=["plev"], coords={"plev": np.arange(len(plevs))})
    # Expand to match dimensions in p_surf, preserving order
    if isinstance(var, np.ndarray) and var.size==hybm.size:
        # If just numpy array, need to make it a data array for it to work
        var = xr.DataArray(var, dims=hybm.dims, coords=hybm.coords, name=var_name)
    if isinstance(p_surf, xr.DataArray):
        for dim in p_surf.dims:
            plevs_expand = plevs_expand.expand_dims({dim: p_surf.coords[dim]})

    idx_lcl_closest = np.abs(plevs_expand - p_desired).argmin(dim='plev')
    var_out = {'plev': plevs_expand.isel(plev=idx_lcl_closest)}     # approx pressure of p_desired used

    # Note that with extrapolate, will obtain values lower than surface
    if isinstance(var, xr.DataArray):
        var_out[var.name] = interp_hybrid_to_pressure(data=var, ps=p_surf, hyam=hyam, hybm=hybm, p0=p0,
                                                      new_levels=plevs, extrapolate=extrapolate, lev_dim=lev_dim,
                                                      variable='other' if extrapolate else None).isel(plev=idx_lcl_closest)
    elif isinstance(var, xr.Dataset):
        for key in var:
            var_out[key] = interp_hybrid_to_pressure(data=var[key], ps=p_surf, hyam=hyam, hybm=hybm, p0=p0,
                                                     new_levels=plevs, extrapolate=extrapolate, lev_dim=lev_dim,
                                                     variable='other' if extrapolate else None).isel(plev=idx_lcl_closest)
    else:
        raise ValueError('Unrecognized var. Needs to be a xr.DataArray or xr.Dataset.')
    for key in var_out:
        # Drop dimension of plev in all variables
        var_out[key] = var_out[key].drop_vars('plev')
    return xr.Dataset(var_out)


def get_var_at_plev(var_env: xr.DataArray, p_env: xr.DataArray, p_desired: xr.DataArray, method: str ='log',
                    lev_dim: str ='lev'):
    """
    Find the value of `var_env` at pressure `p_desired`.

    Similar to `interp_hybrid_to_pressure` but handles the case where want different `p_desired` at each
    latitude and longitude.


    Args:
        var_env: float `n_lat x n_lon x n_lev x ...`</br>
            Variable to find value of at `p_desired`.
        p_env: float `n_lat x n_lon x n_lev x ...`</br>
            Pressure levels corresponding to `var_env`.
        p_desired: float `n_lat x n_lon x ...`</br>
            Pressure levels to find `var_env` at for each coordinate.
        method: Method of interpolation either take log10 of pressure first or leave as raw values.
        lev_dim: String that is the name of level dimension in `var_env` and `p_env`.

    Returns:
        var_desired: float `n_lat x n_lon x ...`</br>
            The value of `var_env` at `p_desired`.
    """
    def _get_var_at_plev(var_env, p_env, p_desired):
        if method == 'log':
            return np.interp(np.log10(p_desired), np.log10(p_env), var_env)
        else:
            return np.interp(p_desired, p_env, var_env)
    if not (p_env.diff(dim=lev_dim) > 0).all():
        # If pressure is not ascending, flip dimension along lev_dim
        # Requirement for np.interp
        print(f'Reversed order of {lev_dim} for interpolation so p_env is ascending')
        lev_dim_ascending = bool((p_env[lev_dim].diff(dim=lev_dim)>0).all())
        p_env = p_env.sortby(lev_dim, ascending=not lev_dim_ascending)
        var_env = var_env.sortby(lev_dim, ascending=not lev_dim_ascending)
        if not (p_env.diff(dim=lev_dim) > 0).all():
            # Sanity check p_env is now ascending
            raise ValueError('Pressure variable not ascending')

    out = xr.apply_ufunc(
        _get_var_at_plev,
        var_env, p_env, p_desired,
        input_core_dims=[[lev_dim], [lev_dim], []],
        output_core_dims=[[]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
        kwargs={}
    )
    return out
