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


def interp_var_at_pressure(var: Union[xr.DataArray, xr.Dataset], p_desired: xr.DataArray, p_surf: xr.DataArray,
                           hyam: xr.DataArray, hybm: xr.DataArray, p0: float,
                           plev_step: float = 1000, extrapolate: bool = False) -> xr.Dataset:
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

    Returns:
        Dataset with `plev` indicating approximate value of `p_desired` used, as well as `var` interpolated
            to that pressure level.
    """
    plevs = np.arange(round_any(float(p_desired.min()), plev_step, 'floor'),
                      round_any(float(p_desired.max()), plev_step, 'ceil')+plev_step/2, plev_step)
    plevs_expand = xr.DataArray(plevs, dims=["plev"], coords={"plev": np.arange(len(plevs))})
    # Expand to match dimensions in p_surf, preserving order
    for dim in p_surf.dims:
        plevs_expand = plevs_expand.expand_dims({dim: p_surf.coords[dim]})

    idx_lcl_closest = np.abs(plevs_expand - p_desired).argmin(dim='plev')
    var_out = {'plev': plevs_expand.isel(plev=idx_lcl_closest)}     # approx pressure of p_desired used

    # Note that with extrapolate, will obtain values lower than surface
    if isinstance(var, xr.DataArray):
        var_out[var.name] = interp_hybrid_to_pressure(data=var, ps=p_surf, hyam=hyam, hybm=hybm, p0=p0,
                                                      new_levels=plevs, extrapolate=extrapolate,
                                                      variable='other' if extrapolate else None).isel(plev=idx_lcl_closest)
    elif isinstance(var, xr.Dataset):
        for key in var:
            var_out[key] = interp_hybrid_to_pressure(data=var[key], ps=p_surf, hyam=hyam, hybm=hybm, p0=p0,
                                                     new_levels=plevs, extrapolate=extrapolate,
                                                     variable='other' if extrapolate else None).isel(plev=idx_lcl_closest)
    else:
        raise ValueError('Unrecognized var. Needs to be a xr.DataArray or xr.Dataset.')
    for key in var_out:
        # Drop dimension of plev in all variables
        var_out[key] = var_out[key].drop_vars('plev')
    return xr.Dataset(var_out)
