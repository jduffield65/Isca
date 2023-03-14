from typing import Tuple, Optional
import numpy as np
import xarray as xr
from ..utils.constants import g
from ..utils.moist_physics import moist_static_energy
from scipy import integrate
from scipy.interpolate import UnivariateSpline


def get_dmse_dt(temp: xr.DataArray, sphum: xr.DataArray, height: xr.DataArray, p_levels: xr.DataArray,
                time: xr.DataArray, zonal_mean: bool = True,
                spline_smoothing_factor: float = 0) -> Tuple[xr.DataArray, xr.DataArray]:
    """
    For a given latitude, this computes the time derivative of the mass weighted vertical integral of the zonal mean
    moist static energy, $<[\\partial_t m]>$, used in the paper to compute the parameter $R_1$.

    Args:
        temp: `float [n_time, n_p_levels, n_lon]`.</br>
            Temperature at each coordinate considered. Units: *Kelvin*.
        sphum: `float [n_time, n_p_levels, n_lon]`.</br>
            Specific humidity at each coordinate considered. Units: *kg/kg*.
        height: `float [n_time, n_p_levels, n_lon]`.</br>
            Geopotential height of each level considered.
        p_levels: `float [n_p_levels]`.</br>
            Pressure levels of atmosphere, `p_levels[0]` is top of atmosphere and `p_levels[-1]` is the surface.
            Units: *Pa*.
        time: `float [n_time]`.</br>
            Time at which data recorded. Units: *second*.
        zonal_mean: `bool`.</br>
            Whether to take zonal average or not. If `False`, returned arrays will have size `[n_time x n_lon]`
        spline_smoothing_factor: `float`.</br>
            If positive, a spline will be fit to smooth `mse_integ` and compute `dmse_dt`. *Typical: 0.001*.</br>
            If 0, the deriviative will be computed using `np.gradient`, but in general I recommend using the spline.

    Returns:
        `mse_integ`: `float [n_time]` or `float [n_time x n_lon]`</br>
            The mass weighted vertical integral of the (zonal mean) moist static energy at each time i.e. $<[m]>$.
            Units: $J/m^2$.
        `dmse_dt`: `float [n_time]` or `float [n_time x n_lon]`</br>
            The time derivative of the mass weighted vertical integral of the (zonal mean) moist static energy at
            each time i.e. $<[\\partial_t m]>$. Units: $W/m^2$.
    """
    # compute zonal mean mse in units of J/kg
    mse = moist_static_energy(temp, sphum, height) * 1000
    if 'lon' in list(temp.coords.keys()) and zonal_mean:
        # Take zonal mean if longitude is a coordinate
        mse = mse.mean(dim='lon')
    mse_integ = integrate.simpson(mse / g, p_levels, axis=1)    # do mass weighted integral over atmosphere
    if spline_smoothing_factor > 0:
        if not zonal_mean:
            raise ValueError('Can only use spline if taking the zonal mean')
        # Divide by mean to fit spline as otherwise numbers too large to fit properly
        spl = UnivariateSpline(time, mse_integ / np.mean(mse_integ), s=spline_smoothing_factor)
        dmse_dt = spl.derivative()(time) * np.mean(mse_integ)      # compute derivative direct from spline fit
        mse_integ = spl(time) * np.mean(mse_integ)    # Set mse_integ to spline version so this is what is returned
    elif spline_smoothing_factor == 0:
        # compute derivative directly from the data
        dmse_dt = np.gradient(mse_integ, time, axis=0)
    else:
        raise ValueError(f'spline_smoothing_factor = {spline_smoothing_factor}, which is negative. It must be >=0.')
    return mse_integ, dmse_dt


def get_dvmse_dy(R_a: xr.DataArray, lh: xr.DataArray, sh: xr.DataArray, dmse_dt: xr.DataArray) -> xr.DataArray:
    """
    This infers the divergence of moist static energy flux, $<[\\partial_y vm]>$, from equation $(2)$ in the paper:

    $<[\\partial_t m]> + <[\\partial_y vm]> = [R_a] + [LH] + [SH]$

    It is not computed directly for reasons outlined in section 2b of the paper.

    Args:
        R_a: `float [n_time]`.</br>
            Atmospheric radiative heating rate i.e. difference between top of atmosphere and surface radiative fluxes.
            Negative means atmosphere is cooling. Units: $W/m^2$.</br>
            This is what is returned by `frierson_atmospheric_heating` if using the grey radiation with the
            *Frierson* scheme.
        lh: `float [n_time]`.</br>
            Surface latent heat flux (up is positive). This is saved by *Isca* if the variable `flux_lhe` in the
            `mixed_layer` module is specified in the diagnostic table. Units: $W/m^2$.
        sh: `float [n_time]`.</br>
            Surface sensible heat flux (up is positive). This is saved by *Isca* if the variable `flux_t` in the
            `mixed_layer` module is specified in the diagnostic table. Units: $W/m^2$.
        dmse_dt: `float [n_time]`.</br>
            The time derivative of the mass weighted vertical integral of the zonal mean moist static energy at
            each time, $<[\\partial_t m]>$. Units: $W/m^2$.

    Returns:
        `float [n_time]`.</br>
            $<[\\partial_y vm]>$ - Mass weighted vertical integral of the zonal mean meridional divergence of
            moist static energy flux, $vm$.
    """
    return R_a + lh + sh - dmse_dt


def get_r1(R_a: xr.DataArray, dmse_dt: xr.DataArray, dvmse_dy: xr.DataArray,
           time: Optional[xr.DataArray] = None, spline_smoothing_factor: float = 0) -> xr.DataArray:
    """
    Returns the non-dimensional number, $R_1 = \\frac{\\partial_t m + \\partial_y vm}{R_a}$.

    Args:
        R_a: `float [n_time]`.</br>
            Atmospheric radiative heating rate i.e. difference between top of atmosphere and surface radiative fluxes.
            Negative means atmosphere is cooling. Units: $W/m^2$.</br>
            This is what is returned by `frierson_atmospheric_heating` if using the grey radiation with the
            *Frierson* scheme.
        dmse_dt: `float [n_time]`.</br>
            The time derivative of the mass weighted vertical integral of the zonal mean moist static energy at
            each time, $<[\\partial_t m]>$. Units: $W/m^2$.
        dvmse_dy: `float [n_time]`.</br>
            $<[\\partial_y vm]>$ - Mass weighted vertical integral of the zonal mean meridional divergence of
            moist static energy flux, $vm$.
        time: `None` or `float [n_time]`.</br>
            Time at which data recorded. Only required if using the `spline_smoothing_factor>0`. Units: *second*.
        spline_smoothing_factor: `float`.</br>
            If positive, a spline will be fit to smooth $R_1$. *Typical: 0.2*.</br>

    Returns:
        `float [n_time]`.</br>
        The non-dimensional number, $R_1 = \\frac{\\partial_t m + \\partial_y vm}{R_a}$.

    """
    r1 = (dmse_dt + dvmse_dy)/R_a
    if spline_smoothing_factor > 0:
        if time is None:
            raise ValueError('Specified spline but no time array given.')
        spl = UnivariateSpline(time, r1, s=spline_smoothing_factor)
        r1 = spl(time)
    elif spline_smoothing_factor < 0:
        raise ValueError(f'spline_smoothing_factor = {spline_smoothing_factor}, which is negative. It must be >=0.')
    return r1
