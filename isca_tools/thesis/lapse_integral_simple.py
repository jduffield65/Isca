import numpy as np
import xarray as xr
from typing import Union, Optional, Tuple, Literal

from .lapse_theory import get_var_at_plev
from ..utils.constants import g, R, lapse_dry
from ..utils.moist_physics import sphum_sat
from ..thesis.adiabat_theory import get_temp_adiabat
from .lapse_integral import get_temp_const_lapse, integral_lapse_dlnp_hydrostatic, integral_and_error_calc, \
    const_lapse_fitting
from ..convection.base import lcl_sigma_bolton_simple, dry_profile_temp
from .mod_parcel_theory import get_temp_mod_parcel


def get_temp_mod_parcel_lapse(p_lev: Union[xr.DataArray, np.ndarray, float],
                              p_low: Union[xr.DataArray, np.ndarray, float],
                              lapse_diff_const: Union[xr.DataArray, np.ndarray, float],
                              temp_parcel_lev: Optional[Union[xr.DataArray, np.ndarray, float]] = None,
                              temp_parcel_surf: Optional[Union[xr.DataArray, np.ndarray, float]] = None,
                              rh_parcel_surf: Optional[Union[xr.DataArray, np.ndarray, float]] = None,
                              p_surf: Optional[Union[xr.DataArray, np.ndarray, float]] = None,
                              ) -> Union[xr.DataArray, np.ndarray, float]:
    """
    This finds the temperature at pressure levels `p_lev` following a lapse rate $\Gamma(p) = \Gamma_p(p, T_p(p)) + \eta$
    where $\Gamma_p(p, T)$ is the parcel (moist adiabatic) lapse rate and $\eta$ is a constant. $T_p(p)$ refers
    to parcel temperature at pressure $p$.

    This assumes hydrostatic balance: $\Gamma(p) = −\\frac{dT}{dz} = \\frac{g}{R} \\frac{d \ln T}{d\ln p}$

    Different from *lapse_integral.py* in that it computes parcel temp, $T_p(p)$ by equating
    $MSE(T_{p,s}, r_s, p_s)$ rather than $MSE^*(T_{LCL})$ to $MSE^*(T_p(p))$.
    Where $T_{p,s}$ is the parcel temperature at the surface starting from environmental LCL temperature.

    Args:
        p_lev: `[n_lev]` Pressure levels to find environmental temperature. Units: Pa.
        p_low: Pressure level where to start the ascent from along the modified parcel profile. Units: Pa.
        lapse_diff_const: Constant, $\eta$, which is added to the parcel lapse rate at each pressure level. Units: K/m.
        temp_parcel_lev: `[n_lev]` Parcel temperature at pressure `p_lev`. Units: K.</br>
            If not provided, will compute according to $MSE^*(T(p)) = MSE(T_{p,s}, r_s, p_s)$
        temp_parcel_surf: Temperature of the parcel at `p_surf`. Only required if `temp_parcel_lev` is `None`.
        rh_parcel_surf: Relative humidity of the parcel at `p_surf`. Only required if `temp_parcel_lev` is `None`.
        p_surf: Pressure at `p_surf`. Units: Pa. Only required if `temp_parcel_lev` is `None`.

    Returns:
        temp_lev: `[n_lev]` Temperature at `p_lev`. Units: K.
    """
    # Compute temperature at p_upper such that lapse rate at all levels is the same as parcel plus `lapse_diff_const`.
    # lapse_parcel_integral = integral_lapse_dlnp_hydrostatic(temp_parcel_lev, p_lev, p_lower, p_upper, temp_lower,
    #                                                         temp_parcel_upper)
    # lapse_parcel_integral = g / R * np.log(temp_parcel_lev / temp_low)
    # return temp_lower * (p_lev / p_low) ** (lapse_diff_const * R / g) * np.exp(R / g * lapse_parcel_integral)
    if temp_parcel_lev is None:
        temp_parcel_lev = np.zeros_like(p_lev)
        sphum_parcel_surf = rh_parcel_surf * sphum_sat(temp_parcel_surf, p_surf)
        for i in range(temp_parcel_lev.size):
            temp_parcel_lev[i] = get_temp_adiabat(temp_parcel_surf, sphum_parcel_surf, p_surf, p_lev[i])
    return get_temp_const_lapse(p_lev, temp_parcel_lev, p_low, lapse_diff_const)


def mod_parcel_lapse_fitting(temp_env_lev: np.ndarray,
                             p_lev: np.ndarray,
                             temp_env_lower: float, p_lower: float,
                             temp_env_upper: float, p_upper: float,
                             temp_parcel_surf: float,
                             rh_parcel_surf: float,
                             p_surf: float,
                             n_lev_above_upper_integral: int = 0,
                             temp_surf_lcl_calc: float = 300,
                             sanity_check: bool = False) -> Tuple[
    float, float, float]:
    """
    Find the constant, $\eta$ that needs adding to parcel lapse rate such that
    $\int_{p_1}^{p_2} \Gamma_{env}(p) d\ln p = \int_{p_1}^{p_2} \Gamma_p(p, T_p(p)) + \eta d\ln p$.
    Then computes the error in this approximation:
    $\int_{p_1}^{p_2} |\Gamma_{env}(p) - \Gamma_p(p, T_p(p)) - \eta| d\ln p$.

    where $\Gamma_p(p, T)$ is the parcel (moist adiabatic) lapse rate and $T_p(p)$ is the parcel temperature
    at pressure $p$ starting at $p_1$.

    Different from *lapse_integral.py* in that it computes parcel temp, $T_p(p)$ by equating
    $MSE(T_{p,s}, r_s, p_s)$ rather than $MSE^*(T_{LCL})$ to $MSE^*(T_p(p))$.
    Where $T_{p,s}$ is the parcel temperature at the surface starting from environmental LCL temperature.

    Args:
        temp_env_lev: `[n_lev]` Environmental temperature at pressure `p_lev`.
        p_lev: `[n_lev]` Model pressure levels in Pa.
        temp_env_lower: Environmental temperature at lower pressure level (nearer surface) `p_lower`.
        p_lower: Pressure level to start profile (near surface).
        temp_env_upper: Environmental temperature at upper pressure level (further from surface) `p_upper`.
        p_upper: Pressure level to end profile (further from surface).
        temp_parcel_surf: Temperature of the parcel at `p_surf`.
        rh_parcel_surf: Relative humidity of the parcel at `p_surf`.
        p_surf: Pressure at `p_surf`. Units: Pa.
        n_lev_above_upper_integral: Will return `integral` and `integral_error` not up to `p_upper` but up to
            the model level pressure `n_lev_above_upper_integral` further from the surface than `p_upper`.
            If `n_lev_above_upper_integral=0`, upper limit of integral will be `p_upper`.
        temp_surf_lcl_calc: Surface temperature to use when computing $\sigma_{LCL}$.
        sanity_check: If `True` will print a sanity check to ensure the calculation is correct.

    Returns:
        lapse_diff_const: Lapse rate adjustment, $\eta$ which needs to be added to $\Gamma_{p}(p, T_p(p))$
            so integral matches that of environmental lapse rate. Units are *K/km*.
        integral: Result of integral $\int_{p_1}^{p_2} \Gamma_{env}(p) d\ln p$. Units are *K/km*.
        integral_error: Result of integral $\int_{p_1}^{p_2} |\Gamma_{env}(p) - \Gamma_p(p) - \eta| d\ln p$.
            Units are *K/km*.
    """
    if n_lev_above_upper_integral != 0:
        if n_lev_above_upper_integral < 0:
            raise ValueError('n_lev_above_upper_integral must be greater than 0')
        if p_lev[1] < p_lev[0]:
            raise ValueError('p_lev[1] must be greater than p_lev[0]')
        ind_upper = np.where(p_lev < p_upper)[0][-n_lev_above_upper_integral]

    # Get parcel upper temp following moist adiabat, using surface MSE
    temp_parcel_upper = get_temp_mod_parcel(rh_parcel_surf, p_surf, p_upper, 0, 0,
                                            temp_parcel_surf, temp_surf_lcl_calc=temp_surf_lcl_calc)


    # Compute integral of deviation between environmental and parcel lapse rate between p_lower and p_upper
    lapse_diff_integral = g / R * (
            np.log(temp_env_upper / temp_env_lower) - np.log(temp_parcel_upper / temp_env_lower))
    # Compute the constant needed to be added to the parcel lapse rate at each level to make above integral equal 0.
    lapse_diff_const = lapse_diff_integral / np.log(p_upper / p_lower)
    # temp_env_approx_lev = get_temp_mod_parcel_lapse(p_lev, p_lower, lapse_diff_const, temp_parcel_lev=temp_parcel_lev)
    temp_env_approx_lev = get_temp_mod_parcel(rh_parcel_surf, p_surf, p_lev, 0, lapse_diff_const,
                                              temp_parcel_surf, temp_surf_lcl_calc=temp_surf_lcl_calc)

    if sanity_check:
        temp_env_approx_upper = get_temp_mod_parcel(rh_parcel_surf, p_surf, p_upper, 0, lapse_diff_const,
                                                    temp_parcel_surf, temp_surf_lcl_calc=temp_surf_lcl_calc)
        lapse_integral = g / R * np.log(temp_env_upper / temp_env_lower)
        # sanity check, this should be the same as lapse_integral
        lapse_integral_approx = integral_lapse_dlnp_hydrostatic(temp_env_approx_lev, p_lev, p_lower, p_upper,
                                                                temp_env_lower, temp_env_upper)
        print(
            f'Actual lapse integral: {lapse_integral * 1000:.3f} K/km\nApprox lapse integral: {lapse_integral_approx * 1000:.3f} K/km')
        # Will use lapse rate such that approx value of T_upper is exact. Check that here
        print(f'Actual temp_upper: {temp_env_upper:.3f} K\nApprox temp_upper: {temp_env_approx_upper:.3f} K')

    # Compute error in the integral
    if n_lev_above_upper_integral == 0:
        lapse_integral, lapse_integral_error = integral_and_error_calc(temp_env_lev, temp_env_approx_lev, p_lev,
                                                                       temp_env_lower, temp_env_lower, p_lower,
                                                                       temp_env_upper, temp_env_upper, p_upper)
    else:
        lapse_integral, lapse_integral_error = integral_and_error_calc(temp_env_lev, temp_env_approx_lev, p_lev,
                                                                       temp_env_lower, temp_env_lower, p_lower,
                                                                       float(temp_env_lev[ind_upper]),
                                                                       float(temp_env_approx_lev[ind_upper]),
                                                                       float(p_lev[ind_upper]))
    return lapse_diff_const * 1000, lapse_integral * 1000, lapse_integral_error * 1000


def fitting_2_layer(temp_env_lev: Union[xr.DataArray, np.ndarray], p_lev: Union[xr.DataArray, np.ndarray],
                    temp_env_lower: float, p_lower: float, rh_lower: float,
                    temp_env_upper: float, p_upper: float,
                    temp_env_upper2: float, p_upper2: float,
                    method_layer1: Literal['const'] = 'const',
                    method_layer2: Literal['const', 'mod_parcel'] = 'const',
                    n_lev_above_upper2_integral: int = 0,
                    temp_surf_lcl_calc: float = 300,
                    sanity_check: bool = False) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray]:
    """
    Applies `const_lapse_fitting` or `mod_parcel_lapse_fitting` to each layer.

    Different from *lapse_integral.py* in that it computes parcel temp, $T_p(p)$ by equating
    $MSE(T_{p,s}, r_s, p_s)$ rather than $MSE^*(T_{LCL})$ to $MSE^*(T_p(p))$.
    Where $T_{p,s}$ is the parcel temperature at the surface starting from environmental LCL temperature.

    Args:
        temp_env_lev: `[n_lev]` Environmental temperature at pressure `p_lev`.
        p_lev: `[n_lev]` Model pressure levels in Pa.
        temp_env_lower: Environmental temperature at lower pressure level (nearer surface) `p_lower`.
        p_lower: Pressure level to start profile (near surface).
        rh_lower: Environmental relative humidity at `p_lower`.
        temp_env_upper: Environmental temperature at the upper pressure level of the first layer (layer closest to surface) `p_upper`.
        p_upper: Pressure level to end profile of first layer (layer closest to surface).
        temp_env_upper2: Environmental temperature at the upper pressure level of the second layer (layer furthest from surface) `p_upper2`.
        p_upper2: Pressure level to end profile of second layer (layer furthest from surface).
        temp_parcel_lev: `[n_lev]` Parcel temperature (following moist adiabat) at pressure `p_lev`.</br>
            Only required if either `method_layer1` or `method_layer2` are `'mod_parcel'`.
        temp_parcel_lower: Parcel temperature at lower pressure level (nearer surface) `p_lower`.</br>
            Only required if either `method_layer1 = 'mod_parcel'`.
        temp_parcel_upper: Parcel temperature at the upper pressure level of the first layer (layer closest to surface) `p_upper`.</br>
            Only required if either `method_layer1` or `method_layer2` are `'mod_parcel'`.
        temp_parcel_upper2: Parcel temperature at the upper pressure level of the second layer (layer furthest from surface) `p_upper2`.</br>
            Only required if either `method_layer2 = 'mod_parcel'`.
        method_layer1: Which fitting method to use for layer 1.
        method_layer2: Which fitting method to use for layer 2.
        n_lev_above_upper2_integral: Will return `integral` and `integral_error` not up to `p_upper2` but up to
            the model level pressure `n_lev_above_upper2_integral` further from the surface than `p_upper2`.
            If `n_lev_above_upper2_integral=0`, upper limit of integral will be `p_upper2`.
        temp_surf_lcl_calc: Surface temperature to use when computing $\sigma_{LCL}$.
        sanity_check: If `True` will print a sanity check to ensure the calculation is correct.

    Returns:
        lapse: Lapse rate info for each layer. Bulk lapse rate if `method_layer='const'` or lapse rate adjustment
            if `method_layer='mod_parcel'`. Units are *K/km*.
        integral: Result of integral $\int_{p_1}^{p_2} \Gamma_{env}(p) d\ln p$ of each layer. Units are *K/km*.
        integral_error: Result of integral $\int_{p_1}^{p_2} |\Gamma_{env}(p) - \Gamma_{approx}| d\ln p$ of each layer.
            Units are *K/km*.
    """
    if (p_upper > p_lower) | (p_upper2 > p_upper):
        return np.asarray([np.nan, np.nan]), np.asarray([np.nan, np.nan]), np.asarray([np.nan, np.nan])
    if method_layer1 == 'const':
        lapse1, lapse_integral1, lapse_integral_error1 = \
            const_lapse_fitting(temp_env_lev, p_lev, temp_env_lower, p_lower, temp_env_upper, p_upper,
                                sanity_check=sanity_check)
    else:
        raise ValueError(f'method_layer1 = {method_layer1} not recognized.')

    if method_layer2 == 'const':
        lapse2, lapse_integral2, lapse_integral_error2 = \
            const_lapse_fitting(temp_env_lev, p_lev, temp_env_upper, p_upper, temp_env_upper2, p_upper2,
                                n_lev_above_upper2_integral, sanity_check=sanity_check)
    elif method_layer2 == 'mod_parcel':
        # Compute parcel temp at p_lower following a dry adiabat from p_upper
        temp_parcel_lower = dry_profile_temp(temp_env_upper, p_upper, p_lower)
        lapse2, lapse_integral2, lapse_integral_error2 = \
            mod_parcel_lapse_fitting(temp_env_lev, p_lev, temp_env_upper, p_upper, temp_env_upper2, p_upper2,
                                     temp_parcel_surf=temp_parcel_lower, rh_parcel_surf=rh_lower, p_surf=p_lower,
                                     n_lev_above_upper_integral=n_lev_above_upper2_integral,
                                     temp_surf_lcl_calc=temp_surf_lcl_calc, sanity_check=sanity_check)
    else:
        raise ValueError(f'method_layer2 = {method_layer2} not recognized.')

    lapse = np.asarray([lapse1, lapse2])
    lapse_integral = np.asarray([lapse_integral1, lapse_integral2])
    lapse_integral_error = np.asarray([lapse_integral_error1, lapse_integral_error2])
    return lapse, lapse_integral, lapse_integral_error


def fitting_2_layer_xr(temp_env_lev: xr.DataArray,
                       p_lev: xr.DataArray,
                       temp_env_lower: xr.DataArray, p_lower: xr.DataArray, rh_lower: xr.DataArray,
                       temp_env_upper2: xr.DataArray, p_upper2: float,
                       method_layer1: Literal['const'] = 'const',
                       method_layer2: Literal['const', 'mod_parcel'] = 'const',
                       n_lev_above_upper2_integral: int = 0,
                       temp_surf_lcl_calc: Optional[Union[float, xr.DataArray]] = 300,
                       sanity_check: bool = False, lev_dim: str = 'lev',
                       layer_dim: str = 'layer') -> Tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    # Different from lapse_integral.py in that it computes LCL approximately from RH.
    # Then also computes parcel temp, $T_p(p)$, by equating MSE(T_{p,s}, r_s, p_s) rather than MSE^*(T_{LCL}) to MSE^*(T_p(p)).
    if temp_surf_lcl_calc is None:
        # If don't specify approx value to use for the LCL calc, then use actual temp_env_lower
        temp_surf_lcl_calc = temp_env_lower
    p_upper = lcl_sigma_bolton_simple(rh_lower, temp_surf_lcl_calc) * p_lower
    temp_env_upper = get_var_at_plev(temp_env_lev, p_lev, p_upper, lev_dim=lev_dim)
    return xr.apply_ufunc(
        fitting_2_layer,
        temp_env_lev, p_lev,  # (lat, lon, lev)
        temp_env_lower, p_lower, rh_lower, temp_env_upper, p_upper, temp_env_upper2,  # (lat, lon)
        input_core_dims=[[lev_dim], [lev_dim], [], [], [], [], [], []],
        # only 'temp_env_lev' and 'lev' varies along 'lev_dim'.
        output_core_dims=[[layer_dim], [layer_dim], [layer_dim]],
        vectorize=True,
        kwargs={'p_upper2': p_upper2, 'method_layer1': method_layer1,
                'method_layer2': method_layer2, 'n_lev_above_upper2_integral': n_lev_above_upper2_integral,
                'temp_surf_lcl_calc': temp_surf_lcl_calc, 'sanity_check': sanity_check}
    )


def get_temp_2_layer_approx(p_lev, temp_env_surf, p_surf, rh_surf, lapse_layer1, lapse_layer2,
                            method_layer1: Literal['const'] = 'const',
                            method_layer2: Literal['const', 'mod_parcel'] = 'const',
                            temp_lower_lcl_calc: Optional[Union[float, xr.DataArray]] = 300) -> np.ndarray:
    # Returns the approximate temperature on model pressure levels for a given 2 layer fitting procedure
    # Expect lapse rates in K/m not K/km
    if temp_lower_lcl_calc is None:
        # If don't specify approx value to use for the LCL calc, then use actual temp_env_lower
        temp_lower_lcl_calc = temp_env_surf
    p_upper = lcl_sigma_bolton_simple(rh_surf, temp_lower_lcl_calc) * p_surf
    if method_layer1 == 'const':
        temp_lev = get_temp_const_lapse(p_lev, temp_env_surf, p_surf, lapse_layer1)
        temp_upper = get_temp_const_lapse(p_upper, temp_env_surf, p_surf, lapse_layer1)
    else:
        raise ValueError(f'method_layer1 = {method_layer1} not recognized.')

    if method_layer2 == 'const':
        temp_lev = np.where(p_lev < p_upper, get_temp_const_lapse(p_lev, temp_upper, p_upper, lapse_layer2),
                            temp_lev)
    elif method_layer2 == 'mod_parcel':
        temp_lev = np.where(p_lev < p_upper, get_temp_mod_parcel(rh_surf, p_surf, p_lev, lapse_layer1-lapse_dry,
                                                                 lapse_layer2, temp_env_surf),
                            temp_lev)
    else:
        raise ValueError(f'method_layer2 = {method_layer2} not recognized.')
    return temp_lev
