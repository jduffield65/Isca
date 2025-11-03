import numpy as np
import xarray as xr
from typing import Union, Optional, Tuple, Literal
from ..utils.constants import g, R
from ..utils.moist_physics import sphum_sat
from ..thesis.adiabat_theory import get_temp_adiabat


def integral_lapse_dlnp_hydrostatic(temp_lev: Union[xr.DataArray, np.ndarray], p_lev: Union[xr.DataArray, np.ndarray],
                                    p1: float, p2: float, T_p1: float, T_p2: float,
                                    temp_ref_lev: Optional[Union[xr.DataArray, np.ndarray]] = None,
                                    temp_ref_p1: Optional[float] = None, temp_ref_p2: Optional[float] = None,
                                    take_abs: bool = False) -> float:
    """
    Compute $\int_{p_1}^{p_2} \Gamma d\ln p$ using the hydrostatic relation (converted to pressure integral) only (no Z required),
    where $\Gamma = -dT/dz$ is the lapse rate.
    Can also compute $\int_{p_1}^{p_2} \Gamma - \Gamma_{ref} d\ln p$

    Uses hydrostatic balance, $d\ln p = -\\frac{g}{RT(p)} dz$, to convert integral into sum over levels.

    If `temp_ref_lev` is None, there is an analytic solution: $\\frac{g}{R} \ln \\left(\\frac{T_2}{T_1}\\right)$,
    but this function will return a numerical estimate still.

    Args:
        temp_lev: xr.DataArray
            Temperature [K], dims include 'lev' (vertical pressure coordinate)
        p_lev: xr.DataArray
            Pressure [Pa], same 'lev' coordinate as temp_lev
        p1: float
            Lower integration limit [Pa]
        p2: float
            Upper integration limit [Pa]
        T_p1: float | None, optional
            Temperature at p1 [K]; if None, will be log-interpolated from temp_lev
        T_p2: float | None, optional
            Temperature at p2 [K]; if None, will be log-interpolated from temp_lev
        temp_ref_lev: Temperature of reference profile at pressure `p_lev`.
        temp_ref_p1: Temperature of reference profile at pressure `p1`.
        temp_ref_p2: Temperature of reference profile at pressure `p2`.
        take_abs: If `True`, and provide `temp_ref_lev`, will compute $\int_{p_1}^{p_2} |\Gamma - \Gamma_{ref}| d\ln p$.
            Otherwise, will compute $\int_{p_1}^{p_2} \Gamma - \Gamma_{ref} d\ln p$.

    Returns:
        integral: Value of the integral
    """
    if isinstance(temp_lev, xr.DataArray):
        temp_lev = temp_lev.values
    if isinstance(p_lev, xr.DataArray):
        p_lev = p_lev.values
    # Ensure descending pressure order (p_lev decreases with height)
    if p_lev[0] < p_lev[-1]:
        temp_lev, p_lev = temp_lev[::-1], p_lev[::-1]
        if temp_ref_lev is not None:
            temp_ref_lev = temp_ref_lev[::-1]

    # Build augmented pressure and temperature arrays
    # P_aug = np.concatenate(([p1], p_lev[(p_lev > min(p1, p2)) & (p_lev < max(p1, p2))], [p2]))
    T_aug = np.concatenate(([T_p1], temp_lev[(p_lev > min(p1, p2)) & (p_lev < max(p1, p2))], [T_p2]))
    # T_aug = np.interp(np.log(P_aug), np.log(p_lev.values), temp_lev.values)
    T_aug[0], T_aug[-1] = T_p1, T_p2  # enforce provided endpoints if given

    # Reference profile, if provided
    if temp_ref_lev is not None:
        if isinstance(temp_ref_lev, xr.DataArray):
            temp_ref_lev = temp_ref_lev.values
        Tref_aug = np.concatenate(([temp_ref_p1], temp_ref_lev[(p_lev > min(p1, p2)) & (p_lev < max(p1, p2))],
                                   [temp_ref_p2]))
        Tref_aug[0], Tref_aug[-1] = temp_ref_p1, temp_ref_p2
    else:
        Tref_aug = None

    def compute_integrand(Tprof):
        # Finite-difference derivative
        dT = np.diff(Tprof)
        Tmean = 0.5 * (Tprof[:-1] + Tprof[1:])
        return dT / Tmean

    if Tref_aug is None:
        integral = g / R * np.sum(compute_integrand(T_aug))
    else:
        if take_abs:
            integral = g / R * np.sum(np.abs(compute_integrand(T_aug) - compute_integrand(Tref_aug)))
        else:
            integral = g / R * np.sum(compute_integrand(T_aug) - compute_integrand(Tref_aug))
    return float(integral)


def get_temp_const_lapse(p_lev: Union[xr.DataArray, np.ndarray, float],
                         temp_low: Union[xr.DataArray, np.ndarray, float],
                         p_low: Union[xr.DataArray, np.ndarray, float],
                         lapse: Union[xr.DataArray, np.ndarray, float]) -> Union[xr.DataArray, np.ndarray, float]:
    """
    Get the temperature at `p_lev` assuming constant lapse rate up from `temp_low` at `p_low`.

    This assumes hydrostatic balance: $\Gamma(p) = −\\frac{dT}{dz} = \\frac{g}{R} \\frac{d \ln T}{d\ln p}$

    Args:
        p_lev: `[n_lev]` Pressure levels to find temperature. Units: Pa.
        temp_low: Temperature at low pressure `p_low`. Units: K.
        p_low: Pressure level where to start ascent from along constant lapse rate profile. Units: Pa.
        lapse: Constant lapse rate, $\Gamma$, to use to find temperature at `p_lev`. Units: K/m.

    Returns:
        temp_lev: `[n_lev]` Temperature at `p_lev`. Units: K.
    """
    return temp_low * (p_lev / p_low) ** (lapse * R / g)


def integral_and_error_calc(temp_env_lev: Union[xr.DataArray, np.ndarray],
                            temp_env_approx_lev: Union[xr.DataArray, np.ndarray],
                            p_lev: Union[xr.DataArray, np.ndarray],
                            temp_env_lower: float, temp_env_approx_lower: float, p_lower: float,
                            temp_env_upper: float, temp_env_approx_upper: float, p_upper: float,
                            ):
    integral = g / R * np.log(temp_env_upper / temp_env_lower)
    error = integral_lapse_dlnp_hydrostatic(temp_env_lev, p_lev, p_lower, p_upper,
                                            temp_env_lower, temp_env_upper, temp_env_approx_lev,
                                            temp_env_approx_lower, temp_env_approx_upper, take_abs=True)
    return integral, error


def const_lapse_fitting(temp_env_lev: Union[xr.DataArray, np.ndarray], p_lev: Union[xr.DataArray, np.ndarray],
                        temp_env_lower: float, p_lower: float,
                        temp_env_upper: float, p_upper: float, n_lev_above_upper_integral: int = 0,
                        sanity_check: bool = False) -> Tuple[float, float, float]:
    """
    Find the bulk lapse rate such that $\int_{p_1}^{p_2} \Gamma_{env}(p) d\ln p = \Gamma_{bulk} \ln (p_2/p_1)$.
    Then computes the error in this approximation: $\int_{p_1}^{p_2} |\Gamma_{env}(p) - \Gamma_{bulk}| d\ln p$.

    Args:
        temp_env_lev: `[n_lev]` Environmental temperature at pressure `p_lev`.
        p_lev: `[n_lev]` Model pressure levels in Pa.
        temp_env_lower: Environmental temperature at lower pressure level (nearer surface) `p_lower`.
        p_lower: Pressure level to start profile (near surface).
        temp_env_upper: Environmental temperature at upper pressure level (further from surface) `p_upper`.
        p_upper: Pressure level to end profile (further from surface).
        n_lev_above_upper_integral: Will return `integral` and `integral_error` not up to `p_upper` but up to
            the model level pressure `n_lev_above_upper_integral` further from the surface than `p_upper`.
            If `n_lev_above_upper_integral=0`, upper limit of integral will be `p_upper`.
        sanity_check: If `True` will print a sanity check to ensure the calculation is correct.

    Returns:
        lapse_bulk: Bulk lapse rate. Units are *K/km*.
        integral: Result of integral $\int_{p_1}^{p_2} \Gamma_{env}(p) d\ln p$. Units are *K/km*.
        integral_error: Result of integral $\int_{p_1}^{p_2} |\Gamma_{env}(p) - \Gamma_{bulk}| d\ln p$.
            Units are *K/km*.
        temp_env_approx_lev: `[n_lev]` Estimate of environmental temperature at pressure `p_lev`.
    """
    # Compute integral of actual environmental lapse rate between p_lower and p_upper
    lapse_integral = g / R * np.log(temp_env_upper / temp_env_lower)
    # Define bulk lapse rate such that a profile following constant lapse rate between p_lower and p_upper
    # would have same value of above integral as actual profile
    lapse_bulk = lapse_integral / np.log(p_upper / p_lower)
    temp_env_approx_lev = get_temp_const_lapse(p_lev, temp_env_lower, p_lower, lapse_bulk)

    if sanity_check:
        # sanity check, this should be the same as lapse_integral
        lapse_integral_approx = integral_lapse_dlnp_hydrostatic(temp_env_approx_lev, p_lev, p_lower, p_upper,
                                                                temp_env_lower, temp_env_upper)
        print(
            f'Actual lapse integral: {lapse_integral * 1000:.3f} K/km\nApprox lapse integral: {lapse_integral_approx * 1000:.3f} K/km')
        # Will use lapse rate such that approx value of T_upper is exact. Check that here
        temp_env_approx_upper = get_temp_const_lapse(p_upper, temp_env_lower, p_lower, lapse_bulk)
        print(f'Actual temp_upper: {temp_env_upper:.3f} K\nApprox temp_upper: {temp_env_approx_upper:.3f} K')

    if n_lev_above_upper_integral == 0:
        lapse_integral, lapse_integral_error = integral_and_error_calc(temp_env_lev, temp_env_approx_lev, p_lev,
                                                                       temp_env_lower, temp_env_lower, p_lower,
                                                                       temp_env_upper, temp_env_upper, p_upper)
    else:
        if n_lev_above_upper_integral < 0:
            raise ValueError('n_lev_above_upper_integral must be greater than 0')
        if p_lev[1] < p_lev[0]:
            raise ValueError('p_lev[1] must be greater than p_lev[0]')
        ind_upper = np.where(p_lev < p_upper)[0][-n_lev_above_upper_integral]
        lapse_integral, lapse_integral_error = integral_and_error_calc(temp_env_lev, temp_env_approx_lev, p_lev,
                                                                       temp_env_lower, temp_env_lower, p_lower,
                                                                       float(temp_env_lev[ind_upper]),
                                                                       float(temp_env_approx_lev[ind_upper]),
                                                                       float(p_lev[ind_upper]))
    return lapse_bulk * 1000, lapse_integral * 1000, lapse_integral_error * 1000


def get_temp_mod_parcel_lapse(p_lev: Union[xr.DataArray, np.ndarray, float],
                              p_low: Union[xr.DataArray, np.ndarray, float],
                              lapse_diff_const: Union[xr.DataArray, np.ndarray, float],
                              temp_parcel_low: Optional[Union[xr.DataArray, np.ndarray, float]] = None,
                              temp_parcel_lev: Optional[Union[xr.DataArray, np.ndarray, float]] = None
                              ) -> Union[xr.DataArray, np.ndarray, float]:
    """
    This finds the temperature at pressure levels `p_lev` following a lapse rate $\Gamma(p) = \Gamma_p(p, T_p(p)) + \eta$
    where $\Gamma_p(p, T)$ is the parcel (moist adiabatic) lapse rate and $\eta$ is a constant. $T_p(p)$ refers
    to parcel temperature at pressure $p$.

    This assumes hydrostatic balance: $\Gamma(p) = −\\frac{dT}{dz} = \\frac{g}{R} \\frac{d \ln T}{d\ln p}$

    Args:
        p_lev: `[n_lev]` Pressure levels to find environmental temperature. Units: Pa.
        p_low: Pressure level where to start the ascent from along the modified parcel profile. Units: Pa.
        lapse_diff_const: Constant, $\eta$, which is added to the parcel lapse rate at each pressure level. Units: K/m.
        temp_parcel_low: Temperature of the parcel at `p_low`. Only required if `temp_parcel_lev` is `None`.
        temp_parcel_lev: `[n_lev]` Parcel temperature at pressure `p_lev`. Units: K.</br>
            If not provided, will compute according to $MSE^*(T(p)) = MSE^*(T(p_{lower}))$

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
        sphum_low = sphum_sat(temp_parcel_low, p_low)
        for i in range(temp_parcel_lev.size):
            temp_parcel_lev[i] = get_temp_adiabat(temp_parcel_low, sphum_low, p_low, p_lev[i])
    return get_temp_const_lapse(p_lev, temp_parcel_lev, p_low, lapse_diff_const)


def mod_parcel_lapse_fitting(temp_env_lev: Union[xr.DataArray, np.ndarray],
                             p_lev: Union[xr.DataArray, np.ndarray],
                             temp_env_lower: float, p_lower: float,
                             temp_env_upper: float, p_upper: float,
                             temp_parcel_lev: Optional[Union[xr.DataArray, np.ndarray]] = None,
                             temp_parcel_lower: Optional[float] = None, temp_parcel_upper: Optional[float] = None,
                             n_lev_above_upper_integral: int = 0,
                             sanity_check: bool = False) -> Tuple[
    float, float, float]:
    """
    Find the constant, $\eta$ that needs adding to parcel lapse rate such that
    $\int_{p_1}^{p_2} \Gamma_{env}(p) d\ln p = \int_{p_1}^{p_2} \Gamma_p(p, T_p(p)) + \eta d\ln p$.
    Then computes the error in this approximation:
    $\int_{p_1}^{p_2} |\Gamma_{env}(p) - \Gamma_p(p, T_p(p)) - \eta| d\ln p$.

    where $\Gamma_p(p, T)$ is the parcel (moist adiabatic) lapse rate and $T_p(p)$ is the parcel temperature
    at pressure $p$ starting at $p_1$.

    Args:
        temp_env_lev: `[n_lev]` Environmental temperature at pressure `p_lev`.
        p_lev: `[n_lev]` Model pressure levels in Pa.
        temp_env_lower: Environmental temperature at lower pressure level (nearer surface) `p_lower`.
        p_lower: Pressure level to start profile (near surface).
        temp_env_upper: Environmental temperature at upper pressure level (further from surface) `p_upper`.
        p_upper: Pressure level to end profile (further from surface).
        temp_parcel_lev: `[n_lev]` Parcel temperature (following moist adiabat) at pressure `p_lev`.</br>
            If not provided, will compute according to $MSE^*(T(p)) = MSE^*(T(p_{lower}))$
        temp_parcel_lower: Parcel temperature at lower pressure level (nearer surface) `p_lower`.</br>
            If not provided, will set to `temp_env_lower`.
        temp_parcel_upper: Parcel temperature at upper pressure level (further from surface) `p_upper`.</br>
            If not provided, will compute according to $MSE^*(T(p_{upper})) = MSE^*(T(p_{lower}))$
        n_lev_above_upper_integral: Will return `integral` and `integral_error` not up to `p_upper` but up to
            the model level pressure `n_lev_above_upper_integral` further from the surface than `p_upper`.
            If `n_lev_above_upper_integral=0`, upper limit of integral will be `p_upper`.
        sanity_check: If `True` will print a sanity check to ensure the calculation is correct.

    Returns:
        lapse_diff_const: Lapse rate adjustment, $\eta$ which needs to be added to $\Gamma_{p}(p, T_p(p))$
            so integral matches that of environmental lapse rate. Units are *K/km*.
        integral: Result of integral $\int_{p_1}^{p_2} \Gamma_{env}(p) d\ln p$. Units are *K/km*.
        integral_error: Result of integral $\int_{p_1}^{p_2} |\Gamma_{env}(p) - \Gamma_p(p) - \eta| d\ln p$.
            Units are *K/km*.
        temp_env_approx_lev: `[n_lev]` Estimate of environmental temperature at pressure `p_lev`.
    """
    if n_lev_above_upper_integral == 0:
        p_integ_upper = p_upper
    else:
        if n_lev_above_upper_integral < 0:
            raise ValueError('n_lev_above_upper_integral must be greater than 0')
        if p_lev[1] < p_lev[0]:
            raise ValueError('p_lev[1] must be greater than p_lev[0]')
        ind_upper = np.where(p_lev < p_upper)[0][-n_lev_above_upper_integral]
        p_integ_upper = p_lev[ind_upper]

    # Get parcel profile following moist adiabat, starting saturated at p_lower
    if temp_parcel_lower is None:
        temp_parcel_lower = temp_env_lower
    sphum_parcel_lower = sphum_sat(temp_parcel_lower, p_lower)
    if temp_parcel_upper is None:
        temp_parcel_upper = get_temp_adiabat(temp_parcel_lower, sphum_parcel_lower, p_lower, p_upper)

    if temp_parcel_lev is None:
        temp_parcel_lev = np.zeros_like(temp_env_lev)
        for i in range(temp_parcel_lev.size):
            if (p_lev[i] > p_lower) | (p_lev[i] < p_integ_upper):
                # If not in the pressure range required for integral, then keep as zero
                continue
            temp_parcel_lev[i] = get_temp_adiabat(temp_parcel_lower, sphum_parcel_lower, p_lower, p_lev[i])

    # Compute integral of deviation between environmental and parcel lapse rate between p_lower and p_upper
    lapse_diff_integral = g / R * (
            np.log(temp_env_upper / temp_env_lower) - np.log(temp_parcel_upper / temp_parcel_lower))
    # Compute the constant needed to be added to the parcel lapse rate at each level to make above integral equal 0.
    lapse_diff_const = lapse_diff_integral / np.log(p_upper / p_lower)
    temp_env_approx_lev = get_temp_mod_parcel_lapse(p_lev, p_lower, lapse_diff_const, temp_parcel_lev=temp_parcel_lev)
    temp_env_approx_upper = get_temp_mod_parcel_lapse(p_upper, p_lower, lapse_diff_const,
                                                      temp_parcel_lev=temp_parcel_upper)

    if sanity_check:
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
                    temp_env_lower: float, p_lower: float,
                    temp_env_upper: float, p_upper: float,
                    temp_env_upper2: float, p_upper2: float,
                    temp_parcel_lev: Optional[Union[xr.DataArray, np.ndarray]] = None,
                    temp_parcel_lower: Optional[float] = None,
                    temp_parcel_upper: Optional[float] = None, temp_parcel_upper2: Optional[float] = None,
                    method_layer1: Literal['const', 'mod_parcel'] = 'const',
                    method_layer2: Literal['const', 'mod_parcel'] = 'const',
                    n_lev_above_upper2_integral: int = 0,
                    sanity_check: bool = False) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray]:
    """
    Applies `const_lapse_fitting` or `mod_parcel_lapse_fitting` to each layer.

    Args:
        temp_env_lev: `[n_lev]` Environmental temperature at pressure `p_lev`.
        p_lev: `[n_lev]` Model pressure levels in Pa.
        temp_env_lower: Environmental temperature at lower pressure level (nearer surface) `p_lower`.
        p_lower: Pressure level to start profile (near surface).
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
        sanity_check: If `True` will print a sanity check to ensure the calculation is correct.

    Returns:
        lapse: Lapse rate info for each layer. Bulk lapse rate if `method_layer='const'` or lapse rate adjustment
            if `method_layer='mod_parcel'`. Units are *K/km*.
        integral: Result of integral $\int_{p_1}^{p_2} \Gamma_{env}(p) d\ln p$ of each layer. Units are *K/km*.
        integral_error: Result of integral $\int_{p_1}^{p_2} |\Gamma_{env}(p) - \Gamma_{approx}| d\ln p$ of each layer.
            Units are *K/km*.
        temp_env_approx_lev: `[n_lev]` Estimate of environmental temperature at pressure `p_lev`.
    """
    if method_layer1 == 'const':
        lapse1, lapse_integral1, lapse_integral_error1 = \
            const_lapse_fitting(temp_env_lev, p_lev, temp_env_lower, p_lower, temp_env_upper, p_upper, sanity_check)
    elif method_layer2 == 'mod_parcel':
        lapse1, lapse_integral1, lapse_integral_error1 = \
            mod_parcel_lapse_fitting(temp_env_lev, p_lev, temp_env_lower, p_lower, temp_env_upper, p_upper,
                                     temp_parcel_lev, temp_parcel_lower, temp_parcel_upper, sanity_check)
    else:
        raise ValueError(f'method_layer1 = {method_layer1} not recognized.')

    if method_layer2 == 'const':
        lapse2, lapse_integral2, lapse_integral_error2 = \
            const_lapse_fitting(temp_env_lev, p_lev, temp_env_upper, p_upper, temp_env_upper2, p_upper2,
                                n_lev_above_upper2_integral, sanity_check)
    elif method_layer2 == 'mod_parcel':
        lapse2, lapse_integral2, lapse_integral_error2 = \
            mod_parcel_lapse_fitting(temp_env_lev, p_lev, temp_env_upper, p_upper, temp_env_upper2, p_upper2,
                                     temp_parcel_lev, temp_parcel_upper, temp_parcel_upper2,
                                     n_lev_above_upper2_integral,
                                     sanity_check)
    else:
        raise ValueError(f'method_layer2 = {method_layer2} not recognized.')

    lapse = np.asarray([lapse1, lapse2])
    lapse_integral = np.asarray([lapse_integral1, lapse_integral2])
    lapse_integral_error = np.asarray([lapse_integral_error1, lapse_integral_error2])
    return lapse, lapse_integral, lapse_integral_error


def get_temp_2_layer_approx(p_lev, temp_env_lower, p_lower, p_upper, lapse_layer1, lapse_layer2,
                            method_layer1: Literal['const', 'mod_parcel'] = 'const',
                            method_layer2: Literal['const', 'mod_parcel'] = 'const') -> np.ndarray:
    # Expect lapse rates in K/m not K/km
    if method_layer1 == 'const':
        temp_lev = get_temp_const_lapse(p_lev, temp_env_lower, p_lower, lapse_layer1)
        temp_upper = get_temp_const_lapse(p_upper, temp_env_lower, p_lower, lapse_layer1)
    elif method_layer2 == 'mod_parcel':
        temp_lev = get_temp_mod_parcel_lapse(p_lev, p_lower, lapse_layer1,
                                             temp_parcel_low=temp_env_lower)
        temp_upper = get_temp_mod_parcel_lapse(p_upper, p_lower, lapse_layer1,
                                               temp_parcel_low=temp_env_lower)
    else:
        raise ValueError(f'method_layer1 = {method_layer1} not recognized.')

    if method_layer2 == 'const':
        temp_lev = np.where(p_lev < p_upper, get_temp_const_lapse(p_lev, temp_upper, p_upper, lapse_layer2),
                            temp_lev)
    elif method_layer2 == 'mod_parcel':
        temp_lev = np.where(p_lev < p_upper, get_temp_mod_parcel_lapse(p_lev, p_upper, lapse_layer2,
                                                                       temp_parcel_low=temp_upper),
                            temp_lev)
    else:
        raise ValueError(f'method_layer2 = {method_layer2} not recognized.')
    return temp_lev


def const_lapse_2_layer_fitting_xr(temp_env_lev: xr.DataArray,
                                   p_lev: xr.DataArray,
                                   temp_env_lower: xr.DataArray, p_lower: xr.DataArray,
                                   temp_env_upper: xr.DataArray, p_upper: xr.DataArray,
                                   temp_env_upper2: xr.DataArray, p_upper2: float,
                                   n_lev_above_upper2_integral: int = 0,
                                   sanity_check: bool = False, lev_dim: str = 'lev',
                                   layer_dim: str = 'layer') -> Tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    return xr.apply_ufunc(
        fitting_2_layer,
        temp_env_lev, p_lev,  # (lat, lon, lev)
        temp_env_lower, p_lower, temp_env_upper, p_upper, temp_env_upper2,  # (lat, lon)
        input_core_dims=[[lev_dim], [lev_dim], [], [], [], [], []],
        # only 'temp_env_lev' and 'lev' varies along 'lev_dim'.
        output_core_dims=[[layer_dim], [layer_dim], [layer_dim]],
        vectorize=True,
        kwargs={'p_upper2': p_upper2, 'n_lev_above_upper2_integral': n_lev_above_upper2_integral,
                'sanity_check': sanity_check}
    )
