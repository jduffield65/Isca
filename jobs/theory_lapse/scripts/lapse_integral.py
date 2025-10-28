import numpy as np
import xarray as xr
from typing import Union, Optional, Tuple, Literal
from isca_tools.utils.constants import g, R


def integral_lapse_dlnp_hydrostatic(temp_lev: Union[xr.DataArray, np.ndarray], p_lev: Union[xr.DataArray, np.ndarray],
                                    p1: float, p2: float, T_p1: float, T_p2: float,
                                    temp_ref_lev: Optional[Union[xr.DataArray, np.ndarray]] = None,
                                    temp_ref_p1: Optional[float] = None, temp_ref_p2: Optional[float] = None,
                                    take_abs: bool = False) -> float:
    """
    Compute ∫_{p1}^{p2} (-dT/dz) dlnp using the hydrostatic relation (converted to pressure integral) only (no Z required).
    Can also compute ∫_{p1}^{p2} lapse - lapse_ref dlnp

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
        take_abs: If `True`, and provide `temp_ref_lev`, will compute ∫_{p1}^{p2} |lapse - lapse_ref| dlnp.
            Otherwise, will compute ∫_{p1}^{p2} lapse - lapse_ref dlnp.

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


def get_temp_const_lapse(p_lev, temp_low, p_low, lapse):
    """
    Get the temperature at `p_lev` assuming constant lapse rate up from `temp_low` at `p_low`.

    Args:
        p_lev:
        temp_low:
        p_low:
        lapse:

    Returns:

    """
    return temp_low * (p_lev / p_low) ** (lapse * R / g)


def const_lapse_fitting(temp_env_lev: Union[xr.DataArray, np.ndarray], p_lev: Union[xr.DataArray, np.ndarray],
                        temp_env_lower: float, p_lower: float,
                        temp_env_upper: float, p_upper: float,
                        sanity_check: bool = False) -> Tuple[
    float, float, float, Union[xr.DataArray, np.ndarray]]:
    """
    Find the bulk lapse rate such that $\int_{p_1}^{p^2} \Gamma_{env}(p) d \lnp = \Gamma_{bulk} \ln (p_2/p_1)$.
    Then computes the error in this approximation: $\int_{p_1}^{p^2} |\Gamma_{env}(p) - \Gamma_{bulk}| d \lnp$.

    Args:
        temp_env_lev: `[n_lev]` Environmental temperature at pressure `p_lev`.
        p_lev: `[n_lev]` Model pressure levels in Pa.
        temp_env_lower: Environmental temperature at lower pressure level (nearer surface) `p_lower`.
        p_lower: Pressure level to start profile (near surface).
        temp_env_upper: Environmental temperature at upper pressure level (further from surface) `p_upper`.
        p_upper: Pressure level to end profile (further from surface).
        sanity_check: If `True` will print a sanity check to ensure the calculation is correct.

    Returns:
        lapse_bulk: Bulk lapse rate. Units are *K/km*.
        integral: Result of integral `$\int_{p_1}^{p^2} \Gamma_{env}(p) d \lnp$. Units are *K/km*.
        integral_error: Result of integral `$\int_{p_1}^{p^2} |\Gamma_{env}(p) - \Gamma_{bulk}| d \lnp$.
            Units are *K/km*.
        temp_env_approx_lev: `[n_lev]` Estimate of environmental temperature at pressure `p_lev`.
    """
    # Compute integral of actual environmental lapse rate between p_lower and p_upper
    lapse_integral = integral_lapse_dlnp_hydrostatic(temp_env_lev, p_lev, p_lower, p_upper,
                                                     temp_env_lower, temp_env_upper)
    # Define bulk lapse rate such that a profile following constant lapse rate between p_lower and p_upper
    # would have same value of above integral as actual profile
    lapse_bulk = lapse_integral / np.log(p_upper / p_lower)
    temp_env_approx_lev = get_temp_const_lapse(p_lev, temp_env_lower, p_lower, lapse_bulk)
    temp_env_approx_upper = get_temp_const_lapse(p_upper, temp_env_lower, p_lower, lapse_bulk)

    if sanity_check:
        # sanity check, this should be the same as lapse_integral
        lapse_integral_approx = integral_lapse_dlnp_hydrostatic(temp_env_approx_lev, p_lev, p_lower, p_upper,
                                                                temp_env_lower, temp_env_upper)
        print(
            f'Actual lapse integral: {lapse_integral * 1000:.3f} K/km\nApprox lapse integral: {lapse_integral_approx * 1000:.3f} K/km')
        # Will use lapse rate such that approx value of T_upper is exact. Check that here
        print(f'Actual temp_upper: {temp_env_upper:.3f} K\nApprox temp_upper: {temp_env_approx_upper:.3f} K')

    # Quantify error in approx of constant lapse rate by integral of absolute deviation between actual lapse rate
    # and constant approx value
    lapse_integral_error = integral_lapse_dlnp_hydrostatic(temp_env_lev, p_lev, p_lower, p_upper,
                                                           temp_env_lower, temp_env_upper, temp_env_approx_lev,
                                                           temp_env_lower, temp_env_approx_upper, take_abs=True)
    return lapse_bulk * 1000, lapse_integral * 1000, lapse_integral_error * 1000, temp_env_approx_lev


def get_temp_mod_parcel_lapse(p_lev, temp_parcel_lev, temp_lower, p_lower, temp_parcel_upper, p_upper,
                              lapse_diff_const):
    # Compute temperature at p_upper such that lapse rate at all levels is the same as parcel plus `lapse_diff_const`.
    lapse_parcel_integral = integral_lapse_dlnp_hydrostatic(temp_parcel_lev, p_lev, p_lower, p_upper, temp_lower,
                                                            temp_parcel_upper)
    return temp_lower * (p_upper / p_lower) ** (lapse_diff_const * R / g) * np.exp(R / g * lapse_parcel_integral)


def mod_parcel_lapse_fitting(temp_env_lev: Union[xr.DataArray, np.ndarray],
                             p_lev: Union[xr.DataArray, np.ndarray],
                             temp_env_lower: float, p_lower: float,
                             temp_env_upper: float, p_upper: float,
                             temp_parcel_lev: Union[xr.DataArray, np.ndarray],
                             temp_parcel_lower: float, temp_parcel_upper: float,
                             sanity_check: bool = False) -> Tuple[
    float, float, float, Union[xr.DataArray, np.ndarray]]:
    """
    Find the constant lapse rate, $\Gamma_{mod}$ that needs adding to parcel lapse rate such that
    $\int_{p_1}^{p^2} \Gamma_{env}(p) d \lnp = \int_{p_1}^{p^2} \Gamma_{parcel}(p) + \Gamma_{mod} d \lnp$.
    Then computes the error in this approximation:
    $\int_{p_1}^{p^2} |\Gamma_{env}(p) - \Gamma_{parcel}(p) - Gamma_{mod}| d \lnp$.

    Args:
        temp_env_lev: `[n_lev]` Environmental temperature at pressure `p_lev`.
        p_lev: `[n_lev]` Model pressure levels in Pa.
        temp_env_lower: Environmental temperature at lower pressure level (nearer surface) `p_lower`.
        p_lower: Pressure level to start profile (near surface).
        temp_env_upper: Environmental temperature at upper pressure level (further from surface) `p_upper`.
        p_upper: Pressure level to end profile (further from surface).
        temp_parcel_lev: `[n_lev]` Parcel temperature (following moist adiabat) at pressure `p_lev`.
        temp_parcel_lower: Parcel temperature at lower pressure level (nearer surface) `p_lower`.
        temp_parcel_upper: Parcel temperature at upper pressure level (further from surface) `p_upper`.
        sanity_check: If `True` will print a sanity check to ensure the calculation is correct.

    Returns:
        lapse_diff_const: Lapse rate adjustment, $\Gamma_{mod}$ which needs to be added to $\Gamma_{parcel}(p)$
            so integral matches that of environmental lapse rate. Units are *K/km*.
        integral: Result of integral `$\int_{p_1}^{p^2} \Gamma_{env}(p) d \lnp$. Units are *K/km*.
        integral_error: Result of integral `$\int_{p_1}^{p^2} |\Gamma_{env}(p) - \Gamma_{parcel}(p) - \Gamma_{mod}| d \lnp$.
            Units are *K/km*.
        temp_env_approx_lev: `[n_lev]` Estimate of environmental temperature at pressure `p_lev`.
    """
    # Compute integral of deviation between environmental and parcel lapse rate between p_lower and p_upper
    lapse_diff_integral = integral_lapse_dlnp_hydrostatic(temp_env_lev, p_lev, p_lower, p_upper, temp_env_lower,
                                                          temp_env_upper, temp_parcel_lev, temp_parcel_lower,
                                                          temp_parcel_upper)
    # Compute the constant needed to be added to the parcel lapse rate at each level to make above integral equal 0.
    lapse_diff_const = lapse_diff_integral / np.log(p_upper / p_lower)
    temp_env_approx_lev = np.asarray([get_temp_mod_parcel_lapse(p_lev, temp_parcel_lev, temp_env_lower, p_lower,
                                                                float(temp_parcel_lev[i]), float(p_lev[i]),
                                                                lapse_diff_const)
                                      for i in range(len(p_lev))])
    temp_env_approx_upper = get_temp_mod_parcel_lapse(p_lev, temp_parcel_lev, temp_env_lower, p_lower,
                                                      temp_parcel_upper, p_upper, lapse_diff_const)

    lapse_integral = integral_lapse_dlnp_hydrostatic(temp_env_lev, p_lev, p_lower, p_upper,
                                                     temp_env_lower, temp_env_upper)
    if sanity_check:
        # sanity check, this should be the same as lapse_integral
        lapse_integral_approx = integral_lapse_dlnp_hydrostatic(temp_env_approx_lev, p_lev, p_lower, p_upper,
                                                                temp_env_lower, temp_env_upper)
        print(
            f'Actual lapse integral: {lapse_integral * 1000:.3f} K/km\nApprox lapse integral: {lapse_integral_approx * 1000:.3f} K/km')
        # Will use lapse rate such that approx value of T_upper is exact. Check that here
        print(f'Actual temp_upper: {temp_env_upper:.3f} K\nApprox temp_upper: {temp_env_approx_upper:.3f} K')

    # Quantify error in approx of constant lapse rate by integral of absolute deviation between actual lapse rate
    # and constant approx value
    lapse_integral_diff_abs = integral_lapse_dlnp_hydrostatic(temp_env_lev, p_lev, p_lower, p_upper,
                                                              temp_env_lower, temp_env_upper, temp_env_approx_lev,
                                                              temp_env_lower, temp_env_approx_upper, take_abs=True)
    return lapse_diff_const * 1000, lapse_integral * 1000, lapse_integral_diff_abs * 1000, temp_env_approx_lev


def fitting_2_layer(temp_env_lev: Union[xr.DataArray, np.ndarray], p_lev: Union[xr.DataArray, np.ndarray],
                    temp_env_lower: float, p_lower: float,
                    temp_env_upper: float, p_upper: float,
                    temp_env_upper2: float, p_upper2: float,
                    temp_parcel_lev: Optional[Union[xr.DataArray, np.ndarray]] = None,
                    temp_parcel_lower: Optional[float] = None,
                    temp_parcel_upper: Optional[float] = None, temp_parcel_upper2: Optional[float] = None,
                    method_layer1: Literal['const', 'mod_parcel'] = 'const',
                    method_layer2: Literal['const', 'mod_parcel'] = 'const',
                    sanity_check: bool = False) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, Union[xr.DataArray, np.ndarray]]:
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
        integral: Result of integral `$\int_{p_1}^{p^2} \Gamma_{env}(p) d \lnp$ of each layer. Units are *K/km*.
        integral_error: Result of integral `$\int_{p_1}^{p^2} |\Gamma_{env}(p) - \Gamma_{approx}| d \lnp$ of each layer.
            Units are *K/km*.
        temp_env_approx_lev: `[n_lev]` Estimate of environmental temperature at pressure `p_lev`.
    """
    if method_layer1 == 'const':
        lapse1, lapse_integral1, lapse_integral_error1, temp_env_approx1 = \
            const_lapse_fitting(temp_env_lev, p_lev, temp_env_lower, p_lower, temp_env_upper, p_upper, sanity_check)
    elif method_layer2 == 'mod_parcel':
        lapse1, lapse_integral1, lapse_integral_error1, temp_env_approx1 = \
            mod_parcel_lapse_fitting(temp_env_lev, p_lev, temp_env_lower, p_lower, temp_env_upper, p_upper,
                                     temp_parcel_lev, temp_parcel_lower, temp_parcel_upper, sanity_check)
    else:
        raise ValueError(f'method_layer1 = {method_layer1} not recognized.')

    if method_layer2 == 'const':
        lapse2, lapse_integral2, lapse_integral_error2, temp_env_approx2 = \
            const_lapse_fitting(temp_env_lev, p_lev, temp_env_upper, p_upper, temp_env_upper2, p_upper2,
                                sanity_check)
    elif method_layer2 == 'mod_parcel':
        lapse2, lapse_integral2, lapse_integral_error2, temp_env_approx2 = \
            mod_parcel_lapse_fitting(temp_env_lev, p_lev, temp_env_upper, p_upper, temp_env_upper2, p_upper2,
                                     temp_parcel_lev, temp_parcel_upper, temp_parcel_upper2, sanity_check)
    else:
        raise ValueError(f'method_layer2 = {method_layer2} not recognized.')

    lapse = np.asarray([lapse1, lapse2])
    lapse_integral = np.asarray([lapse_integral1, lapse_integral2])
    lapse_integral_error = np.asarray([lapse_integral_error1, lapse_integral_error2])
    temp_env_approx = np.where(p_lev < p_upper, temp_env_approx2, temp_env_approx1)
    return lapse, lapse_integral, lapse_integral_error, temp_env_approx
