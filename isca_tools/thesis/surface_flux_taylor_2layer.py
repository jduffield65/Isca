import numpy as np
import xarray as xr
import scipy.optimize
from typing import Union, Tuple, Optional

from ..convection import potential_temp
from ..utils.moist_physics import get_density, sphum_sat, clausius_clapeyron_factor
from ..utils.constants import L_v, c_p, kappa, Stefan_Boltzmann, R, g
from .surface_flux_taylor import reconstruct_flux, first_non_none_key, name_square, name_nl

def get_p_eff(p_surf: Union[xr.DataArray, np.ndarray, float], temp: float=280,
              lapse_rate: float=6.5 / 1000, p_alpha_calc: float=1000*100):
    r"""Calculates the effective pressure for the saturation-specific-humidity distribution in a column.

    Accounts for the concentration of saturation specific humidity, $q^*$,
    near the surface by defining a characteristic pressure that is a fraction
    of the surface pressure.

    Args:
        p_surf: Surface pressure.
        temp: Temperature used to calculate the Clausius--Clapeyron factor, in
            K. Defaults to $280$ K.
        lapse_rate: Atmospheric temperature lapse rate, in $\mathrm{K\,m^{-1}}$.
            Defaults to $6.5 \times 10^{-3}\ \mathrm{K\\,m^{-1}}$.
        p_alpha_calc: Pressure at which to calculate the Clausius--Clapeyron
            factor, in Pa. Defaults to $100000$ Pa. Largely insensitive.

    Returns:
        Effective pressure with the same type, dimensions, and coordinates as
        `p_surf`.

    Notes:
        The effective pressure is calculated as

        $$
        p_{\mathrm{eff}} =
        \frac{\beta + 1}{\beta + 2} p_{\mathrm{surf}},
        $$

        where

        $$
        \beta =
        \alpha(T, p_{\alpha}) \\Gamma
        \frac{R T}{g}.
        $$

        Here, $\alpha$ is the Clausius--Clapeyron factor, $\Gamma$ is the
        lapse rate, $R$ is the dry-air gas constant, and $g$ is gravitational
        acceleration.
    """
    beta = clausius_clapeyron_factor(temp, p_alpha_calc) * lapse_rate * R * temp / g
    return (beta + 1) / (beta + 2) * p_surf


def get_temp_from_sphum_sat(sphum_sat_target: np.ndarray, p: np.ndarray, guess_temp: float = 280):
    r"""Calculates temperature from a target saturation specific humidity.

    Numerically inverts `sphum_sat` to find the temperature at which the
    saturation specific humidity equals `sphum_sat_target`, at pressure `p`.
    Uses `scipy.optimize.fsolve` independently at each element.

    Args:
        sphum_sat_target: Target saturation specific humidity, in
            $\mathrm{kg\,kg^{-1}}$.
        p: Pressure at which to evaluate saturation specific humidity, in
            $\mathrm{Pa}$. Must be broadcast-compatible with
            `sphum_sat_target`.
        guess_temp: Initial temperature guess supplied to the numerical solver,
            in $\mathrm{K}$. Defaults to $280\ \mathrm{K}$.

    Returns:
        Temperature for which the saturation specific humidity satisfies

        $$
        q^*(T, p) = q^*_{\mathrm{target}},
        $$

        in $\mathrm{K}$. The output has the same shape as
        `sphum_sat_target`.

    Notes:
        The function solves

        $$
        f(T) = q^*(T, p) - q^*_{\mathrm{target}} = 0
        $$

        using `scipy.optimize.fsolve`. Convergence depends on the initial
        temperature estimate and on whether the requested target humidity is
        physically attainable at the supplied pressure.
    """
    fit_func = lambda x: sphum_sat(x, p) - sphum_sat_target
    return scipy.optimize.fsolve(fit_func, np.full_like(sphum_sat_target, guess_temp))


def get_latent_heat(
        temp_surf: Union[float, np.ndarray, xr.DataArray],
        temp_atm: Union[float, np.ndarray, xr.DataArray],
        rh_atm: Union[float, np.ndarray, xr.DataArray],
        w_atm: Union[float, np.ndarray, xr.DataArray],
        drag_coef: Union[float, np.ndarray, xr.DataArray],
        p_surf: Union[float, np.ndarray, xr.DataArray],
        p_atm: Union[float, np.ndarray, xr.DataArray],
        evap_prefactor: float = 1,
) -> Union[float, np.ndarray, xr.DataArray]:
    """Compute the surface latent heat flux using a bulk aerodynamic formula.
    This function uses the bulk exchange estimate:

    $$
    LH = \\beta\\, L_v\\, C_E\\, \\rho\\, U\\, \\left(q_s^* - q_a\\right).
    $$

    Here, the near-surface atmospheric state is diagnosed as: $T_a = T_s - T_{dq}$
    and the near-surface atmospheric specific humidity is computed from relative
    humidity, $r_a$: $q_a = r_a q^*(T_a, p_a)$, while the surface saturation specific humidity is $q_s^* = q^*(T_s, p_s)$.

    Args:
        temp_surf: Surface temperature, $T_s$ (K)
        temp_atm: Surface–air temperature disequilibrium, $T_a$ (K)
        rh_atm: Near-surface relative humidity, $r_a$ (unitless, 0–1)
        w_atm: Near-surface wind speed, $U$ (m s$^{-1}$)
        drag_coef: Bulk exchange coefficient, $C_E$ (unitless)
        p_surf: Surface pressure, $p_s$ (Pa)
        p_atm: Near-surface atmospheric pressure, $p_a$ (Pa)
        evap_prefactor: Evaporation prefactor, $\\beta$ (unitless)

    Returns:
        flux_lh: Latent heat flux, $LH$ (W m$^{-2}$)

    """
    rho_atm = get_density(temp_atm, p_atm)
    q_atm = rh_atm * sphum_sat(temp_atm, p_atm)
    q_surf = sphum_sat(temp_surf, p_surf)  # sat specific humidity at surface
    return evap_prefactor * L_v * drag_coef * rho_atm * w_atm * (q_surf - q_atm)


def get_sensitivity_lh(
        temp_surf: Union[float, np.ndarray, xr.DataArray],
        temp_atm: Union[float, np.ndarray, xr.DataArray],
        rh_atm: Union[float, np.ndarray, xr.DataArray],
        w_atm: Union[float, np.ndarray, xr.DataArray],
        drag_coef: Union[float, np.ndarray, xr.DataArray],
        p_surf: Union[float, np.ndarray, xr.DataArray],
        sigma_atm: float,
        evap_prefactor: float = 1,
) -> dict:
    """Compute sensitivities of latent heat flux to bulk-exchange parameters.

    Uses the bulk aerodynamic latent heat flux $LH = \\beta L_v C_E \\rho_a U (q_s^* - q_a)$,
    with $p_a = \\sigma_a p_s$.

    Returns a dictionary containing (i) first-order partial derivatives of $LH$
    with respect to each input parameter (holding the others fixed) and (ii)
    selected second-order / mixed nonlinear terms used in a Taylor-series
    decomposition.

    Args:
        temp_surf: Surface temperature, $T_s$ (K)
        temp_atm: Surface–air temperature disequilibrium, $T_a$ (K)
        rh_atm: Near-surface relative humidity, $RH$ (unitless, 0–1)
        w_atm: Near-surface wind speed, $U$ (m s$^{-1}$)
        drag_coef: Bulk exchange coefficient, $C_E$ (unitless)
        p_surf: Surface pressure, $p_s$ (Pa)
        sigma_atm: Sigma coordinate for the near-surface atmosphere, $\\sigma_a$ (unitless),
            used to set $p_a = \\sigma_a p_s$
        evap_prefactor: Evaporation prefactor, $\\beta$ (unitless)

    Returns:
        sensitivity_factors: Dictionary of sensitivities and nonlinear terms. Values have the same
            type/shape as the broadcasted inputs (float, NumPy array, or xarray DataArray).

            First-order terms (partials):

            - evap_prefactor: $\\partial LH / \\partial \\beta$
            - drag_coef: $\\partial LH / \\partial C_E$
            - w_atm: $\\partial LH / \\partial U$
            - p_surf: $\\partial LH / \\partial p_s$ (set to 0 here)
            - rh_atm: $\\partial LH / \\partial r_a$
            - temp_atm: $\\partial LH / \\partial T_a$
            - temp_surf: $\\partial LH / \\partial T_s$

            Nonlinear / interaction terms:

            - nl_temp_surf_square: quadratic term in $T_s$ (includes the $1/2$ factor)
            - nl_temp_surf_<key>: mixed terms between $T_s$ and each of $\\alpha$, $C_E$, $U$, $p_s$
            - nl_temp_surf_rh_atm: mixed term between $T_s$ and $RH$
            - nl_temp_surf_temp_diseqb: mixed term between $T_s$ and $T_{dq}$

    """
    p_atm = p_surf * sigma_atm
    alpha_surf = clausius_clapeyron_factor(temp_surf, p_surf)
    q_surf = sphum_sat(temp_surf, p_surf)
    alpha_atm = clausius_clapeyron_factor(temp_atm, p_atm)
    q_atm_sat = sphum_sat(temp_atm, p_atm)
    q_atm = rh_atm * q_atm_sat
    rho_atm = get_density(temp_atm, p_atm)
    lh = get_latent_heat(temp_surf, temp_atm, rh_atm, w_atm, drag_coef, p_surf, p_atm, evap_prefactor)
    lh_prefactor = evap_prefactor * L_v * drag_coef * w_atm * rho_atm

    # Differential of lh wrt each param - same order as input args
    out_dict = {'temp_surf': lh_prefactor * alpha_surf * q_surf,
                'temp_atm': -lh / temp_atm - lh_prefactor * q_atm * alpha_atm,
                'rh_atm': -lh_prefactor * q_atm_sat,
                'w_atm': lh / w_atm,
                'drag_coef': lh / drag_coef,
                'p_surf': 0,
                'evap_prefactor': lh / evap_prefactor
                }

    out_dict[name_square('temp_surf')] = lh_prefactor * alpha_surf * q_surf / temp_surf * (alpha_surf * temp_surf - 2)
    out_dict[name_square('temp_surf')] = out_dict[name_square('temp_surf')] * 0.5  # to match the taylor series coef

    out_dict[name_square('temp_atm')] = -out_dict['temp_atm'] / temp_atm + lh / temp_atm ** 2 + \
                                        lh_prefactor * alpha_atm * q_atm * (3 / temp_atm - alpha_atm)
    out_dict[name_square('temp_atm')] = out_dict[name_square('temp_atm')] * 0.5  # to match the taylor series coef

    # Mechanism combinations
    out_dict[name_nl('temp_surf', 'temp_atm')] = -out_dict['temp_surf'] / temp_atm
    return out_dict


def reconstruct_lh(temp_surf_ref: float, temp_atm_ref: float,
                   rh_atm_ref: float, w_atm_ref: float,
                   drag_coef_ref: float, p_surf_ref: float, sigma_atm: float,
                   evap_prefactor_ref: float = 1,
                   temp_surf: Optional[np.ndarray] = None, temp_atm: Optional[np.ndarray] = None,
                   rh_atm: Optional[np.ndarray] = None, w_atm: Optional[np.ndarray] = None,
                   drag_coef: Optional[np.ndarray] = None, p_surf: Optional[np.ndarray] = None,
                   evap_prefactor: Optional[np.ndarray] = None,
                   numerical: bool = False) -> Tuple[float, np.ndarray, np.ndarray, dict]:
    """Reconstruct latent heat flux anomalies from a reference state.

    This function computes a reference latent heat flux $LH_{ref}$ at a scalar
    reference state, then reconstructs anomalies relative to that reference either
    (i) numerically by swapping one (or two) mechanisms at a time into the bulk
    formula or (ii) analytically using a Taylor expansion based on sensitivities
    returned by `get_sensitivity_lh`.

    The near-surface atmospheric pressure is diagnosed with a sigma level:
    $p_a = \\sigma_a p_s$.

    Optional mechanism arrays (e.g. `temp_surf`) are interpreted as alternative
    states to compare against the reference. If an optional mechanism is not
    provided, it is filled with the reference value broadcast to the size of the
    first provided mechanism array.

    Args:
        temp_surf_ref: Reference surface temperature, $T_s$ (K)
        temp_atm_ref: Reference atmospheric temperature, $T_a$ (K)
        rh_atm_ref: Reference near-surface relative humidity, $r_a$ (unitless, 0–1)
        w_atm_ref: Reference near-surface wind speed, $U$ (m s$^{-1}$)
        drag_coef_ref: Reference bulk exchange coefficient, $C_E$ (unitless)
        p_surf_ref: Reference surface pressure, $p_s$ (Pa)
        sigma_atm: Sigma coordinate for the near-surface atmosphere, $\\sigma_a$
            (unitless), used to compute $p_a = \\sigma_a p_s$
        evap_prefactor_ref: Reference evaporation prefactor, $\\beta$ (unitless)
        temp_surf: Alternative surface temperature $T_s$ (K). If None, uses
            `temp_surf_ref` broadcast to the working array size
        temp_atm: Alternative atmospheric temperature $T_a$ (K). If None,
            uses `temp_atm_ref` broadcast to the working array size
        rh_atm: Alternative relative humidity $RH$ (unitless, 0–1). If None, uses
            `rh_atm_ref` broadcast to the working array size
        w_atm: Alternative wind speed $U$ (m s$^{-1}$). If None, uses `w_atm_ref`
            broadcast to the working array size
        drag_coef: Alternative bulk exchange coefficient $C_E$ (unitless). If None,
            uses `drag_coef_ref` broadcast to the working array size
        p_surf: Alternative surface pressure $p_s$ (Pa). If None, uses `p_surf_ref`
            broadcast to the working array size
        evap_prefactor: Alternative evaporation prefactor $\\alpha$ (unitless). If
            None, uses `evap_prefactor_ref` broadcast to the working array size
        numerical: If True, compute contributions by explicitly evaluating
            `get_latent_heat` with one- and two-mechanism substitutions relative to
            the reference. If False, use sensitivities from `get_sensitivity_lh`
            to build a Taylor-series reconstruction (including selected nonlinear
            terms if present in `gamma`)

    Returns:
        lh_ref: Reference latent heat flux $LH_{ref}$ (W m$^{-2}$)
        lh_anom_linear: Sum of linear contributions to the latent heat anomaly
            (W m$^{-2}$)
        lh_anom_nl: Sum of linear plus nonlinear contributions included in the
            reconstruction (W m$^{-2}$)
        info_cont: Dictionary of individual contributions by mechanism and
            interaction term. Always includes `residual`, defined as the
            difference between the full bulk flux anomaly and the reconstructed
            anomaly

    Raises:
        ValueError: If provided optional arrays do not all have the same `.size`
            as the first provided mechanism array, or if the expected key sets do
            not match in the analytical pathway

    """
    return reconstruct_flux(locals(), get_latent_heat, get_sensitivity_lh, sigma_atm, numerical)


def get_sensible_heat(
        temp_surf: Union[float, np.ndarray, xr.DataArray],
        temp_atm: Union[float, np.ndarray, xr.DataArray],
        w_atm: Union[float, np.ndarray, xr.DataArray],
        drag_coef: Union[float, np.ndarray, xr.DataArray],
        p_surf: Union[float, np.ndarray, xr.DataArray],
        p_atm: Union[float, np.ndarray, xr.DataArray],
) -> Union[float, np.ndarray, xr.DataArray]:
    """Compute the surface sensible heat flux using a bulk aerodynamic formula.

    Uses a bulk exchange estimate with near-surface atmospheric temperature
    diagnosed as $T_a$. Air density is computed at
    $(T_a, p_a)$ and the near-surface atmospheric temperature is converted to a
    potential temperature, $\\theta_a$.

    The flux returned by this function is:
    $SH = c_p C_H \\rho_a U (T_s - \\theta_a)$,
    where $c_p$ is the specific heat of air at constant pressure, $C_H$ is a bulk
    transfer coefficient (here `drag_coef`), $\\rho_a$ is near-surface air density,
    and $U$ is near-surface wind speed.

    Args:
        temp_surf: Surface temperature, $T_s$ (K)
        temp_atm: Surface–air temperature disequilibrium, $T_a$ (K).
        w_atm: Near-surface wind speed, $U$ (m s$^{-1}$)
        drag_coef: Bulk transfer coefficient for sensible heat, $C_H$ (unitless)
        p_surf: Surface pressure, $p_s$ (Pa), used as the reference pressure for
            potential temperature
        p_atm: Near-surface atmospheric pressure, $p_a$ (Pa)

    Returns:
        flux_sh: Sensible heat flux, $SH$ (W m$^{-2}$), with the same type/shape as the
            inputs (float, NumPy array, or xarray DataArray), assuming consistent
            broadcasting.

    """
    rho_atm = get_density(temp_atm, p_atm)
    temp_pot_atm = potential_temp(temp_atm, p_atm, p_surf)
    return c_p * drag_coef * rho_atm * w_atm * (temp_surf - temp_pot_atm)


def get_sensitivity_sh(
        temp_surf: Union[float, np.ndarray, xr.DataArray],
        temp_atm: Union[float, np.ndarray, xr.DataArray],
        w_atm: Union[float, np.ndarray, xr.DataArray],
        drag_coef: Union[float, np.ndarray, xr.DataArray],
        p_surf: Union[float, np.ndarray, xr.DataArray],
        sigma_atm: float,
) -> dict:
    """Compute sensitivities of sensible heat flux to bulk-exchange parameters.

    Uses the bulk aerodynamic sensible heat flux $SH = c_p C_H \\rho_a U (T_s - \\theta_a)$,
    with $p_a = \\sigma_a p_s$ where $\\theta_a$ is the
    near-surface atmospheric potential temperature referenced to $p_s$ (as in
    `potential_temp(temp_atm, p_atm, p_surf)` within `get_sensible_heat`).

    Returns a dictionary containing (i) first-order partial derivatives of $SH$
    with respect to each input parameter (holding the others fixed) and (ii)
    selected second-order / mixed nonlinear terms used in a Taylor-series
    decomposition.

    Args:
        temp_surf: Surface temperature, $T_s$ (K)
        temp_atm: Surface–air temperature disequilibrium, $T_a$ (K).
        w_atm: Near-surface wind speed, $U$ (m s$^{-1}$)
        drag_coef: Bulk transfer coefficient for sensible heat, $C_H$ (unitless)
        p_surf: Surface pressure, $p_s$ (Pa)
        sigma_atm: Sigma coordinate for the near-surface atmosphere, $\\sigma_a$ (unitless),
            used to set $p_a = \\sigma_a p_s$

    Returns:
        sensitivity_factors: Dictionary of sensitivities and nonlinear terms. Values have the same
            type/shape as the broadcasted inputs (float, NumPy array, or xarray DataArray).

            First-order terms (partials):

            - temp_surf: $\\partial SH / \\partial T_s$
            - temp_atm: $\\partial SH / \\partial T_a$
            - w_atm: $\\partial SH / \\partial U$
            - drag_coef: $\\partial SH / \\partial C_H$
            - p_surf: $\\partial SH / \\partial p_s$

            Nonlinear / interaction terms:

            - nl_temp_diseqb_square: quadratic term in $T_{dq}$ (includes the $1/2$ factor)
            - nl_temp_surf_temp_diseqb: mixed term between $T_s$ and $T_{dq}$
            - nl_temp_diseqb_<key>: mixed terms between $T_{dq}$ and each of $U$, $C_H$, $p_s$

    """
    p_atm = p_surf * sigma_atm
    rho_atm = get_density(temp_atm, p_atm)
    sh = get_sensible_heat(temp_surf, temp_atm, w_atm, drag_coef, p_surf, p_atm)
    sh_prefactor = c_p * drag_coef * w_atm * rho_atm

    # Differential of sh wrt each param - same order as input args
    out_dict = {'temp_surf': sh_prefactor,
                'temp_atm': -sh_prefactor * temp_surf / temp_atm,
                'w_atm': sh / w_atm,
                'drag_coef': sh / drag_coef,
                'p_surf': sh / p_surf,
                }
    return out_dict


def reconstruct_sh(temp_surf_ref: float, temp_atm_ref: float,
                   w_atm_ref: float,
                   drag_coef_ref: float, p_surf_ref: float, sigma_atm: float,
                   temp_surf: Optional[np.ndarray] = None, temp_atm: Optional[np.ndarray] = None,
                   w_atm: Optional[np.ndarray] = None,
                   drag_coef: Optional[np.ndarray] = None, p_surf: Optional[np.ndarray] = None,
                   numerical: bool = False) -> Tuple[float, np.ndarray, np.ndarray, dict]:
    """Reconstruct sensible heat flux anomalies from a reference state.

    This function computes a reference sensible heat flux $SH_{ref}$ at a scalar
    reference state, then reconstructs anomalies relative to that reference either
    (i) numerically by swapping one (or two) mechanisms at a time into the bulk
    formula or (ii) analytically using a Taylor expansion based on sensitivities
    returned by `get_sensitivity_sh`.

    The near-surface atmospheric pressure is diagnosed with a sigma level:
    $p_a = \\sigma_a p_s$.

    Optional mechanism arrays (e.g. `temp_surf`) are interpreted as alternative
    states to compare against the reference. If an optional mechanism is not
    provided, it is filled with the reference value broadcast to the size of the
    first provided mechanism array.

    Args:
        temp_surf_ref: Reference surface temperature, $T_s$ (K)
        temp_atm_ref: Reference atmospheric temperature, $T_a$ (K)
        w_atm_ref: Reference near-surface wind speed, $U$ (m s$^{-1}$)
        drag_coef_ref: Reference bulk transfer coefficient for sensible heat,
            $C_H$ (unitless)
        p_surf_ref: Reference surface pressure, $p_s$ (Pa)
        sigma_atm: Sigma coordinate for the near-surface atmosphere, $\\sigma_a$
            (unitless), used to compute $p_a = \\sigma_a p_s$
        temp_surf: Alternative surface temperature $T_s$ (K). If None, uses
            `temp_surf_ref` broadcast to the working array size
        temp_atm: Alternative atmospheric temperature $T_a$ (K). If None,
            uses `temp_atm_ref` broadcast to the working array size
        w_atm: Alternative wind speed $U$ (m s$^{-1}$). If None, uses `w_atm_ref`
            broadcast to the working array size
        drag_coef: Alternative bulk transfer coefficient $C_H$ (unitless). If None,
            uses `drag_coef_ref` broadcast to the working array size
        p_surf: Alternative surface pressure $p_s$ (Pa). If None, uses `p_surf_ref`
            broadcast to the working array size
        numerical: If True, compute contributions by explicitly evaluating
            `get_sensible_heat` with one- and two-mechanism substitutions relative
            to the reference. If False, use sensitivities from `get_sensitivity_sh`
            to build a Taylor-series reconstruction (including selected nonlinear
            terms if present in `gamma`)

    Returns:
        sh_ref: Reference sensible heat flux $SH_{ref}$ (W m$^{-2}$)
        sh_anom_linear: Sum of linear contributions to the sensible heat anomaly
            (W m$^{-2}$)
        sh_anom_nl: Sum of linear plus nonlinear contributions included in the
            reconstruction (W m$^{-2}$)
        info_cont: Dictionary of individual contributions by mechanism and
            interaction term. Always includes `residual`, defined as the
            difference between the full bulk flux anomaly and the reconstructed
            anomaly

    Raises:
        ValueError: If provided optional arrays do not all have the same `.size`
            as the first provided mechanism array, or if the expected key sets do
            not match in the analytical pathway

    """
    return reconstruct_flux(locals(), get_sensible_heat, get_sensitivity_sh, sigma_atm, numerical)


def get_temp_rad_atm(olr: Union[float, np.ndarray, xr.DataArray],
                 temp_surf: Union[float, np.ndarray, xr.DataArray],
                 odp_surf: Union[float, np.ndarray, xr.DataArray]) -> Union[float, np.ndarray, xr.DataArray]:
    """
    Compute the (effective) radiative temperature T_r associated with the
    OLR in a
    [gray two-stream framework](https://execlim.github.io/Isca/modules/two_stream_gray_rad.html).

    This function inverts the isothermal-atmosphere form of the two-stream
    solution for the downward flux at the surface plus OLR:

    $$I_-(τ_s) + I_+(τ=0) = 2σ T_r^4 (1 - e^{-τ_s}) + σ e^{-τ_s}T_s^4$$

    where:

    - $I_-(τ_s)$ is the downward longwave flux at the surface (W m^-2),
    - $τ_s$ is the longwave optical depth from TOA to the surface,
    - $σ$ is the Stefan–Boltzmann constant,
    - $T_r$ is the effective radiative temperature (K) that, if the atmosphere
      were isothermal at T_r, would yield the same surface downward flux.
    - $T_s$ is the surface temperature.

    Args:
        olr:
            Upward longwave radiation at top of atmosphere, I_+(τ=0) (W m^-2).
        lwdn_surf:
            Downward longwave radiation at the surface, $I_-(τ_s)$ (W m^-2).
        temp_surf: Surface temperature, $T_s$ (K).
        odp_surf:
            Longwave optical depth at the surface, $τ_s$ (dimensionless).

    Returns:
        temp_rad: Radiative temperature $T_r$ (K) in isothermal atmosphere approximation.
    """
    emission_factor = 1 - np.exp(-odp_surf)
    surf_emission = np.exp(-odp_surf) * Stefan_Boltzmann * temp_surf ** 4
    return ((olr - surf_emission) / (emission_factor * Stefan_Boltzmann)) ** 0.25


def get_lw_atm(
        temp_surf: Union[float, np.ndarray, xr.DataArray],
        temp_rad_surf: Union[float, np.ndarray, xr.DataArray],
        temp_rad_atm: Union[float, np.ndarray, xr.DataArray],
        odp_surf: Union[float, np.ndarray, xr.DataArray],
) -> Union[float, np.ndarray, xr.DataArray]:
    """Compute net longwave flux absorbed by atmosphere in a gray-gas model.

    This implements a simple gray-gas surface longwave budget with an imposed
    surface optical depth. The net longwave absorbed by the atmosphere is written as:

    $LW_{net} = \\sigma T_s^4 - LW^{\\uparrow}(\\tau=0) - LW^{\\downarrow}(\\tau=\\tau_s)$,

    Introducing the emissivity $\\epsilon = 1 - e^{-\\tau_{s}}$, and radiative temperature $T_{rad}$,
    we get:

    $LW_{net} = \\sigma\\ \\epsilon \\[T_s^4 - 2T_{rad}^4 \\]$.

    In this function, the effective radiating temperature is diagnosed from the
    atmospheric temperature using one disequilibrium offset:
    $T_{rad} = T_a - T_{dq,r}$.

    Args:
        temp_surf: Surface temperature, $T_s$ (K)
        temp_atm: Surface–air temperature, $T_a$ (K)
        temp_diseqb_r: Additional radiative disequilibrium offset, $T_{dq,r}$ (K)
        odp_surf: Imposed gray optical depth seen from the surface, $\\tau_{s}$ (unitless)

    Returns:
        lwup_surf_net: Net upward longwave flux at the surface, $LW^{\\uparrow}_{net}$
            (W m$^{-2}$), with the same type/shape as the inputs (float, NumPy array,
            or xarray DataArray), assuming consistent broadcasting.

    """
    emiss_factor = 1 - np.exp(-odp_surf)
    return Stefan_Boltzmann * emiss_factor * (temp_surf ** 4 - temp_rad_atm ** 4 - temp_rad_surf**4)


def get_sensitivity_lw_atm(
        temp_surf: Union[float, np.ndarray, xr.DataArray],
        temp_rad_surf: Union[float, np.ndarray, xr.DataArray],
        temp_rad_atm: Union[float, np.ndarray, xr.DataArray],
        odp_surf: Union[float, np.ndarray, xr.DataArray],
) -> dict:
    """Compute sensitivities of net longwave to gray-gas parameters.

    This function returns first-order partial derivatives and selected second-order
    / mixed nonlinear terms for a gray-gas surface longwave flux with imposed
    surface optical depth.

    The effective radiating temperature is diagnosed as $T_{rad} = T_a - T_{dq,r}$,
    and the gray emissivity factor is $\\epsilon = 1 - e^{-\\tau_s}$, where
    $\\tau_s$ is the imposed optical depth (`odp_surf`).

    The sensitivity factors returned here are intended for use in a Taylor-series
    reconstruction in the same style as `get_sensitivity_sh` (via `name_square`
    and `name_nl` keys).

    Args:
        temp_surf: Surface temperature, $T_s$ (K)
        temp_atm: Surface–air temperature, $T_a$ (K), used in
            $T_{rad} = T_a - T_{dq,r}$
        temp_diseqb_r: Radiative disequilibrium offset, $T_{dq,r}$ (K),
            used in $T_{rad} = T_a - T_{dq,r}$
        odp_surf: Imposed gray optical depth at the surface, $\\tau_s$ (unitless)

    Returns:
        sensitivity_factors: Dictionary of sensitivities and nonlinear terms. Values have the same
            type/shape as the broadcasted inputs (float, NumPy array, or xarray DataArray).

            First-order terms (partials):

            - temp_surf: $\\partial LW_{net}^{\\uparrow} / \\partial T_s$
            - temp_atm: $\\partial LW_{net}^{\\uparrow} / \\partial T_a$
            - temp_diseqb_r: $\\partial LW_{net}^{\\uparrow} / \\partial T_{dq,r}$
            - odp_surf: $\\partial LW_{net}^{\\uparrow} / \\partial \\tau_s$

    """
    emiss_factor = 1 - np.exp(-odp_surf)

    # Differential of sh wrt each param - same order as input args
    out_dict = {'temp_surf': 4 * Stefan_Boltzmann * emiss_factor * temp_surf ** 3,
                'temp_rad_surf': -4 * Stefan_Boltzmann * emiss_factor * temp_rad_surf ** 3,
                'temp_rad_atm': -4 * Stefan_Boltzmann * emiss_factor * temp_rad_atm ** 3,
                'odp_surf': Stefan_Boltzmann * (temp_surf ** 4 - temp_rad_surf ** 4 -
                                                temp_rad_atm ** 4) * np.exp(-odp_surf),
                }

    out_dict[name_square('temp_surf')] = 12 * Stefan_Boltzmann * emiss_factor * temp_surf ** 2
    out_dict[name_square('temp_surf')] = out_dict[name_square('temp_surf')] * 0.5  # to match the taylor series coef

    out_dict[name_square('temp_rad_surf')] = -12 * Stefan_Boltzmann * emiss_factor * temp_rad_surf ** 2
    out_dict[name_square('temp_rad_surf')] = out_dict[name_square('temp_rad_surf')] * 0.5  # to match the taylor series coef

    out_dict[name_square('temp_rad_atm')] = -12 * Stefan_Boltzmann * emiss_factor * temp_rad_atm ** 2
    out_dict[name_square('temp_rad_atm')] = out_dict[name_square('temp_rad_atm')] * 0.5  # to match the taylor series coef

    return out_dict


def reconstruct_lw_atm(temp_surf_ref: float,
                       temp_rad_surf_ref: float, temp_rad_atm_ref: float,
                       odp_surf_ref: float,
                       temp_surf: Optional[np.ndarray] = None,
                       temp_rad_surf: Optional[np.ndarray] = None, temp_rad_atm: Optional[np.ndarray] = None,
                       odp_surf: Optional[np.ndarray] = None,
                       numerical: bool = False) -> Tuple[float, np.ndarray, np.ndarray, dict]:
    """Reconstruct net upward surface longwave anomalies from a reference state.

    This function computes a reference net upward surface longwave flux
    $LW^{\\uparrow}_{net,sfc,ref}$ at a scalar reference state, then reconstructs
    anomalies relative to that reference either (i) numerically by swapping one
    (or two) mechanisms at a time into the gray-gas flux formula or (ii)
    analytically using a Taylor expansion based on sensitivities returned by
    `get_sensitivity_lw`.

    The gray-gas optical depth at the surface is denoted $\\tau_s$ (argument
    `odp_surf`). The effective radiating temperature used by the gray-gas
    parameterization is diagnosed via
    $T_{rad} = T_a - T_{dq,r}$.

    Optional mechanism arrays (e.g. `temp_surf`) are interpreted as alternative
    states to compare against the reference. If an optional mechanism is not
    provided, it is filled with the reference value broadcast to the size of the
    first provided mechanism array.

    Args:
        temp_surf_ref: Reference surface temperature, $T_s$ (K)
        temp_atm_ref: Reference surface–air temperature, $T_a$ (K)
        temp_diseqb_r_ref: Reference radiative disequilibrium offset,
            $T_{dq,r}$ (K)
        odp_surf_ref: Reference imposed gray optical depth, $\\tau_s$ (unitless)
        temp_surf: Alternative surface temperature $T_s$ (K). If None, uses
            `temp_surf_ref` broadcast to the working array size
        temp_atm: Alternative atmospheric temperature $T_{dq}$ (K). If None,
            uses `temp_atm_ref` broadcast to the working array size
        temp_diseqb_r: Alternative radiative disequilibrium offset $T_{dq,r}$ (K).
            If None, uses `temp_diseqb_r_ref` broadcast to the working array size
        odp_surf: Alternative optical depth $\\tau_s$ (unitless). If None, uses
            `odp_surf_ref` broadcast to the working array size
        numerical: If True, compute contributions by explicitly evaluating
            `get_lwup_sfc_net` with one- and two-mechanism substitutions relative
            to the reference. If False, use sensitivities from `get_sensitivity_lw`
            to build a Taylor-series reconstruction (including selected nonlinear
            terms if present in `gamma`)

    Returns:
        lw_ref: Reference net upward surface longwave flux,
            $LW^{\\uparrow}_{net,sfc,ref}$ (W m$^{-2}$)
        lw_anom_linear: Sum of linear contributions to the longwave anomaly
            (W m$^{-2}$)
        lw_anom_nl: Sum of linear plus nonlinear contributions included in the
            reconstruction (W m$^{-2}$)
        info_cont: Dictionary of individual contributions by mechanism and
            interaction term. Always includes `residual`, defined as the
            difference between the full gray-gas flux anomaly and the
            reconstructed anomaly

    Raises:
        ValueError: If provided optional arrays do not all have the same `.size`
            as the first provided mechanism array, or if the expected key sets do
            not match in the analytical pathway

    Notes:
        This wrapper delegates the full computation to `reconstruct_flux`.
        Ensure `reconstruct_flux` is called with the appropriate signature for
        your implementation (e.g. whether it requires `sigma_atm` or passes
        additional keywords to the flux function).

    """
    return reconstruct_flux(locals(), get_lw_atm, get_sensitivity_lw_atm, numerical=numerical)


def get_lw_surf(
        temp_surf: Union[float, np.ndarray, xr.DataArray],
        temp_rad_surf: Union[float, np.ndarray, xr.DataArray],
        odp_surf: Union[float, np.ndarray, xr.DataArray],
) -> Union[float, np.ndarray, xr.DataArray]:
    """Compute net upward longwave flux at the surface in a gray-gas model.

    This implements a simple gray-gas surface longwave budget with an imposed
    surface optical depth. The net upward longwave at the surface is written as:

    $LW^{\\uparrow}_{net} = \\sigma\\left[T_s^4 - LW^{\\downarrow}(\\tau_s)/\\sigma\\right]$,

    where the downwelling longwave is approximated as gray atmospheric emission
    from an effective radiating temperature $T_{rad}$ with emissivity
    $\\epsilon = 1 - e^{-\\tau_{s}}$:

    $LW^{\\downarrow}(\\tau_s) = \\sigma\\, \\epsilon\\, T_{rad}^4
    = \\sigma\\left(1 - e^{-\\tau_{s}}\\right)T_{rad}^4$.

    In this function, the effective radiating temperature is diagnosed from the
    surface temperature using two disequilibrium offsets:
    $T_{rad} = T_a - T_{dq,r}$.

    Note:
        The implementation below returns
        $\\sigma\\left[T_s^4 + \\left(1-e^{-\\tau_{s}}\\right)T_{rad}^4\\right]$.
        This corresponds to treating the atmospheric contribution as an *added*
        upward term; if you intend $LW^{\\uparrow}_{net} = LW^{\\uparrow}(\\tau_s) - LW^{\\downarrow}(\\tau_s)$,
        then the second term typically enters with a minus sign. Keep this sign
        convention consistent with how you define “net upward” elsewhere.

    Args:
        temp_surf: Surface temperature, $T_s$ (K)
        temp_atm: Surface–air temperature, $T_{a}$ (K)
        temp_diseqb_surf: Additional radiative disequilibrium offset, $T_{dq,r}$ (K)
        odp_surf: Imposed gray optical depth seen from the surface, $\\tau_{s}$ (unitless)

    Returns:
        lwup_surf_net: Net upward longwave flux at the surface, $LW^{\\uparrow}_{net}$
            (W m$^{-2}$), with the same type/shape as the inputs (float, NumPy array,
            or xarray DataArray), assuming consistent broadcasting.

    """
    # Effective atmospheric radiating temperature used for gray downwelling LW
    # Gray-gas emissivity for optical depth tau_sfc: epsilon = 1 - exp(-tau_sfc)
    # Downwelling LW at surface would be sigma * epsilon * T_rad^4
    # Surface upwelling LW is sigma * T_s^4
    emiss_factor = 1 - np.exp(-odp_surf)
    return Stefan_Boltzmann * (temp_surf ** 4 - temp_rad_surf ** 4 * emiss_factor)


def get_sensitivity_lw_surf(
        temp_surf: Union[float, np.ndarray, xr.DataArray],
        temp_rad_surf: Union[float, np.ndarray, xr.DataArray],
        odp_surf: Union[float, np.ndarray, xr.DataArray],
) -> dict:
    """Compute sensitivities of net upward surface longwave to gray-gas parameters.

    This function returns first-order partial derivatives and selected second-order
    / mixed nonlinear terms for a gray-gas surface longwave flux with imposed
    surface optical depth.

    The effective radiating temperature is diagnosed as $T_{rad} = T_a - T_{dq,r}$,
    and the gray emissivity factor is $\\epsilon = 1 - e^{-\\tau_s}$, where
    $\\tau_s$ is the imposed optical depth (`odp_surf`).

    The sensitivity factors returned here are intended for use in a Taylor-series
    reconstruction in the same style as `get_sensitivity_sh` (via `name_square`
    and `name_nl` keys).

    Args:
        temp_surf: Surface temperature, $T_s$ (K)
        temp_atm: Surface–air disequilibrium temperature, $T_{dq}$ (K), used in
            $T_{rad} = T_a - T_{dq,r}$
        temp_diseqb_surf: Additional radiative disequilibrium offset, $T_{dq,r}$ (K),
            used in $T_{rad} = T_a - T_{dq,r}$
        odp_surf: Imposed gray optical depth at the surface, $\\tau_s$ (unitless)

    Returns:
        sensitivity_factors: Dictionary of sensitivities and nonlinear terms. Values have the same
            type/shape as the broadcasted inputs (float, NumPy array, or xarray DataArray).

            First-order terms (partials):

            - temp_surf: $\\partial LW_{net,sfc}^{\\uparrow} / \\partial T_s$
            - temp_atm: $\\partial LW_{net,sfc}^{\\uparrow} / \\partial T_a$
            - temp_diseqb_r: $\\partial LW_{net,sfc}^{\\uparrow} / \\partial T_{dq,r}$
            - odp_surf: $\\partial LW_{net,sfc}^{\\uparrow} / \\partial \\tau_s$

            Nonlinear / interaction terms (as included in `out_dict`):

            - nl_temp_surf_square: quadratic term in $T_s$ (includes the $1/2$ factor)
            - nl_temp_atm_square: quadratic term in $T_{dq}$ (includes the $1/2$ factor)
            - nl_temp_diseqb_r_square: quadratic term in $T_{dq,r}$ (includes the $1/2$ factor)
            - nl_odp_surf_square: quadratic term in $\\tau_s$ (includes the $1/2$ factor)
            - nl_temp_surf_temp_diseqb: mixed term between $T_s$ and $T_{dq}$
            - nl_temp_surf_temp_diseqb_r: mixed term between $T_s$ and $T_{dq,r}$
            - nl_temp_surf_odp_surf: mixed term between $T_s$ and $\\tau_s$
            - nl_temp_diseqb_temp_diseqb_r: mixed term between $T_{dq}$ and $T_{dq,r}$
            - nl_temp_diseqb_odp_surf: mixed term between $T_{dq}$ and $\\tau_s$
            - nl_temp_diseqb_r_odp_surf: mixed term between $T_{dq,r}$ and $\\tau_s$

    Notes:
        This function assumes the same sign convention as the corresponding flux
        function used in your reconstruction (e.g. `get_lwup_sfc_net`). Ensure the
        definition of “net upward” longwave used there matches how you interpret
        the derivatives here.

    """
    emiss_factor = 1 - np.exp(-odp_surf)

    # Differential of sh wrt each param - same order as input args
    out_dict = {'temp_surf': 4 * Stefan_Boltzmann * temp_surf ** 3,
                'temp_rad_surf': -4 * Stefan_Boltzmann * temp_rad_surf ** 3 * emiss_factor,
                'odp_surf': -Stefan_Boltzmann * temp_rad_surf ** 4 * np.exp(-odp_surf),
                }

    # Nonlinear contributions
    out_dict[name_square('temp_surf')] = 12 * Stefan_Boltzmann * temp_surf ** 2
    out_dict[name_square('temp_surf')] = out_dict[name_square('temp_surf')] * 0.5  # to match the taylor series coef
    out_dict[name_square('temp_rad_surf')] = -12 * Stefan_Boltzmann * temp_rad_surf ** 2 * emiss_factor
    out_dict[name_square('temp_rad_surf')] = out_dict[name_square('temp_rad_surf')] * 0.5  # to match the taylor series coef

    return out_dict


def reconstruct_lw_surf(temp_surf_ref: float,
                        temp_rad_surf_ref: float, odp_surf_ref: float,
                        temp_surf: Optional[np.ndarray] = None,
                        temp_rad_surf: Optional[np.ndarray] = None,
                        odp_surf: Optional[np.ndarray] = None,
                        numerical: bool = False) -> Tuple[float, np.ndarray, np.ndarray, dict]:
    """Reconstruct net upward surface longwave anomalies from a reference state.

    This function computes a reference net upward surface longwave flux
    $LW^{\\uparrow}_{net,sfc,ref}$ at a scalar reference state, then reconstructs
    anomalies relative to that reference either (i) numerically by swapping one
    (or two) mechanisms at a time into the gray-gas flux formula or (ii)
    analytically using a Taylor expansion based on sensitivities returned by
    `get_sensitivity_lw`.

    The gray-gas optical depth at the surface is denoted $\\tau_s$ (argument
    `odp_surf`). The effective radiating temperature used by the gray-gas
    parameterization is diagnosed via
    $T_{rad} = T_a - T_{dq,r}$.

    Optional mechanism arrays (e.g. `temp_surf`) are interpreted as alternative
    states to compare against the reference. If an optional mechanism is not
    provided, it is filled with the reference value broadcast to the size of the
    first provided mechanism array.

    Args:
        temp_surf_ref: Reference surface temperature, $T_s$ (K)
        temp_atm_ref: Reference surface–air temperature, $T_{dq}$ (K)
        temp_diseqb_surf_ref: Reference additional radiative disequilibrium offset,
            $T_{dq,r}$ (K)
        odp_surf_ref: Reference imposed gray optical depth, $\\tau_s$ (unitless)
        temp_surf: Alternative surface temperature $T_s$ (K). If None, uses
            `temp_surf_ref` broadcast to the working array size
        temp_atm: Alternative atmospheric temperature $T_a$ (K). If None,
            uses `temp_diseqb_ref` broadcast to the working array size
        temp_diseqb_surf: Alternative radiative disequilibrium offset $T_{dq,r}$ (K).
            If None, uses `temp_diseqb_r_ref` broadcast to the working array size
        odp_surf: Alternative optical depth $\\tau_s$ (unitless). If None, uses
            `odp_surf_ref` broadcast to the working array size
        numerical: If True, compute contributions by explicitly evaluating
            `get_lwup_sfc_net` with one- and two-mechanism substitutions relative
            to the reference. If False, use sensitivities from `get_sensitivity_lw`
            to build a Taylor-series reconstruction (including selected nonlinear
            terms if present in `gamma`)

    Returns:
        lw_ref: Reference net upward surface longwave flux,
            $LW^{\\uparrow}_{net,sfc,ref}$ (W m$^{-2}$)
        lw_anom_linear: Sum of linear contributions to the longwave anomaly
            (W m$^{-2}$)
        lw_anom_nl: Sum of linear plus nonlinear contributions included in the
            reconstruction (W m$^{-2}$)
        info_cont: Dictionary of individual contributions by mechanism and
            interaction term. Always includes `residual`, defined as the
            difference between the full gray-gas flux anomaly and the
            reconstructed anomaly

    Raises:
        ValueError: If provided optional arrays do not all have the same `.size`
            as the first provided mechanism array, or if the expected key sets do
            not match in the analytical pathway

    Notes:
        This wrapper delegates the full computation to `reconstruct_flux`.
        Ensure `reconstruct_flux` is called with the appropriate signature for
        your implementation (e.g. whether it requires `sigma_atm` or passes
        additional keywords to the flux function).

    """
    return reconstruct_flux(locals(), get_lw_surf, get_sensitivity_lw_surf, numerical=numerical)
