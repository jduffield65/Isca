import numpy as np
import xarray as xr
from typing import Union, Tuple, Optional, Callable
import copy
import itertools

from ..convection import potential_temp
from ..utils.moist_physics import get_density, sphum_sat, clausius_clapeyron_factor
from ..utils.constants import L_v, c_p, kappa

name_square = lambda x: f"nl_{x}_square"  # name of square cont of individual mechanism
name_nl = lambda x, y: f"nl_{x}_{y}"  # name of combination of two individual mechanism


def get_latent_heat(
        temp_surf: Union[float, np.ndarray, xr.DataArray],
        temp_diseqb: Union[float, np.ndarray, xr.DataArray],
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
        temp_diseqb: Surface–air temperature disequilibrium, $T_{dq}$ (K), used
            in $T_a = T_s - T_{diseqb}$
        rh_atm: Near-surface relative humidity, $r_a$ (unitless, 0–1)
        w_atm: Near-surface wind speed, $U$ (m s$^{-1}$)
        drag_coef: Bulk exchange coefficient, $C_E$ (unitless)
        p_surf: Surface pressure, $p_s$ (Pa)
        p_atm: Near-surface atmospheric pressure, $p_a$ (Pa)
        evap_prefactor: Evaporation prefactor, $\\beta$ (unitless)

    Returns:
        flux_lh: Latent heat flux, $LH$ (W m$^{-2}$)

    """
    temp_atm = temp_surf - temp_diseqb
    rho_atm = get_density(temp_atm, p_atm)
    q_atm = rh_atm * sphum_sat(temp_atm, p_atm)
    q_surf = sphum_sat(temp_surf, p_surf)  # sat specific humidity at surface
    return evap_prefactor * L_v * drag_coef * rho_atm * w_atm * (q_surf - q_atm)


def get_sensitivity_lh(
        temp_surf: Union[float, np.ndarray, xr.DataArray],
        temp_diseqb: Union[float, np.ndarray, xr.DataArray],
        rh_atm: Union[float, np.ndarray, xr.DataArray],
        w_atm: Union[float, np.ndarray, xr.DataArray],
        drag_coef: Union[float, np.ndarray, xr.DataArray],
        p_surf: Union[float, np.ndarray, xr.DataArray],
        sigma_atm: float,
        evap_prefactor: float = 1,
) -> dict:
    """Compute sensitivities of latent heat flux to bulk-exchange parameters.

    Uses the bulk aerodynamic latent heat flux $LH = \\beta L_v C_E \\rho_a U (q_s^* - q_a)$,
    with $p_a = \\sigma_a p_s$ and $T_a = T_s - T_{dq}$.

    Returns a dictionary containing (i) first-order partial derivatives of $LH$
    with respect to each input parameter (holding the others fixed) and (ii)
    selected second-order / mixed nonlinear terms used in a Taylor-series
    decomposition.

    Args:
        temp_surf: Surface temperature, $T_s$ (K)
        temp_diseqb: Surface–air temperature disequilibrium, $T_{dq}$ (K),
            used in $T_a = T_s - T_{dq}$
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
            - temp_diseqb: $\\partial LH / \\partial T_{dq}$
            - temp_surf: $\\partial LH / \\partial T_s$

            Nonlinear / interaction terms:

            - nl_temp_surf_square: quadratic term in $T_s$ (includes the $1/2$ factor)
            - nl_temp_surf_<key>: mixed terms between $T_s$ and each of $\\alpha$, $C_E$, $U$, $p_s$
            - nl_temp_surf_rh_atm: mixed term between $T_s$ and $RH$
            - nl_temp_surf_temp_diseqb: mixed term between $T_s$ and $T_{dq}$

    """
    p_atm = p_surf * sigma_atm
    temp_atm = temp_surf - temp_diseqb
    alpha_surf = clausius_clapeyron_factor(temp_surf, p_surf)
    alpha_atm = clausius_clapeyron_factor(temp_atm, p_atm)
    q_atm_sat = sphum_sat(temp_atm, p_atm)
    q_atm = rh_atm * q_atm_sat
    rho_atm = get_density(temp_atm, p_atm)
    lh = get_latent_heat(temp_surf, temp_diseqb, rh_atm, w_atm, drag_coef, p_surf, p_atm, evap_prefactor)
    lh_prefactor = evap_prefactor * L_v * drag_coef * w_atm * rho_atm

    # Differential of lh wrt each param - same order as input args
    out_dict = {'temp_surf': (alpha_surf - 1 / temp_atm) * lh + lh_prefactor * q_atm * (alpha_surf - alpha_atm),
                'temp_diseqb': lh / temp_atm + lh_prefactor * q_atm * alpha_atm,
                'rh_atm': -lh_prefactor * q_atm_sat,
                'w_atm': lh / w_atm,
                'drag_coef': lh / drag_coef,
                'p_surf': 0,
                'evap_prefactor': lh / evap_prefactor
                }

    # Nonlinear contributions - only temp_surf as this dominates
    out_dict[name_square('temp_surf')] = \
        (alpha_surf - 1 / temp_atm) * out_dict['temp_surf'] + \
        lh * (1 / temp_atm ** 2 - 2 * alpha_surf / temp_surf) + \
        lh_prefactor * q_atm * (2 * alpha_atm / temp_atm -
                                2 * alpha_surf / temp_surf + alpha_atm * (alpha_surf - alpha_atm))
    out_dict[name_square('temp_surf')] = out_dict[name_square('temp_surf')] * 0.5  # to match the taylor series coef

    # Mechanism combinations
    out_dict[name_nl('temp_surf', 'temp_diseqb')] = -lh / temp_atm ** 2 + out_dict['temp_surf'] / temp_atm + \
                                                    lh_prefactor * q_atm * alpha_atm * (alpha_atm - 2 / temp_atm)
    out_dict[name_nl('temp_surf', 'rh_atm')] = -lh_prefactor * alpha_atm * q_atm_sat
    for key in ['w_atm', 'drag_coef', 'p_surf', 'evap_prefactor']:
        out_dict[name_nl('temp_surf', key)] = out_dict[key] * out_dict['temp_surf'] / lh
    return out_dict


def first_non_none_key(d: dict) -> str:
    """
    Return key of the first non-None value in `d`.

    Args:
        d: Dictionary to find first non-None key.

    Returns:
        key: First non-None key.
    """
    for key in d:
        if d[key] is not None:
            return key
    raise ValueError("All dict values are None")


def reconstruct_flux(var_dict: dict, func_flux: Callable, func_sensitivity: Callable,
                     sigma_atm: float, numerical: bool) -> Tuple[float, np.ndarray, np.ndarray, dict]:
    """Reconstruct bulk flux anomalies from a reference state (generic helper).

    This is the general implementation used by `reconstruct_lh` (and can be reused
    for other bulk fluxes such as sensible heat). It takes a dictionary containing
    reference scalars (with names ending in `"_ref"`) plus optional mechanism
    arrays (without `"_ref"`), then reconstructs the flux anomaly relative to the
    reference either:

    - Numerically: by re-evaluating `func_flux` after substituting one mechanism
      at a time (linear terms) and two mechanisms at a time (pairwise nonlinear
      interaction terms)
    - Analytically: by using sensitivity factors from `func_sensitivity` to build
      a Taylor-series reconstruction (including any square and cross terms present
      in the returned sensitivity dictionary)

    The near-surface atmospheric pressure is diagnosed with a sigma level:
    $p_a = \\sigma_a p_s$.

    Input conventions:

    - Reference values must appear in `var_dict` with the suffix `"_ref"`, e.g.
      `"temp_surf_ref"`, `"p_surf_ref"`
    - Mechanism perturbations must appear in `var_dict` without the suffix, e.g.
      `"temp_surf"`, `"p_surf"`, and can be `None` or a NumPy array
    - Any mechanism value that is `None` is filled with its reference value,
      broadcast to the size of the first provided mechanism array. All provided
      mechanism arrays must have the same `.size`

    Args:
        var_dict: Dictionary of variables, typically `locals()` from a wrapper
            such as `reconstruct_lh`. Must include `"<name>_ref"` entries for each
            mechanism. May include optional mechanism arrays under `"<name>"`.
        func_flux: Callable that computes the flux. Must accept the reference
            mechanisms as keyword arguments, and must accept `p_atm` as a keyword
            argument. Example: `get_latent_heat` or `get_sensible_heat`.
        func_sensitivity: Callable returning sensitivity factors used for the
            analytical reconstruction. Must accept the reference mechanisms as
            keyword arguments and `sigma_atm` as a keyword argument. Example:
            `get_sensitivity_lh` or `get_sensitivity_sh`.
        sigma_atm: Sigma coordinate for the near-surface atmosphere, $\\sigma_a$
            (unitless), used to set $p_a = \\sigma_a p_s$.
        numerical: If True, compute contributions by explicit re-evaluation of
            `func_flux`. If False, compute contributions using `func_sensitivity`
            (Taylor-series reconstruction).

    Returns:
        flux_ref: Reference flux evaluated at the reference state (units depend on
            `func_flux`, e.g. W m$^{-2}$).
        flux_anom_linear: Sum of linear contributions to the flux anomaly (same
            units as `flux_ref`).
        flux_anom_nl: Sum of linear plus nonlinear contributions included in the
            reconstruction (same units as `flux_ref`).
        info_cont: Dictionary of individual contributions by mechanism and
            interaction term. Always includes `residual`, defined as the
            difference between the full flux anomaly computed from `vals` and the
            reconstructed anomaly.

    Raises:
        ValueError: If no mechanism arrays are provided (so a broadcast size
            cannot be inferred), if provided mechanism arrays have inconsistent
            `.size`, or if the expected key sets do not match in the analytical
            pathway (`gamma` vs `info_cont`).

    """
    vals_ref = {k.replace('_ref', ''): v for k, v in var_dict.items() if k.endswith("_ref")}
    vals = {k: v for k, v in var_dict.items() if f"{k}" in vals_ref}

    # If no val specified, set to ref value
    key_numpy = first_non_none_key(vals)  # first key of numpy array
    for key in vals:
        if vals[key] is None:
            vals[key] = np.full_like(vals[key_numpy], vals_ref[key])
        elif vals[key].size != vals[key_numpy].size:
            raise ValueError(f"Size mismatch: {key} not the same as {key_numpy}.")

    flux_ref = func_flux(**vals_ref, p_atm=sigma_atm * vals_ref['p_surf'])
    if numerical:
        info_cont = {}
        # Linear contribution of each val
        for key in vals:
            vals_use = copy.deepcopy(vals_ref)
            vals_use[key] = vals[key]
            vals_use['p_atm'] = sigma_atm * vals_use['p_surf']
            info_cont[key] = func_flux(**vals_use) - flux_ref

        # Get non-linear contributions where only two mechanisms are active - include all permutations
        for key1, key2 in itertools.combinations(vals, 2):
            vals_use = copy.deepcopy(vals_ref)
            vals_use[key1] = vals[key1]
            vals_use[key2] = vals[key2]
            vals_use['p_atm'] = sigma_atm * vals_use['p_surf']
            info_cont[name_nl(key1, key2)] = func_flux(**vals_use) - flux_ref
            # Subtract the contribution from the linear mechanisms, so only non-linear contribution remains
            info_cont[name_nl(key1, key2)] -= info_cont[key1] + info_cont[key2]
    else:
        gamma = func_sensitivity(**vals_ref, sigma_atm=sigma_atm)
        vals_anom = {key: vals[key] - vals_ref[key] for key in vals}

        # linear contributions
        info_cont = {key: gamma[key] * vals_anom[key] for key in vals}

        # Adds Squared contribution of individual mechanisms that are in gamma
        for key in vals:
            if name_square(key) in gamma:
                info_cont[name_square(key)] = gamma[name_square(key)] * vals_anom[key] ** 2

        # Adds a nonlinear combination of mechanisms that are included in gamma
        for key1, key2 in itertools.combinations(vals, 2):
            if name_nl(key1, key2) in gamma:
                info_cont[name_nl(key1, key2)] = gamma[name_nl(key1, key2)] * vals_anom[key1] * vals_anom[key2]
        if list(gamma.keys()) != list(info_cont.keys()):
            raise ValueError(f"gamma has keys:\n{list(gamma.keys())}\ninfo_cont has keys:\n{list(info_cont.keys())}")
    final_answer_linear = np.asarray(sum([info_cont[key] for key in info_cont if 'nl' not in key]))
    final_answer_nl = np.asarray(sum([info_cont[key] for key in info_cont]))
    info_cont['residual'] = func_flux(**vals, p_atm=sigma_atm * vals['p_surf']) - flux_ref - final_answer_nl
    return flux_ref, final_answer_linear, final_answer_nl, info_cont


def reconstruct_lh(temp_surf_ref: float, temp_diseqb_ref: float,
                   rh_atm_ref: float, w_atm_ref: float,
                   drag_coef_ref: float, p_surf_ref: float, sigma_atm: float,
                   evap_prefactor_ref: float = 1,
                   temp_surf: Optional[np.ndarray] = None, temp_diseqb: Optional[np.ndarray] = None,
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

    The near-surface atmospheric temperature is diagnosed using a surface–air
    disequilibrium temperature $T_{dq}$:
    $T_a = T_s - T_{dq}$.

    Optional mechanism arrays (e.g. `temp_surf`) are interpreted as alternative
    states to compare against the reference. If an optional mechanism is not
    provided, it is filled with the reference value broadcast to the size of the
    first provided mechanism array.

    Args:
        temp_surf_ref: Reference surface temperature, $T_s$ (K)
        temp_diseqb_ref: Reference disequilibrium temperature, $T_{dq}$ (K)
        rh_atm_ref: Reference near-surface relative humidity, $r_a$ (unitless, 0–1)
        w_atm_ref: Reference near-surface wind speed, $U$ (m s$^{-1}$)
        drag_coef_ref: Reference bulk exchange coefficient, $C_E$ (unitless)
        p_surf_ref: Reference surface pressure, $p_s$ (Pa)
        sigma_atm: Sigma coordinate for the near-surface atmosphere, $\\sigma_a$
            (unitless), used to compute $p_a = \\sigma_a p_s$
        evap_prefactor_ref: Reference evaporation prefactor, $\\beta$ (unitless)
        temp_surf: Alternative surface temperature $T_s$ (K). If None, uses
            `temp_surf_ref` broadcast to the working array size
        temp_diseqb: Alternative disequilibrium temperature $T_{dq}$ (K). If None,
            uses `temp_diseqb_ref` broadcast to the working array size
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
        temp_diseqb: Union[float, np.ndarray, xr.DataArray],
        w_atm: Union[float, np.ndarray, xr.DataArray],
        drag_coef: Union[float, np.ndarray, xr.DataArray],
        p_surf: Union[float, np.ndarray, xr.DataArray],
        p_atm: Union[float, np.ndarray, xr.DataArray],
) -> Union[float, np.ndarray, xr.DataArray]:
    """Compute the surface sensible heat flux using a bulk aerodynamic formula.

    Uses a bulk exchange estimate with near-surface atmospheric temperature
    diagnosed as $T_a = T_s - T_{dq}$. Air density is computed at
    $(T_a, p_a)$ and the near-surface atmospheric temperature is converted to a
    potential temperature, $\\theta_a$.

    The flux returned by this function is:
    $SH = c_p C_H \\rho_a U (T_s - \\theta_a)$,
    where $c_p$ is the specific heat of air at constant pressure, $C_H$ is a bulk
    transfer coefficient (here `drag_coef`), $\\rho_a$ is near-surface air density,
    and $U$ is near-surface wind speed.

    Args:
        temp_surf: Surface temperature, $T_s$ (K)
        temp_diseqb: Surface–air temperature disequilibrium, $T_{dq}$ (K),
            used in $T_a = T_s - T_{dq}$
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
    temp_atm = temp_surf - temp_diseqb
    rho_atm = get_density(temp_atm, p_atm)
    temp_pot_atm = potential_temp(temp_atm, p_atm, p_surf)
    return c_p * drag_coef * rho_atm * w_atm * (temp_surf - temp_pot_atm)


def get_sensitivity_sh(
        temp_surf: Union[float, np.ndarray, xr.DataArray],
        temp_diseqb: Union[float, np.ndarray, xr.DataArray],
        w_atm: Union[float, np.ndarray, xr.DataArray],
        drag_coef: Union[float, np.ndarray, xr.DataArray],
        p_surf: Union[float, np.ndarray, xr.DataArray],
        sigma_atm: float,
) -> dict:
    """Compute sensitivities of sensible heat flux to bulk-exchange parameters.

    Uses the bulk aerodynamic sensible heat flux $SH = c_p C_H \\rho_a U (T_s - \\theta_a)$,
    with $p_a = \\sigma_a p_s$ and $T_a = T_s - T_{dq}$, where $\\theta_a$ is the
    near-surface atmospheric potential temperature referenced to $p_s$ (as in
    `potential_temp(temp_atm, p_atm, p_surf)` within `get_sensible_heat`).

    Returns a dictionary containing (i) first-order partial derivatives of $SH$
    with respect to each input parameter (holding the others fixed) and (ii)
    selected second-order / mixed nonlinear terms used in a Taylor-series
    decomposition.

    Args:
        temp_surf: Surface temperature, $T_s$ (K)
        temp_diseqb: Surface–air temperature disequilibrium, $T_{dq}$ (K), used in
            $T_a = T_s - T_{dq}$
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
            - temp_diseqb: $\\partial SH / \\partial T_{dq}$
            - w_atm: $\\partial SH / \\partial U$
            - drag_coef: $\\partial SH / \\partial C_H$
            - p_surf: $\\partial SH / \\partial p_s$

            Nonlinear / interaction terms:

            - nl_temp_diseqb_square: quadratic term in $T_{dq}$ (includes the $1/2$ factor)
            - nl_temp_surf_temp_diseqb: mixed term between $T_s$ and $T_{dq}$
            - nl_temp_diseqb_<key>: mixed terms between $T_{dq}$ and each of $U$, $C_H$, $p_s$

    """
    p_atm = p_surf * sigma_atm
    temp_atm = temp_surf - temp_diseqb
    rho_atm = get_density(temp_atm, p_atm)
    sh = get_sensible_heat(temp_surf, temp_diseqb, w_atm, drag_coef, p_surf, p_atm)
    sh_prefactor = c_p * drag_coef * w_atm * rho_atm

    # Differential of sh wrt each param - same order as input args
    out_dict = {'temp_surf': sh_prefactor * (1 - temp_surf / temp_atm),
                'temp_diseqb': sh_prefactor * temp_surf / temp_atm,
                'w_atm': sh / w_atm,
                'drag_coef': sh / drag_coef,
                'p_surf': sh / p_surf,
                }
    # Nonlinear contributions - only temp_diseqb as this dominates
    out_dict[name_square('temp_diseqb')] = 2 * out_dict['temp_diseqb'] / temp_atm
    out_dict[name_square('temp_diseqb')] = out_dict[name_square('temp_diseqb')] * 0.5  # to match the taylor series coef

    # Combination of mechanisms
    out_dict[name_nl('temp_surf', 'temp_diseqb')] = sh_prefactor * (1/temp_atm - 2*temp_surf/temp_atm**2)
    for key in ['w_atm', 'drag_coef', 'p_surf']:
        out_dict[name_nl('temp_diseqb', key)] = out_dict[key] * out_dict['temp_diseqb'] / sh

    return out_dict


def reconstruct_sh(temp_surf_ref: float, temp_diseqb_ref: float,
                   w_atm_ref: float,
                   drag_coef_ref: float, p_surf_ref: float, sigma_atm: float,
                   temp_surf: Optional[np.ndarray] = None, temp_diseqb: Optional[np.ndarray] = None,
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

    The near-surface atmospheric temperature is diagnosed using a surface–air
    disequilibrium temperature $T_{dq}$:
    $T_a = T_s - T_{dq}$.

    Optional mechanism arrays (e.g. `temp_surf`) are interpreted as alternative
    states to compare against the reference. If an optional mechanism is not
    provided, it is filled with the reference value broadcast to the size of the
    first provided mechanism array.

    Args:
        temp_surf_ref: Reference surface temperature, $T_s$ (K)
        temp_diseqb_ref: Reference disequilibrium temperature, $T_{dq}$ (K)
        w_atm_ref: Reference near-surface wind speed, $U$ (m s$^{-1}$)
        drag_coef_ref: Reference bulk transfer coefficient for sensible heat,
            $C_H$ (unitless)
        p_surf_ref: Reference surface pressure, $p_s$ (Pa)
        sigma_atm: Sigma coordinate for the near-surface atmosphere, $\\sigma_a$
            (unitless), used to compute $p_a = \\sigma_a p_s$
        temp_surf: Alternative surface temperature $T_s$ (K). If None, uses
            `temp_surf_ref` broadcast to the working array size
        temp_diseqb: Alternative disequilibrium temperature $T_{dq}$ (K). If None,
            uses `temp_diseqb_ref` broadcast to the working array size
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
