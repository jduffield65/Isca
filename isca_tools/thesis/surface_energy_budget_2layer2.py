import numpy as np
import xarray as xr
import inspect
from typing import Union, Tuple, Optional, List

from .surface_flux_taylor_2layer import get_sensitivity_sh, get_sensitivity_lh, \
    get_sensitivity_lw_surf, get_sensitivity_lw_atm
from ..utils.constants import c_p, g, L_v, Stefan_Boltzmann
from ..utils.fourier import coef_conversion
from ..utils.moist_physics import clausius_clapeyron_factor, sphum_sat


def get_feedback_params_analytic(temp_surf: Union[float, np.ndarray, xr.DataArray],
                                 temp_atm: Union[float, np.ndarray, xr.DataArray],
                                 temp_rad_surf: Union[float, np.ndarray, xr.DataArray],
                                 temp_rad_atm: Union[float, np.ndarray, xr.DataArray],
                                 rh_atm: Union[float, np.ndarray, xr.DataArray],
                                 w_atm: Union[float, np.ndarray, xr.DataArray],
                                 drag_coef: Union[float, np.ndarray, xr.DataArray],
                                 p_surf: Union[float, np.ndarray, xr.DataArray],
                                 odp_surf: Union[float, np.ndarray, xr.DataArray],
                                 sigma_atm: float,
                                 temp_col_sphum: Union[float, np.ndarray, xr.DataArray],
                                 p_col_sphum: float,
                                 rh_col: Union[float, np.ndarray, xr.DataArray],
                                 pressure_heat_cap_atmos_calc: Optional[float] = None,
                                 evap_prefactor: float = 1,
                                 coef_amp_rad_surf: Union[float, np.ndarray, xr.DataArray] = 1,
                                 coef_amp_olr: Union[float, np.ndarray, xr.DataArray] = 1,
                                 coef_amp_col_sphum: Union[float, np.ndarray, xr.DataArray] = 1,
                                 approx_lambda_lh: bool = False
                                 ) -> Tuple[Union[float, np.ndarray, xr.DataArray],
Union[float, np.ndarray, xr.DataArray], Union[float, np.ndarray, xr.DataArray],
Union[float, np.ndarray, xr.DataArray], Union[float, np.ndarray, xr.DataArray],
Union[float, np.ndarray, xr.DataArray], Union[float, np.ndarray, xr.DataArray]]:
    # NOTE HAVE CHANGE DEFINITIUON OF coef_amp, now should be approx 1 not approx 0.
    local_vars = locals()
    get_sensitivity = {'lh': get_sensitivity_lh, 'sh': get_sensitivity_sh, 'lw_surf': get_sensitivity_lw_surf,
                       'lw_atm': get_sensitivity_lw_atm}
    gamma = {}
    for key in get_sensitivity:
        arg_names = list(inspect.signature(get_sensitivity[key]).parameters.keys())
        args_use = {name: local_vars[name] for name in arg_names if name in local_vars}
        gamma[key] = get_sensitivity[key](**args_use)

    # Construct two layer feedback parameters from individual flux sensitivity factors
    emiss_factor = 1 - np.exp(-odp_surf)
    lambda_const = gamma['sh']['temp_surf'] + gamma['lh']['temp_surf'] + gamma['lw_surf']['temp_surf']
    lambda_lw1 = 4 * Stefan_Boltzmann * np.exp(-odp_surf) * temp_surf ** 3
    lambda_lw2 = 4 * Stefan_Boltzmann * (temp_surf ** 3 - emiss_factor * coef_amp_rad_surf * temp_rad_surf ** 3)
    flux_prefactor = gamma['sh']['temp_surf'] / c_p
    lambda_sh = flux_prefactor * c_p * (temp_surf / temp_atm - 1)
    alpha_atm = clausius_clapeyron_factor(temp_atm, p_surf * sigma_atm)
    q_atm = rh_atm * sphum_sat(temp_atm, p_surf * sigma_atm)
    q_surf = sphum_sat(temp_surf, p_surf)
    if approx_lambda_lh:
        # Replace alpha_surf with alpha_atm
        lambda_lh = flux_prefactor * L_v * evap_prefactor * (alpha_atm - 1 / temp_atm) * (q_surf - q_atm)
    else:
        alpha_surf = clausius_clapeyron_factor(temp_surf, p_surf)
        lambda_lh = flux_prefactor * L_v * evap_prefactor * (
                alpha_surf * q_surf - alpha_atm * q_atm - (q_surf - q_atm) / temp_atm)
    B = -gamma['lw_atm']['temp_rad_atm'] * coef_amp_olr
    heat_cap_atmos = c_p * pressure_heat_cap_atmos_calc / g
    alpha_col_sphum = clausius_clapeyron_factor(temp_col_sphum, p_col_sphum)
    q_sat_col_sphum = sphum_sat(temp_col_sphum, p_col_sphum)
    if pressure_heat_cap_atmos_calc is None:
        pressure_heat_cap_atmos_calc = p_surf
    # mu is the effect of column specific humidity
    mu = L_v / c_p * alpha_col_sphum * rh_col * q_sat_col_sphum * coef_amp_col_sphum
    return mu, lambda_const, B, lambda_sh, lambda_lh, lambda_lw1, lambda_lw2


def get_heat_cap_lambda_eff(mu: Union[float, np.ndarray, xr.DataArray],
                            lambda_const: Union[float, np.ndarray, xr.DataArray],
                            B: Union[float, np.ndarray, xr.DataArray],
                            lambda_sh: Union[float, np.ndarray, xr.DataArray],
                            lambda_lh: Union[float, np.ndarray, xr.DataArray],
                            lambda_lw1: Union[float, np.ndarray, xr.DataArray],
                            lambda_lw2: Union[float, np.ndarray, xr.DataArray],
                            heat_cap_surf: Union[float, np.ndarray, xr.DataArray],
                            pressure_heat_cap_atmos_calc: float,
                            coef_amp_col: Union[float, np.ndarray, xr.DataArray] = 1,
                            coef_phase_col: Union[float, np.ndarray, xr.DataArray] = 0,
                            coef_phase_olr: Union[float, np.ndarray, xr.DataArray] = 0,
                            sw_abs: Union[float, np.ndarray, xr.DataArray] = 0,
                            albedo: Union[float, np.ndarray, xr.DataArray] = 0,
                            n_year_days: int = 360,
                            day_seconds: int = 86400
                            ) -> Tuple[Union[float, np.ndarray, xr.DataArray],
Union[float, np.ndarray, xr.DataArray]]:
    f = 1 / (n_year_days * day_seconds)
    omega = 2 * np.pi * f
    heat_cap_atmos = c_p * pressure_heat_cap_atmos_calc / g
    heat_cap_atmos_mod = heat_cap_atmos * (coef_amp_col + mu - B * coef_phase_olr / omega / heat_cap_atmos)

    lambda_resid = lambda_lh + lambda_lw2 - lambda_sh
    lambda_mod = lambda_const + B - lambda_resid + omega * heat_cap_atmos * coef_amp_col * coef_phase_col

    scaling_param = (lambda_const - lambda_resid) * (lambda_const - lambda_lw1) / (
            omega ** 2 * heat_cap_atmos_mod ** 2 +
            lambda_mod ** 2)
    heat_cap_scaling0 = 1 + scaling_param * heat_cap_atmos_mod / heat_cap_surf
    heat_cap_eff0 = heat_cap_scaling0 * heat_cap_surf
    lambda_scaling0 = 1 - scaling_param * lambda_mod / lambda_const
    lambda_eff0 = lambda_scaling0 * lambda_const

    sw_abs_mod = sw_abs * (lambda_const - lambda_resid) * lambda_mod / (omega ** 2 * heat_cap_atmos_mod ** 2 +
                                                                        lambda_mod ** 2) / (1 - albedo) / (1 - sw_abs)

    sw_effect_real = 1 - sw_abs_mod + (1 - omega ** 2 * heat_cap_atmos_mod ** 2 / lambda_mod ** 2) * sw_abs_mod ** 2
    # Get cross terms due to effect of sw on imaginary part, need to take account of. Especially important if large
    # heat capacity
    sw_effect_heat_cap = sw_effect_real + lambda_eff0 / lambda_mod * heat_cap_atmos_mod / heat_cap_eff0 * sw_abs_mod * (
            1 - 2 * sw_abs_mod)
    sw_effect_lambda = sw_effect_real - omega * heat_cap_atmos_mod / lambda_mod * omega * heat_cap_eff0 / lambda_eff0 * sw_abs_mod * (
            1 - 2 * sw_abs_mod)
    heat_cap_scaling = heat_cap_scaling0 * sw_effect_heat_cap
    lambda_scaling = lambda_scaling0 * sw_effect_lambda

    return lambda_scaling, heat_cap_scaling


def combine_olr_adv(B: Union[float, np.ndarray, xr.DataArray],
                    lambda_adv: Union[float, np.ndarray, xr.DataArray] = 0,
                    coef_phase_olr: Union[float, np.ndarray, xr.DataArray] = 0,
                    coef_phase_adv: Union[float, np.ndarray, xr.DataArray] = 0,
                    small_phase: bool = False) -> Tuple[Union[float, np.ndarray, xr.DataArray],
Union[float, np.ndarray, xr.DataArray]]:
    if small_phase:
        # Assume phase for col, olr, adv are all small i.e., cos(phase)=1 and sin(phase)=phase
        b = B + lambda_adv
        coef_phase_b = (B * coef_phase_olr + lambda_adv * coef_phase_adv) / b
    else:
        # Combine advection with olr
        b = B * np.cos(coef_phase_olr) + lambda_adv * np.cos(coef_phase_adv)
        coef_phase_b = (B * np.sin(coef_phase_olr) + lambda_adv * np.sin(coef_phase_adv)) / b
    return b, coef_phase_b


def combine_amplitude_phase_factor(
    amplitude: List[xr.DataArray],
    phase: List[xr.DataArray],
    small_phase: bool = False,
) -> tuple[xr.DataArray, xr.DataArray]:
    r"""Combine xarray fields into $A_{\mathrm{final}}(1-i b_{\mathrm{final}})$.

    Each amplitude--phase pair defines a complex quantity:

    $z_k = A_k [\cos(\phi_k) - i\sin(\phi_k)]$.

    The returned fields satisfy:

    $\sum_k z_k = A_{\mathrm{final}}(1-i b_{\mathrm{final}})$.

    Args:
        amplitude: Sequence of amplitude fields $A_k$.
        phase: Sequence of phase fields $\phi_k$, in radians. Each phase
            field corresponds to the amplitude field at the same position.
        small_phase: If ``True``, use the small-phase approximation
            $\cos(\phi_k) \approx 1$ and $\sin(\phi_k) \approx \phi_k$.

    Returns:
        A tuple containing:

        - ``amplitude_final``: The real component
          $A_{\mathrm{final}} = \sum_k A_k\cos(\phi_k)$. With
          ``small_phase=True``, this becomes $\sum_k A_k$.
        - ``b_final``: The factor multiplying $-i$, given by
          $b_{\mathrm{final}} =
          \sum_k A_k\sin(\phi_k) / A_{\mathrm{final}}$. With
          ``small_phase=True``, this becomes
          $\sum_k A_k\phi_k / \sum_k A_k$.

    Raises:
        ValueError: If the amplitude and phase sequences have different
            lengths, or if they are empty.

    Notes:
        Where ``amplitude_final`` is zero, ``b_final`` is undefined and is
        returned as ``NaN``.
    """
    if len(amplitude) != len(phase):
        raise ValueError("`amplitude` and `phase` must have equal lengths.")
    if not amplitude:
        raise ValueError("At least one amplitude/phase pair is required.")

    if small_phase:
        real_part = sum(amplitude)
        imaginary_factor = sum(
            amp * phi for amp, phi in zip(amplitude, phase)
        )
    else:
        real_part = sum(
            amp * np.cos(phi) for amp, phi in zip(amplitude, phase)
        )
        imaginary_factor = sum(
            amp * np.sin(phi) for amp, phi in zip(amplitude, phase)
        )

    b_final = (imaginary_factor / real_part).where(real_part != 0)

    return real_part, b_final


def get_heat_cap_lambda_eff2(mu: Union[float, np.ndarray, xr.DataArray],
                             lambda_const: Union[float, np.ndarray, xr.DataArray],
                             B: Union[float, np.ndarray, xr.DataArray],
                             lambda_sh: Union[float, np.ndarray, xr.DataArray],
                             lambda_lh: Union[float, np.ndarray, xr.DataArray],
                             lambda_lw1: Union[float, np.ndarray, xr.DataArray],
                             lambda_lw2: Union[float, np.ndarray, xr.DataArray],
                             heat_cap_surf: Union[float, np.ndarray, xr.DataArray],
                             pressure_heat_cap_atmos_calc: float,
                             coef_amp_col: Union[float, np.ndarray, xr.DataArray] = 1,
                             coef_phase_col: Union[float, np.ndarray, xr.DataArray] = 0,
                             coef_phase_olr: Union[float, np.ndarray, xr.DataArray] = 0,
                             lambda_adv: Union[float, np.ndarray, xr.DataArray] = 0,
                             coef_phase_adv: Union[float, np.ndarray, xr.DataArray] = 0,
                             sw_abs: Union[float, np.ndarray, xr.DataArray] = 0,
                             albedo: Union[float, np.ndarray, xr.DataArray] = 0,
                             n_year_days: int = 360,
                             day_seconds: int = 86400,
                             small_phase: bool = False,
                             ) -> Tuple[Union[float, np.ndarray, xr.DataArray],
Union[float, np.ndarray, xr.DataArray]]:
    # Different way with everything dimensionless, and add advection
    f = 1 / (n_year_days * day_seconds)
    omega = 2 * np.pi * f
    heat_cap_atmos = c_p * pressure_heat_cap_atmos_calc / g
    lambda_resid = lambda_lh + lambda_lw2 - lambda_sh

    # Combine complex parameters in simple way
    coef_amp_col, coef_phase_col = combine_amplitude_phase_factor([coef_amp_col], [coef_phase_col], small_phase)
    b, coef_phase_b = combine_amplitude_phase_factor([B, lambda_adv], [coef_phase_olr, coef_phase_adv], small_phase)

    # Make all parameters dimensionless by dividing by lambda_const
    x_a = omega * heat_cap_atmos / lambda_const
    x_s = omega * heat_cap_surf / lambda_const
    lambda_resid = lambda_resid / lambda_const
    b = b / lambda_const
    lambda_lw1 = lambda_lw1 / lambda_const

    # In between parameters useful for final answer
    x_a_mod = x_a * (coef_amp_col + mu - b * coef_phase_b / x_a)
    y = 1 + b - lambda_resid + x_a * coef_amp_col * coef_phase_col
    scaling_param = (1 - lambda_resid) * (1 - lambda_lw1) / (x_a_mod ** 2 + y ** 2)

    # Heat cap and lambda with no sw_abs
    x_s_eff0 = x_s * (1 + scaling_param * x_a_mod / x_s)
    y_eff0 = 1 - scaling_param * y

    # Account for sw_abs
    sw_abs_mod = sw_abs * (1 - lambda_resid) / (x_a_mod ** 2 + y ** 2) / (1 - albedo) / (1 - sw_abs)

    sw_effect_real = 1 - y * sw_abs_mod + (y ** 2 - x_a_mod ** 2) * sw_abs_mod ** 2
    sw_effect_imag = sw_abs_mod - 2 * y * sw_abs_mod ** 2

    sw_effect_x = sw_effect_real + y_eff0 * x_a_mod / x_s_eff0 * sw_effect_imag
    sw_effect_y = sw_effect_real - x_s_eff0 * x_a_mod / y_eff0 * sw_effect_imag

    x_s_eff = x_s_eff0 * sw_effect_x
    y_eff = y_eff0 * sw_effect_y

    return lambda_const * y_eff, lambda_const * x_s_eff / omega


def get_heat_cap_lambda_eff3(mu: Union[float, np.ndarray, xr.DataArray],
                             lambda_const: Union[float, np.ndarray, xr.DataArray],
                             B: Union[float, np.ndarray, xr.DataArray],
                             lambda_a: Union[float, np.ndarray, xr.DataArray],
                             lambda_lw: Union[float, np.ndarray, xr.DataArray],
                             heat_cap_surf: Union[float, np.ndarray, xr.DataArray],
                             pressure_heat_cap_atmos_calc: float,
                             coef_amp_col: Union[float, np.ndarray, xr.DataArray] = 1,
                             coef_phase_col: Union[float, np.ndarray, xr.DataArray] = 0,
                             coef_phase_olr: Union[float, np.ndarray, xr.DataArray] = 0,
                             coef_phase_a: Union[float, np.ndarray, xr.DataArray] = 0,
                             lambda_adv: Union[float, np.ndarray, xr.DataArray] = 0,
                             coef_phase_adv: Union[float, np.ndarray, xr.DataArray] = 0,
                             sw_abs: Union[float, np.ndarray, xr.DataArray] = 0,
                             albedo: Union[float, np.ndarray, xr.DataArray] = 0,
                             n_year_days: int = 360,
                             day_seconds: int = 86400,
                             small_phase: bool = False,
                             ) -> Tuple[Union[float, np.ndarray, xr.DataArray],
Union[float, np.ndarray, xr.DataArray]]:
    # Different way with everything dimensionless, and add advection
    f = 1 / (n_year_days * day_seconds)
    omega = 2 * np.pi * f
    heat_cap_atmos = c_p * pressure_heat_cap_atmos_calc / g

    # Combine complex parameters in simple way
    coef_amp_col, coef_phase_col = combine_amplitude_phase_factor([coef_amp_col], [coef_phase_col], small_phase)
    b, coef_phase_b = combine_amplitude_phase_factor([B, lambda_adv, -lambda_a],
                                                         [coef_phase_olr, coef_phase_adv, coef_phase_a], small_phase)

    # Make all parameters dimensionless by dividing by lambda_const
    x_a = omega * heat_cap_atmos / lambda_const
    x_s = omega * heat_cap_surf / lambda_const
    lambda_a = lambda_a / lambda_const
    b = b / lambda_const
    lambda_lw = lambda_lw / lambda_const

    # In between parameters useful for final answer
    x_a_mod = x_a * (coef_amp_col + mu - b * coef_phase_b / x_a)
    y = 1 + b + x_a * coef_amp_col * coef_phase_col
    eta = (1 - lambda_a) / (x_a_mod ** 2 + y ** 2)
    eta_phase = lambda_a * coef_phase_a / (x_a_mod ** 2 + y ** 2)

    # Heat cap and lambda with no sw_abs
    x_s_eff0 = x_s + (1 - lambda_lw) * (eta * x_a_mod - eta_phase * y)
    y_eff0 = 1 - (1 - lambda_lw) * (eta * y + eta_phase * x_a_mod)

    # Account for sw_abs
    sw_abs_mod = sw_abs * eta / (1 - albedo) / (1 - sw_abs)
    sw_abs_phase_mod = sw_abs * eta_phase / (1 - albedo) / (1 - sw_abs)

    sw_effect_real = 1 - y * sw_abs_mod + (y ** 2 - x_a_mod ** 2) * sw_abs_mod ** 2 - x_a_mod * sw_abs_phase_mod
    sw_effect_imag = (sw_abs_mod - 2 * y * sw_abs_mod ** 2) * x_a_mod - y * sw_abs_phase_mod

    sw_effect_x = sw_effect_real + y_eff0 / x_s_eff0 * sw_effect_imag
    sw_effect_y = sw_effect_real - x_s_eff0 / y_eff0 * sw_effect_imag

    x_s_eff = x_s_eff0 * sw_effect_x
    y_eff = y_eff0 * sw_effect_y

    return lambda_const * y_eff, lambda_const * x_s_eff / omega
