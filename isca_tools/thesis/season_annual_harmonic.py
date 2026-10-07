# Functions to help interpret the seasonal cycle of surface temperature in terms of annual harmonic
import xarray as xr
import numpy as np
from typing import Optional, Tuple

def get_phase_amp(coef_sw_amp: xr.DataArray, omega: float, heat_cap_eff: Optional[xr.DataArray]=None,
                  lambda_eff: Optional[xr.DataArray]=None, coef_phase: Optional[xr.DataArray]=None,
                  coef_amp: Optional[xr.DataArray]=None) -> Tuple[xr.DataArray, xr.DataArray]:
    r"""Convert between harmonic temperature response and energy-budget parameters.

    Uses the linear surface energy budget:
        C_eff * dT'/dt = Q'_SW - lambda_eff * T'

    The forcing and steady periodic temperature response are defined as:
        Q'_SW(t) = coef_sw_amp * cos(omega * t)
        T'(t) = coef_amp * cos(omega * t - coef_phase)

    Positive coef_phase denotes a temperature lag relative to the forcing.
    Positive lambda_eff denotes damping.

    Args:
        coef_sw_amp: First-harmonic amplitude of absorbed shortwave forcing
            [W m^-2].
        omega: Angular frequency [rad s^-1], equal to 2*pi / period.
        heat_cap_eff: Effective heat capacity per unit area [J m^-2 K^-1].
            Supply with lambda_eff to calculate temperature phase and amplitude.
        lambda_eff: Effective linear damping coefficient [W m^-2 K^-1].
            Supply with heat_cap_eff.
        coef_phase: First-harmonic temperature phase lag relative to the
            shortwave forcing [rad]. Supply with coef_amp to infer effective
            heat capacity and damping.
        coef_amp: First-harmonic temperature amplitude [K]. Supply with
            coef_phase.

    Returns:
        A tuple of DataArrays containing either:
            (coef_phase, coef_amp), when heat_cap_eff and lambda_eff are supplied.
            (heat_cap_eff, lambda_eff), when coef_phase and coef_amp are supplied.

    Raises:
        ValueError: If neither complete input pair is supplied, or if inputs
            from both pairs are supplied.

    Notes:
        Assumes time-independent coefficients and a steady periodic response.
        Phase is relative to the forcing, not an absolute calendar phase.
        The corresponding time lag is coef_phase / omega.
        Inversion requires nonzero omega and coef_amp.
    """
    if (coef_phase is None) and (coef_amp is None):
        coef_phase = np.arctan2(omega * heat_cap_eff, lambda_eff)
        coef_amp = coef_sw_amp / np.sqrt(omega ** 2 * heat_cap_eff ** 2 + lambda_eff ** 2)
        return coef_phase, coef_amp
    elif (lambda_eff is None) and (heat_cap_eff is None):
        heat_cap_eff = np.sin(coef_phase) / omega / (coef_amp/coef_sw_amp)
        lambda_eff = coef_sw_amp * np.cos(coef_phase) / coef_amp
        return heat_cap_eff, lambda_eff
    else:
        raise ValueError('Incorrect arguments for get_phase_amp')

