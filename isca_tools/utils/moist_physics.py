import numpy as np
from typing import Union
from .constants import lapse_dry, L_v, R, R_v, epsilon, c_p, temp_kelvin_to_celsius


def lcl_temp(temp_surf: np.ndarray, rh_surf: np.ndarray) -> np.ndarray:
    """
    Returns the temperature of the lifting condensation level, *LCL*, given the surface
    temperature and relative humidity.

    Equation comes from *Equation 22* in *Bolton 1980*.

    Args:
        temp_surf: Surface temperature in *Kelvin*.
        rh_surf: Percentage surface relative humidity ($0 < rh < 100$).

    Returns:
        Temperature of *LCL* in *Kelvin*.
    """
    return 1 / (1/(temp_surf-55) - np.log(rh_surf/100)/2840) + 55


def saturation_vapor_pressure(temp: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Computes the saturation vapor pressure, $e_s(T)$, corresponding to a given temperature.

    Uses *Equation 10* in *Bolton 1980*. Valid for $-35^\circ C < T < 35^\circ C$.

    Args:
        temp: Temperature to compute vapor pressure at. Units: *Kelvin*.

    Returns:
        Saturation vapor pressure, $e_s(T)$, in units of *Pa*.
    """
    # Alternative equation from MATLAB exercise M9.2 in Holdon 2004
    # return 611 * np.exp(L_v/R_v * (1/temp_kelvin_to_celsius - 1/temp))
    temp = temp - temp_kelvin_to_celsius       # Convert temperature in kelvin to celsius, as celsius used for this formula.
    # Multiply by 100 below to convert from hPa to Pa.
    return 611.2 * np.exp(17.67 * temp / (temp + 243.5))


def mixing_ratio_from_partial_pressure(partial_pressure: Union[float, np.ndarray],
                                       total_pressure: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Computes the mixing ratio, $w$, from partial pressure, $e$, and total atmospheric pressure, $p$, according to:

    $w = \epsilon \\frac{e}{p-e}$

    Where $\epsilon = R_d/R_v = 0.622$ is the ratio of molecular weight of water to that of dry air.

    This is the same equation used by
    [MetPy](https://unidata.github.io/MetPy/latest/api/generated/metpy.calc.mixing_ratio.html).

    Args:
        partial_pressure: Partial pressure, $e$, in *Pa*.
        total_pressure: Atmospheric pressure at altitude considered, $p$, in *Pa*.

    Returns:
        Mixing ratio, $w$, in units of $kg/kg$.
    """
    return epsilon * partial_pressure / (total_pressure - partial_pressure)


def mixing_ratio_from_sphum(sphum: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Computes the mixing ratio, $w$, from specific humidity, $q$, according to:

    $w = q/(1-q)$

    Args:
        sphum: Specific humidity, $q$, in units of $kg/kg$.

    Returns:
        Mixing ratio, $w$, in units of $kg/kg$.
    """
    return sphum / (1-sphum)


def rh_from_sphum(sphum: Union[float, np.ndarray], temp: Union[float, np.ndarray],
                  total_pressure: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Relative humidity, $rh$, is computed from specific humidity, $q$ according to:

    $rh = w / w_s$

    Where, $w= q/(1-q)$, is the mixing ratio and $w_s$ is the saturation mixing ratio.

    Args:
        sphum: Specific humidity, $q$, in units of $kg/kg$.
        temp: Temperature to compute relative humidity at. Units: *Kelvin*.
        total_pressure: Atmospheric pressure at altitude considered, $p$, in *Pa*.

    Returns:
        Percentage relative humidity ($0 < rh < 100$).
    """
    sat_mix_ratio = mixing_ratio_from_partial_pressure(saturation_vapor_pressure(temp), total_pressure)
    mix_ratio = mixing_ratio_from_sphum(sphum)
    return 100 * mix_ratio / sat_mix_ratio


def lapse_moist(temp: Union[float, np.ndarray], total_pressure: float) -> Union[float, np.ndarray]:
    """
    Returns the saturated moist adiabatic lapse rate, $\Gamma_s = -dT/dz$, at a given temperature.

    Args:
        temp: Temperature to compute lapse rate at. Units: *Kelvin*.
        total_pressure: Atmospheric pressure at altitude considered, $p$, in *Pa*.
    Returns:
        Saturated moist adiabatic lapse rate. Units: $Km^{-1}$.
    """
    e_s = saturation_vapor_pressure(temp)
    q_s = mixing_ratio_from_partial_pressure(e_s, total_pressure)   # saturation mixing ratio
    return lapse_dry * (1 + L_v*q_s / (R * temp)) / (1 + epsilon * L_v**2*q_s/(c_p * R * temp**2))
