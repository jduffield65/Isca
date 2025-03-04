import numpy as np
from typing import Union, Optional
from .constants import L_v, epsilon, c_p, temp_kelvin_to_celsius, g, R, R_v


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
    temp = temp - temp_kelvin_to_celsius  # Convert temperature in kelvin to celsius, as celsius used for this formula.
    # if np.abs(np.asarray(temp)).max() > 35:
    #     warnings.warn('This formula is only valid for $-35^\circ C < T < 35^\circ C$\n'
    #                   'At least one temperature given is outside this range.')
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
        partial_pressure: `float [n_levels]`. Partial pressure at each level, $e$, in *Pa*.
        total_pressure: `float [n_levels]`. Atmospheric pressure at each level, $p$, in *Pa*.

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
    return sphum / (1 - sphum)


def rh_from_sphum(sphum: Union[float, np.ndarray], temp: Union[float, np.ndarray],
                  total_pressure: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Relative humidity, $rh$, is computed from specific humidity, $q$ according to:

    $rh = e / e_s$

    Where, $e = pw/(\epsilon + w)$, is the partial pressure, $e_s$ is the saturation partial pressure and
     $w$ is the mixing ratio.

    Args:
        sphum: `float [n_levels]`. Specific humidity, $q$, at each level considered, in units of $kg/kg$.
        temp: `float [n_levels]`. Temperature at each level considered. Units: *Kelvin*.
        total_pressure: `float [n_levels]`. Atmospheric pressure, $p$, at each level considered in *Pa*.

    Returns:
        Percentage relative humidity ($0 < rh < 100$).
    """
    sat_mix_ratio = mixing_ratio_from_partial_pressure(saturation_vapor_pressure(temp), total_pressure)
    mix_ratio = mixing_ratio_from_sphum(sphum)
    # return 100 * mix_ratio / sat_mix_ratio
    return 100 * mix_ratio / (epsilon + mix_ratio) * (epsilon + sat_mix_ratio) / sat_mix_ratio


def moist_static_energy(temp: np.ndarray, sphum: np.ndarray, height: Union[np.ndarray, float],
                        c_p_const: float = c_p) -> np.ndarray:
    """
    Returns the moist static energy in units of *kJ/kg*.

    Args:
        temp: `float [n_lat, n_p_levels]`. Temperature at each coordinate considered. Units: *Kelvin*.
        sphum: `float [n_lat, n_p_levels]`. Specific humidity at each coordinate considered. Units: *kg/kg*.
        height: `float [n_lat, n_p_levels]` or `float`. Geopotential height of each level considered.
            Just a `float` if only one pressure level considered for each latitude e.g. common to use 2m values.
            Units: *m*.
        c_p_const: Heat capacity constant in units of J/K/kg.
            This gives the option to easily modify the moist static energy but almost always be kept at default `c_p`.

    Returns:
        Moist static energy at each coordinate given.
    """
    return (L_v * sphum + c_p_const * temp + g * height) / 1000


def sphum_sat(temp: Union[float, np.ndarray], pressure: Union[float, np.ndarray]) -> np.ndarray:
    """
    Returns the saturation specific humidity, $q^*$, in *kg/kg*.

    Args:
        temp: Temperature at each coordinate considered. Units: *Kelvin*.
        pressure: Pressure level in *Pa*, temperature corresponds to.
            If all `temp` are at the lowest atmospheric level, then pressure` will be the lowest level pressure i.e.
            a `float`.

    Returns:
        Saturation specific humidity at each coordinate given.
    """
    # Saturation specific humidity
    w_sat = mixing_ratio_from_partial_pressure(saturation_vapor_pressure(temp), pressure)
    q_sat = w_sat / (1 + w_sat)
    return q_sat


def clausius_clapeyron_factor(temp: np.ndarray, pressure: Union[float, np.ndarray]) -> np.ndarray:
    """
    This is the factor $\\alpha$, such that $dq^*/dT = \\alpha q^*$.

    I explicitly compute alpha from the formula for `saturation_vapor_pressure` in the function here.

    Args:
        temp: Temperature at each coordinate considered. Units: *Kelvin*.
        pressure: Pressure level in *Pa*, temperature corresponds to.
            If all `temp` are at the lowest atmospheric level, then pressure` will be the lowest level pressure i.e.
            a `float`.


    Returns:
        Clausius clapeyron factor, $\\alpha$. Units: *Kelvin$^{-1}$*
    """
    lambda_const = 4302.645 / (temp - 29.65) ** 2
    return lambda_const * pressure / epsilon * sphum_sat(temp, pressure) / saturation_vapor_pressure(temp)


def virtual_temp(temp: Union[float, np.ndarray], sphum: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Equation for virtual temperature using [Isca code](https://github.com/jduffield65/Isca/blob/b9249275469583c1723f12ac62333067f9460fea/isca_source/src/coupler/surface_flux.F90#L463).

    The constants `d622`, `d378`, `d608` are to match the
    [Isca code](https://github.com/jduffield65/Isca/blob/b9249275469583c1723f12ac62333067f9460fea/isca_source/src/coupler/surface_flux.F90#L935-L940).

    Args:
        temp: `float [n_p_levels]`
            Temperature in *K* to find virtual potential temperature at.
        sphum: `float [n_p_levels]`
            Specific humidity of parcel at each pressure level in *kg/kg*.

    Returns:
        Virtual temperature at each pressure level in *K*.
    """
    d622 = R / R_v
    d378 = 1 - d622
    d608 = d378 / d622
    return (1 + d608 * sphum) * temp


def get_density(temp: Union[float, np.ndarray], pressure: Union[float, np.ndarray],
                sphum: Optional[Union[float, np.ndarray]] = None) -> Union[float, np.ndarray]:
    """
    Equation for density using ideal gas equation of state: $\\rho = \\frac{p}{RT}$.
    If specific humidity given, will compute density using virtual temperature, $T_v$.

    Args:
        temp: `float [n_p_levels]`
            Temperature in *K* to find density at.
        pressure: `float [n_p_levels]`
            Pressure in *Pa* to find density at.
        sphum: `float [n_p_levels]`
            Specific humidity in *kg/kg* to find density at.

    Returns:
        Density in units of $kg/m^3$.
    """
    if sphum is None:
        return pressure / (R * temp)
    else:
        return pressure / (R * virtual_temp(temp, sphum))
