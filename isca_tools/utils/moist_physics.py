import numpy as np
from typing import Union
from .constants import lapse_dry, L_v, R, R_v, epsilon, c_p, temp_kelvin_to_celsius, kappa, g
from scipy.integrate import odeint
import numpy_indexed


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


def lapse_moist(temp: Union[float, np.ndarray], total_pressure: Union[float, np.ndarray],
                pressure_coords: bool = False) -> Union[float, np.ndarray]:
    """
    Returns the saturated moist adiabatic lapse rate, $\Gamma_s = -dT/dz$, at a given temperature.

    Comes from equation D.10 in Holton 2004.

    Args:
        temp: Temperature to compute lapse rate at. Units: *Kelvin*.
        total_pressure: Atmospheric pressure at altitude considered, $p$, in *Pa*.
        pressure_coords: If `True`, will return $dT/dp$, otherwise will return $-dT/dz$.
    Returns:
        Saturated moist adiabatic lapse rate. Units: $Km^{-1}$.
    """
    e_s = saturation_vapor_pressure(temp)
    w_s = mixing_ratio_from_partial_pressure(e_s, total_pressure)   # saturation mixing ratio
    neg_dT_dz = lapse_dry * (1 + L_v*w_s / (R * temp)) / (1 + epsilon * L_v**2*w_s/(c_p * R * temp**2))
    if pressure_coords:
        return R * temp * neg_dT_dz / (total_pressure * g)
    else:
        return neg_dT_dz


def dry_profile(temp_start:float, p_start: float, p_levels: np.ndarray) -> np.ndarray:
    """
    Returns the temperature of an air parcel at the given pressure levels, assuming it follows the dry adiabat.

    Args:
        temp_start: Starting temperature of parcel. Units: *Kelvin*.
        p_start: Starting pressure of parcel. Units: *Pa*.
        p_levels: `float [n_p_levels]`.</br>
            Pressure levels to find the temperature of the parcel at. Units: *Pa*.

    Returns:
        `float [n_p_levels]`.</br>
        Temperature at each pressure level indicated by `p_levels`.
    """
    return temp_start * (p_levels/p_start)**kappa


def moist_profile(temp_start:float, p_start: float, p_levels: np.ndarray) -> np.ndarray:
    """
    Returns the temperature of an air parcel at the given pressure levels, assuming it follows the saturated moist
    adiabat.

    Args:
        temp_start: Starting temperature of parcel. Units: *Kelvin*.
        p_start: Starting pressure of parcel. Units: *Pa*.
        p_levels: `float [n_p_levels]`.</br>
            Pressure levels to find the temperature of the parcel at. Units: *Pa*.

    Returns:
        `float [n_p_levels]`.</br>
        Temperature at each pressure level indicated by `p_levels`.
    """

    # Solve for all pressure levels with lower value than starting pressure
    p_lower = p_levels[p_levels <= p_start][::-1]
    if len(p_lower) > 0:
        if p_start not in p_lower:
            added_p_start = True
            p_lower = np.concatenate(([p_start], p_lower))
        else:
            added_p_start = False
        temp_lower = odeint(lapse_moist, temp_start, p_lower, args=(True,)).flatten()
        if added_p_start:
            # If had to add starting pressure, remove it so only have temperatures corresponding to pressures in
            # p_levels.
            temp_lower = temp_lower[1:]
            p_lower = p_lower[1:]
    else:
        temp_lower = np.array([])

    # Solve for all pressure levels with higher value than starting pressure
    p_higher = p_levels[p_levels > p_start]
    if len(p_higher) > 0:
        if p_start not in p_higher:
            added_p_start = True
            p_higher = np.concatenate(([p_start], p_higher))
        else:
            added_p_start = False
        temp_higher = odeint(lapse_moist, temp_start, p_higher, args=(True,)).flatten()
        if added_p_start:
            # If had to add starting pressure, remove it so only have temperatures corresponding to pressures in
            # p_levels.
            temp_higher = temp_higher[1:]
            p_higher = p_higher[1:]
    else:
        temp_higher = np.array([])

    # Combine all temperature values found
    p_descend = np.concatenate((p_higher, p_lower))
    temp_descend = np.concatenate((temp_higher, temp_lower))
    # Rearrange temperature values so match how p_levels ordered in input.
    temp_final = temp_descend[numpy_indexed.indices(p_descend, p_levels)].flatten()

    # Old manual way of doing it
    # p_done = np.asarray([p_start]).reshape(-1, 1)
    # p_todo = p_levels
    # temp_levels = np.zeros_like(p_levels)
    # # For each pressure level, compute lapse rate based on pressure level closest to it and then update
    # # temperature of that level assuming lapse rate stays constant between the levels.
    # while len(p_todo) > 0:
    #     # Find the two pressure levels with the minimum distance between them such that on is in the set p_done
    #     # and one is in the set p_todo.
    #     # Want two closest as assuming lapse rate is constant between the levels
    #     diff = np.abs(p_done - p_todo)
    #     done_ind, todo_ind = np.argwhere(diff == np.min(diff))[0]
    #     if done_ind == 0:
    #         # First index is the inputted start values
    #         p_ref = p_start
    #         temp_ref = temp_start
    #     else:
    #         p_ind = np.where(p_levels == p_done[done_ind])[0]
    #         p_ref = p_levels[p_ind]
    #         temp_ref = temp_levels[p_ind]
    #
    #     # Compute lapse rate based on temperature and pressure of level in the done set.
    #     dT_dp = lapse_moist(temp_ref, p_ref, True)
    #     temp_level_ind = np.where(p_levels == p_todo[todo_ind])[0]
    #
    #     # Update temperature at level considering - assume constant lapse rate between levels.
    #     temp_levels[temp_level_ind] = temp_ref + (p_levels[temp_level_ind]-p_ref) * dT_dp
    #
    #     # Transfer pressure level from p_todo to p_done for next iteration
    #     p_done = np.append(p_done, np.asarray(p_todo[todo_ind]).reshape(-1, 1), axis=0)
    #     p_todo = np.delete(p_todo, todo_ind)
    return temp_final
