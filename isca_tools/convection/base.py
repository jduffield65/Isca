from typing import Union, Optional
import numpy as np
import numpy_indexed
from scipy.integrate import odeint
from ..utils.constants import lapse_dry, L_v, R, epsilon, c_p, g, kappa
from ..utils.moist_physics import saturation_vapor_pressure, mixing_ratio_from_partial_pressure, \
    mixing_ratio_from_sphum, sphum_sat, rh_from_sphum


def lcl_temp_bolton(temp_surf: np.ndarray, rh_surf: np.ndarray) -> np.ndarray:
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


def dry_profile_temp(temp_start: float, p_start: float, p_levels: np.ndarray) -> np.ndarray:
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


def dry_profile_pressure(temp_start: float, p_start: float,
                         temp_levels: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Given a series of `temp_levels`, this function returns the pressure corresponding to each temperature, assuming
    it follows a dry adiabat.

    Can also use the pressure of the LCL if the LCL temperature is given as `temp_levels`.

    Args:
        temp_start: Starting temperature of parcel. Units: *Kelvin*.
        p_start: Starting pressure of parcel. Units: *Pa*.
        temp_levels: `float [n_p_levels]`.</br>
            Temperatures of parcel where we want to findthe corresponding pressure. Units: *K*.

    Returns:
        `float [n_p_levels]`.</br>
            Pressure at each temperature indicated by `temp_levels`.
    """
    return p_start * (temp_levels / temp_start) ** (1/kappa)


def moist_profile(temp_start: float, p_start: float, p_levels: np.ndarray) -> np.ndarray:
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


def convection_neutral_profile(temp_start: float, p_start: float, temp_lcl: float,
                               p_levels: np.ndarray) -> np.ndarray:
    """
    This returns the temperature of an air parcel at the given pressure levels, assuming it follows the
    `dry_profile` up until the `lcl_temp` has been reached, followed by the `moist_profile`.

    Args:
        temp_start: Starting temperature of parcel. Units: *Kelvin*.
        p_start: Starting pressure of parcel. Units: *Pa*.
        temp_lcl: Temperature of *LCL* in *K*.
        p_levels: `float [n_p_levels]`.</br>
            Pressure levels to find the temperature of the parcel at. Units: *Pa*.

    Returns:
        `float [n_p_levels]`.</br>
            Temperature at each pressure level indicated by `p_levels`.

    """
    p_lcl = dry_profile_pressure(temp_start, p_start, temp_lcl)
    temp_dry = dry_profile_temp(temp_start, p_start, p_levels)       # Compute dry profile for all pressure values
    temp_moist = moist_profile(temp_lcl, p_lcl, p_levels[p_levels < p_lcl])
    temp_dry[p_levels < p_lcl] = temp_moist     # Replace dry temperature with moist for pressure below p_lcl
    return temp_dry


def potential_temp(temp: Union[float, np.ndarray], pressure: Union[float, np.ndarray],
                   p_ref: float = 1e5) -> Union[float, np.ndarray]:
    """
    Returns the potential temperature: $\\theta = T\\left(\\frac{p_{ref}}{p}\\right)^{\\kappa}$

    Args:
        temp: `float [n_p_levels]`
            Temperature in *K* to find potential temperature at.
        pressure: `float [n_p_levels]`
            Pressure levels in *Pa* corresponding to the temperatures given.
        p_ref: Reference pressure in *Pa* used to compute the potential temperature

    Returns:
        `float [n_p_levels]`
            Potential temperature in *K* at each pressure level.
    """
    return temp * (p_ref / pressure)**kappa


def equivalent_potential_temp(temp: Union[float, np.ndarray], pressure: Union[float, np.ndarray],
                              sphum: Optional[float] = None, p_ref: float = 1e5) -> Union[float, np.ndarray]:
    """
    Returns the virtual potential temperature using equation 9.40 of *Holton 2004* textbook.

    For a saturated parcel, this is $\\theta_e = \\theta \\exp (L_v q_s/c_pT)$.

    For an unsaturated parcel (used if `sphum` given), it is $\\theta_e = \\theta \\exp (L_v q/c_pT_{LCL})$.

    where $q$ ($q_s$) is the (saturation) mixing ratio, $T_{LCL}$ is the lifting condensation level temperature
    (using equation 10 from *Bolton 1980*) and $\\theta$ is the potential temperature.

    Args:
        temp: `float [n_p_levels]`
            Temperature in *K* to find virtual potential temperature at.
        pressure: `float [n_p_levels]`
            Pressure levels in *Pa* corresponding to the temperatures given.
        sphum: `float [n_p_levels]`
            Specific humidity of parcel at each pressure level in *kg/kg*. If not given, assumes saturated.
        p_ref: Reference pressure in *Pa* used to compute the potential temperature

    Returns:
        `float [n_p_levels]`
            Equivalent potential temperature in *K* at given pressure levels.
    """
    if sphum is None:
        mix_ratio = mixing_ratio_from_sphum(sphum_sat(temp, pressure))
        temp_lcl = temp
    else:
        mix_ratio = mixing_ratio_from_sphum(sphum)
        temp_lcl = lcl_temp_bolton(temp, rh_from_sphum(sphum, temp, pressure))
    temp_pot = potential_temp(temp, pressure, p_ref)
    return temp_pot * np.exp(L_v * mix_ratio / (c_p * temp_lcl))
