import numpy as np
from typing import Union
from .constants import lapse_dry, L_v, R, R_v, epsilon, c_p, temp_kelvin_to_celsius, kappa, g
from scipy.integrate import odeint
from scipy import optimize
import numpy_indexed
import warnings


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


def lcl_temp(temp_start: float, p_start: float, sphum_start: float) -> float:
    """
    Function to replicate the way
    [LCL temperature is computed](https://github.com/ExeClim/Isca/blob/9560521e1ba5ce27a13786ffdcb16578d0bd00da/src/
    atmos_param/qe_moist_convection/qe_moist_convection.F90#L1092-L1130)
    in *Isca* with the
    [Simple Betts-Miller](https://jduffield65.github.io/Isca/namelists/convection/qe_moist_convection/)
    convection scheme.

    It is based on two properties of dry ascent:

    Potential temperature is conserved so surface potential temperature = potential temperature at the $LCL$:

    $$\\theta = \\theta_{start}(T_{start}, p_{start}) = T_{start}\\bigg(\\frac{p_{ref}}{p_{start}}\\bigg)^{\kappa} =
    \\theta_{LCL}(T_{LCL}, p_{LCL}) = T_{LCL}\\bigg(\\frac{p_{ref}}{p_{LCL}}\\bigg)^{\kappa}$$

    Mixing ratio, $w$, is conserved in unsaturated adiabatic ascent because there is no precipitation,
    and at the $LCL$, $w_{LCL} = w_{sat}$ because by definition at the $LCL$, the air is saturated:

    $$w = w_{start} = \\frac{q_{start}}{1-q_{start}} = w_{sat} = \\frac{\epsilon e_s(T_{LCL})}{p_{LCL}-e_s(T_{LCL})}$$

    $q$ is specific humidity, $\epsilon = R_{dry}/R_v$ is the ratio of gas constant for dry air to vapour and
    $\kappa = R_{dry}/c_p$. $p_{ref}$ is taken to be $100,000 Pa$ to be consistent with the
    [value used in Isca](https://github.com/ExeClim/Isca/blob/9560521e1ba5ce27a13786ffdcb16578d0bd00da/src/
    atmos_param/qe_moist_convection/qe_moist_convection.F90#L74-L75).

    So we have two equations for two unknowns, $T_{LCL}$ and $p_{LCL}$. By eliminating $p_{LCL}$,
    we can get an equation where RHS is just a function of $T_{LCL}$ and the LHS consists only of known quantities:

    $$\\theta(T_{start})^{\kappa}p_{ref} \\frac{w(q_{start})}{w(q_{start}) + \epsilon} =
    T_{LCL}^{-1/\kappa}e_s(T_{LCL})$$

    This can then be solved using Newton iteration to get the value of $T_{LCL}$.

    Args:
        temp_start: Starting temperature of parcel, $T_{start}$. Units: *Kelvin*.
        p_start: Pressure, $p_{start}$, in *Pa* corresponding to starting point of dry ascent i.e. near surface
            pressure.
        sphum_start: Starting specific humidity of parcel, $q_{start}$. Units: *kg/kg*.

    Returns:
        Temperature of *LCL* in *Kelvin*.
    """
    def lcl_opt_func(temp_lcl, p_start, temp_start, sphum_start):
        # Function to optimize
        p_ref = 1e5
        r = mixing_ratio_from_sphum(sphum_start)
        theta = temp_start * (p_ref / p_start) ** kappa  # potential temperature
        value = theta ** (-1 / kappa) * p_ref * r / (epsilon + r)

        return value - saturation_vapor_pressure(temp_lcl) / temp_lcl ** (1 / kappa)
    return optimize.newton(lcl_opt_func, 270, args=(p_start, temp_start, sphum_start))


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
    return sphum / (1-sphum)


def rh_from_sphum(sphum: Union[float, np.ndarray], temp: Union[float, np.ndarray],
                  total_pressure: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Relative humidity, $rh$, is computed from specific humidity, $q$ according to:

    $rh = w / w_s$

    Where, $w= q/(1-q)$, is the mixing ratio and $w_s$ is the saturation mixing ratio.

    Args:
        sphum: `float [n_levels]`. Specific humidity, $q$, at each level considered, in units of $kg/kg$.
        temp: `float [n_levels]`. Temperature at each level considered. Units: *Kelvin*.
        total_pressure: `float [n_levels]`. Atmospheric pressure, $p$, at each level considered in *Pa*.

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


def dry_profile(temp_start: float, p_start: float, p_levels: np.ndarray) -> np.ndarray:
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


def convection_neutral_profile(temp_start: float, p_start: float, sphum_start: float,
                               p_levels: np.ndarray) -> np.ndarray:
    """
    This returns the temperature of an air parcel at the given pressure levels, assuming it follows the
    `dry_profile` up until the `lcl_temp` has been reached, followed by the `moist_profile`.

    Args:
        temp_start: Starting temperature of parcel. Units: *Kelvin*.
        p_start: Starting pressure of parcel. Units: *Pa*.
        sphum_start: Specific humidity, $q_{start}$, in *kg/kg* at $p_{start}$.
        p_levels: `float [n_p_levels]`.</br>
            Pressure levels to find the temperature of the parcel at. Units: *Pa*.

    Returns:
        `float [n_p_levels]`.</br>
        Temperature at each pressure level indicated by `p_levels`.

    """
    temp_lcl = lcl_temp(temp_start, p_start, sphum_start)
    p_lcl = p_start * (temp_lcl/temp_start)**(1/kappa)      # Pressure corresponding to LCL from dry adiabat equation
    temp_dry = dry_profile(temp_start, p_start, p_levels)       # Compute dry profile for all pressure values
    temp_moist = moist_profile(temp_lcl, p_lcl, p_levels[p_levels < p_lcl])
    temp_dry[p_levels < p_lcl] = temp_moist     # Replace dry temperature with moist for pressure below p_lcl
    return temp_dry


def moist_static_energy(temp: np.ndarray, sphum: np.ndarray, height: Union[np.ndarray, float]) -> np.ndarray:
    """
    Returns the moist static energy in units of *kJ/kg*.

    Args:
        temp: `float [n_lat, n_p_levels]`. Temperature at each coordinate considered. Units: *Kelvin*.
        sphum: `float [n_lat, n_p_levels]`. Specific humidity at each coordinate considered. Units: *kg/kg*.
        height: `float [n_lat, n_p_levels]` or `float`. Geopotential height of each level considered.
        Just a `float` if only one pressure level considered for each latitude e.g. common to use 2m values. Units: *m*.

    Returns:
        Moist static energy at each coordinate given
    """
    return (L_v * sphum + c_p * temp + g * height) / 1000
