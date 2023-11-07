import numpy as np
from scipy import optimize
from typing import Union
from ..utils.constants import kappa, epsilon, L_v, c_p, R_v
from ..utils.moist_physics import (mixing_ratio_from_sphum, saturation_vapor_pressure,
                                   mixing_ratio_from_partial_pressure, potential_temp)


def lcl_temp(temp_start: float, p_start: float, sphum_start: float, p_ref: float = 1e5) -> float:
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
        p_ref: Reference pressure, $p_{ref}$ in *Pa*. It is a parameter in the `qe_moist_convection` namelist.

    Returns:
        Temperature of *LCL* in *Kelvin*.
    """
    def lcl_opt_func(temp_lcl, p_start, temp_start, sphum_start, p_ref):
        # Function to optimize
        r = mixing_ratio_from_sphum(sphum_start)
        theta = potential_temp(temp_start, p_start, p_ref)
        # theta = temp_start * (p_ref / p_start) ** kappa
        value = theta ** (-1 / kappa) * p_ref * r / (epsilon + r)

        return value - saturation_vapor_pressure(temp_lcl) / temp_lcl ** (1 / kappa)
    return optimize.newton(lcl_opt_func, 270, args=(p_start, temp_start, sphum_start, p_ref))


def lcl_p(temp_lcl: Union[float, np.ndarray], temp_start: Union[float, np.ndarray], p_start: Union[float, np.ndarray]):
    """

    Args:
        temp_lcl:
        temp_start:
        p_start:

    Returns:
        Pressure corresponding to potential temperature
    """
    return p_start * (temp_lcl / temp_start) ** (1/kappa)


def ref_temp_above_lcl(temp_lcl: float, p_lcl: float, p_full: np.ndarray) -> np.ndarray:
    """
    Function to replicate the way the
    [reference temperature profile above the LCL](https://github.com/ExeClim/Isca/blob/
    7acc5d2c10bfa8f116d9e0f90d535a3067f898cd/src/atmos_param/qe_moist_convection/qe_moist_convection.F90#L621C7-L651)
    is computed in *Isca* with the
    [Simple Betts-Miller](https://jduffield65.github.io/Isca/namelists/convection/qe_moist_convection/)
    convection scheme.

    Args:
        temp_lcl: Temperature at the Lifting Condensation Level (LCL) in *K*.
        p_lcl: Pressure of the LCL in *Pa*. This will not be one of the pressure levels in the model, and will
            be larger than any value in `p_full`.
        p_full: `float [n_p_levels]`.</br>
            Full model pressure levels in ascending order. `p_full[0]` represents space and `p_full[-1]` is the smallest
            pressure level in the model that is above the LCL.</br>Units: *Pa*.

    Returns:
        `float [n_p_levels]`.</br>
            Reference temperature in *K* at each pressure level indicated in `p_full`.
    """
    if np.ediff1d(p_full).min() < 0:
        raise ValueError('p_full needs to be in ascending order.')
    if p_lcl < p_full[-1]:
        raise ValueError(f'p_lcl ({p_lcl}Pa) should be larger than the last value in p_full ({p_full[-1]}Pa).')

    # Add LCL pressure to end of pressure array as it is the largest
    p_full = np.concatenate((p_full, [p_lcl]))
    temp_ref = np.zeros_like(p_full)
    temp_ref[-1] = temp_lcl
    mix_ratio_ref = np.zeros_like(p_full)
    mix_ratio_lcl = mixing_ratio_from_partial_pressure(saturation_vapor_pressure(temp_lcl), p_lcl)
    mix_ratio_ref[-1] = mix_ratio_lcl
    # pressure is in ascending order so iterate from largest pressure (last index) which is LCL, to lowest
    # (first index) which is space
    for k in range(len(p_full) - 2, -1, -1):
        # First get estimate for temperature and mixing ratio half-way between the pressure levels
        # using larger pressure level
        a = kappa * temp_ref[k+1] + L_v/c_p * mix_ratio_ref[k+1]
        b = L_v**2 * mix_ratio_ref[k+1] / (c_p * R_v * temp_ref[k+1]**2)
        dtlnp = a/(1+b)
        temp_half = temp_ref[k+1] + dtlnp * np.log(p_full[k] / p_full[k + 1]) / 2
        mix_ratio_half = mixing_ratio_from_partial_pressure(saturation_vapor_pressure(temp_half),
                                                            (p_full[k] + p_full[k + 1]) / 2)

        # Use halfway values to compute temperature at smaller pressure level
        a = kappa * temp_half + L_v/c_p * mix_ratio_half
        b = L_v**2 * mix_ratio_half / (c_p * R_v * temp_half**2)
        dtlnp = a/(1+b)
        temp_ref[k] = temp_ref[k+1] + dtlnp * np.log(p_full[k] / p_full[k+1])

        # Use temperature at smaller pressure to compute mixing ratio at smaller pressure
        mix_ratio_ref[k] = mixing_ratio_from_partial_pressure(temp_ref[k], p_full[k])
    return temp_ref[:-1]        # don't return LCL value
