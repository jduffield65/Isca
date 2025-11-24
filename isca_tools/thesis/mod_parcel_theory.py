import numpy as np
from typing import Optional, Tuple, Union, Literal
from .lapse_integral import get_temp_const_lapse
from ..convection.base import lcl_sigma_bolton_simple, dry_profile_temp
from ..utils.constants import c_p, L_v, R, g, lapse_dry
from ..utils.moist_physics import clausius_clapeyron_factor, sphum_sat, moist_static_energy
from ..utils.numerical import hybrid_root_find


def temp_mod_parcel_fit_func(temp_ft: float, temp_surf: float, rh_surf: float,
                             p_surf: float, p_ft: float, lapse_mod_D: float = 0, lapse_mod_M: float = 0,
                             temp_surf_lcl_calc: Optional[float] = 300,
                             method: Literal['add', 'multiply'] = 'add') -> float:
    """
    In the modified parcel framework, equating surface and free tropospheric moist static energy leads to the
    exact vertical coupling equation:

    $$h_{\mathrm{p}}^{\\dagger} = (c_p + R^{\\dagger})T_{\mathrm{FTp}} + L_vq^*(T_{\mathrm{FTp}}, p_{FT}) =
    (c_p - R^{\\dagger})T_{\mathrm{sp}} + L_v q^*(T_{\mathrm{sp}}, p_s)$$

    where $R^{\dagger} = R\\ln(p_s/p_{FT})/2$ and the parcel temperatures are related to the environmental
    temperatures by:

    * $T_{\mathrm{sp}} = \sigma_{LCL}^{R\eta_D/g} T_s$
    * $T_{\mathrm{FTp}} = (\sigma_{LCL} / \sigma_{FT})^{R\eta_M/g} T_{FT}$

    And an approximate formula derived from *Bolton 1980* is used to relate $\sigma_{LCL}$ to surface relative humidity.

    Note that in this definition of *parcel*, we neglect the error in relating $z$ to temperature.

    This function returns the RHS minus the LHS of this equation to then give to `scipy.optimize.fsolve` to find
    $T_{\mathrm{FTp}}$ or $T_{\mathrm{sp}}$.

    Args:
        temp_ft: float
            Environmental temperature at `pressure_ft` in Kelvin.
        temp_surf:
            Environmental temperature at `pressure_surf` in Kelvin.
        rh_surf:
            Environmental pseudo relative humidity, $q_s/q^*(T_s, p_s)$ at `pressure_surf` in *kg/kg*.
        p_surf:
            Pressure at near-surface, $p_s$, in *Pa*.
        p_ft:
            Pressure at free troposphere level, $p_{FT}$, in *Pa*.
        lapse_mod_D:
            The quantity $\eta_D$ such that the lapse rate between $p_s$ and LCL is $\Gamma_D + \eta_D$ with
            $\Gamma_D$ being the dry adiabatic lapse rate.</br>
            Units: *K/m*
        lapse_mod_M:
            The quantity $\eta_M$ such that the lapse rate above the LCL is $\Gamma_M(p) + \eta_M$ with
            $\Gamma_M(p)$ being the moist adiabatic lapse rate at pressure $p$.</br>
            Units: *K/m*</bm>
            If `method='multiply'`, then expect dimensions $\eta_M$ such that lapse rate above LCL is
            $\Gamma_M(p) \\times \eta_M$.
        temp_surf_lcl_calc:
            Surface temperature to use when computing $\sigma_{LCL}$. If `None`, uses `temp_surf`.
        method:
            How to modify moist adiabat lapse rate using `lapse_mod_M`.</br>
            `add` so it is $\Gamma_M(p) + \eta_M$</br>
            `multiply` so it is $\Gamma_M(p) \\times \eta_M$

    Returns:
        modMSE_diff: difference between parcel surface and free troposphere saturated modified MSE.
    """
    if temp_surf_lcl_calc is None:
        temp_surf_lcl_calc = temp_surf
    sigma_lcl = lcl_sigma_bolton_simple(rh_surf, temp_surf_lcl_calc)
    sigma_ft = p_ft / p_surf
    temp_parcel_surf = temp_surf * sigma_lcl ** (R * lapse_mod_D / g)
    if method == 'multiply':
        temp_lcl = dry_profile_temp(temp_parcel_surf, p_surf, sigma_lcl * p_surf)
        temp_parcel_ft = (temp_ft / temp_lcl) ** (1 / (1 + lapse_mod_M)) * temp_lcl
    else:
        temp_parcel_ft = temp_ft * (sigma_lcl / sigma_ft) ** (R * lapse_mod_M / g)
    R_mod = R * np.log(p_surf / p_ft) / 2
    # Could optimize by computing all the above outside this function which is called a lot by root_scalar
    mse_mod_surf = moist_static_energy(temp_parcel_surf, rh_surf * sphum_sat(temp_parcel_surf, p_surf),
                                       height=0, c_p_const=c_p - R_mod)
    mse_mod_ft = moist_static_energy(temp_parcel_ft, sphum_sat(temp_parcel_ft, p_ft), height=0,
                                     c_p_const=c_p + R_mod)
    return mse_mod_surf - mse_mod_ft


def get_temp_mod_parcel(rh_surf: float,
                        p_surf: float, p_ft: float,
                        lapse_mod_D: float = 0, lapse_mod_M: float = 0,
                        temp_surf: Optional[float] = None, temp_ft: Optional[float] = None,
                        temp_surf_lcl_calc: Optional[float] = 300, guess_lapse: float = lapse_dry,
                        valid_range: float = 100,
                        method: Literal['add', 'multiply'] = 'add') -> Union[float, np.ndarray]:
    """
    This returns the free tropospheric (or surface) temperature $T_{FT}$ ($T_s$),
    such that the parcel modified MSE, $h_{\mathrm{p}}^{\\dagger}$ is equal at the surface and free troposphere.

    The parcel temperature can be obtained with both `lapse_mod_D` and `lapse_mod_M` set to zero.

    If any variable given is a numpy array, the returned value will be a numpy array of the same shape.
    If more than one variable is a numpy array, they must be the same shape.

    Args:
        rh_surf:
            Environmental pseudo relative humidity, $q_s/q^*(T_s, p_s)$ at `pressure_surf` in *kg/kg*.
            Either a single value or one for each temperature.
        p_surf:
            Pressure at near-surface, $p_s$, in *Pa*. Either a single value or one for each temperature.
        p_ft:
            Pressure at free troposphere level, $p_{FT}$, in *Pa*. Either a single value or one for each temperature.
        lapse_mod_D:
            The quantity $\eta_D$ such that the lapse rate between $p_s$ and LCL is $\Gamma_D + \eta_D$ with
            $\Gamma_D$ being the dry adiabatic lapse rate.</br>
            Either a single value or one for each temperature.
            Units: *K/m*
        lapse_mod_M:
            The quantity $\eta_M$ such that the lapse rate above the LCL is $\Gamma_M(p) + \eta_M$ with
            $\Gamma_M(p)$ being the moist adiabatic lapse rate at pressure $p$.</br>
            Either a single value or one for each temperature.
            Units: *K/m*</bm>
            If `method='multiply'`, then expect dimensions $\eta_M$ such that lapse rate above LCL is
            $\Gamma_M(p) \\times \eta_M$.
        temp_surf:
            Environmental temperature at `pressure_surf` in Kelvin. If `None`, this is the temperature returned.
        temp_ft:
            Environmental temperature at `pressure_ft` in Kelvin. If `None`, this is the temperature returned.
        temp_surf_lcl_calc:
            Surface temperature to use when computing $\sigma_{LCL}$. If `None`, uses `temp_surf`.
        guess_lapse:
            Initial guess for temperature will be found assuming this bulk lapse rate
            from `temp_surf` or `temp_ft`. Units: *K/m*
        valid_range:
            Valid temperature range in Kelvin for temperature. Allow +/- this much from the initial guess.
        method:
            How to modify moist adiabat lapse rate using `lapse_mod_M`.</br>
            `add` so it is $\Gamma_M(p) + \eta_M$</br>
            `multiply` so it is $\Gamma_M(p) \\times \eta_M$


    Returns:
        temp: Environmental temperature in Kelvin at the pressure level `p_surf` or `p_ft` not provided.
    """
    if (temp_ft is None) == (temp_surf is None):
        raise ValueError("Exactly one of temp_ft or temp_surf must be None.")

    if temp_ft is None:
        def residual(x):
            return temp_mod_parcel_fit_func(
                temp_ft=x,
                temp_surf=temp_surf,
                rh_surf=rh_surf,
                p_surf=p_surf,
                p_ft=p_ft,
                lapse_mod_D=lapse_mod_D,
                lapse_mod_M=lapse_mod_M,
                temp_surf_lcl_calc=temp_surf_lcl_calc, method=method
            )

        # good physical starting point, assuming a bulk lapse rate
        guess_temp = get_temp_const_lapse(p_ft, temp_surf, p_surf, guess_lapse)
    else:
        def residual(x):
            return temp_mod_parcel_fit_func(
                temp_ft=temp_ft,
                temp_surf=x,
                rh_surf=rh_surf,
                p_surf=p_surf,
                p_ft=p_ft,
                lapse_mod_D=lapse_mod_D,
                lapse_mod_M=lapse_mod_M,
                temp_surf_lcl_calc=temp_surf_lcl_calc, method=method
            )

        guess_temp = get_temp_const_lapse(p_surf, temp_ft, p_ft, guess_lapse)

    try:
        sol = hybrid_root_find(residual, guess_temp, valid_range)
    except ValueError as e:
        sol = np.nan
    return sol


def get_scale_factor_theory_numerical(temp_surf_ref: np.ndarray, temp_surf_quant: np.ndarray, r_ref: np.ndarray,
                                      r_quant: np.ndarray, temp_ft_quant: np.ndarray,
                                      lapse_mod_D_quant: np.ndarray,
                                      lapse_mod_M_quant: np.ndarray,
                                      p_ft_ref: float,
                                      p_surf_ref: np.ndarray, p_surf_quant: Optional[np.ndarray] = None,
                                      lapse_mod_D_ref: Optional[np.ndarray] = None,
                                      lapse_mod_M_ref: Optional[np.ndarray] = None,
                                      temp_surf_lcl_calc: float = 300,
                                      guess_lapse: float = lapse_dry,
                                      valid_range: float = 100) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Calculates the theoretical near-surface temperature change for percentile $x$, $\delta \hat{T}_s(x)$, relative
    to the reference temperature change, $\delta \\tilde{T}_s$. The theoretical scale factor is given by the linear
    sum of mechanisms assumed independent: either anomalous values in current climate, $\Delta$, or due to the
    variation in that parameter with warming, $\delta$.

    ??? note "Reference Quantities"
        The reference quantities, $\\tilde{\chi}$ are free to be chosen by the user. For ease of interpretation,
        I propose the following, where $\overline{\chi}$ is the mean value of $\chi$ across all days:

        * $\\tilde{T}_s = \overline{T_s}; \delta \\tilde{T}_s = \delta \overline{T_s}$
        * $\\tilde{r}_s = \overline{r_s}; \delta \\tilde{r}_s = 0$
        * $\\tilde{p}_s = \overline{p_s}; \delta \\tilde{p}_s = 0$
        * $\\tilde{\eta_D} = 0; \delta \\tilde{\eta_D} = 0$
        * $\\tilde{\eta_M} = 0; \delta \\tilde{\eta_M} = 0$

        Given the choice of these five reference variables and their changes with warming, the reference free
        troposphere temperature, $\\tilde{T}_{FT}$, can be computed according to the definition of $\\tilde{h}^{\dagger}$:

        $\\tilde{h}^{\dagger} = (c_p - R^{\dagger})\\tilde{T}_{sP} + L_v \\tilde{q}_s =
            (c_p + R^{\dagger}) \\tilde{T}_{FT} + L_v q^*(\\tilde{T}_{FTP}, p_{FT})$

        Poor choice of reference quantities may cause the theoretical scale factor to be a bad approximation. If this
        is the case, `get_approx_terms` can be used to investigate what is causing the theory to break down.

    Args:
        temp_surf_ref: `float [n_exp]` $\\tilde{T}_s$</br>
            Reference near surface temperature of each simulation, corresponding to a different
            optical depth, $\kappa$. Units: *K*. We assume `n_exp=2`.
        temp_surf_quant: `float [n_exp, n_quant]` $T_s(x)$ </br>
            `temp_surf_quant[i, j]` is the percentile `quant_use[j]` of near surface temperature of
            experiment `i`. Units: *K*.</br>
            Note that `quant_use` is not provided as not needed by this function, but is likely to be
            `np.arange(1, 100)` - leave out `x=0` as doesn't really make sense to consider $0^{th}$ percentile
            of a quantity.
        r_ref: `float [n_exp]` $\\tilde{r}_s$</br>
            Reference near surface relative humidity of each simulation. Units: dimensionless (from 0 to 1).
        r_quant: `float [n_exp, n_quant]` $r_s[x]$</br>
            `r_quant[i, j]` is near-surface relative humidity, averaged over all days with near-surface temperature
             corresponding to the quantile `quant_use[j]`, for experiment `i`. Units: dimensionless.
        temp_ft_quant: `float [n_exp, n_quant]` $T_{FT}[x]$</br>
            `temp_ft_quant[i, j]` is temperature at `pressure_ft`, averaged over all days with near-surface temperature
             corresponding to the quantile `quant_use[j]`, for experiment `i`. Units: *kg/kg*.
        lapse_mod_D_quant: `float [n_exp, n_quant]` $\eta_D[x]$</br>
            The quantity $\eta_D$ such that the lapse rate between $p_s$ and LCL is
            $\Gamma_D + \eta_D$ with $\Gamma_D$ being the dry adiabatic lapse rate.</br>,
            `[i, j]` is averaged over all days with near-surface temperature
            corresponding to the quantile `quant_use[j]`, for experiment `i`. Units: *K/m*.
        lapse_mod_M_quant: `float [n_exp, n_quant]` $\eta_M[x]$</br>
            The quantity $\eta_M$ such that the lapse rate above the LCL is $\Gamma_M(p) + \eta_M$ with
            $\Gamma_M(p)$ being the moist adiabatic lapse rate at pressure $p$.</br>,
            `[i, j]` is averaged over all days with near-surface temperature
            corresponding to the quantile `quant_use[j]`, for experiment `i`. Units: *K/m*.
        p_ft_ref:
            Pressure at free troposphere level for reference day, $p_{FT}$, in *Pa*.
        p_surf_ref:
            Pressure at near-surface for reference day, $p_s$, in *Pa*.
        p_surf_quant: `float [n_exp, n_quant]` $p_s[x]$</br>
            `[i, j]` is surface pressure averaged over all days with near-surface temperature
            corresponding to the quantile `quant_use[j]`, for experiment `i`. Units: *Pa*.</br>
            If not supplied, will set to `p_surf_ref` for all quantiles.
        lapse_mod_D_ref: `float [n_exp]` $\\tilde{\eta}_D$</br>
            Reference value of $\eta_D$. If not given, it will set to 0. Units: *K/m*.
        lapse_mod_M_ref: `float [n_exp]` $\\tilde{\eta}_M$</br>
            Reference value of $\eta_M$. If not given, it will set to 0. Units: *K/m*.
        temp_surf_lcl_calc:
            Surface temperature to use when computing $\sigma_{LCL}$. If `None`, uses `temp_surf`.
        guess_lapse:
            Initial guess for parcel temperature will be found assuming this bulk lapse rate
            from `temp_surf` or `temp_ft`. Units: *K/m*
        valid_range:
            Valid temperature range in Kelvin for temperature. Allow +/- this much from the initial guess.

    Returns:
        scale_factor: `float [n_quant]`</br>
            `scale_factor[i]` refers to the temperature difference between experiments
            for percentile `quant_use[i]`, relative to the reference temperature change, $\delta \\tilde{T_s}$.</br>
            This is the sum of all contributions in `info_cont` and should exactly match the simulated scale factor.
        scale_factor_linear: `float [n_quant]`</br>
            This is the sum of all contributions in `info_cont` apart from those with
            the `nl_` prefix and `error_av_change`. It provides a simpler theoretical estimate as a sum
            of changing each variable independently.
        info_cont: Dictionary containing a contribution from each mechanism. This gives
            the contribution from each physical mechanism to the overall scale factor.</br>
    """
    n_exp, n_quant = temp_surf_quant.shape
    if lapse_mod_D_ref is None:
        lapse_mod_D_ref = np.zeros(n_exp)
    if lapse_mod_M_ref is None:
        lapse_mod_M_ref = np.zeros(n_exp)
    if p_surf_quant is None:
        p_surf_quant = np.full_like(temp_surf_quant, p_surf_ref[:, np.newaxis])

    def get_temp(rh_surf=r_ref[0], p_surf=p_surf_ref[0],
                 lapse_mod_D=lapse_mod_D_ref[0], lapse_mod_M=lapse_mod_M_ref[0],
                 temp_surf=None, temp_ft=None):
        # Has useful default values as ref in cold climate. So to find effect of a mechanism, just
        # change that variable.
        # Still gives option to compute surf or FT temp
        return get_temp_mod_parcel(rh_surf, p_surf, p_ft_ref, lapse_mod_D, lapse_mod_M, temp_surf, temp_ft,
                                   temp_surf_lcl_calc, guess_lapse, valid_range)

    get_temp = np.vectorize(get_temp)  # may need optimizing in future
    # Compute temp_ft_ref using base climate reference rh and lapse_mod
    # No approx error in temp_surf_ref as use it to compute temp_ft_ref
    temp_ft_ref = get_temp(temp_surf=temp_surf_ref)

    def get_temp_change(rh_surf=r_ref[0], rh_surf_change=r_ref[1] - r_ref[0],
                        p_surf=p_surf_ref[0], p_surf_change=p_surf_ref[1] - p_surf_ref[0],
                        lapse_mod_D=lapse_mod_D_ref[0],
                        lapse_mod_D_change=lapse_mod_D_ref[1] - lapse_mod_D_ref[0],
                        lapse_mod_M=lapse_mod_M_ref[0],
                        lapse_mod_M_change=lapse_mod_M_ref[1] - lapse_mod_M_ref[0],
                        temp_ft_change=temp_ft_ref[1] - temp_ft_ref[0],
                        temp_surf=temp_surf_ref[0]):
        # Default variables are such that if alter one variable, will give the temp change cont due to that change
        # and only that change
        temp_ft0 = get_temp(rh_surf, p_surf, lapse_mod_D, lapse_mod_M, temp_surf)
        temp_surf_change_theory = get_temp(rh_surf + rh_surf_change, p_surf + p_surf_change,
                                           lapse_mod_D + lapse_mod_D_change, lapse_mod_M + lapse_mod_M_change,
                                           temp_ft=temp_ft0 + temp_ft_change) - temp_surf
        return temp_surf_change_theory

    temp_surf_quant_approx = get_temp(r_quant, p_surf_quant, lapse_mod_D_quant, lapse_mod_M_quant,
                                      temp_ft=temp_ft_quant)

    def get_temp_change_nl(rh_surf=r_quant[0], rh_surf_change=r_quant[1] - r_quant[0],
                           p_surf=p_surf_quant[0], p_surf_change=p_surf_quant[1] - p_surf_quant[0],
                           lapse_mod_D=lapse_mod_D_quant[0],
                           lapse_mod_D_change=lapse_mod_D_quant[1] - lapse_mod_D_quant[0],
                           lapse_mod_M=lapse_mod_M_quant[0],
                           lapse_mod_M_change=lapse_mod_M_quant[1] - lapse_mod_M_quant[0],
                           temp_ft_change=temp_ft_quant[1] - temp_ft_quant[0],
                           temp_surf=temp_surf_quant_approx[0],
                           temp_surf_change_actual=temp_surf_quant_approx[1] - temp_surf_quant_approx[0],
                           temp_surf_ref_change=temp_surf_ref[1] - temp_surf_ref[0]):
        # Returns the change due to one variable, accounting for nl combinations with other variables
        temp_ft0 = get_temp(rh_surf, p_surf, lapse_mod_D, lapse_mod_M, temp_surf)
        # Compute the surface temp change, with given variables. One of which should be held at ref value
        temp_surf_change_theory = get_temp(rh_surf + rh_surf_change, p_surf + p_surf_change,
                                           lapse_mod_D + lapse_mod_D_change, lapse_mod_M + lapse_mod_M_change,
                                           temp_ft=temp_ft0 + temp_ft_change) - temp_surf
        # Change due to variable held at ref value is given by actual change minus the change with variable held at ref
        # Add ref surface change to give absolute value of the contribution
        temp_surf_change_var = temp_surf_change_actual - temp_surf_change_theory + temp_surf_ref_change
        return temp_surf_change_var

    # Temp_ft change is different if account for ref value changes or not
    temp_ft_ref_change = {'base': temp_ft_ref[1] - temp_ft_ref[0],
                          'r_ref': get_temp(temp_surf=temp_surf_ref, rh_surf=r_ref)[1] - temp_ft_ref[0],
                          'p_surf_ref': get_temp(temp_surf=temp_surf_ref, p_surf=p_surf_ref)[1] - temp_ft_ref[0],
                          'lapse_mod_D_ref':
                              get_temp(temp_surf=temp_surf_ref, lapse_mod_D=lapse_mod_D_ref)[1] - temp_ft_ref[0],
                          'lapse_mod_M_ref':
                              get_temp(temp_surf=temp_surf_ref, lapse_mod_M=lapse_mod_M_ref)[1] - temp_ft_ref[0],
                          'all': get_temp(temp_surf=temp_surf_ref, rh_surf=r_ref, p_surf=p_surf_ref,
                                          lapse_mod_D=lapse_mod_D_ref, lapse_mod_M=lapse_mod_M_ref)[1] - temp_ft_ref[0]
                          }

    info_cont = {key: np.full(n_quant, temp_surf_ref[1] - temp_surf_ref[0]) for key in
                 ['r_ref_change', 'p_surf_ref_change', 'lapse_mod_D_ref_change', 'lapse_mod_M_ref_change']}

    for key in ['r_ref', 'p_surf_ref', 'lapse_mod_D_ref', 'lapse_mod_M_ref']:
        # Reference quantities change with warming but nothing else, and all ref quantities in current climate
        if temp_ft_ref_change[key] == temp_ft_ref_change['base']:
            continue
        info_cont[f'{key}_change'][:] = \
            get_temp_change(rh_surf_change=r_ref[1 if key == 'r_ref' else 0] - r_ref[0],
                            p_surf_change=p_surf_ref[1 if key == 'p_surf_ref' else 0] - p_surf_ref[0],
                            lapse_mod_D_change=lapse_mod_D_ref[1 if key == 'lapse_mod_D_ref' else 0] -
                                               lapse_mod_D_ref[0],
                            lapse_mod_M_change=lapse_mod_M_ref[1 if key == 'lapse_mod_M_ref' else 0] -
                                               lapse_mod_M_ref[0],
                            temp_ft_change=temp_ft_ref_change[key]) - temp_surf_ref[0]

    # temp_ft changes with warming | All ref quantities in current climate
    info_cont['temp_ft_change'] = get_temp_change(temp_ft_change=temp_ft_quant[1] - temp_ft_quant[0])
    # RH changes with warming | All ref quantities in current climate | temp_ft changes due temp_surf_ref
    info_cont['r_change'] = get_temp_change(rh_surf_change=r_quant[1] - r_quant[0])
    # p_surf changes with warming | All ref quantities in current climate | temp_ft changes due temp_surf_ref
    info_cont['p_surf_change'] = get_temp_change(p_surf_change=p_surf_quant[1] - p_surf_quant[0])
    # lapse_mod_D changes with warming | All ref quantities in current climate | temp_ft changes due temp_surf_ref
    info_cont['lapse_mod_D_change'] = get_temp_change(lapse_mod_D_change=lapse_mod_D_quant[1] - lapse_mod_D_quant[0])
    # lapse_mod_M changes with warming | All ref quantities in current climate | temp_ft changes due temp_surf_ref
    info_cont['lapse_mod_M_change'] = get_temp(lapse_mod_M_change=lapse_mod_M_quant[1] - lapse_mod_M_quant[0])

    # All ref quantities in current climate except temp_surf | temp_ft changes due temp_surf_ref
    # Subtract temp_surf_quant not temp_surf_ref because it is starting surface temp for this mechanism
    info_cont['temp_anom'] = get_temp_change(temp_surf=temp_surf_quant_approx[0])
    # All ref quantities in current climate except rh_surf | temp_ft changes due temp_surf_ref
    info_cont['r_anom'] = get_temp_change(rh_surf=r_quant[0])
    # All ref quantities in current climate except p_surf | temp_ft changes due temp_surf_ref
    info_cont['p_surf_anom'] = get_temp_change(p_surf=p_surf_quant[0])
    # All ref quantities in current climate except lapse_mod_D | temp_ft changes due temp_surf_ref
    info_cont['lapse_mod_D_anom'] = get_temp_change(lapse_mod_D=lapse_mod_D_quant[0])
    # All ref quantities in current climate except lapse_mod_M | temp_ft changes due temp_surf_ref
    info_cont['lapse_mod_M_anom'] = get_temp_change(lapse_mod_M=lapse_mod_M_quant[0])

    # Non-linear mechanisms - find by providing the ref change for the given quantity
    info_cont['nl_temp_ft_change'] = get_temp_change_nl(temp_ft_change=temp_ft_ref_change['all'])
    info_cont['nl_r_change'] = get_temp_change_nl(rh_surf_change=r_ref[1] - r_ref[0])
    info_cont['nl_p_surf_change'] = get_temp_change_nl(p_surf_change=p_surf_ref[1] - p_surf_ref[0])
    info_cont['nl_lapse_mod_D_change'] = get_temp_change_nl(lapse_mod_D_change=lapse_mod_D_ref[1] - lapse_mod_D_ref[0])

    info_cont['nl_lapse_mod_M_change'] = get_temp_change_nl(lapse_mod_M_change=lapse_mod_M_ref[1] - lapse_mod_M_ref[0])

    info_cont['nl_temp_anom'] = get_temp_change_nl(temp_surf=temp_surf_ref[0])
    info_cont['nl_r_anom'] = get_temp_change_nl(rh_surf=r_ref[0])
    info_cont['nl_p_surf_anom'] = get_temp_change_nl(p_surf=p_surf_ref[0])
    info_cont['nl_lapse_mod_D_anom'] = get_temp_change_nl(lapse_mod_D=lapse_mod_D_ref[0])
    info_cont['nl_lapse_mod_M_anom'] = get_temp_change_nl(lapse_mod_M=lapse_mod_M_ref[0])

    for key in info_cont:
        if 'nl' in key:
            info_cont[key] -= (info_cont[key.replace('nl_', '')] - (temp_surf_ref[1] - temp_surf_ref[0]))

    # Have residual because no guarantee combined nl contributions give total change
    info_cont['nl_residual'] = temp_surf_quant_approx[1] - temp_surf_quant_approx[0] - \
                               np.asarray(sum([info_cont[key] - (temp_surf_ref[1] - temp_surf_ref[0])
                                               for key in info_cont]))

    # OLD - alternative way of doing nl mechanisms, by combining all change and anom mechanisms separately
    # # Non-linear change mechanisms
    # info_cont['nl_change'] = get_temp(
    #     temp_ft=temp_ft0['base'] + temp_ft_quant[1] - temp_ft_quant[0],
    #     rh_surf=r_ref[0] + r_quant[1] - r_quant[0],
    #     p_surf=p_surf_ref[0] + p_surf_quant[1] - p_surf_quant[0],
    #     lapse_mod_D=lapse_mod_D_ref[0] + lapse_mod_D_quant[1] - lapse_mod_D_quant[0],
    #     lapse_mod_M=lapse_mod_M_ref[0] + lapse_mod_M_quant[1] - lapse_mod_M_quant[0]) - temp_surf_ref[0]
    # for key in ['temp_ft', 'r', 'p_surf', 'lapse_mod_D', 'lapse_mod_M']:
    #     info_cont['nl_change'] -= (info_cont[f'{key}_change'] - (temp_surf_ref[1] - temp_surf_ref[0]))
    #
    # # Non-linear anom mechanisms
    # info_cont['nl_anom'] = \
    #     get_temp(temp_ft=temp_ft_quant[0] + temp_ft_ref_change['base'],
    #              rh_surf=r_quant[0], p_surf=p_surf_quant[0],
    #              lapse_mod_D=lapse_mod_D_quant[0],
    #              lapse_mod_M=lapse_mod_M_quant[0]) - temp_surf_quant_approx[0]
    # for key in ['temp', 'r', 'p_surf', 'lapse_mod_D', 'lapse_mod_M']:
    #     info_cont['nl_anom'] -= (info_cont[f'{key}_anom'] - (temp_surf_ref[1] - temp_surf_ref[0]))
    #
    # # Non-linear anom+change mechanisms - have anom and their changes together (basically residual)
    # info_cont['nl_anom_change'] = \
    #     temp_surf_quant_approx[1] - temp_surf_quant_approx[0]
    # for key in ['temp', 'r', 'p_surf', 'lapse_mod_D', 'lapse_mod_M', 'nl']:
    #     info_cont['nl_anom_change'] -= (info_cont[f'{key}_anom'] - (temp_surf_ref[1] - temp_surf_ref[0]))
    # for key in ['temp_ft', 'r', 'p_surf', 'lapse_mod_D', 'lapse_mod_M', 'nl']:
    #     info_cont['nl_anom_change'] -= (info_cont[f'{key}_change'] - (temp_surf_ref[1] - temp_surf_ref[0]))

    # Account for the fact that the average variables may not lead to the average surface temp due to averaging error
    info_cont['error_av_change'] = temp_surf_quant - temp_surf_quant_approx
    info_cont['error_av_change'] = info_cont['error_av_change'][1] - info_cont['error_av_change'][0] + temp_surf_ref[
        1] - temp_surf_ref[0]
    for key in info_cont:
        info_cont[key] /= (temp_surf_ref[1] - temp_surf_ref[0])  # Make it so it gives scale factor contribution

    final_answer = np.asarray(sum([info_cont[key] - 1 for key in info_cont])) + 1
    final_answer_linear = np.asarray(sum([info_cont[key] - 1 for key in info_cont if
                                          (('nl' not in key) and ('error' not in key))])) + 1
    return final_answer, final_answer_linear, info_cont
