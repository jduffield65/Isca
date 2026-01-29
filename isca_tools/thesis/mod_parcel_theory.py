import numpy as np
from typing import Optional, Tuple, Union, Literal
import itertools
from .lapse_integral import get_temp_const_lapse
from .adiabat_theory import get_theory_prefactor_terms
from ..convection.base import lcl_sigma_bolton_simple, dry_profile_temp
from ..utils.constants import c_p, L_v, R, g, lapse_dry
from ..utils.moist_physics import clausius_clapeyron_factor, sphum_sat, moist_static_energy
from ..utils.numerical import hybrid_root_find


def temp_mod_parcel_fit_func(temp_ft: float, temp_surf: float, rh_surf: float,
                             p_surf: float, p_ft: float, lapse_mod_D: float = 0, lapse_mod_M: float = 0,
                             temp_surf_lcl_calc: Optional[float] = 300,
                             lapse_coords: Literal['z', 'lnp'] = 'z',
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
        lapse_coords: The coordinate system used for `lapse_mod_D` and `lapse_mod_M`. If `z`, then expect in *K/m*.
            If `lnp`, expect in log pressure coordinates, units of *K*. This is obtained from the z coordinate
            version $\eta_z$ through: $\eta_{D\ln p} = RT_s\eta_{Dz}/g$ and
            $\eta_{M\ln p} = RT_{FT}\eta_{Mz}/g$.
        method:
            How to modify moist adiabat lapse rate using `lapse_mod_M`.</br>
            `add` so it is $\Gamma_M(p) + \eta_M$</br>
            `multiply` so it is $\Gamma_M(p) \\times \eta_M$.

    Returns:
        modMSE_diff: difference between parcel surface and free troposphere saturated modified MSE.
    """
    if temp_surf_lcl_calc is None:
        temp_surf_lcl_calc = temp_surf
    if lapse_coords == 'lnp':
        # Convert lapse rate parameters into z coordinate form with units K/m
        lapse_mod_D = lapse_mod_D * g / R / temp_surf
        lapse_mod_M = lapse_mod_M * g / R / temp_ft
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
                        lapse_coords: Literal['z', 'lnp'] = 'z',
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
        lapse_coords: The coordinate system used for `lapse_mod_D` and `lapse_mod_M`. If `z`, then expect in *K/m*.
            If `lnp`, expect in log pressure coordinates, units of *K*. This is obtained from the z coordinate
            version $\eta_z$ through: $\eta_{D\ln p} = RT_s\eta_{Dz}/g$ and
            $\eta_{M\ln p} = RT_{FT}\eta_{Mz}/g$.
        method:
            How to modify moist adiabat lapse rate using `lapse_mod_M`.</br>
            `add` so it is $\Gamma_M(p) + \eta_M$</br>
            `multiply` so it is $\Gamma_M(p) \\times \eta_M$


    Returns:
        temp: Environmental temperature in Kelvin at the pressure level `p_surf` or `p_ft` not provided.
    """
    if rh_surf < 0:
        return np.nan
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
                lapse_coords=lapse_coords,
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
                lapse_coords=lapse_coords,
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
                                      valid_range: float = 100,
                                      lapse_coords: Literal['z', 'lnp'] = 'z') -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    **Recommended to use `get_scale_factor_theory_numerical2` instead**

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
        lapse_coords: The coordinate system used for `lapse_mod_D` and `lapse_mod_M`. If `z`, then expect in *K/m*.
            If `lnp`, expect in log pressure coordinates, units of *K*. This is obtained from the z coordinate
            version $\eta_z$ through: $\eta_{D\ln p} = RT_s\eta_{Dz}/g$ and
            $\eta_{M\ln p} = RT_{FT}\eta_{Mz}/g$.

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
                                   temp_surf_lcl_calc, guess_lapse, valid_range, lapse_coords)

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

    # Compute the expected surface temperature given the variables and our mod_parcel framework.
    # Will likely differ from temp_surf_quant if averaging done.
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
    info_cont['lapse_mod_M_change'] = get_temp_change(lapse_mod_M_change=lapse_mod_M_quant[1] - lapse_mod_M_quant[0])

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

    # Remove the linear contribution from the non-linear mechanisms
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


def get_scale_factor_theory_numerical2(temp_surf_ref: np.ndarray, temp_surf_quant: np.ndarray, rh_ref: float,
                                       rh_quant: np.ndarray, temp_ft_quant: np.ndarray,
                                       p_ft: float,
                                       p_surf_ref: float, p_surf_quant: Optional[np.ndarray] = None,
                                       lapse_D_quant: Optional[np.ndarray] = None,
                                       lapse_M_quant: Optional[np.ndarray] = None,
                                       sCAPE_quant: Optional[np.ndarray] = None,
                                       temp_surf_lcl_calc: float = 300,
                                       guess_lapse: float = lapse_dry,
                                       valid_range: float = 100,
                                       lapse_coords: Literal['z', 'lnp'] = 'z'
                                       ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    **Recommended over `get_scale_factor_theory_numerical`. Also allows for old `sCAPE` framework.**

    Calculates the theoretical near-surface temperature change for percentile $x$, $\delta \hat{T}_s(x)$, relative
    to the reference temperature change, $\delta \\tilde{T}_s$. The theoretical scale factor is given by the linear
    sum of mechanisms assumed independent: either anomalous values in current climate, $\Delta$, or due to the
    variation in that parameter with warming, $\delta$. Then we also includes a non linear
    contribution from all combinations of two mechanisms.

    Can give a theoretical scale factor for either the modParcel framework involving `lapse_D` and `lapse_M`
    or simple CAPE framework involving `sCAPE`.

    Numerical estimate found from equating two equations for modified MSE,
    $h^{\dagger} = f_1(T_{FT}, p_s, sCAPE) = f_2(T_s, r_s, p_s)$
    So if you know all variables but $T_s$, can invert to compute $T_s$.
    Compute for each climate and take the difference to compute $\delta T_s$. To isolate effect of each mechanism,
    keep all variables at the reference value, and set that one variable to the actual value.

    ??? note "List of mechanisms - keys in `info_dict`"
        The list of mechanisms considered for differential warming, acting independently are:

        * `temp_ft_change`: Change in free tropospheric temperature
        * `rh_change`: Change in surface relative humidity
        * `p_surf_change`: Change in surface pressure
        * `temp_surf_anom`: Surface temperature anomaly in current climate
        * `rh_anom`: Surface relative humidity anomaly in current climate
        * `p_surf_anom`: Surface pressure anomaly in current climate

        If provide `lapse_D_quant` and `lapse_M_quant`, will also include:

        * `lapse_D_change`: Change in boundary layer modified lapse rate parameter, $\eta_D$
        * `lapse_M_change`: Change in aloft modified lapse rate parameter, $\eta_M$
        * `lapse_D_anom`: $\eta_D$ anomaly in current climate
        * `lapse_M_anom`: $\eta_M$ anomaly in current climate

        If provide `sCAPE_quant`, will also include:

        * `sCAPE_change`: Change in simple CAPE proxy, sCAPE
        * `sCAPE_anom`: sCAPE anomaly in current climate

        In `info_dict`, there is a key for each of these, as well as `nl_{key1}_{key2}` for the non linear conbinations
        of two mechanisms.


    ??? note "Reference Quantities"
        The reference quantities are constrained to obey the following,
        where $\overline{\chi}$ is the mean value of $\chi$ across all days:

        * $\\tilde{T}_s = \overline{T_s}; \delta \\tilde{T}_s = \delta \overline{T_s}$
        * $\\tilde{r}_s = \overline{r_s}; \delta \\tilde{r}_s = 0$
        * $\\tilde{p}_s = \overline{p_s}; \delta \\tilde{p}_s = 0$
        * $\\tilde{\eta_D} = 0; \delta \\tilde{\eta_D} = 0$
        * $\\tilde{\eta_M} = 0; \delta \\tilde{\eta_M} = 0$

        Given the choice of these five reference variables and their changes with warming, the reference free
        troposphere temperature, $\\tilde{T}_{FT}$, can be computed according to the definition of $\\tilde{h}^{\dagger}$:

        $\\tilde{h}^{\dagger} = (c_p - R^{\dagger})\\tilde{T}_{sP} + L_v \\tilde{q}_s =
            (c_p + R^{\dagger}) \\tilde{T}_{FT} + L_v q^*(\\tilde{T}_{FTP}, p_{FT})$

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
            Only used to get `nl_error_av_change`, to get error due to averaging i.e. why actual `temp_surf_quant`
            differs from that computed from all other quant variables.
        rh_ref: `float [n_exp]` $\\tilde{r}_s$</br>
            Reference near surface relative humidity for cold simulaion. `r_ref_change` is set to zero.
            Units: dimensionless (from 0 to 1).
        rh_quant: `float [n_exp, n_quant]` $r_s[x]$</br>
            `rh_quant[i, j]` is near-surface relative humidity, averaged over all days with near-surface temperature
             corresponding to the quantile `quant_use[j]`, for experiment `i`. Units: dimensionless.
        temp_ft_quant: `float [n_exp, n_quant]` $T_{FT}[x]$</br>
            `temp_ft_quant[i, j]` is temperature at `pressure_ft`, averaged over all days with near-surface temperature
             corresponding to the quantile `quant_use[j]`, for experiment `i`. Units: *kg/kg*.
        p_ft:
            Pressure at free troposphere level, $p_{FT}$, in *Pa*.
        p_surf_ref:
            Pressure at near-surface for reference day in colder simulation, $p_s$, in *Pa*.
            `p_surf_ref_change` set to zero.
        p_surf_quant: `float [n_exp, n_quant]` $p_s[x]$</br>
            `[i, j]` is surface pressure averaged over all days with near-surface temperature
            corresponding to the quantile `quant_use[j]`, for experiment `i`. Units: *Pa*.</br>
            If not supplied, will set to `p_surf_ref` for all quantiles.
        lapse_D_quant: `float [n_exp, n_quant]` $\eta_D[x]$</br>
            The quantity $\eta_D$ such that the lapse rate between $p_s$ and LCL is
            $\Gamma_D + \eta_D$ with $\Gamma_D$ being the dry adiabatic lapse rate.</br>,
            `[i, j]` is averaged over all days with near-surface temperature
            corresponding to the quantile `quant_use[j]`, for experiment `i`. Units: *K/m*.
            If don't provide, will use sCAPE version of scale factor theory.
        lapse_M_quant: `float [n_exp, n_quant]` $\eta_M[x]$</br>
            The quantity $\eta_M$ such that the lapse rate above the LCL is $\Gamma_M(p) + \eta_M$ with
            $\Gamma_M(p)$ being the moist adiabatic lapse rate at pressure $p$.</br>,
            `[i, j]` is averaged over all days with near-surface temperature
            corresponding to the quantile `quant_use[j]`, for experiment `i`. Units: *K/m*.
            If don't provide, will use sCAPE version of scale factor theory.
        sCAPE_quant: `float [n_exp, n_quant]` $sCAPE[x]$</br>
            $sCAPE = R^{\dagger} (T_{FT,parc} - T_{FT})$ in units of *J/kg*
            Proxy for CAPE, to account for deviation of parcel and environmental temperature at `p_ft`.
            If don't provide, will use modParc version of scale factor theory using `lapse_D_quant` and `lapse_M_quant`.
        temp_surf_lcl_calc:
            Surface temperature to use when computing $\sigma_{LCL}$. If `None`, uses `temp_surf`.
            Makes no difference if give `sCAPE_quant`
        guess_lapse:
            Initial guess for parcel temperature will be found assuming this bulk lapse rate
            from `temp_surf` or `temp_ft`. Units: *K/m*
        valid_range:
            Valid temperature range in Kelvin for temperature. Allow +/- this much from the initial guess.
        lapse_coords: The coordinate system used for `lapse_D` and `lapse_M`. If `z`, then expect in *K/m*.
            If `lnp`, expect in log pressure coordinates, units of *K*. This is obtained from the z coordinate
            version $\eta_z$ through: $\eta_{D\ln p} = RT_s\eta_{Dz}/g$ and
            $\eta_{M\ln p} = RT_{FT}\eta_{Mz}/g$.
            Makes no difference if give `sCAPE_quant`.

    Returns:
        scale_factor: `float [n_quant]`</br>
            `scale_factor[i]` refers to the temperature difference between experiments
            for percentile `quant_use[i]`, relative to the reference temperature change, $\delta \\tilde{T_s}$.</br>
            This is the sum of all contributions in `info_cont` and should exactly match the simulated scale factor.
        scale_factor_non_linear: `float [n_quant]`</br>
            This is the sum of all contributions in `info_cont` apart from the
            the `nl_residual` and `error_av_change` contributions.
            It only includes nl combinations of two variables.
        scale_factor_linear: `float [n_quant]`</br>
            This is the sum of all contributions in `info_cont` apart from those with
            the `nl_` prefix and `error_av_change`. It provides a simpler theoretical estimate as a sum
            of changing each variable independently.
        info_cont: Dictionary containing a contribution from each mechanism. This gives
            the contribution from each physical mechanism to the overall scale factor.</br>
    """
    if (sCAPE_quant is None) == (lapse_D_quant is None and lapse_M_quant is None):
        # Deal with the case where both sCAPE and lapse provided, or neither.
        raise ValueError('Must provide either `sCAPE_quant` or `lapse_M_quant` and `lapse_D_quant` only')

    if p_surf_quant is None:
        p_surf_quant = np.full_like(temp_surf_quant, p_surf_ref[:, np.newaxis])

    # Set values used for our reference quantities
    lapse_D_ref = 0
    lapse_D_ref_change = 0
    lapse_M_ref = 0
    lapse_M_ref_change = 0
    p_surf_ref_change = 0
    rh_ref_change = 0
    if sCAPE_quant is not None:
        # ref conditions in sCAPE mode, is all values are zero i.e. reference is parcel
        sCAPE_ref = 0
        lapse_D_quant = 0  # lapse parameters always zero in sCAPE mode
        lapse_M_quant = 0
    else:
        sCAPE_ref = None
    sCAPE_ref_change = 0

    def get_temp(rh=rh_ref, p_surf=p_surf_ref,
                 lapse_D=lapse_D_ref, lapse_M=lapse_M_ref,
                 sCAPE=sCAPE_ref, temp_surf=None, temp_ft=None):
        # Has useful default values as ref in cold climate. So to find effect of a mechanism, just
        # change that variable.
        # Still gives option to compute surf or FT temp
        if sCAPE is not None:
            lapse_D = 0
            lapse_M = 0
            R_mod = R / 2 * np.log(p_surf / p_ft)
            temp_ft_dev = sCAPE / R_mod  # Value of T_ft_parc - T_ft_env given formula for sCAPE
            if temp_surf is None:
                # When finding surface temperature, use parcel temperature at FT, then follow parcel profile to surface
                # with lapse_D = 0; lapse_M=0.
                temp_ft = temp_ft + temp_ft_dev

        temp_sol = get_temp_mod_parcel(rh, p_surf, p_ft, lapse_D, lapse_M, temp_surf, temp_ft,
                                       temp_surf_lcl_calc, guess_lapse, valid_range, lapse_coords)
        if (sCAPE is not None) and (temp_surf is not None):
            # In this case, will have computed the parcel temperature at the FT
            # Get environmental temperature from parcel by subtracting (T_ft_parc - T_ft_env)
            temp_sol = temp_sol - temp_ft_dev
        return temp_sol

    get_temp = np.vectorize(get_temp)  # may need optimizing in future
    # Compute temp_ft_ref using base climate reference rh and lapse_mod
    # No approx error in temp_surf_ref as use it to compute temp_ft_ref
    temp_ft_ref = get_temp(temp_surf=temp_surf_ref)

    def get_temp_change(rh=rh_ref, rh_change=rh_ref_change,
                        p_surf=p_surf_ref, p_surf_change=p_surf_ref_change,
                        lapse_D=lapse_D_ref, lapse_D_change=lapse_D_ref_change,
                        lapse_M=lapse_M_ref, lapse_M_change=lapse_M_ref_change,
                        sCAPE=sCAPE_ref, sCAPE_change=sCAPE_ref_change,
                        temp_ft_change=temp_ft_ref[1] - temp_ft_ref[0], temp_surf=temp_surf_ref[0]):
        # Default variables are such that if alter one variable, will give the temp change cont due to that change
        # and only that change
        # Given conditions in base climate, compute temperature at p_ft from that
        temp_ft0 = get_temp(rh, p_surf, lapse_D, lapse_M, sCAPE, temp_surf)
        # Given this ft temperature, and imposed change at p_ft, compute change at surface
        temp_surf_change_theory = get_temp(rh + rh_change, p_surf + p_surf_change,
                                           lapse_D + lapse_D_change, lapse_M + lapse_M_change,
                                           None if sCAPE is None else sCAPE + sCAPE_change,
                                           temp_ft=temp_ft0 + temp_ft_change) - temp_surf
        return temp_surf_change_theory

    # Compute the expected surface temperature given the variables and our mod_parcel framework.
    # Will likely differ from temp_surf_quant if averaging done.
    temp_surf_quant_approx = get_temp(rh_quant, p_surf_quant, lapse_D_quant, lapse_M_quant,
                                      sCAPE_quant, temp_ft=temp_ft_quant)

    # Record quantity responsible for each mechanism
    # Use temp_surf_quant_approx not temp_surf_quant because we are computing the temperature change
    # in temp_surf_quant_approx, with deviation due to averaging accounted for later by error_av_change term
    var = {'temp_ft_change': temp_ft_quant[1] - temp_ft_quant[0], 'temp_surf_anom': temp_surf_quant_approx[0],
           'rh_change': rh_quant[1] - rh_quant[0], 'rh_anom': rh_quant[0],
           'p_surf_change': p_surf_quant[1] - p_surf_quant[0], 'p_surf_anom': p_surf_quant[0]}

    if sCAPE_quant is None:
        # For modParc mode, include mechanisms from lapse rates
        var['lapse_D_change'] = lapse_D_quant[1] - lapse_D_quant[0]
        var['lapse_D_anom'] = lapse_D_quant[0]
        var['lapse_M_change'] = lapse_M_quant[1] - lapse_M_quant[0]
        var['lapse_M_anom'] = lapse_M_quant[0]
    else:
        # In old mode, include mechanism from sCAPE
        var['sCAPE_change'] = sCAPE_quant[1] - sCAPE_quant[0]
        var['sCAPE_anom'] = sCAPE_quant[0]

    info_cont = {}
    # Get linear mechanisms where only one mechanism is active
    for key in var:
        info_cont[key] = get_temp_change(**{key.replace('_anom', ''): var[key]})

    # Get non-linear contributions where only two mechanisms are active - include all permutations
    for key1, key2 in itertools.combinations(var, 2):
        info_cont[f"nl_{key1}_{key2}"] = get_temp_change(**{key1.replace('_anom', ''): var[key1],
                                                            key2.replace('_anom', ''): var[key2]})
        # Subtract the contribution from the linear mechanisms, so only non-linear contribution remains
        info_cont[f"nl_{key1}_{key2}"] -= (info_cont[key1] - (temp_surf_ref[1] - temp_surf_ref[0]))
        info_cont[f"nl_{key1}_{key2}"] -= (info_cont[key2] - (temp_surf_ref[1] - temp_surf_ref[0]))

    # Have residual because no guarantee combined nl contributions give total change
    info_cont['nl_residual'] = temp_surf_quant_approx[1] - temp_surf_quant_approx[0] - \
                               np.asarray(sum([info_cont[key] - (temp_surf_ref[1] - temp_surf_ref[0])
                                               for key in info_cont]))

    # Account for the fact that the average variables may not lead to the average surface temp due to averaging error
    # I.e. that theory was for change in temp_surf_quant_approx not temp_surf_quant
    info_cont['nl_error_av_change'] = temp_surf_quant - temp_surf_quant_approx
    info_cont['nl_error_av_change'] = info_cont['nl_error_av_change'][1] - info_cont['nl_error_av_change'][0] + \
                                      temp_surf_ref[1] - temp_surf_ref[0]
    for key in info_cont:
        # Make it so it gives scale factor contribution, will be 1 if no contribution
        info_cont[key] /= (temp_surf_ref[1] - temp_surf_ref[0])

    final_answer = np.asarray(sum([info_cont[key] - 1 for key in info_cont])) + 1  # with error term, should be exact
    final_answer_nl = np.asarray(sum([info_cont[key] - 1 for key in info_cont if
                                      (('residual' not in key) and ('error' not in key))])) + 1
    final_answer_linear = np.asarray(sum([info_cont[key] - 1 for key in info_cont if 'nl' not in key])) + 1
    return final_answer, final_answer_nl, final_answer_linear, info_cont


def get_sensitivity_factors(temp_surf: float, rh_surf: float,
                            pressure_surf: float, pressure_ft: float, temp_surf_lcl_calc: float = 300) -> dict:
    """
    Calculates the dimensionless sensitivity $\gamma$ parameters such that the theoretical scaling factor is given by:

    $$
    \\begin{align}
    \\frac{\delta T_s(x)}{\delta\\tilde{T}_s} \\approx
    &\gamma_{\delta T_{FT}} \\frac{\delta T_{FT}[x]}{\delta \\tilde{T}_s}
    + \gamma_{\Delta T_s}\\frac{\Delta T_s(x)}{\\tilde{T}_s}
    - \gamma_{\delta r} \\frac{\\tilde{T}_s}{\\tilde{r}_s} \\frac{\delta r_s[x]}{\delta \\tilde{T}_s}
    - \gamma_{\Delta r} \\frac{\Delta r_s[x]}{\\tilde{r}_s} + \\\\
    &\gamma_{\delta p} \\frac{\\tilde{T}_s}{\\tilde{p}_s}\\frac{\delta p_s[x]}{\delta \\tilde{T}_s}
    - \gamma_{\Delta p} \\frac{\Delta p_s[x]}{\\tilde{p}_s} +
    \gamma_{\delta \eta_M}\\frac{\delta \eta_M[x]}{\delta \overline{T}_s} +
    \gamma_{\delta \eta_D}\\frac{\delta \eta_D[x]}{\delta \overline{T}_s} -
    \gamma_{\Delta \eta_D}\\frac{\eta_D[x]}{\overline{T}_s}
    \\end{align}
    $$

    These $\gamma$ parameters quantify the significance of different physical mechanisms in causing a change
    in the near-surface temperature distribution.

    Args:
        temp_surf: Temperature at `pressure_surf`.
        rh_surf: Relative humidity at `pressure_surf`.
        pressure_surf: Pressure at which to compute the sensitivity factors.
        pressure_ft: Pressure at free troposphere level, $p_{FT}$, in *Pa*.
        temp_surf_lcl_calc: Surface temperature to use when computing $\sigma_{LCL}$.

    Returns:
        gamma: Dictionary containing sensitivity parameters. All are a single dimensionless `float`. The keys
            refer to the possible physical mechanisms responsible which can contribute to differential surface warming:

            * `temp_ft_change`: Change in free tropospheric temperature
            * `rh_change`: Change in surface relative humidity
            * `p_surf_change`: Change in surface pressure
            * `temp_surf_anom`: Surface temperature anomaly in current climate
            * `rh_anom`: Surface relative humidity anomaly in current climate
            * `p_surf_anom`: Surface pressure anomaly in current climate
            * `lapse_D_change`: Change in boundary layer modified lapse rate parameter, $\eta_D$
            * `lapse_M_change`: Change in aloft modified lapse rate parameter, $\eta_M$
            * `lapse_D_anom`: $\eta_D$ anomaly in current climate

            The sensitivity factor for the `sCAPE_change` mechanism is also returned, although this is
            for a previous framing with `sCAPE` replacing `lapse_D` and `lapse_M`.
    """
    sphum = rh_surf * sphum_sat(temp_surf, pressure_surf)
    # Compute FT temp according to parcel profile i.e. lapse_D and lapse_M = 0 - so does not matter if z or lnp.
    temp_ft = get_temp_mod_parcel(rh_surf, pressure_surf, pressure_ft,
                                  temp_surf=temp_surf, temp_surf_lcl_calc=temp_surf_lcl_calc)
    _, _, _, beta_ft1, beta_ft2, _, _ = get_theory_prefactor_terms(temp_ft, pressure_surf, pressure_ft)
    _, _, _, beta_s1, beta_s2, _, mu = get_theory_prefactor_terms(temp_surf, pressure_surf, pressure_ft, sphum)
    sigma_lcl = lcl_sigma_bolton_simple(rh_surf, temp_surf_lcl_calc)

    gamma = {}
    gamma['temp_ft_change'] = beta_ft1 / beta_s1
    gamma['rh_change'] = L_v * sphum / (beta_s1 * temp_surf)
    gamma['p_surf_change'] = R / (2 * beta_s1) * (1 + temp_ft / temp_surf) + L_v * sphum / (beta_s1 * temp_surf)
    gamma['sCAPE_change'] = gamma['temp_ft_change'] * 1         # *1 so is not a copy
    gamma['lapse_D_change'] = -np.log(sigma_lcl)
    gamma['lapse_M_change'] = gamma['temp_ft_change'] * np.log(sigma_lcl * pressure_surf / pressure_ft)

    gamma['temp_surf_anom'] = beta_ft2 / beta_ft1 * beta_s1 / beta_ft1 * temp_surf / temp_ft - beta_s2 / beta_s1
    gamma['rh_anom'] = mu - beta_ft2 / beta_ft1 * L_v * sphum / (beta_ft1 * temp_ft)
    gamma['p_surf_anom'] = R / 2 * (beta_ft2 / beta_ft1 * (temp_surf + temp_ft) / (beta_ft1 * temp_ft) -
                                    1 / beta_s1 - 1 / beta_ft1) + L_v * sphum * beta_ft2 / beta_ft1 ** 2 / temp_ft - mu
    gamma['lapse_D_anom'] = gamma['temp_surf_anom'] * gamma['lapse_D_change']
    return gamma


def get_scale_factor_theory(temp_surf_ref: np.ndarray, temp_surf_quant: np.ndarray, rh_ref: float,
                            rh_quant: np.ndarray, temp_ft_quant: np.ndarray,
                            p_ft: float,
                            p_surf_ref: float, p_surf_quant: Optional[np.ndarray] = None,
                            lapse_D_quant: Optional[np.ndarray] = None,
                            lapse_M_quant: Optional[np.ndarray] = None,
                            sCAPE_quant: Optional[np.ndarray] = None,
                            temp_surf_lcl_calc: float = 300) -> Tuple[np.ndarray, dict, dict, dict]:
    """
    Calculates the theoretical scaling factor given by:

    $$
    \\begin{align}
    \\frac{\delta T_s(x)}{\delta\\tilde{T}_s} \\approx
    &\gamma_{\delta T_{FT}} \\frac{\delta T_{FT}[x]}{\delta \\tilde{T}_s}
    + \gamma_{\Delta T_s}\\frac{\Delta T_s(x)}{\\tilde{T}_s}
    - \gamma_{\delta r} \\frac{\\tilde{T}_s}{\\tilde{r}_s} \\frac{\delta r_s[x]}{\delta \\tilde{T}_s}
    - \gamma_{\Delta r} \\frac{\Delta r_s[x]}{\\tilde{r}_s} + \\\\
    &\gamma_{\delta p} \\frac{\\tilde{T}_s}{\\tilde{p}_s}\\frac{\delta p_s[x]}{\delta \\tilde{T}_s}
    - \gamma_{\Delta p} \\frac{\Delta p_s[x]}{\\tilde{p}_s} +
    \gamma_{\delta \eta_M}\\frac{\delta \eta_M[x]}{\delta \overline{T}_s} +
    \gamma_{\delta \eta_D}\\frac{\delta \eta_D[x]}{\delta \overline{T}_s} -
    \gamma_{\Delta \eta_D}\\frac{\eta_D[x]}{\overline{T}_s}
    \\end{align}
    $$

    ??? note "List of mechanisms - keys in `info_dict`"
        The list of mechanisms considered for differential warming, acting independently are:

        * `temp_ft_change`: Change in free tropospheric temperature
        * `rh_change`: Change in surface relative humidity
        * `p_surf_change`: Change in surface pressure
        * `temp_surf_anom`: Surface temperature anomaly in current climate
        * `rh_anom`: Surface relative humidity anomaly in current climate
        * `p_surf_anom`: Surface pressure anomaly in current climate

        If provide `lapse_D_quant` and `lapse_M_quant`, will also include:

        * `lapse_D_change`: Change in boundary layer modified lapse rate parameter, $\eta_D$
        * `lapse_M_change`: Change in aloft modified lapse rate parameter, $\eta_M$
        * `lapse_D_anom`: $\eta_D$ anomaly in current climate

        If provide `sCAPE_quant`, will also include:

        * `sCAPE_change`: Change in simple CAPE proxy, sCAPE

    ??? note "Reference Quantities"
        The reference quantities are constrained to obey the following,
        where $\overline{\chi}$ is the mean value of $\chi$ across all days:

        * $\\tilde{T}_s = \overline{T_s}; \delta \\tilde{T}_s = \delta \overline{T_s}$
        * $\\tilde{r}_s = \overline{r_s}; \delta \\tilde{r}_s = 0$
        * $\\tilde{p}_s = \overline{p_s}; \delta \\tilde{p}_s = 0$
        * $\\tilde{\eta_D} = 0; \delta \\tilde{\eta_D} = 0$
        * $\\tilde{\eta_M} = 0; \delta \\tilde{\eta_M} = 0$

        Given the choice of these five reference variables and their changes with warming, the reference free
        troposphere temperature, $\\tilde{T}_{FT}$, can be computed according to the definition of $\\tilde{h}^{\dagger}$:

        $\\tilde{h}^{\dagger} = (c_p - R^{\dagger})\\tilde{T}_{sP} + L_v \\tilde{q}_s =
            (c_p + R^{\dagger}) \\tilde{T}_{FT} + L_v q^*(\\tilde{T}_{FTP}, p_{FT})$

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
            Only used to get `nl_error_av_change`, to get error due to averaging i.e. why actual `temp_surf_quant`
            differs from that computed from all other quant variables.
        rh_ref: `float [n_exp]` $\\tilde{r}_s$</br>
            Reference near surface relative humidity for cold simulaion. `r_ref_change` is set to zero.
            Units: dimensionless (from 0 to 1).
        rh_quant: `float [n_exp, n_quant]` $r_s[x]$</br>
            `rh_quant[i, j]` is near-surface relative humidity, averaged over all days with near-surface temperature
             corresponding to the quantile `quant_use[j]`, for experiment `i`. Units: dimensionless.
        temp_ft_quant: `float [n_exp, n_quant]` $T_{FT}[x]$</br>
            `temp_ft_quant[i, j]` is temperature at `pressure_ft`, averaged over all days with near-surface temperature
             corresponding to the quantile `quant_use[j]`, for experiment `i`. Units: *kg/kg*.
        p_ft:
            Pressure at free troposphere level, $p_{FT}$, in *Pa*.
        p_surf_ref:
            Pressure at near-surface for reference day in colder simulation, $p_s$, in *Pa*.
            `p_surf_ref_change` set to zero.
        p_surf_quant: `float [n_exp, n_quant]` $p_s[x]$</br>
            `[i, j]` is surface pressure averaged over all days with near-surface temperature
            corresponding to the quantile `quant_use[j]`, for experiment `i`. Units: *Pa*.</br>
            If not supplied, will set to `p_surf_ref` for all quantiles.
        lapse_D_quant: `float [n_exp, n_quant]` $\eta_D[x]$</br>
            The quantity $\eta_D$ such that the lapse rate between $p_s$ and LCL is
            $\Gamma_D + \eta_D$ with $\Gamma_D$ being the dry adiabatic lapse rate.</br>,
            `[i, j]` is averaged over all days with near-surface temperature
            corresponding to the quantile `quant_use[j]`, for experiment `i`. Units: *K/m*.
            If don't provide, will use sCAPE version of scale factor theory.
        lapse_M_quant: `float [n_exp, n_quant]` $\eta_M[x]$</br>
            The quantity $\eta_M$ such that the lapse rate above the LCL is $\Gamma_M(p) + \eta_M$ with
            $\Gamma_M(p)$ being the moist adiabatic lapse rate at pressure $p$.</br>,
            `[i, j]` is averaged over all days with near-surface temperature
            corresponding to the quantile `quant_use[j]`, for experiment `i`. Units: *K/m*.
            If don't provide, will use sCAPE version of scale factor theory.
        sCAPE_quant: `float [n_exp, n_quant]` $sCAPE[x]$</br>
            $sCAPE = R^{\dagger} (T_{FT,parc} - T_{FT})$ in units of *J/kg*
            Proxy for CAPE, to account for deviation of parcel and environmental temperature at `p_ft`.
            If don't provide, will use modParc version of scale factor theory using `lapse_D_quant` and `lapse_M_quant`.
        temp_surf_lcl_calc:
            Surface temperature to use when computing $\sigma_{LCL}$. If `None`, uses `temp_surf`.
            Makes no difference if give `sCAPE_quant`

    Returns:
        scale_factor: `float [n_quant]`</br>
            This is the sum of all contributions in `info_cont` apart from those with
            It provides a simple theoretical estimate as a sum of changing each variable independently.
        gamma: The sensitivity $\gamma$ factors output by `get_sensitivity_factors`.
        info_var: For each mechanism, with dimensionless sensitivity factor in `gamma`,
            this gives the variable that mutliplies $\gamma$ to give `info_cont`.
            For each mechanism, this is a `float [n_quant]` numpy array.
        info_cont: Dictionary containing a contribution from each mechanism. This gives
            the contribution from each physical mechanism to the overall scale factor.</br>
            For each mechanism, this is a `float [n_quant]` numpy array.
    """
    if (sCAPE_quant is None) == (lapse_D_quant is None and lapse_M_quant is None):
        # Deal with the case where both sCAPE and lapse provided, or neither.
        raise ValueError('Must provide either `sCAPE_quant` or `lapse_M_quant` and `lapse_D_quant` only')

    if p_surf_quant is None:
        p_surf_quant = np.full_like(temp_surf_quant, p_surf_ref[:, np.newaxis])

    gamma = get_sensitivity_factors(temp_surf_ref[0], rh_ref, p_surf_ref, p_ft, temp_surf_lcl_calc)
    R_mod_ref = R/2 * np.log(p_surf_ref/p_ft)
    temp_surf_ref_change = temp_surf_ref[1] - temp_surf_ref[0]

    info_var = {'temp_ft_change': np.diff(temp_ft_quant, axis=0).squeeze() / temp_surf_ref_change,
                'rh_change': np.diff(rh_quant, axis=0).squeeze() / rh_ref * temp_surf_ref[0] / temp_surf_ref_change,
                'p_surf_change': np.diff(p_surf_quant, axis=0).squeeze() / p_surf_ref * temp_surf_ref[0] / temp_surf_ref_change,
                'temp_surf_anom': (temp_surf_quant[0] - temp_surf_ref[0]) / temp_surf_ref[0],
                'rh_anom': (rh_quant[0] - rh_ref) / rh_ref,
                'p_surf_anom': (p_surf_quant[0] - p_surf_ref) / p_surf_ref}

    # Add lapse or sCAPE depending on method using
    if sCAPE_quant is None:
        info_var['lapse_D_change'] = np.diff(lapse_D_quant, axis=0).squeeze() / temp_surf_ref_change
        info_var['lapse_M_change'] = np.diff(lapse_M_quant, axis=0).squeeze() / temp_surf_ref_change
        info_var['lapse_D_anom'] = lapse_D_quant[0] / temp_surf_ref[0]
    else:
        info_var['sCAPE_change'] = np.diff(sCAPE_quant, axis=0).squeeze() / R_mod_ref / temp_surf_ref_change
    coef_sign = {'temp_ft_change': 1, 'rh_change': -1, 'p_surf_change': 1, 'sCAPE_change': 1, 'lapse_D_change': 1,
                 'lapse_M_change': 1, 'temp_surf_anom': 1, 'rh_anom': -1, 'p_surf_anom': -1, 'lapse_D_anom': -1}

    # Get contribution from each term - will be 1 if no contribution to match numerical version
    info_cont = {}
    for key in info_var:
        info_cont[key] = coef_sign[key] * gamma[key] * info_var[key]
        if key != 'temp_ft_change':
            info_cont[key] += 1
    final_answer = np.asarray(sum([info_cont[key]-1 for key in info_cont])) + 1
    return final_answer, gamma, info_var, info_cont
