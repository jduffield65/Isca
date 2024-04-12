import numpy as np
import scipy.optimize
from ..utils.moist_physics import clausius_clapeyron_factor, sphum_sat, moist_static_energy
from ..utils.constants import c_p, L_v, R
from typing import Tuple, Union


def temp_adiabat_fit_func(temp_ft_adiabat: float, temp_surf: float, sphum_surf: float,
                          pressure_surf: float, pressure_ft: float) -> float:
    """
    Adiabatic Free Troposphere temperature, $T_{A,FT}$, is defined such that surface moist static energy, $h$
    is equal to the saturated moist static energy, $h^*$, evaluated at $T_{A,FT}$ and free troposphere pressure,
    $p_{FT}$ i.e. $h = h^*(T_{A,FT}, p_{FT})$.

    Using the following approximate relationship between $z_{A, FT}$ and $T_{A, FT}$:

    $$z_{A,FT} - z_s \\approx \\frac{R^{\\dagger}}{g}(T_s + T_{A, FT})$$

    where $R^{\dagger} = \\ln(p_s/p_{FT})/2$, we can obtain $T_{A, FT}$
    by solving the following equation for modified MSE, $h^{\\dagger}$:

    $$h^{\\dagger} = (c_p - R^{\\dagger})T_s + L_v q_s =
    (c_p + R^{\\dagger})T_{A, FT} + L_vq^*(T_{A, FT})$$

    This function returns the LHS minus the RHS of this equation to then give to `scipy.optimize.fsolve` to find
    $T_{A, FT}$.

    Args:
        temp_ft_adiabat: float
            Adiabatic temperature at `pressure_ft` in Kelvin.
        temp_surf:
            Actual temperature at `pressure_surf` in Kelvin.
        sphum_surf:
            Actual specific humidity at `pressure_surf` in *kg/kg*.
        pressure_surf:
            Pressure at near-surface in *Pa*.
        pressure_ft:
            Pressure at free troposphere level in *Pa*.
    Returns:
        MSE discrepancy: difference between surface and free troposphere saturated adiabatic MSE.
    """
    R_mod = R * np.log(pressure_surf / pressure_ft) / 2
    mse_mod_surf = moist_static_energy(temp_surf, sphum_surf, height=0, c_p_const=c_p - R_mod)
    mse_mod_ft = moist_static_energy(temp_ft_adiabat, sphum_sat(temp_ft_adiabat, pressure_ft), height=0,
                                     c_p_const=c_p + R_mod)
    return mse_mod_surf - mse_mod_ft


def get_temp_adiabat(temp_surf: float, sphum_surf: float, pressure_surf: float, pressure_ft: float,
                     guess_temp_adiabat: float = 273):
    """
    This returns the adiabatic temperature at `pressure_ft`, $T_{A, FT}$, such that surface moist static
    energy equals free troposphere saturated moist static energy: $h = h^*(T_{A, FT}, p_{FT})$.

    Args:
        temp_surf:
            Temperature at `pressure_surf` in Kelvin.
        sphum_surf:
            Specific humidity at `pressure_surf` in *kg/kg*.
        pressure_surf:
            Pressure at near-surface in *Pa*.
        pressure_ft:
            Pressure at free troposphere level in *Pa*.
        guess_temp_adiabat:
            Initial guess for what adiabatic temperature at `pressure_ft` should be.
    Returns:
        Adiabatic temperature at `pressure_ft` in Kelvin.
    """
    return scipy.optimize.fsolve(temp_adiabat_fit_func, guess_temp_adiabat,
                                 args=(temp_surf, sphum_surf, pressure_surf, pressure_ft))


def decompose_temp_adiabat_anomaly(temp_surf_mean: np.ndarray, temp_surf_quant: np.ndarray, sphum_mean: np.ndarray,
                                   sphum_quant: np.ndarray, temp_ft_mean: np.ndarray, temp_ft_quant: np.ndarray,
                                   pressure_surf, pressure_ft) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    The theory for $\delta T(x)$ involves the adiabatic temperature anomaly, $\Delta T_A$. This can be decomposed
    into more physically meaningful quantities:

    $$\Delta T_A(x) = T_A(x) - \overline{T_A} = \overline{T_{CE}} - T_{CE}(x) + \Delta T_{FT}(x)$$

    where:

    * $\overline{T_{CE}} = \overline{T_{FT}} - \overline{T_A}$ represents the deviation of the mean free tropospheric
    temperature from the adiabatic temperature. If at convective equilibrium, this would be zero. If the mean day had
    CAPE, this would be negative, as the lapse rate would be steeper than that expected by convection.
    * $T_{CE}(x) = T_{FT}(x) - T_A(x)$ represents the deviation of the free tropospheric
    temperature from the adiabatic temperature conditioned on percentile $x$ of near-surface temperature.
    * $\Delta T_{FT}(x) = T_{FT}(x) - \overline{T_{FT}}$ represents the gradient of the free tropospheric temperature.
    Near the tropics, we expect a weak temperature gradient (WTG) so this term would be small.

    Args:
        temp_surf_mean: `float [n_exp]`</br>
            Average near surface temperature of each simulation, corresponding to a different
            optical depth, $\kappa$. Units: *K*.
        temp_surf_quant: `float [n_exp, n_quant]`</br>
            `temp_surf_quant[i, j]` is the percentile `quant_use[j]` of near surface temperature of
            experiment `i`. Units: *K*.</br>
            Note that `quant_use` is not provided as not needed by this function, but is likely to be
            `np.arange(1, 100)` - leave out `x=0` as doesn't really make sense to consider $0^{th}$ percentile
            of a quantity.
        sphum_mean: `float [n_exp]`</br>
            Average near surface specific humidity of each simulation. Units: *kg/kg*.
        sphum_quant: `float [n_exp, n_quant]`</br>
            `sphum_quant[i, j]` is near-surface specific humidity, averaged over all days with near-surface temperature
             corresponding to the quantile `quant_use[j]`, for experiment `i`. Units: *kg/kg*.
        temp_ft_mean: `float [n_exp]`</br>
            Average temperature at `pressure_ft` in Kelvin.
        temp_ft_quant: `float [n_exp, n_quant]`</br>
            `temp_ft_quant[i, j]` is temperature at `pressure_ft`, averaged over all days with near-surface temperature
             corresponding to the quantile `quant_use[j]`, for experiment `i`. Units: *kg/kg*.
        pressure_surf:
            Pressure at near-surface in *Pa*.
        pressure_ft:
            Pressure at free troposphere level in *Pa*.

    Returns:
        `temp_adiabat_anom`: `float [n_exp, n_quant]`</br>
            Adiabatic temperature anomaly at `pressure_ft`, $\Delta T_A(x) = T_A(x) - \overline{T_A}$.
        `temp_ce_mean`: `float [n_exp]`</br>
            Deviation of mean temperature at `pressure_ft` from mean adiabatic temperature:
            $\overline{T_{CE}} = \overline{T_{FT}} - \overline{T_A}$.
        `temp_ce_quant`: `float [n_exp, n_quant]`</br>
            Deviation of temperature at `pressure_ft` from adiabatic temperature: $T_{CE}(x) = T_{FT}(x) - T_A(x)$.
            Conditioned on percentile of near-surface temperature.
        `temp_ft_anom`: `float [n_exp, n_quant]`</br>
            Temperature anomaly at `pressure_ft`, $\Delta T_{FT}(x) = T_{FT}(x) - \overline{T_{FT}}$.

    """
    n_exp, n_quant = temp_surf_quant.shape
    temp_adiabat_mean = np.zeros_like(temp_surf_mean)
    temp_adiabat_quant = np.zeros_like(temp_surf_quant)
    for i in range(n_exp):
        temp_adiabat_mean[i] = get_temp_adiabat(temp_surf_mean[i], sphum_mean[i], pressure_surf, pressure_ft)
        for j in range(n_quant):
            temp_adiabat_quant[i, j] = get_temp_adiabat(temp_surf_quant[i, j], sphum_quant[i, j], pressure_surf,
                                                        pressure_ft)
    temp_adiabat_anom = temp_adiabat_quant - temp_adiabat_mean[:, np.newaxis]
    temp_ce_quant = temp_ft_quant - temp_adiabat_quant
    temp_ce_mean = temp_ft_mean - temp_adiabat_mean
    temp_ft_anom = temp_ft_quant - temp_ft_mean
    return temp_adiabat_anom, temp_ce_mean, temp_ce_quant, temp_ft_anom


def get_delta_mse_mod_anom_theory(temp_surf_mean: np.ndarray, temp_surf_quant: np.ndarray, sphum_mean: np.ndarray,
                                  sphum_quant: np.ndarray, pressure_surf: float, pressure_ft: float,
                                  taylor_terms: str = 'linear') -> Tuple[np.ndarray, dict, np.ndarray]:
    """
    This function returns an approximation in the change in modified MSE anomaly,
    $\delta \Delta h^{\dagger} = \delta (h^{\dagger}(x) - \overline{h^{\dagger}})$, with warming -
    the basis of a theory for $\delta T_s(x)$.

    Doing a second order taylor expansion of $h^{\dagger}$ in the base climate,
    about adiabatic temperature $\overline{T_A}$, we can get:

    $$\\Delta h^{\\dagger}(x) \\approx \\beta_1 \\Delta T_A + \\frac{1}{2\\overline{T_A}}
    \\beta_2 \\Delta T_A^2$$

    Terms in equation:
        * $h^{\\dagger} = (c_p - R^{\\dagger})T_s + L_v q_s = (c_p + R^{\\dagger})T_A + L_vq^*(T_A, p_{FT})$ where
        $R^{\dagger} = R\\ln(p_s/p_{FT})/2$
        * $\\Delta T_A = T_A(x) - \overline{T_A}$
        * $\\beta_1 = \\frac{d\\overline{h^{\\dagger}}}{d\\overline{T_A}} = c_p + R^{\dagger} + L_v \\alpha q^*$
        * $\\beta_2 = \\overline{T_A}\\frac{d\\beta_1}{d\\overline{T_A}} = L_v \\alpha q^*(\\alpha \\overline{T_A} - 2)$
        * All terms on RHS are evaluated at the free tropospheric adiabatic temperature, $T_A$. I.e.
        $q^* = q^*(T_A, p_{FT})$ where $p_{FT}$ is the free tropospheric pressure.

    Doing a second taylor expansion on this equation for a change with warming between simulations, $\delta$, we can
    decompose $\delta \\Delta h^{\\dagger}(x)$ into terms involving $\delta \Delta T_A$ and $\delta \overline{T_A}$

    We can then use a third taylor expansion to relate $\delta \overline{T_A}$ to $\delta \overline{h^{\\dagger}}$:

    $$\\delta \\overline{T_A} \\approx \\frac{\\delta \\overline{h^{\\dagger}}}{\\beta_1} -
    \\frac{1}{2} \\frac{\\beta_2}{\\beta_1^3 \\overline{T_A}} (\\delta \\overline{h^{\\dagger}})^2$$

    Overall, we get $\delta \Delta h^{\dagger}$ as a function of $\delta \Delta T_A$, $\delta \overline{h^{\\dagger}}$
    and quantities evaluated at the base climate. The `taylor_terms` variable can be used to specify how many terms
    we want to keep.

    The simplest equation with `taylor_terms = 'linear'` is:

    $$\\delta \\Delta h^{\\dagger} \\approx \\beta_1 \\delta \\Delta T_A +
    \\frac{\\beta_2}{\\beta_1}\\frac{\\overline{T_A}}{\\Delta T_A} \\delta \\overline{h^{\\dagger}}$$

    Args:
        temp_surf_mean: `float [n_exp]`</br>
            Average near surface temperature of each simulation, corresponding to a different
            optical depth, $\kappa$. Units: *K*. We assume `n_exp=2`.
        temp_surf_quant: `float [n_exp, n_quant]`</br>
            `temp_surf_quant[i, j]` is the percentile `quant_use[j]` of near surface temperature of
            experiment `i`. Units: *K*.</br>
            Note that `quant_use` is not provided as not needed by this function, but is likely to be
            `np.arange(1, 100)` - leave out `x=0` as doesn't really make sense to consider $0^{th}$ percentile
            of a quantity.
        sphum_mean: `float [n_exp]`</br>
            Average near surface specific humidity of each simulation. Units: *kg/kg*.
        sphum_quant: `float [n_exp, n_quant]`</br>
            `sphum_quant[i, j]` is near-surface specific humidity, averaged over all days with near-surface temperature
             corresponding to the quantile `quant_use[j]`, for experiment `i`. Units: *kg/kg*.
        pressure_surf:
            Pressure at near-surface in *Pa*.
        pressure_ft:
            Pressure at free troposphere level in *Pa*.
        taylor_terms:
            The approximations in this equation arise from the three taylor series mentioned above, we can specify
            how many terms we want to keep, with one of the 3 options below:

            * `linear`: Only keep the two terms which are linear in all three taylor series i.e.
            $\\delta \\Delta h^{\\dagger} \\approx \\beta_1 \\delta \\Delta T_A +
            \\frac{\\beta_2}{\\beta_1}\\frac{\\overline{T_A}}{\\Delta T_A} \\delta \\overline{h^{\\dagger}}$
            * `squared`: Keep five additional terms corresponding to *LLS*, *SLL*, *LSL*, *LNL* and *SNL* terms in the
            taylor series. SNL means second order in the first taylor series mentioned above, non-linear
            (i.e. $\\delta \\Delta T_A\\delta \\overline{h^{\\dagger}}$ terms)  in the second
            and linear in the third. These 5 terms are the most significant non-linear terms.
            * `full`: In addition to the terms in `squared`, we keep the usually small *LSS* and *SLS* terms.
    Returns:
        `delta_mse_mod_anomaly`: `float [n_quant]`</br>
            $\delta \Delta h^{\dagger}$ conditioned on each quantile of near-surface temperature. Units: *kJ/kg*.
        `info_dict`: Dictionary with 5 keys: `temp_adiabat_anom`, `mse_mod_mean`, `mse_mod_mean_squared`,
            `mse_mod_mean_cubed`, `non_linear`.</br>

            For each key, a list containing a prefactor computed in the base climate and a change between simulations is
            returned. I.e. for `info_dict[non_linear][1]` would be $\\delta \\Delta T_A\\delta \\overline{h^{\\dagger}}$
            and the total contribution of non-linear terms to $\delta \Delta h^{\dagger}$ would be
            `info_dict[non_linear][0] * info_dict[non_linear][1]`. In the `linear` case this would be zero,
            and `info_dict[temp_adiabat_anom][0]`$=\\beta_1$ and `info_dict[mse_mod_mean][0]`$=
            \\frac{\\beta_2}{\\beta_1}\\frac{\\overline{T_A}}{\\Delta T_A}$ would be the only non-zero prefactors.

            Units of prefactor multiplied by change is *kJ/kg*.
        `temp_adiabat_anom`: `float [n_exp, n_quant]`</br>
            The adiabatic free troposphere temperature anomaly, $\Delta T_A$ for each experiment, as may be of use.
            Units: *Kelvin*.
    """
    # Compute adiabatic temperatures
    n_exp, n_quant = temp_surf_quant.shape
    temp_adiabat_mean = np.zeros_like(temp_surf_mean)
    temp_adiabat_quant = np.zeros_like(temp_surf_quant)
    for i in range(n_exp):
        temp_adiabat_mean[i] = get_temp_adiabat(temp_surf_mean[i], sphum_mean[i], pressure_surf, pressure_ft)
        for j in range(n_quant):
            temp_adiabat_quant[i, j] = get_temp_adiabat(temp_surf_quant[i, j], sphum_quant[i, j], pressure_surf,
                                                        pressure_ft)
    temp_adiabat_anom = temp_adiabat_quant - temp_adiabat_mean[:, np.newaxis]
    delta_temp_adiabat_anom = temp_adiabat_anom[1] - temp_adiabat_anom[0]

    # Parameters needed for taylor expansions - most compute using adiabatic temperature in free troposphere.
    R_mod = R * np.log(pressure_surf / pressure_ft) / 2
    q_sat = sphum_sat(temp_adiabat_mean[0], pressure_ft)
    alpha = clausius_clapeyron_factor(temp_adiabat_mean[0], pressure_ft)
    # beta params all have same units (J/kg/K) and related to dh/dT_A, d^2h/dT_A^2 and d^3h/dT_A^3 respectively,
    # where T_A is adiabatic temperature and h is surface MSE.
    beta_1 = c_p + R_mod + L_v * alpha * q_sat
    beta_2 = L_v * alpha * q_sat * (alpha * temp_adiabat_mean[0] - 2)
    beta_3 = L_v * alpha * q_sat * ((alpha * temp_adiabat_mean[0]) ** 2 - 6 * alpha * temp_adiabat_mean[0] + 6)

    # Compute modified MSE - need in units of J/kg at the moment hence multiply by 1000
    mse_mod_mean = moist_static_energy(temp_surf_mean, sphum_mean, height=0, c_p_const=c_p - R_mod) * 1000
    delta_mse_mod = mse_mod_mean[1] - mse_mod_mean[0]

    # Decompose Taylor Expansions - 3 in total
    # l means linear, s means squared and n means non-linear
    # first index is for Delta expansion i.e. base climate - quantile about mean
    # second index is for delta expansion i.e. difference between climates
    # third index is for conversion between delta_temp_adiabat_mean and delta_mse_mod_mean
    # I neglect all terms that are more than squared in two or more of these taylor expansions
    if taylor_terms.lower() not in ['linear', 'squared', 'full']:
        raise ValueError(f'taylor_terms given is{taylor_terms}, but must be linear, squared or full.')

    term_ll = beta_1 * delta_temp_adiabat_anom
    term_lll = beta_2 / beta_1 * temp_adiabat_anom[0] / temp_adiabat_mean[0] * delta_mse_mod
    if taylor_terms != 'linear':
        term_lls = -0.5 * beta_2 ** 2 / beta_1 ** 3 * temp_adiabat_anom[0] / temp_adiabat_mean[
            0] ** 2 * delta_mse_mod ** 2
        term_sll = 0.5 * beta_3 / beta_1 * (temp_adiabat_anom[0] / temp_adiabat_mean[0]) ** 2 * delta_mse_mod
        term_lsl = 0.5 * beta_3 / beta_1 ** 2 * temp_adiabat_anom[0] / temp_adiabat_mean[0] ** 2 * delta_mse_mod ** 2
        term_lnl = beta_2 / beta_1 / temp_adiabat_mean[0] * delta_temp_adiabat_anom * delta_mse_mod
        term_snl = beta_3 / beta_1 * temp_adiabat_anom[0] / temp_adiabat_mean[
            0] ** 2 * delta_temp_adiabat_anom * delta_mse_mod
    else:
        term_lls = 0
        term_sll = 0
        term_lsl = 0
        term_lnl = 0
        term_snl = 0

    # Extra squared-squared terms
    if taylor_terms == 'full':
        term_lss = -0.5 * beta_3 * beta_2 / beta_1 ** 4 * temp_adiabat_anom[0] / temp_adiabat_mean[
            0] ** 3 * delta_mse_mod ** 3
        term_sls = -0.25 * beta_3 * beta_2 / beta_1 ** 3 * temp_adiabat_anom[0] ** 2 / temp_adiabat_mean[
            0] ** 3 * delta_mse_mod ** 2
        # The two below are very small so should exclude
        # term_ss = 0.5 * beta_2/temp_adiabat_mean[0] * delta_temp_anom**2
        # term_lns = -0.5 * beta_2**2/beta_1**3/temp_adiabat_mean[0]**2 * delta_temp_anom * delta_mse**2
    else:
        term_lss = 0
        term_sls = 0

    # Keep track of contribution to different changes
    # Have a prefactor based on current climate and a change between simulations for each factor.
    info_dict = {'temp_adiabat_anom': [term_ll / delta_temp_adiabat_anom / 1000, delta_temp_adiabat_anom],
                 'mse_mod_mean': [(term_lll + term_sll) / delta_mse_mod / 1000, delta_mse_mod],
                 'mse_mod_mean_squared': [(term_lls + term_lsl + term_sls) / delta_mse_mod ** 2 / 1000,
                                          delta_mse_mod ** 2],
                 'mse_mod_mean_cubed': [term_lss / delta_mse_mod ** 3 / 1000, delta_mse_mod ** 3],
                 'non_linear': [(term_lnl + term_snl) / (delta_temp_adiabat_anom * delta_mse_mod) / 1000,
                                delta_temp_adiabat_anom * delta_mse_mod]
                 }

    final_answer = term_ll + term_lll + term_lls + term_sll + term_lsl + term_lnl + term_snl + term_lss + term_sls
    final_answer = final_answer / 1000  # convert to units of kJ/kg
    return final_answer, info_dict, temp_adiabat_anom


def get_delta_temp_quant_theory(temp_surf_mean: np.ndarray, temp_surf_quant: np.ndarray, sphum_mean: np.ndarray,
                                sphum_quant: np.ndarray, pressure_surf: float, pressure_ft: float,
                                taylor_terms_delta_mse_mod_anom: str = 'linear',
                                taylor_terms_delta_mse_mod: str = 'linear',
                                rh_option: str = 'full'
                                ) -> Union[Tuple[np.ndarray, np.ndarray, dict], Tuple[np.ndarray, np.ndarray]]:
    """
    Returns a theoretical prediction for change in a given percentile, $x$, of near-surface temperature. In the simplest
    case with `taylor_terms_delta_mse_mod_anom = 'linear'`, `taylor_terms_delta_mse_mod='linear'` and
    `rh_option = 'approx_anomaly'`, the equation for the theory is:

    $$\delta T(x) = \gamma_T \delta \overline{T} + \gamma_{\Delta r} \delta (\overline{r} - r(x)) +
    \\frac{\\beta_1 \\delta \\Delta T_A(x)}{c_p - R^{\\dagger} + L_v \\alpha q} +
    \\frac{\\beta_2}{\\beta_1}\\frac{\\Delta T_A(x)}{\overline{T_A}} \gamma_T \delta \overline{T}$$

    A simpler theory with $\\Delta T_A = \\delta \\Delta T_A = 0$, so just including the first two terms is also
    returned.

    Terms in equation:
        * $\gamma_T = \\frac{c_p - R^{\dagger} + L_v \\bar{\\alpha} \\bar{q}}{c_p - R^{\dagger} + L_v \\alpha q}$
        * $\gamma_{\Delta r} = \\frac{L_v \\overline{q^*}}{c_p - R^{\dagger} + L_v \\alpha q}$
        * $q^*$ and $\\alpha$ are evaluated at the surface i.e. $q^* = q^*(T_s, p_s)$.
        * $R^{\dagger} = R\\ln(p_s/p_{FT})/2$
        * $\\Delta T_A = T_A(x) - \overline{T_A}$ where $T_A$ is the adiabatic temperature at the free troposphere
            pressure of $p_{FT}$.
        * $\\beta_1 = \\frac{d\\overline{h^{\\dagger}}}{d\\overline{T_A}} = c_p + R^{\dagger} + L_v \\alpha_A q_A^*$
        * $\\beta_2 = \\overline{T_A}\\frac{d\\beta_1}{d\\overline{T_A}} = L_v \\alpha_A q^*(\\alpha_A \\overline{T_A} - 2)$
        * All terms in $\\beta$ are evaluated at $T_A$. I.e. $q_A^* = q^*(T_A, p_{FT})$.

    Args:
        temp_surf_mean: `float [n_exp]`</br>
            Average near surface temperature of each simulation, corresponding to a different
            optical depth, $\kappa$. Units: *K*. We assume `n_exp=2`.
        temp_surf_quant: `float [n_exp, n_quant]`</br>
            `temp_surf_quant[i, j]` is the percentile `quant_use[j]` of near surface temperature of
            experiment `i`. Units: *K*.</br>
            Note that `quant_use` is not provided as not needed by this function, but is likely to be
            `np.arange(1, 100)` - leave out `x=0` as doesn't really make sense to consider $0^{th}$ percentile
            of a quantity.
        sphum_mean: `float [n_exp]`</br>
            Average near surface specific humidity of each simulation. Units: *kg/kg*.
        sphum_quant: `float [n_exp, n_quant]`</br>
            `sphum_quant[i, j]` is near-surface specific humidity, averaged over all days with near-surface temperature
             corresponding to the quantile `quant_use[j]`, for experiment `i`. Units: *kg/kg*.
        pressure_surf:
            Pressure at near-surface in *Pa*.
        pressure_ft:
            Pressure at free troposphere level in *Pa*.
        taylor_terms_delta_mse_mod_anom:
            How many taylor series terms to keep in the `get_delta_mse_mod_anom_theory` function to obtain approximation
            for change in modified moist static energy anomaly: $\\delta \\Delta h^{\\dagger}$.
        taylor_terms_delta_mse_mod:
            How many taylor series terms to keep in the expansions for changes in modified moist static energy:
            $\\delta h^{\\dagger}(x)$ and $\\delta \overline{h^{\\dagger}}$ ($\\alpha$ and $q^*$ are both evaluated
            at the surface i.e. $\\alpha(T_s, p_s)$).

            * `linear`: $\\delta h^{\\dagger} \\approx (c_p - R^{\\dagger} + L_v \\alpha q)\\delta T_s +
                L_v \\alpha q^* \\delta r$
            * `squared`: Includes the additional term $0.5 L_v \\alpha q (\\alpha - 2 / T_s) \\delta T_s^2$
        rh_option:
            Relative humidity changes, $\delta r$, provide a secondary effect, so we give a number of options to approximate them:

            * `full`: No approximations to $\delta r$ terms.
            * `approx`: Neglect all $\\Delta T_A \delta r$ terms, in essence non-linear terms.
            * `none`: Set all relative humidity changes to zero.
            * If `anomaly` is also included e.g. `full_anomaly` or `approx_anomaly`, the $\delta r(x)$ term will be
                multiplied by $\overline{q^*}$ rather than $q^*(x)$. This then gives a $\delta (r(x) - \overline{r})$
                term rather than two separate relative humidity terms.
    Returns:
        `delta_temp_quant`: `float [n_quant]`</br>
            `delta_temp_quant[i]` refers to the theoretical temperature difference between experiments
            for percentile `quant_use[j]`.
        `delta_temp_quant_old`: `float [n_quant]`</br>
            Theoretical temperature difference using old theory which assumes $\\Delta T_A = \\delta \\Delta T_A = 0$.
        `info_dict`: Only returned if `taylor_terms_delta_mse_mod='linear'` otherwise too complicated.

            Dictionary with 8 keys for each term in the theory: `temp_adiabat_anom_change`, `temp_mean_change`,
            `r_mean_change`, `r_quant_change`, `temp_mean_squared_change`, `temp_mean_cubed_change`,
            `non_linear_change` and `temp_adiabat_anom_0`.</br>

            For each key with the `_change` suffix, a list containing a prefactor computed in the base climate and a
            change between simulations is returned. The sum of all these prefactors multiplied by the changes equals
            the full theory.

            There is also a `temp_adiabat_anom_0` key. Here `temp_adiabat_anom_0[1]` is $\Delta T_A$ in the base climate
            and `temp_adiabat_anom_0[0]` the prefactor which multiplies the $\Delta T_A$ term.
            If `taylor_terms_delta_mse_mod_anom` is not `linear`, then this term is not accurate - very complicated in
            this case and get $\Delta T_A^2$ terms.
    """
    R_mod = R * np.log(pressure_surf / pressure_ft) / 2
    q_sat_surf_quant = sphum_sat(temp_surf_quant, pressure_surf)  # [n_exp, n_quant]
    q_sat_surf_mean = sphum_sat(temp_surf_mean, pressure_surf)  # [n_exp]
    alpha_surf_quant = clausius_clapeyron_factor(temp_surf_quant[0], pressure_surf)  # float
    alpha_surf_mean = clausius_clapeyron_factor(temp_surf_mean[0], pressure_surf)  # float

    r_quant = sphum_quant / q_sat_surf_quant  # [n_exp, n_quant]
    r_mean = sphum_mean / q_sat_surf_mean  # [n_exp]

    delta_temp_surf_mean = temp_surf_mean[1] - temp_surf_mean[0]
    delta_r_mean = r_mean[1] - r_mean[0]
    delta_r_quant = r_quant[1] - r_quant[0]

    delta_mse_mod_mean_rh_term = L_v * q_sat_surf_mean[0] * delta_r_mean
    delta_mse_mod_mean_temp_term = (c_p - R_mod + L_v * alpha_surf_mean * sphum_mean[0]) * delta_temp_surf_mean

    # Get coefs and terms such that LHS of equation is
    # coef_delta_temp_quant * delta_temp_quant + coef_delta_temp_quant_squared * delta_temp_quant**2
    # and RHS is some of all remaining variables with term prefix.
    coef_delta_temp_quant = c_p - R_mod + L_v * alpha_surf_quant * sphum_quant[0]
    if taylor_terms_delta_mse_mod == 'squared':
        # Add extra term in taylor expansion of delta_mse_mod if requested
        delta_mse_mod_mean_temp_term += 0.5 * L_v * alpha_surf_mean * sphum_mean[0] * \
                                        (alpha_surf_mean - 2 / temp_surf_mean[0]) * delta_temp_surf_mean ** 2
        coef_delta_temp_quant_squared = 0.5 * L_v * alpha_surf_quant * sphum_quant[0] * (alpha_surf_quant -
                                                                                         2 / temp_surf_quant[0])

    if 'full' in rh_option:
        delta_mse_mod_mean_use = delta_mse_mod_mean_temp_term + delta_mse_mod_mean_rh_term
    elif 'approx' in rh_option:
        # Neglect rh terms when multiplied by an info[0] term, as very small.
        delta_mse_mod_mean_use = delta_mse_mod_mean_temp_term
    elif 'none' in rh_option:
        # Equivalent to setting both delta_r_mean and delta_r_quant to 0
        delta_mse_mod_mean_use = delta_mse_mod_mean_temp_term
        delta_mse_mod_mean_rh_term = 0
        delta_r_quant = delta_r_quant * 0
    else:
        raise ValueError(f"rh_option given is {rh_option}, but must contain 'full', 'approx' or 'none'")

    if 'anomaly' in rh_option:
        # Use mean so can easily combine relative humidity terms for quant and mean.
        term_r_quant = -L_v * q_sat_surf_mean[0] * delta_r_quant
    else:
        term_r_quant = -L_v * q_sat_surf_quant[0] * delta_r_quant

    _, info, temp_adiabat_anom = get_delta_mse_mod_anom_theory(temp_surf_mean, temp_surf_quant, sphum_mean,
                                                               sphum_quant, pressure_surf, pressure_ft,
                                                               taylor_terms_delta_mse_mod_anom)
    for key in info:
        info[key][0] = info[key][0] * 1000  # turn prefactor units into J/kg
    term_temp_adiabat_anom = info['temp_adiabat_anom'][0] * info['temp_adiabat_anom'][1]
    term_mse_mod_mean = delta_mse_mod_mean_temp_term + delta_mse_mod_mean_rh_term + \
                        info['mse_mod_mean'][0] * delta_mse_mod_mean_use
    term_mse_mean_squared = info['mse_mod_mean_squared'][0] * delta_mse_mod_mean_use ** 2
    term_mse_mod_mean_cubed = info['mse_mod_mean_cubed'][0] * delta_mse_mod_mean_use ** 3
    # Non-linear changes are $\delta \Delta T_A \delta T_s_mean$ changes
    term_non_linear = info['non_linear'][0] * info['temp_adiabat_anom'][1] * delta_mse_mod_mean_use

    terms_sum = term_r_quant + term_temp_adiabat_anom + term_mse_mod_mean + term_mse_mean_squared + \
                term_mse_mod_mean_cubed + term_non_linear
    # Old theory is when assume Delta T_A=0 and \delta \Delta T_A=0
    terms_sum_old = term_r_quant + delta_mse_mod_mean_temp_term + delta_mse_mod_mean_rh_term

    final_answer = []
    for coef_rhs in [terms_sum, terms_sum_old]:
        if taylor_terms_delta_mse_mod == 'squared':
            # Solve quadratic equation, taking the positive solution
            final_answer += [(-coef_delta_temp_quant + np.sqrt(coef_delta_temp_quant ** 2 -
                                                               4 * coef_delta_temp_quant_squared * (-coef_rhs))
                              ) / (2 * coef_delta_temp_quant_squared)]
        else:
            final_answer += [coef_rhs / coef_delta_temp_quant]

    if taylor_terms_delta_mse_mod == 'linear':
        # There are only 4 terms which contribute to near-surface temperature change in the linear case.
        # The sum of these 4 change terms will equal the final answer.
        # For each change, record prefactor formed from base climate variables and the change
        # (delta term for variable in question).
        # As well as changes, also record terms which depend on adiabatic temp anomaly in the base climate as
        # temp_adiabat_anom_0.
        out_info = {'temp_adiabat_anom_change': ([info['temp_adiabat_anom'][0] / coef_delta_temp_quant,
                                                  info['temp_adiabat_anom'][1]],),
                    'temp_mean_change': [(1 + info['mse_mod_mean'][0]) * delta_mse_mod_mean_temp_term / (
                            delta_temp_surf_mean * coef_delta_temp_quant), delta_temp_surf_mean],
                    'r_mean_change': [delta_mse_mod_mean_rh_term / (delta_r_mean * coef_delta_temp_quant),
                                      delta_r_mean],
                    'r_quant_change': [term_r_quant / coef_delta_temp_quant, delta_r_quant],
                    'temp_mean_squared_change': [
                        info['mse_mod_mean_squared'][0] * delta_mse_mod_mean_temp_term ** 2 / (
                                delta_temp_surf_mean ** 2 * coef_delta_temp_quant), delta_temp_surf_mean ** 2],
                    'temp_mean_cubed_change': [
                        info['mse_mod_mean_cubed'][0] * delta_mse_mod_mean_temp_term ** 3 / (
                                delta_temp_surf_mean ** 3 * coef_delta_temp_quant), delta_temp_surf_mean ** 3],
                    'non_linear_change': [
                        info['non_linear'][0] * delta_mse_mod_mean_temp_term / (
                                delta_temp_surf_mean * coef_delta_temp_quant),
                        info['temp_adiabat_anom'][1] * delta_temp_surf_mean],
                    'temp_adiabat_anom_0': [info['mse_mod_mean'][0] * delta_mse_mod_mean_use / temp_adiabat_anom[0],
                                            temp_adiabat_anom[0]]}
        if 'full' in rh_option:
            out_info['r_mean_change'][0] += info['mse_mod_mean'][0] * delta_mse_mod_mean_rh_term / (
                    delta_r_mean * coef_delta_temp_quant)

        return final_answer[0], final_answer[1], out_info
    else:
        return final_answer[0], final_answer[1]
