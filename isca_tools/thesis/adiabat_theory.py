import numpy as np
import scipy.optimize
from ..utils.moist_physics import clausius_clapeyron_factor, sphum_sat, moist_static_energy
from ..utils.constants import c_p, L_v, R, g
from typing import Tuple, Union, Optional


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
    temp_ft_anom = temp_ft_quant - temp_ft_mean[:, np.newaxis]
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
    \\frac{\\beta_2}{\\beta_1}\\frac{\\Delta T_A}{\\overline{T_A}} \\delta \\overline{h^{\\dagger}}$$

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
            \\frac{\\beta_2}{\\beta_1}\\frac{\\Delta T_A}{\\overline{T_A}} \\delta \\overline{h^{\\dagger}}$
            * `squared_0`: Keep terms linear and squared in first expansion and then just linear terms:
            *LL*, *LLL*, *SL*, *SLL*.
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
            \\frac{\\beta_2}{\\beta_1}\\frac{\\Delta T_A}{\\overline{T_A}}$ would be the only non-zero prefactors.

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
    if taylor_terms.lower() not in ['linear', 'squared_0', 'squared', 'full']:
        raise ValueError(f'taylor_terms given is {taylor_terms}, but must be linear, squared_0, squared or full.')

    term_ll = beta_1 * delta_temp_adiabat_anom
    term_lll = beta_2 / beta_1 * temp_adiabat_anom[0] / temp_adiabat_mean[0] * delta_mse_mod
    if taylor_terms == 'squared_0':
        term_sl = beta_2 * temp_adiabat_anom[0] / temp_adiabat_mean[0] * delta_temp_adiabat_anom
        term_sll = 0.5 * beta_3 / beta_1 * (temp_adiabat_anom[0] / temp_adiabat_mean[0]) ** 2 * delta_mse_mod
        term_lls = 0
        term_lsl = 0
        term_lnl = 0
        term_snl = 0
    elif taylor_terms != 'linear':
        term_sl = beta_2 * temp_adiabat_anom[0] / temp_adiabat_mean[0] * delta_temp_adiabat_anom
        term_sll = 0.5 * beta_3 / beta_1 * (temp_adiabat_anom[0] / temp_adiabat_mean[0]) ** 2 * delta_mse_mod
        term_lls = -0.5 * beta_2 ** 2 / beta_1 ** 3 * temp_adiabat_anom[0] / temp_adiabat_mean[
            0] ** 2 * delta_mse_mod ** 2
        term_lsl = 0.5 * beta_3 / beta_1 ** 2 * temp_adiabat_anom[0] / temp_adiabat_mean[0] ** 2 * delta_mse_mod ** 2
        term_lnl = beta_2 / beta_1 / temp_adiabat_mean[0] * delta_temp_adiabat_anom * delta_mse_mod
        term_snl = beta_3 / beta_1 * temp_adiabat_anom[0] / temp_adiabat_mean[
            0] ** 2 * delta_temp_adiabat_anom * delta_mse_mod
    else:
        term_sl = 0
        term_sll = 0
        term_lls = 0
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
    info_dict = {'temp_adiabat_anom': [(term_ll + term_sl) / delta_temp_adiabat_anom / 1000, delta_temp_adiabat_anom],
                 'mse_mod_mean': [(term_lll + term_sll) / delta_mse_mod / 1000, delta_mse_mod],
                 'mse_mod_mean_squared': [(term_lls + term_lsl + term_sls) / delta_mse_mod ** 2 / 1000,
                                          delta_mse_mod ** 2],
                 'mse_mod_mean_cubed': [term_lss / delta_mse_mod ** 3 / 1000, delta_mse_mod ** 3],
                 'non_linear': [(term_lnl + term_snl) / (delta_temp_adiabat_anom * delta_mse_mod) / 1000,
                                delta_temp_adiabat_anom * delta_mse_mod]
                 }

    final_answer = term_ll + term_lll + term_sl + term_lls + term_sll + term_lsl + term_lnl + term_snl + term_lss + term_sls
    final_answer = final_answer / 1000  # convert to units of kJ/kg
    return final_answer, info_dict, temp_adiabat_anom


def do_delta_mse_mod_taylor_expansion(temp_surf: np.ndarray, sphum_surf: np.ndarray,
                                      pressure_surf: float, pressure_ft: float, taylor_terms: str = 'linear',
                                      temp_use_rh_term: Optional[Union[np.ndarray, float]] = None
                                      ) -> Tuple[Union[np.ndarray, float], dict]:
    """
    Does a taylor expansion of the change in modified moist static energy, $\delta h^{\dagger}$:

    $$\\delta h^{\\dagger} \\approx (c_p - R^{\\dagger} + L_v \\alpha q)\\delta T_s + L_v q^* \\delta r +
    0.5 L_v \\alpha q (\\alpha - 2 / T_s) \\delta T_s^2$$

    The last term is only included if `taylor_terms == 'linear'`.

    Args:
        temp_surf: `float [n_exp]` or `float [n_exp, n_quant]` </br>
            Near surface temperature of each simulation, corresponding to a different
            optical depth, $\kappa$. Units: *K*. We assume `n_exp=2`.
        sphum_surf: `float [n_exp]` or `float [n_exp, n_quant]` </br>
            Near surface specific humidity of each simulation. Units: *kg/kg*.
        pressure_surf:
            Pressure at near-surface in *Pa*.
        pressure_ft:
            Pressure at free troposphere level in *Pa*.
        taylor_terms:
            How many taylor series terms to keep in the expansions for changes in modified moist static energy:

            * `linear`: $\\delta h^{\\dagger} \\approx (c_p - R^{\\dagger} + L_v \\alpha q)\\delta T_s +
                L_v \\alpha q^* \\delta r$
            * `squared`: Includes the additional term $0.5 L_v \\alpha q (\\alpha - 2 / T_s) \\delta T_s^2$
        temp_use_rh_term: `float` or `float [n_quant]` </br>
            Can specify temperature to evaluate $q^*(T, p_s)$ in the $L_v q^* \\delta r$ term. May want to do
            this if want to approximate $q^*(T(x), p_s) \\approx q^*(\overline{T}, p_s)$ for this term, so can combine
            $\\delta r(x)$ and $\\delta \overline{r}$ terms. If `None`, then will use `temp_surf[0]`.

    Returns:
        `delta_mse_mod`: `float` or `float [n_quant]` </br>
            Approximation of $\delta h^{\\dagger}$. Units are *kJ/kg*.
        `info_dict`: Dictionary containing 3 keys: `rh`, `temp`, `temp_squared`. </br>
            For each key, there is a list with first value being the prefactor and second the change in the expansion.
            The sum of all these prefactors multiplied by the changes equals the full theory.
            Units of prefactors are *kJ/kg* divided by units of the change term.
    """
    R_mod = R * np.log(pressure_surf / pressure_ft) / 2
    q_sat = sphum_sat(temp_surf, pressure_surf)
    alpha = clausius_clapeyron_factor(temp_surf[0], pressure_surf)  # float
    rh = sphum_surf / q_sat

    delta_temp = temp_surf[1] - temp_surf[0]
    delta_rh = rh[1] - rh[0]

    if temp_use_rh_term is None:
        coef_rh = L_v * q_sat[0]
    else:
        coef_rh = L_v * sphum_sat(temp_use_rh_term, pressure_surf)
    coef_temp = (c_p - R_mod + L_v * alpha * sphum_surf[0])
    if taylor_terms == 'squared':
        # Add extra term in taylor expansion of delta_mse_mod if requested
        coef_temp_squared = 0.5 * L_v * alpha * sphum_surf[0] * \
                            (alpha - 2 / temp_surf[0])
    elif taylor_terms == 'linear':
        coef_temp_squared = sphum_surf[0] * 0  # set to 0 for all values
    else:
        raise ValueError(f"taylor_terms given is {taylor_terms}, but must be 'linear' or 'squared'")

    final_answer = (coef_rh * delta_rh + coef_temp * delta_temp + coef_temp_squared * delta_temp ** 2) / 1000
    info_dict = {'rh': [coef_rh / 1000, delta_rh], 'temp': [coef_temp / 1000, delta_temp],
                 'temp_squared': [coef_temp_squared / 1000, delta_temp ** 2]}
    return final_answer, info_dict


def get_delta_temp_quant_theory(temp_surf_mean: np.ndarray, temp_surf_quant: np.ndarray, sphum_mean: np.ndarray,
                                sphum_quant: np.ndarray, pressure_surf: float, pressure_ft: float,
                                taylor_terms_delta_mse_mod_anom: str = 'linear',
                                taylor_terms_delta_mse_mod: str = 'linear',
                                rh_option: str = 'full', sphum_option: str = 'full',
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
        * All terms in $\\beta$ are evaluated at $\overline{T_A}$. I.e. $q_A^* = q^*(\overline{T_A}, p_{FT})$.

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
                L_v q^* \\delta r$
            * `squared`: Includes the additional term $0.5 L_v \\alpha q (\\alpha - 2 / T_s) \\delta T_s^2$
        rh_option:
            Relative humidity changes, $\delta r$, provide a secondary effect, so we give a number of options to
            approximate them:

            * `full`: No approximations to $\delta r$ terms.
            * `approx`: Neglect all $\\Delta T_A \delta r$ terms, in essence non-linear terms.
            * `none`: Set all relative humidity changes to zero.
            * If `anomaly` is also included e.g. `full_anomaly` or `approx_anomaly`, the $\delta r(x)$ term will be
                multiplied by $\overline{q^*}$ rather than $q^*(x)$. This then gives a $\delta (r(x) - \overline{r})$
                term rather than two separate relative humidity terms.
        sphum_option:
            Only relavent if `taylor_terms_delta_mse_mod = 'linear`. Specific humidity as function of quantile, $q(x)$
            appears in denominator due to $\beta_{s1}(x)$ term. Below, give options to replace this with
            $\overline{\beta_{s1}}$ in some cases to remove $x$ dependence.

            * `full`: No approximations to $\beta_{s1}(x)$ terms.
            * `approx`: Keep $\beta_{s1}(x)$ as denominator for the primary $\delta \overline{T_s}$ term, but
                replace with $\overline{\beta_{s1}}$ elsewhere (where relative humidity, $\Delta T_A$ or
                $\delta \Delta T_A$ appear).
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
    info_taylor_mean = do_delta_mse_mod_taylor_expansion(temp_surf_mean, sphum_mean, pressure_surf, pressure_ft,
                                                         taylor_terms_delta_mse_mod)[1]
    delta_temp_surf_mean = info_taylor_mean['temp'][1]
    delta_r_mean = info_taylor_mean['rh'][1]
    delta_mse_mod_mean_rh_term = info_taylor_mean['rh'][0] * delta_r_mean * 1000
    beta_s1_mean = info_taylor_mean['temp'][0] * 1000
    delta_mse_mod_mean_temp_term = beta_s1_mean * delta_temp_surf_mean + \
                                   info_taylor_mean['temp_squared'][0] * delta_temp_surf_mean ** 2 * 1000
    info_taylor_quant = \
        do_delta_mse_mod_taylor_expansion(temp_surf_quant, sphum_quant, pressure_surf,
                                          pressure_ft, taylor_terms_delta_mse_mod,
                                          temp_use_rh_term=temp_surf_mean[0] if 'anomaly' in rh_option else None)[1]
    delta_r_quant = info_taylor_quant['rh'][1]
    beta_s1 = info_taylor_quant['temp'][0] * 1000
    coef_delta_temp_quant_squared = info_taylor_quant['temp_squared'][0] * 1000
    term_r_quant = -info_taylor_quant['rh'][0] * delta_r_quant * 1000

    if 'full' in rh_option:
        delta_mse_mod_mean_use = delta_mse_mod_mean_temp_term + delta_mse_mod_mean_rh_term
    elif 'approx' in rh_option:
        # Neglect rh terms when multiplied by an info[0] term, as very small.
        delta_mse_mod_mean_use = delta_mse_mod_mean_temp_term
    elif 'none' in rh_option:
        # Equivalent to setting both delta_r_mean and delta_r_quant to 0
        delta_mse_mod_mean_use = delta_mse_mod_mean_temp_term
        delta_mse_mod_mean_rh_term = 0
        term_r_quant = term_r_quant * 0
        delta_r_mean = delta_r_mean * 0
        delta_r_quant = delta_r_quant * 0
    else:
        raise ValueError(f"rh_option given is {rh_option}, but must contain 'full', 'approx' or 'none'")

    _, info, temp_adiabat_anom = get_delta_mse_mod_anom_theory(temp_surf_mean, temp_surf_quant, sphum_mean,
                                                               sphum_quant, pressure_surf, pressure_ft,
                                                               taylor_terms_delta_mse_mod_anom)
    for key in info:
        info[key][0] = info[key][0] * 1000  # turn prefactor units into J/kg
    beta_a1 = info['temp_adiabat_anom'][0]

    # Record all terms in RHS of equation where LHS includes \delta temp_surf(x)
    # temp_a_prefactor is terms with climatological Delta T_A in prefactor
    terms_rhs = {'temp_s_mean': delta_mse_mod_mean_temp_term,
                 'rh_mean': delta_mse_mod_mean_rh_term,
                 'rh_quant': term_r_quant,
                 'temp_a_anom': beta_a1 * info['temp_adiabat_anom'][1],
                 'temp_s_mean_temp_a0': info['mse_mod_mean'][0] * delta_mse_mod_mean_temp_term,
                 'rh_mean_temp_a0': info['mse_mod_mean'][0] * delta_mse_mod_mean_rh_term if 'full' in rh_option else 0,
                 'squared': info['mse_mod_mean_squared'][0] * delta_mse_mod_mean_use ** 2,
                 'cubed': info['mse_mod_mean_cubed'][0] * delta_mse_mod_mean_use ** 3,
                 'non_linear': info['non_linear'][0] * info['temp_adiabat_anom'][1] * delta_mse_mod_mean_use
                 }
    # what to divide rhs terms by - only used if taylor_terms_delta_mse_mod is linear.
    if sphum_option == 'full':
        terms_rhs_denom = {key: beta_s1 for key in terms_rhs}
    elif sphum_option == 'approx':
        terms_rhs_denom = {key: beta_s1_mean for key in terms_rhs}
        terms_rhs_denom['temp_s_mean'] = beta_s1
    else:
        raise ValueError(f'sphum_option needs to be full or approx but was {sphum_option}')
    # term_mse_mod_mean1 = delta_mse_mod_mean_temp_term + delta_mse_mod_mean_rh_term  # Term in old theory
    # term_mse_mod_mean2 = info['mse_mod_mean'][0] * delta_mse_mod_mean_use  # with \Delta T_A in prefactor
    # term_mse_mean_squared = info['mse_mod_mean_squared'][0] * delta_mse_mod_mean_use ** 2
    # term_mse_mod_mean_cubed = info['mse_mod_mean_cubed'][0] * delta_mse_mod_mean_use ** 3
    # # Non-linear changes are $\delta \Delta T_A \delta T_s_mean$ changes
    # term_non_linear = info['non_linear'][0] * info['temp_adiabat_anom'][1] * delta_mse_mod_mean_use

    # terms_sum = term_r_quant + term_temp_adiabat_anom + term_mse_mod_mean1 + term_mse_mod_mean2 + term_mse_mean_squared + \
    #             term_mse_mod_mean_cubed + term_non_linear
    # # Old theory is when assume Delta T_A=0 and \delta \Delta T_A=0
    # terms_sum_old = term_r_quant + term_mse_mod_mean1

    final_answer = {}
    for key in ['full', 'old']:
        if taylor_terms_delta_mse_mod == 'squared':
            if key == 'old':
                # have to use sum not np.sum for sum of list like this
                coef_rhs = sum([terms_rhs[var] for var in ['temp_s_mean', 'rh_mean', 'rh_quant']])
            else:
                coef_rhs = sum([terms_rhs[var] for var in terms_rhs])
            # Solve quadratic equation, taking the positive solution
            final_answer[key] = (-beta_s1 + np.sqrt(beta_s1 ** 2 - 4 * coef_delta_temp_quant_squared * (-coef_rhs))
                                 ) / (2 * coef_delta_temp_quant_squared)
        else:
            if key == 'old':
                final_answer[key] = sum([terms_rhs[var] / terms_rhs_denom[var] for var in
                                         ['temp_s_mean', 'rh_mean', 'rh_quant']])
            else:
                final_answer[key] = sum([terms_rhs[var] / terms_rhs_denom[var] for var in terms_rhs])

    if taylor_terms_delta_mse_mod == 'linear':
        # There are only 4 terms which contribute to near-surface temperature change in the linear case.
        # The sum of these 4 change terms will equal the final answer.
        # For each change, record prefactor formed from base climate variables and the change
        # (delta term for variable in question).
        # As well as changes, also record terms which depend on adiabatic temp anomaly in the base climate as
        # temp_adiabat_anom_0.
        out_info = {'temp_adiabat_anom_change': [terms_rhs['temp_a_anom'] / (terms_rhs_denom['temp_a_anom']*
                                                                             info['temp_adiabat_anom'][1]),
                                                 info['temp_adiabat_anom'][1]],
                    'temp_mean_change': [sum(terms_rhs[var] / terms_rhs_denom[var] for var in
                                                ['temp_s_mean', 'temp_s_mean_temp_a0'])/delta_temp_surf_mean,
                                         delta_temp_surf_mean],
                    'r_mean_change': [0 if 'none' in rh_option else
                                      sum(terms_rhs[var] / terms_rhs_denom[var] for var in
                                             ['rh_mean', 'rh_mean_temp_a0'])/delta_r_mean, delta_r_mean],
                    'r_quant_change': [term_r_quant * 0 if 'none' in rh_option else
                                       terms_rhs['rh_quant'] / (terms_rhs_denom['rh_quant'] * delta_r_quant),
                                       delta_r_quant],
                    'temp_mean_squared_change': [
                        info['mse_mod_mean_squared'][0] * delta_mse_mod_mean_temp_term ** 2 / (
                                delta_temp_surf_mean ** 2 * terms_rhs_denom['squared']), delta_temp_surf_mean ** 2],
                    'temp_mean_cubed_change': [
                        info['mse_mod_mean_cubed'][0] * delta_mse_mod_mean_temp_term ** 3 / (
                                delta_temp_surf_mean ** 3 * terms_rhs_denom['cubed']), delta_temp_surf_mean ** 3],
                    'non_linear_change': [
                        info['non_linear'][0] * delta_mse_mod_mean_temp_term / (
                                delta_temp_surf_mean * terms_rhs_denom['non_linear']),
                        info['temp_adiabat_anom'][1] * delta_temp_surf_mean],
                    'temp_adiabat_anom_0': [sum(terms_rhs[var] / terms_rhs_denom[var] for var in
                                             ['temp_s_mean_temp_a0', 'rh_mean_temp_a0'])/temp_adiabat_anom[0],
                                            temp_adiabat_anom[0]]}

        return final_answer['full'], final_answer['old'], out_info
    else:
        return final_answer['full'], final_answer['old']


def get_delta_temp_quant_z_theory(temp_surf_mean: np.ndarray, temp_surf_quant: np.ndarray, sphum_mean: np.ndarray,
                                  sphum_quant: np.ndarray, temp_ft_mean: np.ndarray, temp_ft_quant: np.ndarray,
                                  z_ft_mean: np.ndarray, z_ft_quant: np.ndarray,
                                  pressure_surf: float, pressure_ft: float,
                                  taylor_terms_delta_mse_mod: str = 'linear',
                                  rh_option: str = 'full'
                                  ) -> Union[Tuple[np.ndarray, np.ndarray, dict], Tuple[np.ndarray, np.ndarray]]:
    """
    Returns a theoretical prediction for change in a given percentile, $x$, of near-surface temperature. Here, with
    $\Delta z_{FT}$ rather than $\Delta T_{FT}$ so more relavent to extratropics. In the simplest
    case with `taylor_terms_delta_mse_mod='linear'` and
    `rh_option = 'approx_anomaly'`, the equation for the theory is:

    $$\delta T(x) = \gamma_{T}' \delta \overline{T} + \gamma_{\Delta r}' \delta (\overline{r} - r(x)) +
    \\frac{\\beta_1 \\delta \\Delta T_A(x)}{c_p - R^{\\dagger} + L_v \\alpha q + \\beta_1} +
    \\frac{\\beta_2}{\\beta_1}\\frac{\\Delta T_A'(x)}{\overline{T_A}} \gamma_{T_2} \delta \overline{T}$$

    A simpler theory with $T_{CE} = \\overline{T_{CE}} = \\Delta z_{FT} = 0$, as well as for $\delta$ versions of these
    is also returned.

    Terms in equation:
        * $\gamma_T' = \\frac{c_p - R^{\dagger} + L_v \\bar{\\alpha} \\bar{q} + \\beta_1}{c_p - R^{\dagger} +
        L_v \\alpha q + \\beta_1}$
        * $\gamma_{T_2} = \\frac{c_p - R^{\dagger} + L_v \\bar{\\alpha} \\bar{q}}{c_p - R^{\dagger} +
        L_v \\alpha q + \\beta_1}$
        * $\gamma_{\Delta r}' = \\frac{L_v \\overline{q^*}}{c_p - R^{\dagger} + L_v \\alpha q + \\beta_1}$
        * $q^*$ and $\\alpha$ are evaluated at the surface i.e. $q^* = q^*(T_s, p_s)$.
        * $R^{\dagger} = R\\ln(p_s/p_{FT})/2$
        * $\\Delta T_A = T_{CE}(x) - \\overline{T_{CE}} + \\Delta T_{FT}$ and all
        terms are evaluated in the base climate.
        * $\\delta \\Delta T_A' = \\delta T_{CE}(x) - \\delta \\overline{T_{CE}} +
        \\frac{g}{R^{\\dagger}}\\delta \\Delta z_{FT}$
        * $\\beta_1 = \\frac{d\\overline{h^{\\dagger}}}{d\\overline{T_A}} = c_p + R^{\dagger} + L_v \\alpha_A q_A^*$
        * $\\beta_2 = \\overline{T_A}\\frac{d\\beta_1}{d\\overline{T_A}} = L_v \\alpha_A q^*(\\alpha_A \\overline{T_A} - 2)$
        * All terms in $\\beta$ are evaluated at $\overline{T_A}$. I.e. $q_A^* = q^*(\overline{T_A}, p_{FT})$.

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
        temp_ft_mean: `float [n_exp]`</br>
            Average temperature at `pressure_ft` in Kelvin.
        temp_ft_quant: `float [n_exp, n_quant]`</br>
            `temp_ft_quant[i, j]` is temperature at `pressure_ft`, averaged over all days with near-surface temperature
             corresponding to the quantile `quant_use[j]`, for experiment `i`. Units: *kg/kg*.
        z_ft_mean: `float [n_exp]`</br>
            Average geopotential height at `pressure_ft` in *m*.
        z_ft_quant: `float [n_exp, n_quant]`</br>
            `z_ft_quant[i, j]` is geopotential height at `pressure_ft`, averaged over all days with near-surface
            temperature corresponding to the quantile `quant_use[j]`, for experiment `i`. Units: *m*.
        pressure_surf:
            Pressure at near-surface in *Pa*.
        pressure_ft:
            Pressure at free troposphere level in *Pa*.
        taylor_terms_delta_mse_mod:
            How many taylor series terms to keep in the expansions for changes in modified moist static energy:
            $\\delta h^{\\dagger}(x)$ and $\\delta \overline{h^{\\dagger}}$ ($\\alpha$ and $q^*$ are both evaluated
            at the surface i.e. $\\alpha(T_s, p_s)$).

            * `linear`: $\\delta h^{\\dagger} \\approx (c_p - R^{\\dagger} + L_v \\alpha q)\\delta T_s +
                L_v q^* \\delta r$
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
            Theoretical temperature difference using old theory which assumes
            $T_{CE} = \\overline{T_{CE}} = \\Delta z_{FT} = 0$, as well as for $\delta$ versions of these.
        `info_dict`: Only returned if `taylor_terms_delta_mse_mod='linear'` otherwise too complicated.

            Dictionary with 5 keys for each term in the theory: `temp_adiabat_anom_change`, `temp_mean_change`,
            `r_mean_change`, `r_quant_change` and `temp_adiabat_anom_0`.</br>

            For each key with the `_change` suffix, a list containing a prefactor computed in the base climate and a
            change between simulations is returned. The sum of all these prefactors multiplied by the changes equals
            the full theory.

            There is also a `temp_adiabat_anom_0` key. Here `temp_adiabat_anom_0[1]` is
            $\\overline{T_{CE}} - T_{CE}(x) + \\frac{g}{R^{\\dagger}} \\Delta z_{FT} - \\Delta T_s$ in the base climate
            and `temp_adiabat_anom_0[0]` the prefactor which multiplies this term.
    """
    R_mod = R * np.log(pressure_surf / pressure_ft) / 2
    info_taylor_mean = do_delta_mse_mod_taylor_expansion(temp_surf_mean, sphum_mean, pressure_surf, pressure_ft,
                                                         taylor_terms_delta_mse_mod)[1]
    delta_temp_surf_mean = info_taylor_mean['temp'][1]
    delta_r_mean = info_taylor_mean['rh'][1]
    # turn prefactor units into J/kg by multiplying by 1000.
    delta_mse_mod_mean_rh_term = info_taylor_mean['rh'][0] * delta_r_mean * 1000
    delta_mse_mod_mean_temp_term = info_taylor_mean['temp'][0] * delta_temp_surf_mean * 1000 + \
                                   info_taylor_mean['temp_squared'][0] * delta_temp_surf_mean ** 2 * 1000
    info_taylor_quant = \
        do_delta_mse_mod_taylor_expansion(temp_surf_quant, sphum_quant, pressure_surf,
                                          pressure_ft, taylor_terms_delta_mse_mod,
                                          temp_use_rh_term=temp_surf_mean[0] if 'anomaly' in rh_option else None)[1]
    delta_r_quant = info_taylor_quant['rh'][1]
    coef_delta_temp_quant = info_taylor_quant['temp'][0] * 1000
    coef_delta_temp_quant_squared = info_taylor_quant['temp_squared'][0] * 1000
    term_r_quant = -info_taylor_quant['rh'][0] * delta_r_quant * 1000

    if 'full' in rh_option:
        delta_mse_mod_mean_use = delta_mse_mod_mean_temp_term + delta_mse_mod_mean_rh_term
    elif 'approx' in rh_option:
        # Neglect rh terms when multiplied by an info[0] term, as very small.
        delta_mse_mod_mean_use = delta_mse_mod_mean_temp_term
    elif 'none' in rh_option:
        # Equivalent to setting both delta_r_mean and delta_r_quant to 0
        delta_mse_mod_mean_use = delta_mse_mod_mean_temp_term
        delta_mse_mod_mean_rh_term = 0
        term_r_quant = term_r_quant * 0
        delta_r_mean = delta_r_mean * 0
        delta_r_quant = delta_r_quant * 0
    else:
        raise ValueError(f"rh_option given is {rh_option}, but must contain 'full', 'approx' or 'none'")

    temp_adiabat_anom, temp_ce_mean, temp_ce_quant = \
        decompose_temp_adiabat_anomaly(temp_surf_mean, temp_surf_quant, sphum_mean,
                                       sphum_quant, temp_ft_mean, temp_ft_quant, pressure_surf, pressure_ft)[:3]
    delta_temp_ce_mean = temp_ce_mean[1] - temp_ce_mean[0]
    delta_temp_ce_quant = temp_ce_quant[1] - temp_ce_quant[0]
    z_ft_anom = z_ft_quant - z_ft_mean[:, np.newaxis]
    delta_z_ft_anom = z_ft_anom[1] - z_ft_anom[0]

    info = get_delta_mse_mod_anom_theory(temp_surf_mean, temp_surf_quant, sphum_mean,
                                         sphum_quant, pressure_surf, pressure_ft, 'linear')[1]
    for key in info:
        info[key][0] = info[key][0] * 1000  # turn prefactor units into J/kg
    beta_1 = info['temp_adiabat_anom'][0]
    # get prefactor to \Delta T_A \delta h_mod_mean term which is beta_2/beta_1/temp_adiabat_mean[0]
    term_mse_mod_mean2_prefactor = info['mse_mod_mean'][0] / temp_adiabat_anom[0]
    coef_delta_temp_quant = coef_delta_temp_quant + beta_1  # +beta_1 is a modification for z theory

    # Combine terms on RHS of equation, with changes due to z term.
    term_temp_adiabat_anom = beta_1 * (delta_temp_ce_mean - delta_temp_ce_quant + g / R_mod * delta_z_ft_anom)
    # extra beta_1 term below is a modification for z theory
    term_mse_mod_mean1 = delta_mse_mod_mean_temp_term + delta_mse_mod_mean_rh_term + beta_1 * delta_temp_surf_mean
    # get adiabatic temp anomaly term with z_ft_anom replacing temp_ft_anom
    # temp_adiabat_anom_z0 = temp_ce_mean[0] - temp_ce_quant[0] + g / R_mod * z_ft_anom[0] - temp_surf_anom0
    term_mse_mod_mean2 = term_mse_mod_mean2_prefactor * temp_adiabat_anom[0] * delta_mse_mod_mean_use

    terms_sum = term_r_quant + term_temp_adiabat_anom + term_mse_mod_mean1 + term_mse_mod_mean2
    # Old theory is when assume both \Delta T_A=0 as well as \delta z_anom=0 as well as the \delta temp_ce terms.
    terms_sum_old = term_r_quant + term_mse_mod_mean1
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
        out_info = {'temp_adiabat_anom_change': [beta_1 / coef_delta_temp_quant, delta_temp_ce_mean -
                                                 delta_temp_ce_quant + g / R_mod * delta_z_ft_anom],
                    'temp_mean_change': [(beta_1 + (1 + term_mse_mod_mean2_prefactor * temp_adiabat_anom[0]) *
                                          delta_mse_mod_mean_temp_term / delta_temp_surf_mean) / coef_delta_temp_quant,
                                         delta_temp_surf_mean],
                    'r_mean_change': [0 if 'none' in rh_option else
                                      delta_mse_mod_mean_rh_term / (delta_r_mean * coef_delta_temp_quant),
                                      delta_r_mean],
                    'r_quant_change': [term_r_quant * 0 if 'none' in rh_option else
                                       term_r_quant / (delta_r_quant * coef_delta_temp_quant), delta_r_quant],
                    'temp_adiabat_anom_0': [term_mse_mod_mean2 / (temp_adiabat_anom[0] * coef_delta_temp_quant),
                                            temp_adiabat_anom[0]]}
        if 'full' in rh_option:
            out_info['r_mean_change'][0] += term_mse_mod_mean2_prefactor * temp_adiabat_anom[0] * \
                                            delta_mse_mod_mean_rh_term / (delta_r_mean * coef_delta_temp_quant)
        return final_answer[0], final_answer[1], out_info
    else:
        return final_answer[0], final_answer[1]
