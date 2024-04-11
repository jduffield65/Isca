import numpy as np
import scipy.optimize
from ..utils.moist_physics import clausius_clapeyron_factor, sphum_sat, moist_static_energy
from ..utils.constants import c_p, L_v, R
from typing import Tuple


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


def get_delta_mse_mod_anom_theory(temp_surf_mean: np.ndarray, temp_surf_quant: np.ndarray, sphum_mean: np.ndarray,
                                  sphum_quant: np.ndarray, pressure_surf: float, pressure_ft: float,
                                  taylor_terms: str = 'linear') -> Tuple[np.ndarray, dict]:
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
        $R^{\dagger} = \\ln(p_s/p_{FT})/2$
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
            $\delta \Delta h^{\dagger}$ conditioned on each quantile of near-surface temperature.
        `info_dict`: Dictionary with 5 keys: `temp_adiabat_anom`, `mse_mod_mean`, `mse_mod_mean_squared`,
            `mse_mod_mean_cubed`, `non_linear`.</br>

            For each key, a list containing a prefactor computed in the base climate and a change between simulations is
            returned. I.e. for `info_dict[non_linear][1]` would be $\\delta \\Delta T_A\\delta \\overline{h^{\\dagger}}$
            and the total contribution of non-linear terms to $\delta \Delta h^{\dagger}$ would be
            `info_dict[non_linear][0] * info_dict[non_linear][1]`. In the `linear` case this would be zero,
            and `info_dict[temp_adiabat_anom][0]`$=\\beta_1$ and `info_dict[mse_mod_mean][0]`$=
            \\frac{\\beta_2}{\\beta_1}\\frac{\\overline{T_A}}{\\Delta T_A}$ would be the only non-zero prefactors.

    """
    # Compute adiabatic temperatures
    n_exp, n_quant = temp_surf_quant.shape
    temp_adiabat_mean = np.zeros_like(temp_surf_mean)
    temp_adiabat_quant = np.zeros_like(temp_surf_quant)
    for i in range(n_exp):
        temp_adiabat_mean[i] = get_temp_adiabat(temp_surf_mean[i], sphum_mean[i], pressure_surf, pressure_ft)
        for j in range(n_quant):
            temp_adiabat_quant[i, j] = get_temp_adiabat(temp_surf_quant[i, j], sphum_quant[i, j], pressure_surf, pressure_ft)
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

    # Compute modified MSE
    mse_mod_mean = moist_static_energy(temp_surf_mean, sphum_mean, height=0, c_p_const=c_p - R_mod)
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
        term_lls = -0.5 * beta_2 ** 2 / beta_1 ** 3 * temp_adiabat_anom[0] / temp_adiabat_mean[0] ** 2 * delta_mse_mod ** 2
        term_sll = 0.5 * beta_3 / beta_1 * (temp_adiabat_anom[0] / temp_adiabat_mean[0]) ** 2 * delta_mse_mod
        term_lsl = 0.5 * beta_3 / beta_1 ** 2 * temp_adiabat_anom[0] / temp_adiabat_mean[0] ** 2 * delta_mse_mod ** 2
        term_lnl = beta_2 / beta_1 / temp_adiabat_mean[0] * delta_temp_adiabat_anom * delta_mse_mod
        term_snl = beta_3 / beta_1 * temp_adiabat_anom[0] / temp_adiabat_mean[0] ** 2 * delta_temp_adiabat_anom * delta_mse_mod
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
                 'mse_mod_mean_squared': [(term_lls + term_lsl + term_sls) / delta_mse_mod ** 2 / 1000, delta_mse_mod ** 2],
                 'mse_mod_mean_cubed': [term_lss / delta_mse_mod ** 3 / 1000, delta_mse_mod ** 3],
                 'non_linear': [(term_lnl + term_snl) / (delta_temp_adiabat_anom * delta_mse_mod) / 1000,
                                delta_temp_adiabat_anom * delta_mse_mod]
                 }

    final_answer = term_ll + term_lll + term_lls + term_sll + term_lsl + term_lnl + term_snl + term_lss + term_sls
    final_answer = final_answer / 1000  # convert to units of kJ/kg
    return final_answer, info_dict
