import numpy as np
from ..utils.moist_physics import clausius_clapeyron_factor, sphum_sat
from ..utils.constants import c_p, L_v


def get_delta_temp_quant_theory(temp_mean: np.ndarray, sphum_mean: np.ndarray, temp_quant: np.ndarray,
                                sphum_quant: np.ndarray, pressure_surface: float, const_rh: bool = False) -> np.ndarray:
    """
    Computes the theoretical temperature difference between simulations of neighbouring optical depth values for each
    percentile, $\delta T(x)$, according to the assumption that changes in MSE are equal to the change in mean MSE,
    $\delta h(x) = \delta \overline{h}$:

    $$\delta T(x) = \gamma^T \delta \overline{T} + \gamma^{\Delta r} \delta (\overline{r} - r(x))$$

    If data from `n_exp` optical depth values provided, `n_exp-1` theoretical temperature differences will be returned
    for each percentile.

    Args:
        temp_mean: `float [n_exp]`</br>
            Average near surface temperature of each simulation, corresponding to a different
            optical depth, $\kappa$. Units: *K*.
        sphum_mean: `float [n_exp]`</br>
            Average near surface specific humidity of each simulation. Units: *kg/kg*.
        temp_quant: `float [n_exp, n_quant]`</br>
            `temp_quant[i, j]` is the percentile `quant_use[j]` of near surface temperature of
            experiment `i`. Units: *K*.</br>
            Note that `quant_use` is not provided as not needed by this function, but is likely to be
            `np.arange(1, 100)` - leave out `x=0` as doesn't really make sense to consider $0^{th}$ percentile
            of a quantity.
        sphum_quant: `float [n_exp, n_quant]`</br>
            `sphum_quant[i, j]` is the percentile `quant_use[j]` of near surface specific humidity of
            experiment `i`. Units: *kg/kg*.
        pressure_surface: Near surface pressure level. Units: *Pa*.
        const_rh: If `True`, will return the constant relative humidity version of the theory, i.e.
            $\gamma^T \delta \overline{T}$. Otherwise, will return the full theory.

    Returns:
        `float [n_exp-1, n_quant]`.</br>
            `delta_temp_quant_theory[i, j]` refers to the theoretical temperature difference between experiment `i` and
            `i+1` for percentile `quant_use[j]`.
    """
    alpha_quant = clausius_clapeyron_factor(temp_quant, pressure_surface)
    alpha_mean = clausius_clapeyron_factor(temp_mean, pressure_surface)
    sphum_quant_sat = sphum_sat(temp_quant, pressure_surface)
    sphum_mean_sat = sphum_sat(temp_mean, pressure_surface)
    r_quant = sphum_quant / sphum_quant_sat
    r_mean = sphum_mean / sphum_mean_sat
    delta_temp_mean = np.expand_dims(np.diff(temp_mean), axis=-1)
    delta_r_mean = np.expand_dims(np.diff(r_mean), axis=-1)
    delta_r_quant = np.diff(r_quant, axis=0)

    denom = c_p + L_v * alpha_quant * sphum_quant
    gamma_t = np.expand_dims(c_p + L_v * alpha_mean * sphum_mean, axis=-1) / denom
    gamma_rdiff = L_v/denom * np.expand_dims(sphum_mean_sat, axis=-1)
    if const_rh:
        delta_temp_quant_theory = gamma_t[:-1] * delta_temp_mean
    else:
        delta_temp_quant_theory = gamma_t[:-1] * delta_temp_mean + gamma_rdiff[:-1] * (delta_r_mean - delta_r_quant)
    return delta_temp_quant_theory
