import numpy as np
from ..utils.moist_physics import clausius_clapeyron_factor, sphum_sat
from ..utils.constants import c_p, L_v
from typing import Tuple, Optional


def get_delta_temp_quant_theory(temp_quant_ocean: np.ndarray, sphum_quant_ocean: np.ndarray, temp_quant_land: np.ndarray,
                                sphum_quant_land: np.ndarray, pressure_surface: float, const_rh: bool = False,
                                delta_mse_ratio: Optional[np.ndarray] = None,
                                taylor_level: str = 'linear_rh_diff') -> np.ndarray:
    """
    Computes the theoretical temperature difference between simulations of neighbouring optical depth values for each
    percentile, $\delta T(x)$, according to the assumption that changes in MSE are equal to the change in mean MSE,
    $\delta h(x) = \delta \overline{h}$:

    $$\delta T_L(x) = \gamma^T \delta T_O(x) + \gamma^{\Delta r} \delta (r_O(x) - r_L(x))$$

    This above equation is for the default settings, but a more accurate equation can be used with the `delta_mse_ratio`
    and `taylor_level` arguments.

    If data from `n_exp` optical depth values provided, `n_exp-1` theoretical temperature differences will be returned
    for each percentile.

    Args:
        temp_quant_ocean: `float [n_exp, n_quant]`</br>
            `temp_quant_ocean[i, j]` is the percentile `quant_use[j]` of near surface ocean temperature of
            experiment `i`. Units: *K*.</br>
            Note that `quant_use` is not provided as not needed by this function, but is likely to be
            `np.arange(1, 100)` - leave out `x=0` as doesn't really make sense to consider $0^{th}$ percentile
            of a quantity.
        sphum_quant_ocean: `float [n_exp, n_quant]`</br>
            `sphum_quant_ocean[i, j]` is the percentile `quant_use[j]` of near surface ocean specific humidity of
            experiment `i`. Units: *kg/kg*.
        temp_quant_land: `float [n_exp, n_quant]`</br>
            `temp_quant_land[i, j]` is the percentile `quant_use[j]` of near surface land temperature of
            experiment `i`. Units: *K*.</br>
        sphum_quant_land: `float [n_exp, n_quant]`</br>
            `sphum_quant_land[i, j]` is the percentile `quant_use[j]` of near surface land specific humidity of
            experiment `i`. Units: *kg/kg*.
        pressure_surface: Near surface pressure level. Units: *Pa*.
        const_rh: If `True`, will return the constant relative humidity version of the theory, i.e.
            $\gamma^T \delta \overline{T}$. Otherwise, will return the full theory.
        delta_mse_ratio: `float [n_exp-1, n_quant]`</br>
            `delta_mse_ratio[i]` is the change in $x$ percentile of MSE divided by the change in the mean MSE
            between experiment `i` and `i+1`: $\delta h(x)/\delta \overline{h}$.
            If not given, it is assumed to be equal to 1 for all $x$.
        taylor_level: This specifies the level of approximation that goes into the taylor series for $\delta q(x)$
            and $\delta \overline{q}$:

            - `squared`: Includes squared, $\delta T^2$, nonlinear, $\delta T \delta r$, and linear terms.
            - `nonlinear`: Includes nonlinear, $\delta T \delta r$, and linear terms.
            - `linear`: Includes just linear terms so $\delta T_L(x) = \\gamma^T \delta T_O(x) +
                \\gamma^{r_O} \delta r_O(x) +\\gamma^{r_L} \\delta r_L(x)$
            - `linear_rh_diff`: Same as `linear`, but does another approximation to combine relative humidity
                contributions so: $\delta T_L(x) = \gamma^T \delta T_O(x) +
                \gamma^{\Delta r} \delta (r_O(x) - r_L(x))$

    Returns:
        `float [n_exp-1, n_quant]`.</br>
            `delta_temp_quant_theory[i, j]` refers to the theoretical temperature difference between experiment `i` and
            `i+1` for percentile `quant_use[j]`.
    """
    n_exp, n_quant = temp_quant_land.shape
    alpha_quant_l = clausius_clapeyron_factor(temp_quant_land, pressure_surface)
    alpha_quant_o = clausius_clapeyron_factor(temp_quant_ocean, pressure_surface)
    sphum_quant_sat_l = sphum_sat(temp_quant_land, pressure_surface)
    sphum_quant_sat_o = sphum_sat(temp_quant_ocean, pressure_surface)
    r_quant_l = sphum_quant_land / sphum_quant_sat_l
    r_quant_o = sphum_quant_ocean / sphum_quant_sat_o
    delta_temp_quant_o = np.diff(temp_quant_ocean, axis=0)

    delta_r_quant_o = np.diff(r_quant_o, axis=0)
    delta_r_quant_l = np.diff(r_quant_l, axis=0)
    if const_rh:
        # get rid of relative humidity contribution if constant rh
        delta_r_quant_o = 0 * delta_r_quant_o
        delta_r_quant_l = 0 * delta_r_quant_l

    # Pad all delta variables so same size as temp_quant - will not use this in calculation but just makes it easier
    pad_array = ((0, 1), (0, 0))
    if delta_mse_ratio is None:
        delta_mse_ratio = np.ones_like(temp_quant_land)
    else:
        # make delta_mse_ratio the same size as all other quant variables
        delta_mse_ratio = np.pad(delta_mse_ratio, pad_width=pad_array)
    delta_temp_quant_o = np.pad(delta_temp_quant_o, pad_width=pad_array)
    delta_r_quant_o = np.pad(delta_r_quant_o, pad_width=pad_array)
    delta_r_quant_l = np.pad(delta_r_quant_l, pad_width=pad_array)

    if taylor_level == 'squared':
        # Keep squared, linear and non-linear terms in taylor expansion of delta_sphum_quant
        coef_a = 0.5 * L_v * alpha_quant_l * sphum_quant_land * (alpha_quant_l - 2 / temp_quant_land[0])
        coef_b = c_p + L_v * alpha_quant_l * (sphum_quant_land + sphum_quant_sat_l * delta_r_quant_l)
        coef_c = L_v * sphum_quant_sat_l * delta_r_quant_l - delta_mse_ratio * (
            0.5 * L_v * alpha_quant_o * sphum_quant_ocean * (alpha_quant_o - 2 / temp_quant_ocean) * delta_temp_quant_o ** 2 +
            (c_p + L_v * alpha_quant_o * (sphum_quant_ocean + sphum_quant_sat_o * delta_r_quant_o)) * delta_temp_quant_o +
            L_v * sphum_quant_sat_o * delta_r_quant_o)
        delta_temp_quant_theory = np.asarray([[np.roots([coef_a[i, j], coef_b[i, j], coef_c[i, j]])[1]
                                               for j in range(n_quant)] for i in range(n_exp - 1)])
    elif taylor_level in ['nonlinear', 'non-linear']:
        # Keep linear and non-linear terms in taylor expansion of delta_sphum_quant
        coef_b = c_p + L_v * alpha_quant_l * (sphum_quant_land + sphum_quant_sat_l * delta_r_quant_l)
        coef_c = L_v * sphum_quant_sat_l * delta_r_quant_l - delta_mse_ratio * (
            (c_p + L_v * alpha_quant_o * (sphum_quant_ocean + sphum_quant_sat_o * delta_r_quant_o)) * delta_temp_quant_o +
            L_v * sphum_quant_sat_o * delta_r_quant_o)
        delta_temp_quant_theory = -coef_c[:-1] / coef_b[:-1]
    elif taylor_level == 'linear':
        # Only keep linear terms in taylor expansion of delta_sphum_quant
        coef_b = c_p + L_v * alpha_quant_l * sphum_quant_land
        coef_c = L_v * sphum_quant_sat_l * delta_r_quant_l - delta_mse_ratio * (
            (c_p + L_v * alpha_quant_o * sphum_quant_ocean) * delta_temp_quant_o +
            L_v * sphum_quant_sat_o * delta_r_quant_o)
        delta_temp_quant_theory = -coef_c[:-1] / coef_b[:-1]
    elif taylor_level == 'linear_rh_diff':
        # combine mean and quantile RH changes with same prefactor
        # This is a further taylor expansion of sphum_quant_sat around sphum_quant_mean
        gamma_t, gamma_rdiff = get_gamma(temp_quant_ocean, sphum_quant_ocean, temp_quant_land, sphum_quant_land, pressure_surface)
        delta_temp_quant_theory = (gamma_t * delta_mse_ratio * delta_temp_quant_o +
                                   gamma_rdiff * (delta_mse_ratio * delta_r_quant_o -
                                                  delta_r_quant_l))[:-1]
    else:
        raise ValueError(f"taylor_level given is {taylor_level}. This is not valid, it must be either: 'squared', "
                         f"'nonlinear', 'linear' or 'linear_rh_diff'")
    return delta_temp_quant_theory


def get_gamma(temp_quant_ocean: np.ndarray, sphum_quant_ocean: np.ndarray, temp_quant_land: np.ndarray,
              sphum_quant_land: np.ndarray, pressure_surface: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    This function returns the sensitivity parameters in the theory.
    One for changes in ocean temperature, $\\delta T_O(x)$, and  one for difference between land and ocean
    relative humidity, $\delta (r_O(x) - r_L(x))$:

    $$\gamma^T = \\frac{c_p + L_v \\alpha_O q_O}{c_p + L_v \\alpha_L q_L};\quad
    \gamma^{\Delta r} = \\frac{L_v q_{O, sat}}{c_p + L_v \\alpha_L q_L}$$

    Args:
        temp_quant_ocean: `float [n_exp, n_quant]`</br>
            `temp_quant_ocean[i, j]` is the percentile `quant_use[j]` of near surface ocean temperature of
            experiment `i`. Units: *K*.</br>
            Note that `quant_use` is not provided as not needed by this function, but is likely to be
            `np.arange(1, 100)` - leave out `x=0` as doesn't really make sense to consider $0^{th}$ percentile
            of a quantity.
        sphum_quant_ocean: `float [n_exp, n_quant]`</br>
            `sphum_quant_ocean[i, j]` is the percentile `quant_use[j]` of near surface ocean specific humidity of
            experiment `i`. Units: *kg/kg*.
        temp_quant_land: `float [n_exp, n_quant]`</br>
            `temp_quant_land[i, j]` is the percentile `quant_use[j]` of near surface land temperature of
            experiment `i`. Units: *K*.</br>
        sphum_quant_land: `float [n_exp, n_quant]`</br>
            `sphum_quant_land[i, j]` is the percentile `quant_use[j]` of near surface land specific humidity of
            experiment `i`. Units: *kg/kg*.
        pressure_surface: Near surface pressure level. Units: *Pa*.

    Returns:
        `gamma_t`: `float [n_exp, n_quant]`</br>
            The sensitivity to change in ocean temperature for each experiment and quantile.
        `gamma_rdiff`: `float [n_exp, n_quant]`</br>
            The sensitivity to change in relative humidity difference from land to ocean for each experiment and
            quantile.
    """
    alpha_quant_l = clausius_clapeyron_factor(temp_quant_land, pressure_surface)
    alpha_quant_o = clausius_clapeyron_factor(temp_quant_ocean, pressure_surface)
    sphum_quant_sat_o = sphum_sat(temp_quant_ocean, pressure_surface)
    denom = c_p + L_v * alpha_quant_l * sphum_quant_land
    gamma_t = (c_p + L_v * alpha_quant_o * sphum_quant_ocean) / denom
    gamma_rdiff = L_v / denom * sphum_quant_sat_o
    return gamma_t, gamma_rdiff
