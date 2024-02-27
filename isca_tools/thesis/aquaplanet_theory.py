import numpy as np
from ..utils.moist_physics import clausius_clapeyron_factor, sphum_sat
from ..utils.constants import c_p, L_v
from typing import Tuple, Optional


def get_delta_temp_quant_theory(temp_mean: np.ndarray, sphum_mean: np.ndarray, temp_quant: np.ndarray,
                                sphum_quant: np.ndarray, pressure_surface: float, const_rh: bool = False,
                                delta_mse_ratio: Optional[np.ndarray] = None,
                                taylor_level: str = 'linear_rh_diff') -> np.ndarray:
    """
    Computes the theoretical temperature difference between simulations of neighbouring optical depth values for each
    percentile, $\delta T(x)$, according to the assumption that changes in MSE are equal to the change in mean MSE,
    $\delta h(x) = \delta \overline{h}$:

    $$\delta T(x) = \gamma^T \delta \overline{T} + \gamma^{\Delta r} \delta (\overline{r} - r(x))$$

    This above equation is for the default settings, but a more accurate equation can be used with the `delta_mse_ratio`
    and `taylor_level` arguments.

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
        delta_mse_ratio: `float [n_exp-1, n_quant]`</br>
            `delta_mse_ratio[i]` is the change in $x$ percentile of MSE divided by the change in the mean MSE
            between experiment `i` and `i+1`: $\delta h(x)/\delta \overline{h}$.
            If not given, it is assumed to be equal to 1 for all $x$.
        taylor_level: This specifies the level of approximation that goes into the taylor series for $\delta q(x)$
            and $\delta \overline{q}$:

            - `squared`: Includes squared, $\delta T^2$, nonlinear, $\delta T \delta r$, and linear terms.
            - `nonlinear`: Includes nonlinear, $\delta T \delta r$, and linear terms.
            - `linear`: Includes just linear terms so $\delta T(x) = \\gamma^T \delta \\overline{T} +
                \\gamma^{\\bar{r}} \delta \\overline{r} +\\gamma^{r} \\delta r(x)$
            - `linear_rh_diff`: Same as `linear`, but does another approximation to combine relative humidity
                contributions so: $\delta T(x) = \gamma^T \delta \overline{T} +
                \gamma^{\Delta r} \delta (\overline{r} - r(x))$

    Returns:
        `float [n_exp-1, n_quant]`.</br>
            `delta_temp_quant_theory[i, j]` refers to the theoretical temperature difference between experiment `i` and
            `i+1` for percentile `quant_use[j]`.
    """
    n_exp, n_quant = temp_quant.shape
    alpha_quant = clausius_clapeyron_factor(temp_quant, pressure_surface)
    alpha_mean = clausius_clapeyron_factor(temp_mean, pressure_surface)
    sphum_quant_sat = sphum_sat(temp_quant, pressure_surface)
    sphum_mean_sat = sphum_sat(temp_mean, pressure_surface)
    r_quant = sphum_quant / sphum_quant_sat
    r_mean = sphum_mean / sphum_mean_sat
    delta_temp_mean = np.diff(temp_mean)

    delta_r_mean = np.diff(r_mean)
    delta_r_quant = np.diff(r_quant, axis=0)
    if const_rh:
        # get rid of relative humidity contribution if constant rh
        delta_r_mean = 0 * delta_r_mean
        delta_r_quant = 0 * delta_r_quant

    # Pad all delta variables so same size as temp_quant - will not use this in calculation but just makes it easier
    pad_array = ((0, 1), (0, 0))
    if delta_mse_ratio is None:
        delta_mse_ratio = np.ones_like(temp_quant)
    else:
        # make delta_mse_ratio the same size as all other quant variables
        delta_mse_ratio = np.pad(delta_mse_ratio, pad_width=pad_array)
    delta_temp_mean = np.pad(delta_temp_mean, pad_width=pad_array[0])
    delta_r_mean = np.pad(delta_r_mean, pad_width=pad_array[0])
    delta_r_quant = np.pad(delta_r_quant, pad_width=pad_array)

    if taylor_level == 'squared':
        # Keep squared, linear and non-linear terms in taylor expansion of delta_sphum_quant
        coef_a = 0.5 * L_v * alpha_quant * sphum_quant * (alpha_quant - 2 / temp_quant[0])
        coef_b = c_p + L_v * alpha_quant * (sphum_quant + sphum_quant_sat * delta_r_quant)
        coef_c = L_v * sphum_quant_sat * delta_r_quant - delta_mse_ratio * np.expand_dims(
                0.5 * L_v * alpha_mean * sphum_mean * (alpha_mean - 2 / temp_mean) * delta_temp_mean ** 2 +
                (c_p + L_v * alpha_mean * (sphum_mean + sphum_mean_sat * delta_r_mean)) * delta_temp_mean +
                L_v * sphum_mean_sat * delta_r_mean, axis=-1)
        delta_temp_quant_theory = np.asarray([[np.roots([coef_a[i, j], coef_b[i, j], coef_c[i, j]])[1]
                                               for j in range(n_quant)] for i in range(n_exp - 1)])
    elif taylor_level in ['nonlinear', 'non-linear']:
        # Keep linear and non-linear terms in taylor expansion of delta_sphum_quant
        coef_b = c_p + L_v * alpha_quant * (sphum_quant + sphum_quant_sat * delta_r_quant)
        coef_c = L_v * sphum_quant_sat * delta_r_quant - delta_mse_ratio * np.expand_dims(
                (c_p + L_v * alpha_mean * (sphum_mean + sphum_mean_sat * delta_r_mean)) * delta_temp_mean +
                L_v * sphum_mean_sat * delta_r_mean, axis=-1)
        delta_temp_quant_theory = -coef_c[:-1] / coef_b[:-1]
    elif taylor_level == 'linear':
        # Only keep linear terms in taylor expansion of delta_sphum_quant
        coef_b = c_p + L_v * alpha_quant * sphum_quant
        coef_c = L_v * sphum_quant_sat * delta_r_quant - delta_mse_ratio * np.expand_dims(
                (c_p + L_v * alpha_mean * sphum_mean) * delta_temp_mean + L_v * sphum_mean_sat * delta_r_mean, axis=-1)
        delta_temp_quant_theory = -coef_c[:-1] / coef_b[:-1]
    elif taylor_level == 'linear_rh_diff':
        # combine mean and quantile RH changes with same prefactor
        # This is a further taylor expansion of sphum_quant_sat around sphum_quant_mean
        gamma_t, gamma_rdiff = get_gamma(temp_mean, sphum_mean, temp_quant, sphum_quant, pressure_surface)
        delta_temp_quant_theory = (gamma_t * delta_mse_ratio * np.expand_dims(delta_temp_mean, axis=-1) +
                                   gamma_rdiff * (delta_mse_ratio * np.expand_dims(delta_r_mean, axis=-1) -
                                                  delta_r_quant))[:-1]
    else:
        raise ValueError(f"taylor_level given is {taylor_level}. This is not valid, it must be either: 'squared', "
                         f"'nonlinear', 'linear' or 'linear_rh_diff'")
    return delta_temp_quant_theory


def get_gamma(temp_mean: np.ndarray, sphum_mean: np.ndarray, temp_quant: np.ndarray,
              sphum_quant: np.ndarray, pressure_surface: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    This function returns the sensitivity parameters in the theory.
    One for changes in mean temperature, $\\delta \\overline{T}$, and  one for difference to the mean relative humidity,
    $\delta (\overline{r} - r(x))$:

    $$\gamma^T = \\frac{c_p + L_v \\bar{\\alpha} \\bar{q}}{c_p + L_v \\alpha q};\quad
    \gamma^{\Delta r} = \\frac{L_v \\overline{q_{sat}}}{c_p + L_v \\alpha q}$$

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

    Returns:
        `gamma_t`: `float [n_exp, n_quant]`</br>
            The sensitivity to change in mean temperature for each experiment and quantile.
        `gamma_rdiff`: `float [n_exp, n_quant]`</br>
            The sensitivity to change in relative humidity difference from the mean for each experiment and quantile.
    """
    alpha_quant = clausius_clapeyron_factor(temp_quant, pressure_surface)
    alpha_mean = clausius_clapeyron_factor(temp_mean, pressure_surface)
    sphum_mean_sat = sphum_sat(temp_mean, pressure_surface)
    denom = c_p + L_v * alpha_quant * sphum_quant
    gamma_t = np.expand_dims(c_p + L_v * alpha_mean * sphum_mean, axis=-1) / denom
    gamma_rdiff = L_v / denom * np.expand_dims(sphum_mean_sat, axis=-1)
    return gamma_t, gamma_rdiff


def get_lambda_2_theory(temp_ft_quant: np.ndarray, temp_ft_mean: np.ndarray, z_quant: np.ndarray, z_mean: np.ndarray,
                        pressure_ft: float) -> Tuple[np.ndarray, dict, dict]:
    """
    Compute the approximation for $\lambda_2 = \delta h^*_{FT}(x)/\delta \overline{h^*_{FT}}$ used
    for the extratropical part of the theory between two simulations (`n_exp` should be 2):

    $$\\lambda_2 \\approx (1+\\frac{\\overline{T}\\delta \\overline{\\kappa}}{\\delta \\overline{z}})
    \\frac{c_p + L_v\\alpha(x) q^*(x)}{c_p + L_v\\overline{\\alpha} \\overline{q^*}}
    \\frac{\\delta z(x)}{\\delta \\overline{z}} -
    \\frac{c_p + L_v\\alpha(x) q^*(x)}{c_p + L_v\\overline{\\alpha} \\overline{q^*}}
    \\frac{\\overline{T} \\delta \\kappa(x)}{\\delta \\overline{z}}$$

    where $\kappa=z/T$ and $z$ and $T$ are the free-troposphere geopotential height and temperature.

    Args:
        temp_ft_quant: `float [n_exp, n_lat, n_quant]`</br>
            Free troposphere temperature for each experiment, latitude and quantile.
        temp_ft_mean: `float [n_exp, n_lat]`</br>
            Mean free troposphere temperature for each experiment and latitude.
        z_quant: `float [n_exp, n_lat, n_quant]`</br>
            Free troposphere geopotential height for each experiment, latitude and quantile.
        z_mean: `float [n_exp, n_lat]`</br>
            Mean free troposphere geopotential height for each experiment and latitude.
        pressure_ft: Free troposphere pressure level. Units: *Pa*.

    Returns:
        $\lambda_2$ Approximation: `float [n_lat, n_quant]`</br>
            Approximation of $\lambda_2$ at each latitude and quantile.
        `prefactors`: Dictionary containing the prefactors that go into the approximation. All variables are
            evaluated at the colder simulation.

            - `z1`: `float [n_exp, n_lat, 1]`</br>
                $1+\\frac{\\overline{T}\\delta \\overline{\\kappa}}{\\delta \\overline{z}}$
            - `z2`: `float [n_exp, n_lat, n_quant]`</br>
                $\\frac{c_p + L_v\\alpha(x) q^*(x)}{c_p + L_v\\overline{\\alpha} \\overline{q^*}}$
            - `kappa`: `float [n_exp, n_lat, n_quant]`</br>
                $-\\frac{c_p + L_v\\alpha(x) q^*(x)}{c_p + L_v\\overline{\\alpha} \\overline{q^*}} \\times \\overline{T}$
        `delta_var`: Dictionary containing the following changes between simulations.

            - `z_quant`: `float [n_exp, n_lat, n_quant]`</br>
                    $\delta z(x)$</br>
            - `z_mean`: `float [n_exp, n_lat, 1]`</br>
                    $\delta \overline{z}$</br>
            - `kappa_quant`: `float [n_exp, n_lat, n_quant]`</br>
                    $\delta \kappa(x)$</br>
            - `kappa_mean`: `float [n_exp, n_lat, 1]`</br>
                    $\delta \overline{\kappa}$
    """
    kappa_quant = z_quant / temp_ft_quant
    kappa_mean = z_mean / temp_ft_mean
    # Need to expand dims of mean variables so can multiply quant arrays
    temp_ft_mean = np.expand_dims(temp_ft_mean, axis=-1)

    delta_var = {'z_quant': z_quant[1] - z_quant[0], 'z_mean': np.expand_dims(z_mean[1] - z_mean[0], axis=-1),
                 'kappa_quant': kappa_quant[1] - kappa_quant[0],
                 'kappa_mean': np.expand_dims(kappa_mean[1] - kappa_mean[0], axis=-1)}

    alpha_quant = clausius_clapeyron_factor(temp_ft_quant[0], pressure_ft)
    alpha_mean = clausius_clapeyron_factor(temp_ft_mean[0], pressure_ft)
    q_quant = sphum_sat(temp_ft_quant[0], pressure_ft)
    q_mean = sphum_sat(temp_ft_mean[0], pressure_ft)

    prefactors = {'z1': 1+temp_ft_mean[0] * delta_var['kappa_mean'] / delta_var['z_mean'],
                  'z2': (c_p + L_v * alpha_quant * q_quant) / (c_p + L_v * alpha_mean * q_mean)}
    prefactors['kappa'] = -prefactors['z2'] * temp_ft_mean[0]

    lambda_2_approx = prefactors['z1'] * prefactors['z2'] * delta_var['z_quant'] / delta_var['z_mean'] + \
        prefactors['kappa'] * delta_var['kappa_quant'] / delta_var['z_mean']
    return lambda_2_approx, prefactors, delta_var
