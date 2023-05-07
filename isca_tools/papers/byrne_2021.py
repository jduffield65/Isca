import numpy as np
import numpy_indexed
import xarray as xr
from ..utils.moist_physics import clausius_clapeyron_factor, sphum_sat, moist_static_energy
from ..utils.constants import c_p, L_v
from typing import List
from scipy.stats import percentileofscore


def get_delta_temp_quant_theory(temp_mean_land: np.ndarray, sphum_mean_land: np.ndarray, temp_quant_land_x: np.ndarray,
                                temp_quant_ocean_p: np.ndarray, sphum_quant_land_x: np.ndarray,
                                sphum_quant_ocean_p: np.ndarray, quant_use: np.ndarray, px: np.ndarray,
                                pressure_surface: float, const_rh: bool = False) -> np.ndarray:
    """
    Computes the theoretical temperature difference between simulations of neighbouring optical depth values for each
    percentile, $\delta T_L^x$, according to equation 5 in *Byrne 2021*:

    $$(1 + \epsilon \delta r_L^x)\delta T_L^x = \gamma^{T_O}\delta T_O + \gamma^{r_O}\delta r_O -
    \eta \delta \overline{r_L}$$

    If data from `n_exp` optical depth values provided, `n_exp-1` theoretical temperature differences will be returned
    for each percentile.

    Args:
        temp_mean_land: `float [n_exp]`</br>
            Average near surface land temperature of each simulation, corresponding to a different
            optical depth, $\kappa$.
        sphum_mean_land: `float [n_exp]`</br>
            Average near surface specific humidity temperature of each simulation.
        temp_quant_land_x: `float [n_exp, n_quant]`</br>
            `temp_quant_land_x[i, j]` is the near surface land temperature of experiment `i`, averaged over all days
            exceeding the percentile `quant_use[j]` of temperature
            ($x$ means averaged over the temperature percentile $x$). Units: *K*.
        temp_quant_ocean_p: `float [n_exp, n_quant]`</br>
            `temp_quant_ocean_p[i, j]` is the percentile `quant_use[j]` of near surface ocean temperature of
            experiment `i` ($p$ means at percentile $p$ of given quantity). Units: *K*.
        sphum_quant_land_x: `float [n_exp, n_quant]`</br>
            `sphum_quant_land_x[i, j]` is the near surface land specific humidity of experiment `i`, averaged over
            all days exceeding the percentile `quant_use[j]` of temperature. Units: *kg/kg*.
        sphum_quant_ocean_p: `float [n_exp, n_quant]`</br>
            `sphum_quant_ocean_p[i, j]` is the percentile `quant_use[j]` of near surface ocean specific humidity of
            experiment `i`. Units: *kg/kg*.
        quant_use: `int [n_quant]`. This contains the percentiles, that the above variables correspond to. It must
            contain all values of `p_x` in it.
        px: `int [n_exp, n_quant]`</br>
            `p_x[i, j]` is the percentile of MSE corresponding to the MSE averaged over all days exceeding
            the percentile quant_use[i] of temperature in experiment `i`.
        pressure_surface: Near surface pressure level. Units: *Pa*.
        const_rh: If `True`, will return the constant relative humidity version of the theory, i.e.
            $\gamma^{T_O} \delta T_O$. Otherwise, will return the full theory.

    Returns:
        `float [n_exp-1, n_quant]`.</br>
            `delta_temp_quant_theory[i, j]` refers to the theoretical temperature difference between experiment `i` and
            `i+1` for percentile `quant_use[j]`.
    """
    n_exp = temp_mean_land.shape[0]
    alpha_l = clausius_clapeyron_factor(temp_quant_land_x, pressure_surface)
    sphum_quant_sat_l = sphum_sat(temp_quant_land_x, pressure_surface)
    sphum_mean_sat_l = np.expand_dims(sphum_sat(temp_mean_land, pressure_surface), axis=-1)
    r_quant_l = sphum_quant_land_x / sphum_quant_sat_l
    r_mean_l = np.expand_dims(sphum_mean_land, axis=-1) / sphum_mean_sat_l
    delta_r_mean_l = np.diff(r_mean_l, axis=0)
    delta_r_quant_l = np.diff(r_quant_l, axis=0)

    # Ocean constants required - these are for the percentile px which corresponds to the average above the x percentile in temperature
    p_x_ind = np.asarray([numpy_indexed.indices(quant_use, px[i]) for i in range(n_exp)])
    temp_quant_o = np.asarray([temp_quant_ocean_p[i, p_x_ind[i]] for i in range(n_exp)])
    delta_temp_o = np.diff(temp_quant_o, axis=0)
    sphum_quant_o = np.asarray([sphum_quant_ocean_p[i, p_x_ind[i]] for i in range(n_exp)])
    sphum_quant_sat_o = sphum_sat(temp_quant_o, pressure_surface)
    r_quant_o = sphum_quant_o / sphum_quant_sat_o
    delta_r_quant_o = np.diff(r_quant_o, axis=0)
    alpha_o = clausius_clapeyron_factor(temp_quant_o, pressure_surface)

    e_const = L_v * alpha_l * sphum_quant_sat_l / (c_p + L_v * alpha_l * sphum_quant_land_x)
    nabla = sphum_mean_sat_l / sphum_quant_sat_l * e_const / alpha_l
    gamma_t = (c_p + L_v * alpha_o * sphum_quant_o) / (c_p + L_v * alpha_l * sphum_quant_land_x)
    gamma_r_o = L_v * sphum_quant_sat_o / (c_p + L_v * alpha_l * sphum_quant_land_x)

    if const_rh:
        delta_temp_quant_theory = gamma_t[:-1] * delta_temp_o
    else:
        delta_temp_quant_theory = (gamma_t[:-1] * delta_temp_o + gamma_r_o[:-1] * delta_r_quant_o - nabla[:-1] * delta_r_mean_l
                                   ) / (1 + e_const[:-1] * delta_r_quant_l)
    return delta_temp_quant_theory


def get_px(ds: List[xr.Dataset], mse_quant_x: np.ndarray, quant_use: np.ndarray, as_int: bool = False) -> np.ndarray:
    """

    Args:
        ds: `[n_exp]`.
            ds[i] is the dataset for experiment `i` containing only variables at the lowest pressure level.
            It must include `temp`, `sphum` and `height`.
        mse_quant_x: `[n_exp, n_quant]`.</br>
            `mse_quant_x[i, j]` is the near MSE of experiment `i`, averaged over all days
            exceeding the percentile `quant_use[j]` of temperature. Units: *kJ/kg*.
        quant_use: `int [n_quant]`. This contains the percentiles, that `mse_quant_x` corresponds to.
        as_int: If `True`, will round the percentile to the nearest integer.

    Returns:
        `float [n_exp, n_quant]` or `int [n_exp, n_quant]`.</br>
            `p_x[i, j]` is the percentile of MSE corresponding to the MSE averaged over all days exceeding
            the percentile quant_use[i] of temperature in experiment `i`.
    """
    n_exp = len(ds)
    px = np.zeros((n_exp, len(quant_use)))
    for i in range(n_exp):
        mse_all = moist_static_energy(ds[i].temp, ds[i].sphum, ds[i].height)
        for j, quant in enumerate(quant_use):
            px[i, j] = percentileofscore(mse_all, mse_quant_x[i, j])
    if as_int:
        px = np.round(px).astype(int)
    return px
