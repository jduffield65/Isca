import numpy as np
import numpy_indexed
import xarray as xr
from ..utils.moist_physics import clausius_clapeyron_factor, sphum_sat, moist_static_energy
from ..utils.constants import c_p, L_v
from typing import List, Tuple, Optional, Union
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
            optical depth, $\kappa$. Units: *K*.
        sphum_mean_land: `float [n_exp]`</br>
            Average near surface specific humidity of each simulation. Units: *kg/kg*.
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
            Note that `px` for the warmest simulation is not used but makes sense to have same shape as other variables.
        pressure_surface: Near surface pressure level. Units: *Pa*.
        const_rh: If `True`, will return the constant relative humidity version of the theory, i.e.
            $\gamma^{T_O} \delta T_O$. Otherwise, will return the full theory.

    Returns:
        `float [n_exp-1, n_quant]`.</br>
            `delta_temp_quant_theory[i, j]` refers to the theoretical temperature difference between experiment `i` and
            `i+1` for percentile `quant_use[j]`.
    """
    n_exp = temp_mean_land.shape[0]
    sphum_quant_sat_l = sphum_sat(temp_quant_land_x, pressure_surface)
    sphum_mean_sat_l = np.expand_dims(sphum_sat(temp_mean_land, pressure_surface), axis=-1)
    r_quant_l = sphum_quant_land_x / sphum_quant_sat_l
    r_mean_l = np.expand_dims(sphum_mean_land, axis=-1) / sphum_mean_sat_l
    delta_r_mean_l = np.diff(r_mean_l, axis=0)
    delta_r_quant_l = np.diff(r_quant_l, axis=0)

    # Ocean constants required - these are for the percentile px which corresponds
    # to the average above the x percentile in temperature
    p_x_ind = np.asarray([numpy_indexed.indices(quant_use, px[i]) for i in range(n_exp)])

    # For change in ocean variables, use quantile p_x from colder simulation for each set of subsequent simulations
    # Idea is to only use information from colder simulation to predict warmer one
    delta_temp_o = np.zeros_like(delta_r_quant_l)
    delta_r_quant_o = np.zeros_like(delta_r_quant_l)
    for i in range(n_exp-1):
        delta_temp_o[i] = temp_quant_ocean_p[i+1, p_x_ind[i]] - temp_quant_ocean_p[i, p_x_ind[i]]
        delta_r_quant_o[i] = \
            sphum_quant_ocean_p[i+1, p_x_ind[i]]/sphum_sat(temp_quant_ocean_p[i+1, p_x_ind[i]], pressure_surface) - \
            sphum_quant_ocean_p[i, p_x_ind[i]]/sphum_sat(temp_quant_ocean_p[i, p_x_ind[i]], pressure_surface)

    gamma_t, gamma_r_o, e_param, eta_param = get_gamma(temp_mean_land, temp_quant_land_x, temp_quant_ocean_p,
                                                   sphum_quant_land_x, sphum_quant_ocean_p, quant_use, px,
                                                   pressure_surface)

    if const_rh:
        delta_temp_quant_theory = gamma_t[:-1] * delta_temp_o
    else:
        delta_temp_quant_theory = (gamma_t[:-1] * delta_temp_o + gamma_r_o[:-1] * delta_r_quant_o -
                                   eta_param[:-1] * delta_r_mean_l) / (1 + e_param[:-1] * delta_r_quant_l)
    return delta_temp_quant_theory


def get_gamma(temp_mean_land: np.ndarray, temp_quant_land_x: np.ndarray, temp_quant_ocean_p: np.ndarray,
              sphum_quant_land_x: np.ndarray, sphum_quant_ocean_p: np.ndarray, quant_use: np.ndarray, px: np.ndarray,
              pressure_surface: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    This function returns the sensitivity parameters in the theory.
    One for changes in ocean temperature, $\\delta T_O$, and one for ocean relative humidity,
    $\delta r_O$:

    $$\gamma^{T_O} = \\frac{c_p + L_v \\alpha_O q_O}{c_p + L_v \\alpha_L q_L^x};\quad
    \gamma^{r_O} = \\frac{L_v q_{O, sat}}{c_p + L_v \\alpha_L q^x_L}$$

    Two more parameters are also returned:

    $$\\epsilon = \\frac{L_v \\alpha_L q^x_{L,sat}}{c_p + L_v \\alpha_L q^x_L};\quad
    \\eta = \\frac{\\epsilon}{\\alpha_L}\\frac{\overline{q_{L,sat}}}{q^x_{L,sat}}$$

    These can then be combined to give the sensitivity parameter to a 1% change in mean land relative humidity,
    $\delta \overline{r_L}$:

    $$\gamma^{r_L} = -\\frac{\eta}{100 + \\epsilon}$$

    Args:
        temp_mean_land: `float [n_exp]`</br>
            Average near surface land temperature of each simulation, corresponding to a different
            optical depth, $\kappa$. Units: *K*.
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
            Note that `px` for the warmest simulation is not used but makes sense to have same shape as other variables.
        pressure_surface: Near surface pressure level. Units: *Pa*.

    Returns:
        `gamma_t`: `float [n_exp, n_quant]`</br>
            The sensitivity to change in ocean temperature for each experiment and quantile.
        `gamma_r_o`: `float [n_exp, n_quant]`</br>
            The sensitivity to change in ocean relative humidity difference from the mean for each experiment and
            quantile.
        `e_param`: `float [n_exp, n_quant]`</br>
            The $\\epsilon = \\frac{L_v \\alpha_L q^x_{L,sat}}{c_p + L_v \\alpha_L q^x_L}$ parameter.
        `eta_param`: `float [n_exp, n_quant]`</br>
            The $\\eta = \\frac{\\epsilon}{\\alpha_L}\\frac{\overline{q_{L,sat}}}{q^x_{L,sat}}$ parameter.

    """
    n_exp = temp_mean_land.shape[0]
    alpha_l = clausius_clapeyron_factor(temp_quant_land_x, pressure_surface)
    sphum_quant_sat_l = sphum_sat(temp_quant_land_x, pressure_surface)
    sphum_mean_sat_l = np.expand_dims(sphum_sat(temp_mean_land, pressure_surface), axis=-1)

    # Ocean constants required - these are for the percentile px which corresponds
    # to the average above the x percentile in temperature
    p_x_ind = np.asarray([numpy_indexed.indices(quant_use, px[i]) for i in range(n_exp)])
    temp_quant_o = np.asarray([temp_quant_ocean_p[i, p_x_ind[i]] for i in range(n_exp)])
    sphum_quant_o = np.asarray([sphum_quant_ocean_p[i, p_x_ind[i]] for i in range(n_exp)])
    sphum_quant_sat_o = sphum_sat(temp_quant_o, pressure_surface)

    alpha_o = clausius_clapeyron_factor(temp_quant_o, pressure_surface)
    e_param = L_v * alpha_l * sphum_quant_sat_l / (c_p + L_v * alpha_l * sphum_quant_land_x)
    eta_param = sphum_mean_sat_l / sphum_quant_sat_l * e_param / alpha_l
    gamma_t = (c_p + L_v * alpha_o * sphum_quant_o) / (c_p + L_v * alpha_l * sphum_quant_land_x)
    gamma_r_o = L_v * sphum_quant_sat_o / (c_p + L_v * alpha_l * sphum_quant_land_x)
    return gamma_t, gamma_r_o, e_param, eta_param


def get_px(ds: List[xr.Dataset], mse_quant_x: np.ndarray, quant_use: np.ndarray, as_int: bool = False) -> np.ndarray:
    """
    Returns the percentile of Moist Static Energy corresponding to the MSE averaged over all days exceeding
    a given percentile of temperature.

    Args:
        ds: `[n_exp]`.</br>
            ds[i] is the dataset for experiment `i` containing only variables at the lowest pressure level.
            Also, the `lon`, `lat`, `time` coordinates must be collapsed into a single coordinate.
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
        if 'pfull' in ds[i].dims:
            # get rid of pressure coordinate if exists
            ds_use = ds[i].sel(pfull=np.inf, method='nearest', drop=True)
            mse_all = moist_static_energy(ds_use.temp, ds_use.sphum, ds_use.height)
        elif len(ds[i].temp.shape) != 1:
            raise ValueError(f'Dataset has coordinates:\n{ds[i].coords}\nbut should only have 1')
        else:
            mse_all = moist_static_energy(ds[i].temp, ds[i].sphum, ds[i].height)
        for j, quant in enumerate(quant_use):
            px[i, j] = percentileofscore(mse_all, mse_quant_x[i, j])
    if as_int:
        px = np.round(px).astype(int)
    return px


def get_quant_ind(var: Union[xr.DataArray, np.ndarray], percentile: int, range_below: float = 0,
                  range_above: float = np.inf, av_dim: Optional[Union[List, str, int]]=None,
                  return_mask: bool = False) -> np.ndarray:
    """
    This functions returns the indices of all occurrences whereby the value of `var` is between the
    `percentile-range_below` and `percentile+range_above` percentile.

    By default, `range_below=0` and `range_above=inf` so it just returns all indices above the given `percentile`.

    Args:
        var: `float [n_lon_lat_time]`</br>
            Variable to find quantiles for, usually surface temperature. The first coordinate of this variable
            must be called `lon_lat_time`, which is all longitudes, latitudes and times collapsed into a single
            coordinate.
        percentile: The percentile around which, you want to find the indices.
        range_below: All indices will `var` in the percentile range between `percentile-range_below` and
            `percentile+range_above` will be returned.
        range_above: All indices will `var` in the percentile range between `percentile-range_below` and
            `percentile+range_above` will be returned.
        av_dim: Dimension to find quantile over, should be string if `var` is xarray or integer if `var` is numpy
            array. If not given, will find quantile over 'lon_lat_time' or 'lon_time' dimension.
        return_mask: If `True`, will return boolean mask indicating which coordinates are in the quant range.</br>
            Otherwise will return the indices

    Returns:
        `int [n_ind]`</br>
            All indices where `var` is in the correct percentile range.</br>
            Or boolean of length `n_lon_lat_time` if `return_mask` is `True`.</br>
    """
    quant_min = np.clip(percentile-range_below, 0, 100)
    quant_max = np.clip(percentile+range_above, 0, 100)
    if isinstance(var, np.ndarray):
        quantile_thresh_min = np.quantile(var, quant_min/100, axis=av_dim)
        quantile_thresh_max = np.quantile(var, quant_max/100, axis=av_dim)
    else:
        if av_dim is None:
            if 'lon_lat_time' in var.dims:
                av_dim = 'lon_lat_time'
            elif 'lon_time' in var.dims:
                av_dim = 'lon_time'
            else:
                raise ValueError('No suitable dimension to average over - neither lon_lat_time nor lon_time in var')
        # else:
        #     if av_dim not in var.dims:
        #         raise ValueError(f'No suitable dimension to average over - {av_dim} is not in var')
        quantile_thresh_min = var.quantile(quant_min / 100, dim=av_dim, keep_attrs=True)
        quantile_thresh_max = var.quantile(quant_max / 100, dim=av_dim, keep_attrs=True)
    mask = np.logical_and(var > quantile_thresh_min, var <= quantile_thresh_max)
    if return_mask:
        return mask
    else:
        return np.where(mask)[0]
