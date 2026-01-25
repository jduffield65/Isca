from ..utils.constants import g, R, lapse_dry
from ..utils.base import round_any
from ..utils.numerical import hybrid_root_find
from ..convection.base import lcl_sigma_bolton_simple
import numpy as np
import xarray as xr
from typing import Union
from geocat.comp.interpolation import interp_hybrid_to_pressure
import itertools


def get_bulk_lapse_rate(temp1: xr.DataArray, temp2: xr.DataArray, p1: Union[xr.DataArray, float],
                        p2: Union[xr.DataArray, float]):
    """
    Compute the bulk environmental lapse rate, $\Gamma$, between pressure `p1` at environmental temperature `temp1`
    and `p2` at environmental temperature `temp2`:

    $$\Gamma = \\frac{g}{R}\ln\\left(\\frac{T_1}{T_2}\\right)/\ln\\left(\\frac{p_1}{p_2}\\right)$$

    This equation assumes hydrostatic equilibrium, ideal gas equation of state and that $\Gamma$ is constant
    between `p1` and `p2`.

    Args:
        temp1: Temperature at pressure `p1`. Units: *K*.
        temp2: Temperature at pressure `p2`. Units: *K*.
        p1: Pressure at environmental temperature `temp1`. Units: *Pa*.
        p2: Pressure at environmental temperature `temp2`. Units: *Pa*.

    Returns:
        Bulk environmental lapse rate, positive if `temp2<temp1` and `p2<p1`. Units are *K/m*.
    """
    return g / R * np.log(temp1 / temp2) / np.log(p1 / p2)


def reconstruct_temp(temp3: Union[xr.DataArray, np.ndarray, float], p1: Union[xr.DataArray, np.ndarray, float],
                     p2: Union[xr.DataArray, np.ndarray, float],
                     p3: Union[xr.DataArray, np.ndarray, float],
                     lapse_12: Union[xr.DataArray, np.ndarray, float],
                     lapse_23: Union[xr.DataArray, np.ndarray, float]):
    """
    The temperature, $T_1$, at $p_1$ can be reconstructed from the lapse rate, $\Gamma_{12}$, between $p_1$ and $p_2$;
    the lapse rate $\Gamma_{23}$, between $p_2$ and $p_3$; and the temperature at $p_3$, $T_3$:

    $$
    T_1 = T_{3}\\left((\\frac{p_2}{p_1})^{\Gamma_{23}-\Gamma_{12}}(\\frac{p_3}{p_1})^{-\Gamma_{23}}\\right)^{R/g}
    $$

    Args:
        temp3: Temperature at pressure `p3`. Units: *K*.
        p1: Pressure at level to reconstruct `temp1`. Units: *Pa*.
        p2: Pressure at environmental temperature `temp2`. Units: *Pa*.
        p3: Pressure at environmental temperature `temp3`. Units: *Pa*.
        lapse_12: Bulk environmental lapse rate between `p1` and `p2`. Units are *K/m*.
        lapse_23: Bulk environmental lapse rate between `p2` and `p3`. Units are *K/m*.

    Returns:
        temp1: Temperature at pressure `p1`. Units: *K*.
    """
    sigma_12 = p2 / p1  # if p1 is surface, this should be <1
    sigma_13 = p3 / p1
    return temp3 * (sigma_12 ** (lapse_23 - lapse_12) * sigma_13 ** (-lapse_23)) ** (R / g)


def interp_var_at_pressure(var: Union[xr.DataArray, xr.Dataset, np.ndarray], p_desired: Union[xr.DataArray, np.ndarray],
                           p_surf: Union[xr.DataArray, np.ndarray],
                           hyam: xr.DataArray, hybm: xr.DataArray, p0: float,
                           plev_step: float = 1000, extrapolate: bool = False,
                           lev_dim: str = 'lev', var_name: str = 'new_var') -> xr.Dataset:
    """
    Function to get the value of variable `var` at the pressure `p_desired`, where `p_desired` is expected to
    be a different value at each lat and lon.

    Args:
        var: Variable to do interpolation of. Should have `lev` dimension as well as lat, lon and possibly time.
            If give Dataset, will do interpolation on all variables.
        p_desired: Desired pressure to find `var` at.
            Should have same dimension as `var` but no `lev`. Units: *Pa*.
        p_surf: Surface pressure.
            Should have same dimension as `var` but no `lev`. Units: *Pa*.
        hyam: Hybrid a coefficients. Should have dimension of `lev` only.
        hybm: Hybrid b coefficients. Should have dimension of `lev` only.
        p0: Reference pressure. Units: *Pa*.
        plev_step: Will find var at value closest to `p_desired` on pressure grid with this spacing,
            so sets accuracy of interpolation.
        extrapolate: If True, below ground extrapolation for variable will be done, otherwise will return nan.
        lev_dim: String that is the name of level dimension in input data.
        var_name: String that is the name of variable in input data. Only used if `var` is numpy array

    Returns:
        Dataset with `plev` indicating approximate value of `p_desired` used, as well as `var` interpolated
            to that pressure level.
    """
    plevs = np.arange(round_any(float(p_desired.min()), plev_step, 'floor'),
                      round_any(float(p_desired.max()), plev_step, 'ceil') + plev_step / 2, plev_step)
    plevs_expand = xr.DataArray(plevs, dims=["plev"], coords={"plev": np.arange(len(plevs))})
    # Expand to match dimensions in p_surf, preserving order
    if isinstance(var, np.ndarray) and var.size == hybm.size:
        # If just numpy array, need to make it a data array for it to work
        var = xr.DataArray(var, dims=hybm.dims, coords=hybm.coords, name=var_name)
    if isinstance(p_surf, xr.DataArray):
        for dim in p_surf.dims:
            plevs_expand = plevs_expand.expand_dims({dim: p_surf.coords[dim]})

    idx_lcl_closest = np.abs(plevs_expand - p_desired).argmin(dim='plev')
    var_out = {'plev': plevs_expand.isel(plev=idx_lcl_closest)}  # approx pressure of p_desired used

    # Note that with extrapolate, will obtain values lower than surface
    if isinstance(var, xr.DataArray):
        var_out[var.name] = interp_hybrid_to_pressure(data=var, ps=p_surf, hyam=hyam, hybm=hybm, p0=p0,
                                                      new_levels=plevs, extrapolate=extrapolate, lev_dim=lev_dim,
                                                      variable='other' if extrapolate else None).isel(
            plev=idx_lcl_closest)
    elif isinstance(var, xr.Dataset):
        for key in var:
            var_out[key] = interp_hybrid_to_pressure(data=var[key], ps=p_surf, hyam=hyam, hybm=hybm, p0=p0,
                                                     new_levels=plevs, extrapolate=extrapolate, lev_dim=lev_dim,
                                                     variable='other' if extrapolate else None).isel(
                plev=idx_lcl_closest)
    else:
        raise ValueError('Unrecognized var. Needs to be a xr.DataArray or xr.Dataset.')
    for key in var_out:
        # Drop dimension of plev in all variables
        var_out[key] = var_out[key].drop_vars('plev')
    return xr.Dataset(var_out)


def _get_var_at_plev(var_env, p_env, p_desired, method='log'):
    if method == 'log':
        return np.interp(np.log10(p_desired), np.log10(p_env), var_env)
    else:
        return np.interp(p_desired, p_env, var_env)


def get_var_at_plev(var_env: Union[xr.Dataset, xr.DataArray], p_env: xr.DataArray, p_desired: xr.DataArray,
                    method: str = 'log',
                    lev_dim: str = 'lev'):
    """
    Find the value of `var_env` at pressure `p_desired`.

    Similar to `interp_hybrid_to_pressure` but handles the case where want different `p_desired` at each
    latitude and longitude.


    Args:
        var_env: float `n_lat x n_lon x n_lev x ...`</br>
            Variable to find value of at `p_desired`.
        p_env: float `n_lat x n_lon x n_lev x ...`</br>
            Pressure levels corresponding to `var_env`.
        p_desired: float `n_lat x n_lon x ...`</br>
            Pressure levels to find `var_env` at for each coordinate.
        method: Method of interpolation either take log10 of pressure first or leave as raw values.
        lev_dim: String that is the name of level dimension in `var_env` and `p_env`.

    Returns:
        var_desired: float `n_lat x n_lon x ...`</br>
            The value of `var_env` at `p_desired`.
    """
    if not (p_env.diff(dim=lev_dim) > 0).all():
        # If pressure is not ascending, flip dimension along lev_dim
        # Requirement for np.interp
        print(f'Reversed order of {lev_dim} for interpolation so p_env is ascending')
        lev_dim_ascending = bool((p_env[lev_dim].diff(dim=lev_dim) > 0).all())
        p_env = p_env.sortby(lev_dim, ascending=not lev_dim_ascending)
        var_env = var_env.sortby(lev_dim, ascending=not lev_dim_ascending)
        if not (p_env.diff(dim=lev_dim) > 0).all():
            # Sanity check p_env is now ascending
            raise ValueError('Pressure variable not ascending')

    out = xr.apply_ufunc(
        _get_var_at_plev,
        var_env, p_env, p_desired,
        input_core_dims=[[lev_dim], [lev_dim], []],
        output_core_dims=[[]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
        kwargs={"method": method}
    )
    return out


def get_ds_in_pressure_range(ds: xr.Dataset, pressure_min: xr.DataArray,
                             pressure_max: xr.DataArray, n_pressure: int = 20, pressure_var_name: str = 'P',
                             method: str = 'log', lev_dim: str = 'lev',
                             pressure_dim_name_out: str = 'plev_ind') -> xr.Dataset:
    """
    Extracts dataset variables interpolated (or sampled) at multiple evenly spaced
    pressure levels between `pressure_min` and `pressure_max` for each point.

    Args:
        ds: Input dataset containing at least a pressure variable (e.g. 'P')
            and one or more other variables dependent on pressure.
            Expected dims: (..., lev)
        pressure_min: Lower pressure bound for range [Pa].</br>
            Shape: same as all non-'lev' dims of ds.
        pressure_max: Upper pressure bound for range [Pa].</br>
            Shape: same as all non-'lev' dims of ds.
        n_pressure: Number of evenly spaced pressure levels to sample between
            pressure_min and pressure_max.
        pressure_var_name: Name of pressure variable in `ds`.
        method: Method of interpolation either take log10 of pressure first or leave as raw values.
        lev_dim: Name of model level dimension in `pressure_var_name`.
        pressure_dim_name_out: Name for the new pressure dimension in the output dataset.</br>
            The out dimension with this name will have the value `np.arange(n_pressure)`.

    Returns:
        ds_out: Dataset sampled at `n_pressure` intermediate pressure levels between
            `pressure_min` and `pressure_max`, concatenated along a new dimension
            named `pressure_dim_name_out`.</br>
            Output dims: (..., plev_ind)
    """
    if pressure_var_name not in ds.data_vars:
        raise ValueError(f'Pressure ({pressure_var_name}) not in dataset.')
    if len(ds.data_vars) == 1:
        raise ValueError(f'Must have another variable other than {pressure_var_name} in dataset.')
    ds_out = []
    pressure_range = pressure_max - pressure_min
    for i in range(n_pressure):
        p_use = pressure_min + i / np.clip(n_pressure - 1, 1, np.inf) * pressure_range
        ds_use = get_var_at_plev(ds.drop_vars(pressure_var_name), ds[pressure_var_name], p_use, method=method,
                                 lev_dim=lev_dim)
        ds_use[pressure_var_name] = p_use
        ds_out.append(ds_use)
    return xr.concat(ds_out,
                     dim=xr.DataArray(np.arange(n_pressure), name=pressure_dim_name_out, dims=pressure_dim_name_out))


from typing import Optional, Tuple


def get_scale_factor_theory(temp_surf_ref: np.ndarray, temp_surf_quant: np.ndarray, rh_ref: float,
                            rh_quant: np.ndarray, temp_ft_quant: np.ndarray,
                            p_ft: float,
                            p_surf_ref: float, p_surf_quant: Optional[np.ndarray],
                            lapse_D_quant: np.ndarray,
                            lapse_ALz_ref: float,
                            lapse_AL_anom_quant: np.ndarray,
                            temp_surf_lcl_calc: float = 300) -> Tuple[np.ndarray, dict, dict, dict]:
    """
    TODO: These are old comments from copied modParc code
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
        lapse_AL_anom_quant: `float [n_exp, n_quant]` $\eta_M[x]$</br>
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
    if p_surf_quant is None:
        p_surf_quant = np.full_like(temp_surf_quant, p_surf_ref[:, np.newaxis])

    # Gamma computed onlu from quantities in current climate, hence temp_surf_ref[0]
    gamma = {}
    sigma_lcl = lcl_sigma_bolton_simple(rh_ref, temp_surf_lcl_calc)
    sigma_exponent = np.log(sigma_lcl) / np.log(rh_ref)
    temp_ft_ref = reconstruct_temp(temp_surf_ref[0], p_ft, sigma_lcl * p_surf_ref, p_surf_ref, lapse_ALz_ref, lapse_dry)

    gamma['temp_ft_change'] = 1 + R * lapse_dry / g * np.log(1 / sigma_lcl) + R * lapse_ALz_ref / g * np.log(
        sigma_lcl * p_surf_ref / p_ft)
    gamma['rh_change'] = R * sigma_exponent / g * (lapse_dry - lapse_ALz_ref) * temp_ft_ref / temp_surf_ref[0]
    gamma['p_surf_change'] = R * lapse_ALz_ref / g * temp_ft_ref / temp_surf_ref[0]
    gamma['lapse_D_change'] = -np.log(sigma_lcl)
    gamma['lapse_AL_change'] = np.log(sigma_lcl * p_surf_ref / p_ft)

    gamma['rh_anom'] = gamma['temp_ft_change'] * R * sigma_exponent / g * (lapse_dry - lapse_ALz_ref)
    gamma['p_surf_anom'] = gamma['temp_ft_change'] * R * lapse_ALz_ref / g

    temp_surf_ref_change = temp_surf_ref[1] - temp_surf_ref[0]

    info_var = {'temp_ft_change': np.diff(temp_ft_quant, axis=0).squeeze() / temp_surf_ref_change,
                'rh_change': np.diff(rh_quant, axis=0).squeeze() / rh_ref * temp_surf_ref[0] / temp_surf_ref_change,
                'p_surf_change': np.diff(p_surf_quant, axis=0).squeeze() / p_surf_ref * temp_surf_ref[
                    0] / temp_surf_ref_change,
                'rh_anom': (rh_quant[0] - rh_ref) / rh_ref,
                'p_surf_anom': (p_surf_quant[0] - p_surf_ref) / p_surf_ref,
                'lapse_D_change': np.diff(lapse_D_quant, axis=0).squeeze() / temp_surf_ref_change,
                'lapse_AL_change': np.diff(lapse_AL_anom_quant, axis=0).squeeze() / temp_surf_ref_change}

    coef_sign = {'temp_ft_change': 1, 'rh_change': -1, 'p_surf_change': 1, 'lapse_D_change': 1,
                 'lapse_AL_change': 1, 'rh_anom': -1, 'p_surf_anom': 1}

    # Get contribution from each term - will be 1 if no contribution to match numerical version
    info_cont = {}
    for key in info_var:
        info_cont[key] = coef_sign[key] * gamma[key] * info_var[key]
        if key != 'temp_ft_change':
            info_cont[key] += 1
    final_answer = np.asarray(sum([info_cont[key] - 1 for key in info_cont])) + 1
    return final_answer, gamma, info_var, info_cont


def get_scale_factor_theory_numerical(temp_surf_ref: np.ndarray, temp_surf_quant: np.ndarray, rh_ref: float,
                                      rh_quant: np.ndarray, temp_ft_quant: np.ndarray,
                                      p_ft: float,
                                      p_surf_ref: float, p_surf_quant: Optional[np.ndarray],
                                      lapse_D_quant: np.ndarray,
                                      lapse_ALz_ref: float,
                                      lapse_AL_anom_quant: np.ndarray,
                                      temp_surf_lcl_calc: float = 300,
                                      valid_range: float = 100) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    TODO: These are old comments from copied modParc code
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
    if p_surf_quant is None:
        p_surf_quant = np.full_like(temp_surf_quant, p_surf_ref[:, np.newaxis])

    # Set values used for our reference quantities
    lapse_D_ref = 0
    lapse_D_ref_change = 0
    lapse_AL_anom_ref = 0
    lapse_AL_anom_ref_change = 0
    p_surf_ref_change = 0
    rh_ref_change = 0

    def get_temp(rh=rh_ref, p_surf=p_surf_ref,
                 lapse_D=lapse_D_ref, lapse_ALz_ref=lapse_ALz_ref,
                 lapse_AL_anom=lapse_AL_anom_ref,
                 temp_surf=None, temp_ft=None):
        # Has useful default values as ref in cold climate. So to find effect of a mechanism, just
        # change that variable.
        # Still gives option to compute surf or FT temp
        p_lcl = p_surf * lcl_sigma_bolton_simple(rh, temp_surf_lcl_calc)
        if temp_ft is not None:
            lapse_Dz = lapse_D * g / R / temp_ft
            lapse_ALz_anom = lapse_AL_anom * g / R / temp_ft
            temp_sol = reconstruct_temp(temp_ft, p_surf, p_lcl, p_ft, lapse_dry + lapse_Dz,
                                        lapse_ALz_ref + lapse_ALz_anom)
        else:
            def residual(x):
                lapse_Dz = lapse_D * g / R / x
                lapse_ALz_anom = lapse_AL_anom * g / R / x
                temp_surf_compute = reconstruct_temp(x, p_surf, p_lcl, p_ft, lapse_dry + lapse_Dz,
                                                     lapse_ALz_ref + lapse_ALz_anom)
                return temp_surf - temp_surf_compute

            # Guess temp_ft using reference conditions
            guess_temp = reconstruct_temp(temp_surf, p_ft, p_lcl, p_surf, lapse_ALz_ref, lapse_dry)
            try:
                temp_sol = hybrid_root_find(residual, guess_temp, valid_range)
            except ValueError as e:
                temp_sol = np.nan
        return temp_sol

    get_temp = np.vectorize(get_temp)  # may need optimizing in future
    # Compute temp_ft_ref using base climate reference rh and lapse_mod
    # No approx error in temp_surf_ref as use it to compute temp_ft_ref
    temp_ft_ref = get_temp(temp_surf=temp_surf_ref)

    def get_temp_change(rh=rh_ref, rh_change=rh_ref_change,
                        p_surf=p_surf_ref, p_surf_change=p_surf_ref_change,
                        lapse_D=lapse_D_ref, lapse_D_change=lapse_D_ref_change,
                        lapse_ALz_ref=lapse_ALz_ref,
                        lapse_AL=lapse_AL_anom_ref, lapse_AL_change=lapse_AL_anom_ref_change,
                        temp_ft_change=temp_ft_ref[1] - temp_ft_ref[0], temp_surf=temp_surf_ref[0]):
        # Default variables are such that if alter one variable, will give the temp change cont due to that change
        # and only that change
        # Given conditions in base climate, compute temperature at p_ft from that
        temp_ft0 = get_temp(rh, p_surf, lapse_D, lapse_ALz_ref, lapse_AL, temp_surf)
        # Given this ft temperature, and imposed change at p_ft, compute change at surface
        temp_surf_change_theory = get_temp(rh + rh_change, p_surf + p_surf_change,
                                           lapse_D + lapse_D_change, lapse_ALz_ref,
                                           lapse_AL + lapse_AL_change,
                                           temp_ft=temp_ft0 + temp_ft_change) - temp_surf
        return temp_surf_change_theory

    # Compute the expected surface temperature given the variables and our mod_parcel framework.
    # Will likely differ from temp_surf_quant if averaging done.
    temp_surf_quant_approx = get_temp(rh_quant, p_surf_quant, lapse_D_quant, lapse_ALz_ref,
                                      lapse_AL_anom_quant, temp_ft=temp_ft_quant)

    # Record quantity responsible for each mechanism
    # Use temp_surf_quant_approx not temp_surf_quant because we are computing the temperature change
    # in temp_surf_quant_approx, with deviation due to averaging accounted for later by error_av_change term
    var = {'temp_ft_change': temp_ft_quant[1] - temp_ft_quant[0], 'temp_surf_anom': temp_surf_quant_approx[0],
           'rh_change': rh_quant[1] - rh_quant[0], 'rh_anom': rh_quant[0],
           'p_surf_change': p_surf_quant[1] - p_surf_quant[0], 'p_surf_anom': p_surf_quant[0],
           'lapse_D_change': lapse_D_quant[1] - lapse_D_quant[0],
           'lapse_D_anom': lapse_D_quant[0],
           'lapse_AL_change': lapse_AL_anom_quant[1] - lapse_AL_anom_quant[0],
           'lapse_AL_anom': lapse_AL_anom_quant[0]}

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
