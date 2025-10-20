import xarray as xr
import numpy as np
from typing import Union, Optional, Tuple, Literal
from ..utils.base import weighted_RMS
from ..utils.constants import lapse_dry, g
from ..utils.numerical import interp_nan
from ..utils.moist_physics import moist_static_energy, sphum_sat

def get_mse_env(temp_env: xr.DataArray, p_env: xr.DataArray, z_env: xr.DataArray,
                 temp_at_lcl: xr.DataArray, p_lcl: xr.DataArray,
                 prof_type: Literal['full', 'above_lcl', 'below_lcl'] = 'full') -> xr.DataArray:
    """
    Returns environmental MSE profile as a function of pressure $p$, which satisfies the following if
    `prof_type = full`:

    * Above LCL: $MSE_{env}(p) = MSE^*(p)$
    * Below LCL: $MSE_{env}(p) = DSE(p) + L_v q^*_{LCL}$

    Where $q^*_{LCL}$ is the saturation-specific humidity evaluated at $T_{env}(p_{lcl})$.
    The two profiles are the same at the LCL.

    Args:
        temp_env: `[n_lev]` Environment temperature in Kelvin.
        p_env: `[n_lev]` Environment pressure in Pa.
        z_env: `[n_lev]` Environment geopotential height in m.
        temp_at_lcl: Environment temperature at `p_lcl` in Kelvin.
        p_lcl: Pressure of LCL in Pa.
        prof_type: If `above_lcl`, will return $MSE^*(p)$ for all $p$.</br>
            If `below_lcl`, will return $DSE(p) + L_v q^*_{LCL}$ for all $p$.</br>
            If `full` will return different profile above and below LCL.

    Returns:
        mse_env: `[n_lev]` Environment MSE in kJ/kg.
    """
    mse_above_lcl = moist_static_energy(temp_env, sphum_sat(temp_env, p_env), z_env)
    mse_below_lcl = moist_static_energy(temp_env, sphum_sat(temp_at_lcl, p_lcl), z_env)
    if prof_type == 'full':
        return xr.where(p_env <= p_lcl, mse_above_lcl, mse_below_lcl)
    elif prof_type == 'above_lcl':
        return mse_above_lcl
    elif prof_type == 'below_lcl':
        return mse_below_lcl
    else:
        raise ValueError('prof_type must be either full or above_lcl or below_lcl')


def get_lnb_lev_ind(temp_env: xr.DataArray, z_env: xr.DataArray, p_env: xr.DataArray, p_max: float = 400 * 100,
                    lapse_thresh: float = 5, lapse_change_thresh: float = 2, n_iter: int = 5,
                    lev_dim: str = 'lev') -> xr.DataArray:
    """
    Finds the index of level of neutral buoyancy in dimension `lev_dim` that satisfies the following conditions
    (basically so no stratospheric influence between LNB and surface):

    * LNB must be a pressure lower than `p_max`.
    * LNB is level above which is the first layer with negative lapse rate.
    * If lapse rate in level immediately below this deviates from the lapse rate in the level below that
    or two below that by more than `lapse_change_thresh`, then LNB is moved to a lower level by 1.
    This process is repeated `n_iter` times.
    * Lapse rate in level immediately below LNB must be less than `lapse_thresh`.

    Args:
        temp_env: Environmental temperature profile in Kelvin.
        z_env: Environment geopotential height profile in m.
        p_env: Environment pressure profile in Pa
            (assumed different for each location i.e. same dimensions as `temp_env` and `z_env`).
        p_max: Pressure of LNB cannot exceed this value (i.e. further from surface than this).
        lapse_thresh: Lapse rate in level immediately below LNB must be less than `lapse_thresh`.
        lapse_change_thresh: If lapse rate in level immediately below LNB deviates from the lapse rate
            in the level below that or two below that by more than this,
            then LNB is moved to a lower level by 1.
        n_iter: Number of iterations to run `lapse_change_thresh` process.
        lev_dim: Name of dimension for vertical model level.

    Returns:
        lnb_ind: Index of level of neutral buoyancy in dimension `lev_dim` for each location.
    """
    lapse = -temp_env.diff(dim=lev_dim, label='lower') / z_env.diff(dim=lev_dim, label='lower') * 1000
    lapse = lapse.reindex_like(temp_env)  # make same shape
    lapse = lapse.fillna(lapse_dry * 1000)  # ensure final value satisfies lapse criteria
    lapse = lapse.where(p_env < p_max)
    mask = lapse < 0
    lnb_ind = (mask.where(mask, other=np.nan) * np.arange(lapse.lev.size)).max(dim=lev_dim).astype(int)
    # lnb_ind = np.where(lapse < 0)[0][-1]
    # If lapse rate has very big variation, push LNB closer to surface
    for j in range(n_iter):
        is_large_lapse_diff = lapse.isel(**{lev_dim: lnb_ind + 2}) - lapse.isel(**{lev_dim: lnb_ind + 1}) > lapse_change_thresh
        is_large_lapse_diff = is_large_lapse_diff & (lapse.isel(**{lev_dim: lnb_ind + 1}) < lapse_thresh)
        is_large_lapse_diff2 = lapse.isel(**{lev_dim: lnb_ind + 3}) - lapse.isel(**{lev_dim: lnb_ind + 1}) > lapse_change_thresh
        is_large_lapse_diff2 = is_large_lapse_diff2 & (lapse.isel(**{lev_dim: lnb_ind + 1}) < lapse_thresh)
        is_large_lapse_diff = is_large_lapse_diff | is_large_lapse_diff2
        lnb_ind = lnb_ind + is_large_lapse_diff.astype(int)
    lnb_ind = lnb_ind + 1  # make it up to and including this level
    return lnb_ind


def get_pnorm(p: Union[xr.DataArray, np.ndarray, float], p_low: Union[xr.DataArray, np.ndarray, float],
              p_high: Union[xr.DataArray, np.ndarray, float]) -> Union[xr.DataArray, np.ndarray, float]:
    """
    Given the pressure, $p$, this returns a normalized pressure coordinate going from 0 at $p_{low}$ to 1 at $p_{high}$:

    $$p_{norm} = \\frac{\log_{10}p - \log_{10}p_{low}}{\log_{10}p_{high} - \log_{10}p_{low}}$$

    Args:
        p: Pressure in Pa.
        p_low: Low level pressure (closer to surface) in Pa.
        p_high: High level pressure (further from surface so $p_{high} < p_{low}$) in Pa.

    Returns:
        pnorm: Value of pressure $p$ in normalized pressure coordinates between 0 and 1.
    """
    return (np.log10(p) - np.log10(p_low)) / (np.log10(p_high) - np.log10(p_low))


def get_p_from_pnorm(pnorm: Union[xr.DataArray, np.ndarray, float], p_low: Union[xr.DataArray, np.ndarray, float],
                        p_high: Union[xr.DataArray, np.ndarray, float]) -> Union[xr.DataArray, np.ndarray, float]:
    """
    Given the normalized pressure coordinate `pnorm`, this inverts `get_pnorm` to give the physical pressure in Pa.

    Args:
        pnorm: Normalized pressure coordinate (dimensionless).
        p_low: Low level pressure used to compute `pnorm` (closer to surface) in Pa.
        p_high: High level pressure used to compute `pnorm` (further from surface so $p_{high} < p_{low}$) in Pa.

    Returns:
        p: Pressure corresponding to `pnorm` in Pa.
    """
    return 10**(pnorm * (np.log10(p_high) - np.log10(p_low)) + np.log10(p_low))


def interp_var_to_pnorm(var: xr.DataArray, p: xr.DataArray, var_at_low: xr.DataArray, p_low: Union[xr.DataArray, float],
                        var_at_high: xr.DataArray, p_high: Union[xr.DataArray, float], lnb_ind: xr.DataArray,
                        d_pnorm: float = 0.1, pnorm_custom_grid: Optional[np.ndarray] = None, extrapolate: bool = True,
                        insert_low: bool = True, insert_high: bool = True,
                        lev_dim: str = 'lev', pnorm_dim_name: str = 'pnorm',
                        subtract_var_at_low: bool = True) -> Tuple[xr.DataArray, xr.DataArray]:
    """
    Interpolate `var` as a function of pressure, $p$, into $p_{norm}$ that goes from 0 at
    $p_{low}$ to 1 at $p_{high}$ according to:

    $$p_{norm} = \\frac{\log_{10}p - \log_{10}p_{low}}{\log_{10}p_{high} - \log_{10}p_{low}}$$

    Args:
        var: `[n_lev]` Variable to interpolate.
        p: `[n_lev]` Pressure in Pa.
        var_at_low: Value of `var` at `p_low`.
        p_low: Pressure level that will correspond to $p_{norm}=0$.
        var_at_high: Value of `var` at `p_high`.
        p_high: Pressure level that will correspond to $p_{norm}=1$.
        lnb_ind: The value of `var` at model levels further from the surface than this will be set to `nan`
            or extrapolated from below this level if `extrapolate=True`.
        d_pnorm: Desired spacing in $p_{norm}$ coordinates.
        pnorm_custom_grid: Option to provide the $p_{norm}$ coordinates. Will override the `d_pnorm`.
        extrapolate: Whether to extrapolate above LNB.
        insert_low: Whether to add `var_at_high` to `var` before interpolation.
        insert_high: Whether to add `var_at_low` to `var` before interpolation.
        lev_dim: Name of dimension for vertical model level.
        pnorm_dim_name: Name of the output `pnorm` dimension.
        subtract_var_at_low: If `True` will subtract `var_at_low` from the interpolated `var`.

    Returns:
        var: `[n_pnorm]` Value of `var` on the new $ p_{norm} $ grid.
        n_extrapolate: Number of model levels used above LNB that were extrapolated.
            Only non-zero if `extrapolate` is True.
    """
    small = 1   # a small value in units of Pa, much less than spacing of model levels
    var = var.where(p >= p.isel(**{lev_dim: lnb_ind}) - small)  # set var to nan above LNB
    # Define target pnorm-grid
    if pnorm_custom_grid is None:
        pnorm = np.arange(0, 1 + d_pnorm, d_pnorm)
    else:
        pnorm = pnorm_custom_grid


    def _interp_onecell(var_prof, p_prof, var_at_low, p_low, var_at_high, p_high):
        # Skip missing
        if np.all(np.isnan(var_prof)):
            return np.full_like(pnorm, np.nan, dtype=float)

        # Shift to pnorm grid
        logp_target = np.log10(get_p_from_pnorm(pnorm, p_low, p_high))
        # Add to dataset used to perform interpolation
        if insert_low and (np.min(np.abs(p_prof - p_low)) > small): # only insert if not already present even if request
            ind_low = np.searchsorted(p_prof, p_low)
            p_prof = np.insert(p_prof, ind_low, p_low)
            var_prof = np.insert(var_prof, ind_low, var_at_low)
        if insert_high and (np.min(np.abs(p_prof - p_high)) > small):
            ind_high = np.searchsorted(p_prof, p_high)
            p_prof = np.insert(p_prof, ind_high, p_high)
            var_prof = np.insert(var_prof, ind_high, var_at_high)

        var_target = np.interp(logp_target, np.log10(p_prof), var_prof)
        n_extrap = 0
        if extrapolate:
            var_target, extrap_ind = interp_nan(logp_target, var_target)
            n_extrap = extrap_ind.size
        if subtract_var_at_low:
            var_target = var_target - var_at_low
        return var_target, n_extrap

    out = xr.apply_ufunc(
        _interp_onecell,
        var, p, var_at_low, p_low, var_at_high, p_high,
        input_core_dims=[[lev_dim], [lev_dim], [], [], [], []],
        output_core_dims=[[pnorm_dim_name], []],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float, int],
        kwargs={}
    )
    out = list(out)
    out[0] = out[0].assign_coords(**{pnorm_dim_name: pnorm})
    return out[0], out[1]


def get_mse_prof_rms(temp_env: xr.DataArray, p_env: xr.DataArray, z_env: xr.DataArray,
                     p_thickness: xr.DataArray, temp_at_split: Optional[xr.DataArray] = None,
                     p_split: Optional[xr.DataArray] = None, z_at_split: Optional[np.ndarray] = None,
                     lev_dim: str = 'lev',
                     split_dim: str = 'lev') -> xr.DataArray:
    """
    For each possible split level, will compute the RMS error of $MSE_{env} - MSE^*_{split}$ where:

    * Above Split: $MSE_{env}(p) = MSE^*(p)$
    * Below Split: $MSE_{env}(p) = DSE(p) + L_v q^*_{split}$

    Idea being that LCL is split level with minimum RMS error.

    Args:
        temp_env: Environmental temperature [K], dims (..., lev_dim)
        p_env: Environmental pressure [Pa], dims (..., lev_dim)
        z_env: Geopotential height [m], dims (..., lev_dim)
        p_thickness: Pressure thickness between levels [Pa], dims (..., lev_dim)
        temp_at_split: Temperature to use as `temp_at_lcl` in `get_mse_env`, dims (..., split_dim).</br>
            If `None`, sets to `temp_env`.
        p_split: Pressure to use as `p_lcl` in `get_mse_env`, dims (..., split_dim).</br>
            If `None`, sets to `p_env`.
        z_at_split: Geopotential height corresponding to `p_split`, dims (..., split_dim).</br>
            If `None`, sets to `z_env`.
        lev_dim: Model level dimension in `temp_env`, `p_env`, `z_env`, and `p_thickness`.
        split_dim: Dimension corresponding to different split levels in `temp_at_split` and `p_split`.</br>
            If `temp_at_split` is `None`, will set to `lev_dim`.

    Returns:
        mse_prof_error: Mass weighted RMS difference for each possible split level, dims (..., split_dim)
    """
    if (temp_at_split is None) and (p_split is None) and (z_at_split is None):
        temp_at_split = temp_env
        p_split = p_env
        z_at_split = z_env
        split_dim = lev_dim
    elif (temp_at_split is None) or (p_split is None) or (z_at_split is None):
        raise ValueError('Either all or none of temp_at_split, p_split, and z_at_split must be specified.')

    def _core(temp_env, p_env, z_env, p_thickness, temp_at_split, p_split, z_at_split):
        # temp_env, p_env, z_env, p_thickness: (lev,)
        norm = []
        for i in range(temp_at_split.shape[0]):
            if np.isnan(temp_at_split[i]):
                norm.append(np.nan)
                continue
            var = get_mse_env(temp_env, p_env, z_env,
                              temp_at_split[i], p_split[i], 'full')
            var = var - moist_static_energy(temp_at_split[i],
                                            sphum_sat(temp_at_split[i], p_split[i]),
                                            z_at_split[i])


            # Set MSE above LCL to mass weighted mean in this layer, not equal to LCL MSE^* when computing optimal
            # LCL. Because idea is MSE should be constant above, and DSE should be constant below
            # var[p_env < p_split[i]] = smooth_threshold(var[p_env < p_split[i]], x_thresh=1)
            # var[p_env < p_split[i]] -= np.sum((var * p_thickness)[p_env < p_split[i]]/g) / np.sum(p_thickness[p_env < p_split[i]]/g)
            # var[p_env > p_split[i]] -= np.sum((var * p_thickness)[p_env > p_split[i]] / g) / np.sum(
            #     p_thickness[p_env > p_split[i]] / g)
            # weight with p_thickness/g across lev
            # var = np.clip(var, -10,10)  # don't allow extremely large error from any one level
            norm.append(weighted_RMS(var, p_thickness / g))
        return np.array(norm)

    # Apply across non-lev dims
    norm = xr.apply_ufunc(
        _core,
        temp_env,
        p_env,
        z_env,
        p_thickness,
        temp_at_split,
        p_split,
        z_at_split,
        input_core_dims=[[lev_dim], [lev_dim], [lev_dim], [lev_dim], [split_dim], [split_dim], [split_dim]],
        output_core_dims=[[split_dim]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
    )

    # Attach coordinates (pressure levels as p_lcl)
    norm = norm.assign_coords({split_dim:(split_dim, p_split[split_dim].values)})
    norm.name = "mse_prof_error"
    return norm
