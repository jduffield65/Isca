import numpy as np
import xarray as xr
import inspect
from typing import Union, Literal, Optional, Tuple, List, Callable
import scipy.optimize

from isca_tools.thesis.surface_flux_taylor_2layer import get_temp_rad_atm, reconstruct_lh, reconstruct_sh, \
    reconstruct_lw_atm, \
    name_square, name_nl, get_latent_heat, get_sensible_heat, get_lw_atm, get_sensitivity_lh, \
    get_sensitivity_sh, get_sensitivity_lw_atm, get_lw_surf, get_sensitivity_lw_surf, reconstruct_lw_surf, \
    get_temp_from_sphum_sat
from isca_tools.thesis.surface_flux_taylor import get_temp_rad as get_temp_rad_surf
from tqdm.notebook import tqdm
from isca_tools.utils.constants import c_p_ocean, rho_ocean
from isca_tools.utils.moist_physics import sphum_sat
from isca_tools.utils.numerical import get_fit_coef_complex, spline_deriv_periodic, fit_linear_zero_mean
from isca_tools.utils.radiation import get_heat_capacity, opd_lw_gray, frierson_sw_optical_depth, get_frierson_sw_abs
from isca_tools.utils.xarray import wrap_with_apply_ufunc, update_dim_slice, raise_if_common_dims_not_identical
from isca_tools import load_namelist, load_dataset
from jobs.theory_lapse.cesm.thesis_figs.scripts.utils import convert_ds_of_dicts
from jobs.thesis_season.thesis_figs.utils import get_annual_zonal_mean, month_ticks, width, label_lat, label_time, \
    lat_min, lat_max, ax_lims_lat, day_seconds, get_fourier_fit_xr, fourier_series_xr

var_keep = ['temp', 't_surf', 'swdn_sfc', 'lwup_sfc', 'lwdn_sfc',
            'flux_lhe', 'flux_t', 'q_surf', 'ps', 'q_surf', 'w_atm', 'q_atm', 'olr']
exp_dir = lambda x: f'thesis_season/column/depth={x}/fix_rh'


def load_ds(depth: Literal[5, 20, 'both'] = 'both', var_keep: List = var_keep,
            lat_min: float = lat_min, lat_max: float = lat_max, exp_name: Optional[Union[str, List]] = None,
            low_lev_only: bool = True, first_month_file=121) -> xr.Dataset:
    """Load and preprocess near-surface fields for one or two mixed-layer depths.

    Loads the Isca experiment dataset(s), keeps selected variables, subsets a latitude
    band, and retains only the lowest model level. Adds coordinates and derived
    diagnostics used for surface-flux breakdown (e.g., near-surface RH, net surface
    flux, radiative temperature, and temperature disequilibrium terms).
    Rename the variables `temp` to `temp_atm`, `t_surf` to `temp_surf`, `ps` to `p_surf`.

    Args:
        depth: Mixed-layer depth to load, one of 5, 20, or "both".
        reduced_evap: If True, load the reduced-evaporation version of the 5 m
            experiment (no effect for 20 m).
        var_keep: Variables to keep from the raw dataset. Missing variables are
            removed from this list during loading.
        lat_min: Minimum latitude (degrees) for subset.
        lat_max: Maximum latitude (degrees) for subset.

    Returns:
        ds: An `xarray.Dataset` concatenated along dimension "depth" (length 1 or 2),
            containing original and derived variables. Key additions include:
            `p_atm`, `heat_capacity`, `evap_prefactor`, `odp_surf`, `rh_atm`, `lw_sfc`,
            `flux_net`, `temp_rad`, `temp_diseqb`, and `temp_diseqb_r`.

    Raises:
        ValueError: If `depth` is not 5, 20, or "both".
        KeyError: If required namelist entries are missing (some are given defaults).

    Notes:
        - The "depth" dimension labels experiments (mixed-layer depths), not vertical
          levels. The vertical selection is done via `pfull=np.inf` (nearest).
        - `var_keep` is mutated in-place when variables are missing; pass a copy if
          you want to preserve the original list.
    """
    # Load dataset
    if exp_name is None:
        if depth == 5:
            exp_name = [exp_dir(5)]
        elif depth == 20:
            exp_name = [exp_dir(20)]
        elif depth == 'both':
            exp_name = [exp_dir(5), exp_dir(20)]
        else:
            raise ValueError('Depth must be either 5 or 20 or "both"')
    elif isinstance(exp_name, str):
        exp_name = [exp_name]

    # Get low level sigma level
    namelist = load_namelist(exp_name[0])
    sigma_levels_half = np.asarray(namelist['vert_coordinate_nml']['bk'])
    sigma_levels_full = np.convolve(sigma_levels_half, np.ones(2) / 2, 'valid')

    n_exp = len(exp_name)
    # Think best to use one hemisphere as from Roach expect slight difference between hemispheres
    lat_range = slice(lat_min, lat_max)  # only consider NH and outside deep tropics
    ds = []
    evap_prefactor = []
    for i in tqdm(range(n_exp)):
        ds_use = load_dataset(exp_name[i], first_month_file=first_month_file)
        try:
            ds_use = ds_use[var_keep]
        except KeyError:
            remove_keys = []
            for key in var_keep:
                if key not in ds_use:
                    print(f'Removing {key} from var_keep')
                    remove_keys += [key]
            for key in remove_keys:
                var_keep.remove(key)
            ds_use = ds_use[var_keep]
        ds_use = ds_use.sel(lat=lat_range)
        ds_use['hybm'] = ds_use.temp.isel(time=0, lat=0, lon=0) * 0 + sigma_levels_full
        if low_lev_only:
            ds_use = ds_use.sel(pfull=np.inf, method='nearest')  # only keep lowest level
        ds.append(ds_use.load())  # only keep after spin up
        try:
            evap_prefactor.append(load_namelist(exp_name[i])['surface_flux_nml']['land_evap_prefactor'])
        except KeyError:
            evap_prefactor.append(1)  # default value
    mixed_layer_depth = [load_namelist(exp_name[i])['mixed_layer_nml']['depth'] for i in range(n_exp)]
    mixed_layer_depth = xr.DataArray(mixed_layer_depth, dims="depth", name='depth')
    ds = xr.concat(ds, dim=mixed_layer_depth)
    ds['heat_capacity'] = get_heat_capacity(c_p_ocean, rho_ocean, ds.depth)
    ds['evap_prefactor'] = xr.DataArray(evap_prefactor, dims="depth", coords={"depth": ds["depth"]})
    ds['hybm'] = ds.hybm.isel(depth=0)
    ds.attrs['drag_coef'] = namelist['surface_flux_nml']['drag_const']  # drag coef is a constant here
    if 'rh_flux_q' in namelist['surface_flux_nml']:
        # Constant RH used in latent heat calculations
        ds.attrs['rh_flux_q'] = namelist['surface_flux_nml']['rh_flux_q']
    # Rename temp vars to used in surface flux functions
    ds = ds.rename_vars({'temp': 'temp_atm', 't_surf': 'temp_surf', 'ps': 'p_surf',
                         'hybm': 'sigma_atm'})
    if not low_lev_only:
        ds = ds.rename_vars({'temp_atm': 'temp'})
        ds['temp_atm'] = ds.temp.sel(pfull=np.inf, method='nearest')

    # Get optical depth at surface - assume same for both experiments
    odp_info = {'odp': 1, 'ir_tau_eq': 6, 'ir_tau_pole': 1.5, 'linear_tau': 0.1, 'wv_exponent': 4,
                'atm_abs': 0}  # default vals
    for key in odp_info:  # If provided, update
        if key in namelist['two_stream_gray_rad_nml']:
            odp_info[key] = namelist['two_stream_gray_rad_nml'][key]
    ds['odp_surf'] = opd_lw_gray(ds.lat, kappa=odp_info['odp'], tau_eq=odp_info['ir_tau_eq'],
                                 tau_pole=odp_info['ir_tau_pole'], frac_linear=odp_info['linear_tau'],
                                 k_exponent=odp_info['wv_exponent'])  # optical depth as function of latitude
    # ds.attrs['odp_sw'] = float(frierson_sw_optical_depth(ds.p_surf.isel(depth=0, time=0, lon=0, lat=0), odp_info['atm_abs']))
    # ds.attrs['sw_abs'] = 1 - np.exp(-ds.attrs['odp_sw'])      # fraction of sw absorbed
    ds.attrs['atm_abs'] = odp_info['atm_abs']
    ds.attrs['sw_abs'] = float(get_frierson_sw_abs(odp_info['atm_abs'], ds.p_surf.isel(depth=0, time=0, lon=0, lat=0)))
    ds.attrs['albedo'] = namelist['mixed_layer_nml']['albedo_value']
    # Compute variables required for flux breakdown
    if low_lev_only:
        ds['p_atm'] = ds.p_surf * ds.sigma_atm
    else:
        ds['p_atm'] = ds.p_surf * ds.sigma_atm.sel(pfull=np.inf, method='nearest')
    ds['rh_atm'] = ds.q_atm / sphum_sat(ds.temp_atm, ds.p_atm)
    ds['lw_atm'] = ds.lwup_sfc - ds.lwdn_sfc - ds.olr
    ds['lw_surf'] = ds.lwup_sfc - ds.lwdn_sfc
    ds['temp_rad_surf'] = get_temp_rad_surf(ds.lwdn_sfc, ds.odp_surf)
    ds['temp_diseqb_surf'] = ds.temp_atm - ds.temp_rad_surf
    ds['temp_rad_atm'] = get_temp_rad_atm(ds.olr, ds.temp_surf, ds.odp_surf)
    ds['temp_diseqb_atm'] = ds.temp_atm - ds.temp_rad_atm
    return ds


def get_flux(ds: xr.Dataset, flux_name: Literal['lh', 'sh', 'lw_atm', 'lw_surf'] = 'lh',
             calc: bool = False, use_rh_flux_q: bool = False) -> xr.DataArray:
    """Return a surface turbulent/radiative flux from a Dataset.

    This helper provides two modes:

    1. `calc=False` (default): return the flux directly from pre-existing fields in
       `ds` (fast; no recalculation).
    2. `calc=True`: recompute the flux diagnostically from input variables/parameters
       using the appropriate flux function. Inputs are pulled from `ds` variables
       first, and if missing, from `ds.attrs`.

    Args:
        ds:
            Input dataset containing either the flux fields directly (when `calc=False`)
            or the required input variables/attributes for the chosen calculation
            (when `calc=True`).
        flux_name:
            Which flux to return.

            - `'lh'`: latent heat flux (surface) returned directly as `ds.flux_lhe`
              or computed via `get_latent_heat`.
            - `'sh'`: sensible heat flux (surface) returned directly as `ds.flux_sh`
              or computed via `get_sensible_heat`.
            - `'lw'`: net upward longwave at the surface returned directly as
              `ds.lwup_sfc - ds.lwdn_sfc` or computed via `get_lwup_sfc_net`.
        calc:
            If True, calculate the flux from required inputs using the corresponding
            flux function. If False, return the precomputed/direct flux expression
            from `ds`.
        use_rh_flux_q:
            For latent heat calculation, this will use `rh_flux_q` in attributes rather than `rh_atm`
            to compute the latent heat flux.

    Returns:
        flux: Flux as an `xr.DataArray`. Dimensions/coords follow the underlying dataset
            variables used (e.g. typically includes `time`, `lat`, `lon`).

    Raises:
        KeyError:
            If `calc=False` and the required direct field(s) are missing from `ds`
            (e.g. `flux_lhe`, `flux_sh`, `lwup_sfc`, `lwdn_sfc`).
        ValueError:
            If `calc=True` and any required argument for the chosen flux function is
            missing from both `ds` and `ds.attrs`.

    Notes:
        - For `calc=True`, the required inputs are inferred from the signature of the
          selected flux function (`get_latent_heat`, `get_sensible_heat`,
          `get_lwup_sfc_net`). This makes the interface robust to changes in the
          underlying calculation functions, but it means `ds`/`ds.attrs` must use
          matching argument names.
        - For `'lw'` in direct mode, the returned value is computed on the fly as
          `ds.lwup_sfc - ds.lwdn_sfc`, so it does not require an explicit stored
          "net" variable.
    """
    if calc:
        if flux_name == 'lh':
            if use_rh_flux_q:
                ds = ds.copy(deep=True)  # so not to overwrite
                if 'rh_flux_q' in ds.attrs:
                    ds['rh_atm'] = ds.rh_atm * 0 + ds.rh_flux_q
                else:
                    raise ValueError('ds does not contain rh_flux_q')
        flux_func = {'lh': get_latent_heat, 'sh': get_sensible_heat, 'lw_atm': get_lw_atm,
                     'lw_sfc': get_lw_surf}[flux_name]
        arg_names = list(inspect.signature(flux_func).parameters.keys())
        var = {}
        for key in arg_names:
            if key in ds:
                var[key] = ds[key]
            elif key in ds.attrs:
                var[key] = ds.attrs[key]
            else:
                raise ValueError(f'ds does not contain the variable "{key}"')
        return flux_func(**var)
    else:
        return {'lh': ds.flux_lhe, 'sh': ds.flux_t, 'lw_atm': ds.lwup_sfc - ds.lwdn_sfc - ds.olr,
                'lw_surf': ds.lwup_sfc - ds.lwdn_sfc}[flux_name]


def get_flux_sensitivity(ds: xr.Dataset, flux_name: Literal['lh', 'sh', 'lw_atm', 'lw_surf'] = 'lh',
                         use_rh_flux_q: bool = False) -> xr.Dataset:
    """
    Return flux sensitivity (Taylor-series coefficients) for a chosen flux decomposition.

    This is a thin wrapper that selects the appropriate sensitivity routine
    (`get_sensitivity_lh`, `get_sensitivity_sh`, or `get_sensitivity_lw`), gathers
    its required inputs from `ds` variables (or `ds.attrs`), and returns the
    resulting coefficients as an `xr.Dataset`.

    Args:
        ds: Dataset containing the required inputs as variables and/or attributes.
        flux_name: Flux decomposition to use: `'lh'`, `'sh'`, or `'lw'`.
        use_rh_flux_q:
            For latent heat calculation, this will use `rh_flux_q` in attributes rather than `rh_atm`
            to compute the latent heat flux.

    Returns:
        Dataset of Taylor-series coefficients returned by the selected sensitivity function.

    Raises:
        ValueError: If any required input is missing from both `ds` and `ds.attrs`.
    """
    func_use = {'lh': get_sensitivity_lh, 'sh': get_sensitivity_sh, 'lw_atm': get_sensitivity_lw_atm,
                'lw_sfc': get_sensitivity_lw_surf}[flux_name]
    arg_names = inspect.signature(func_use).parameters.keys()
    if flux_name == 'lh':
        if use_rh_flux_q:
            ds = ds.copy(deep=True)  # so not to overwrite
            if 'rh_flux_q' in ds.attrs:
                ds['rh_atm'] = ds.rh_atm * 0 + ds.rh_flux_q
            else:
                raise ValueError('ds does not contain rh_flux_q')
    var = {}
    for key in arg_names:
        if key in ds:
            var[key] = ds[key]
        elif key in ds.attrs:
            var[key] = ds.attrs[key]
        else:
            raise ValueError(f'ds does not contain the variable "{key}"')
    return xr.Dataset(func_use(**var))


def reconstruct_flux_xr(ds: xr.Dataset, ds_ref: xr.Dataset,
                        flux_name: Literal['lh', 'sh', 'lw_atm', 'lw_surf'] = 'lh',
                        numerical: bool = False,
                        time_dim: str = 'time', use_rh_flux_q: bool = False
                        ) -> Tuple[xr.DataArray, xr.DataArray, xr.DataArray, xr.Dataset]:
    reconstruct_flux = {'lh': reconstruct_lh, 'sh': reconstruct_sh, 'lw_atm': reconstruct_lw_atm,
                        'lw_surf': reconstruct_lw_surf}[flux_name]
    arg_names = list(inspect.signature(reconstruct_flux).parameters.keys())
    arg_names = [key for key in arg_names if key != 'numerical']  # treat numerical as kwarg
    input_core_dims = [[] if (('_ref' in arg) or (arg in ['sigma_atm'])) else [time_dim] for arg in
                       arg_names]
    output_core_dims = [[], [time_dim], [time_dim], []]
    reconstruct_flux_wrap = wrap_with_apply_ufunc(reconstruct_flux, input_core_dims=input_core_dims,
                                                  output_core_dims=output_core_dims)

    if flux_name == 'lh':
        # Add time dimension to drag and evap, as don't have initially
        drag_coef = ds.temp_surf * 0 + ds_ref.drag_coef
        evap_prefactor = ds.temp_surf * 0 + ds_ref.evap_prefactor
        if use_rh_flux_q:
            ds = ds.copy(deep=True)  # so not to overwrite
            ds_ref = ds_ref.copy(deep=True)
            if 'rh_flux_q' in ds.attrs:
                ds['rh_atm'] = ds.rh_atm * 0 + ds.rh_flux_q
            else:
                raise ValueError('ds does not contain rh_flux_q')
            if 'rh_flux_q' in ds_ref.attrs:
                ds_ref['rh_atm'] = ds_ref.rh_atm * 0 + ds_ref.rh_flux_q
            else:
                raise ValueError('ds does not contain rh_flux_q')
        flux_ref, flux_anom_linear, flux_anom_nl, info_cont = \
            reconstruct_flux_wrap(ds_ref.temp_surf, ds_ref.temp_atm, ds_ref.rh_atm, ds_ref.w_atm, ds_ref.drag_coef,
                                  ds_ref.p_surf, ds_ref.sigma_atm, ds_ref.evap_prefactor,
                                  ds.temp_surf, ds.temp_atm, ds.rh_atm, ds.w_atm, drag_coef, ds.p_surf,
                                  evap_prefactor, numerical=numerical)
    elif flux_name == 'sh':
        # Add time dimension to drag, as don't have initially
        drag_coef = ds.temp_surf * 0 + ds_ref.drag_coef
        flux_ref, flux_anom_linear, flux_anom_nl, info_cont = \
            reconstruct_flux_wrap(ds_ref.temp_surf, ds_ref.temp_atm, ds_ref.w_atm, ds_ref.drag_coef,
                                  ds_ref.p_surf, ds_ref.sigma_atm, ds.temp_surf, ds.temp_atm,
                                  ds.w_atm, drag_coef, ds.p_surf, numerical=numerical)
    elif flux_name == 'lw_atm':
        # Add time dimension to odp_surf, as don't have initially
        odp_surf = ds.temp_surf * 0 + ds_ref.odp_surf
        flux_ref, flux_anom_linear, flux_anom_nl, info_cont = \
            reconstruct_flux_wrap(ds_ref.temp_surf, ds_ref.temp_rad_surf, ds_ref.temp_rad_atm,
                                  ds_ref.odp_surf, ds.temp_surf, ds.temp_rad_surf, ds.temp_rad_atm,
                                  odp_surf, numerical=numerical)
    elif flux_name == 'lw_surf':
        # Add time dimension to odp_surf, as don't have initially
        odp_surf = ds.temp_surf * 0 + ds_ref.odp_surf
        flux_ref, flux_anom_linear, flux_anom_nl, info_cont = \
            reconstruct_flux_wrap(ds_ref.temp_surf, ds_ref.temp_rad_surf,
                                  ds_ref.odp_surf, ds.temp_surf, ds.temp_rad_surf,
                                  odp_surf, numerical=numerical)
    else:
        raise ValueError(f'Unknown flux_name="{flux_name}". Must be one of "lh", "sh", "lw_atm", "lw_sfc".')
    info_cont = xr.Dataset(convert_ds_of_dicts(info_cont, ds.time, 'time'))
    return flux_ref, flux_anom_linear, flux_anom_nl, info_cont


# Xarray versions of functions
get_fit_coef_complex_xr = wrap_with_apply_ufunc(get_fit_coef_complex, input_core_dims=[['time'], ['time'], ['time']],
                                                output_core_dims=[[], []])

get_temp_from_sphum_sat_xr = wrap_with_apply_ufunc(get_temp_from_sphum_sat, input_core_dims=[[], []],
                                                   output_core_dims=[[]])

spline_deriv_periodic_xr = wrap_with_apply_ufunc(spline_deriv_periodic, input_core_dims=[['time'], ['time']],
                                                 output_core_dims=[['time']])

fit_linear_zero_mean_xr_1 = wrap_with_apply_ufunc(
    lambda x1, y: fit_linear_zero_mean(x1, y, x2=None)[0],
    input_core_dims=[['time'], ['time']],
    output_core_dims=[[]],
)

fit_linear_zero_mean_xr_2 = wrap_with_apply_ufunc(
    lambda x1, y, x2: fit_linear_zero_mean(x1, y, x2=x2),
    input_core_dims=[['time'], ['time'], ['time']],
    output_core_dims=[[], []],
)

def fit_linear_zero_mean_xr(x1, y, x2=None):
    r"""Fits one or two mean-centred predictors to a mean-centred response.

    Removes the temporal mean from each supplied variable before fitting a
    linear model with no intercept. With one predictor, fits

    $$
    y' = c_1 x_1',
    $$

    where primes denote anomalies relative to the time mean. With two
    predictors, fits

    $$
    y' = c_1 x_1' + c_2 x_2'.
    $$

    Args:
        x1: First predictor. Must contain a `time` dimension.
        y: Response variable. Must contain a `time` dimension and be
            broadcast-compatible with `x1`.
        x2: Optional second predictor. Must contain a `time` dimension and be
            broadcast-compatible with `x1` and `y`.

    Returns:
        If `x2` is `None`, returns the fitted coefficient $c_1$ as an
        `xarray.DataArray`.

        If `x2` is provided, returns a tuple `(c1, c2)` containing the fitted
        coefficients for `x1` and `x2`, respectively. Each coefficient is an
        `xarray.DataArray` over all dimensions except `time`.

    Notes:
        Mean-centering is performed independently over the `time` dimension:

        $$
        x_i' = x_i - \overline{x_i},
        \qquad
        y' = y - \overline{y}.
        $$

        Consequently, fitting without an intercept to the centred variables is
        equivalent to fitting a linear model with an intercept to the original
        variables.
    """
    if x2 is None:
        return fit_linear_zero_mean_xr_1(
            x1 - x1.mean(dim="time"),
            y - y.mean(dim="time"),
        )
    else:
        return fit_linear_zero_mean_xr_2(
            x1 - x1.mean(dim="time"),
            y - y.mean(dim="time"),
            x2 - x2.mean(dim="time"),
        )
