import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import inspect
from typing import Union, Literal, Optional, Tuple, List, Callable
import itertools
from matplotlib.ticker import FuncFormatter

from isca_tools.thesis.surface_flux_taylor import get_temp_rad, reconstruct_lh, reconstruct_sh, reconstruct_lw, \
    name_square, name_nl, get_latent_heat, get_sensible_heat, get_lwup_sfc_net, get_sensitivity_lh, \
    get_sensitivity_sh, get_sensitivity_lw
from isca_tools.utils import numerical
from tqdm.notebook import tqdm

from isca_tools.thesis.surface_energy_budget import get_temp_extrema_numerical, get_temp_fourier_analytic, \
    get_temp_fourier_analytic2, get_temp_extrema_theory, get_param_dimensionless, get_temp_shift_params
from isca_tools.utils import area_weighting, annual_mean
import isca_tools.utils.fourier as fourier
from isca_tools.utils.constants import c_p_water, rho_water
from isca_tools.utils.moist_physics import sphum_sat
from isca_tools.utils.radiation import get_heat_capacity, opd_lw_gray
from isca_tools.utils.xarray import wrap_with_apply_ufunc, update_dim_slice, raise_if_common_dims_not_identical
from isca_tools import load_namelist, load_dataset
from jobs.theory_lapse.cesm.thesis_figs.scripts.utils import convert_ds_of_dicts

# Plotting info
width = {'one_col': 3.2, 'two_col': 5.5}  # width in inches
# Default parameters
label_lat = "Latitude [deg]"
label_error = 'Error [%]'
label_time = 'Day of year'
label_time_shift = lambda x ='ex': '$\sin(2\pi f\Delta_{\\text{'+x+'}})$'
label_amp = lambda x='ex': '$A_{\\text{'+x+'}}$ [KK$^{-1}$]'
ax_lims_time = [-1, 360]
leg_handlelength = 1.5
month_ticks = (np.arange(15, 12 * 30 + 15, 30), ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'])
style_map = {
    # --- change (solid) ---
    "simulated": ("k", "-", "Simulated"),
    "direct1": ("k", "-", "Direct"),
    "direct2": ("C1", "-", "Direct"),
    "direct5": ("C3", "-", "Direct"),
    "linear": ("C0", ":", "Linear"),
    "linear_phase": ("C0", "--", "Linear-phase"),
    "square_phase": ("C0", "-.", "Square-phase"),
    "square_phase+": ("C0", "-", "Square-phase+"),
    "poly10_phase+": ("C2", "-", "Poly10-phase+"),
    "lw": ("C1", "-", "$\\text{LW}^{\\uparrow}_{\\text{net}}$"),
    "lh": ("C0", ":", "LH$^{\\uparrow}$"),
    "sh": ("C2", ":", "SH$^{\\uparrow}$"),
    "net": ("k", "-", "Sum"),
}

# Order reflects the order of vars in get_sensitivity - important for nonlinear combinations
style_map_var = {'temp_surf': ("C3", "-", "$T_s$", "K"),
                 'temp_diseqb': ("C1", "-", "$T_{dq}$", "K"),
                 'rh_atm': ('C2', "-", "$r_a$", "Unitless"),
                 'w_atm': ('C0', '-', '$U_a$', "ms$^{-1}$"),
                 'p_surf': ('C5', '-', '$p_s$', "Pa"),
                 'temp_diseqb_r': ("C4", "-", "$T_{dqr}$", "K")}

style_map_var_nl = {name_nl('temp_surf', 'w_atm'): ("C0", '--', '$T_sU_a$'),
                    name_square('temp_surf'): ("C3", "-", "$T_s$", "K")}

# General info
smooth_n_days = 50  # default smoothing window in days
resample = False  # Don't do resample in polyfit_phase as complicated
deg_max = 2  # In fitting go up to maximum of T^2 dependence of surface fluxes
# Lowest power is last in deg to match polyfit
deg_vals = xr.DataArray(['phase', 'cos', 'sin'] + np.arange(deg_max + 1).tolist()[::-1], dims="deg", name="deg")
day_seconds = 86400
lat_min = 30
lat_max = 90
ax_lims_lat = [lat_min, lat_max]
var_keep = ['temp', 't_surf', 'swdn_sfc', 'lwup_sfc', 'lwdn_sfc',
            'flux_lhe', 'flux_t', 'q_surf', 'ps', 'q_surf', 'w_atm', 'q_atm']
exp_dir = lambda x, y=False: f'thesis_season/depth={x}/k=1_const_drag{"_evap=0_1" if y else ""}'


def load_ds(depth: Literal[5, 20, 'both'] = 'both', reduced_evap: bool = False, var_keep: List = var_keep,
            lat_min: float = lat_min, lat_max: float = lat_max, exp_name: Optional[Union[str, List]]=None) -> xr.Dataset:
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
            exp_name = [exp_dir(5, reduced_evap)]
        elif depth == 20:
            exp_name = [exp_dir(20)]
        elif depth == 'both':
            exp_name = [exp_dir(5, reduced_evap), exp_dir(20)]
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
        ds_use = load_dataset(exp_name[i], first_month_file=121)
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
        ds_use = ds_use.sel(pfull=np.inf, method='nearest')  # only keep lowest level
        ds.append(ds_use.load())  # only keep after spin up
        try:
            evap_prefactor.append(load_namelist(exp_name[i])['surface_flux_nml']['land_evap_prefactor'])
        except KeyError:
            evap_prefactor.append(1)  # default value
    mixed_layer_depth = [load_namelist(exp_name[i])['mixed_layer_nml']['depth'] for i in range(n_exp)]
    mixed_layer_depth = xr.DataArray(mixed_layer_depth, dims="depth", name='depth')
    ds = xr.concat(ds, dim=mixed_layer_depth)
    ds['heat_capacity'] = get_heat_capacity(c_p_water, rho_water, ds.depth)
    ds['evap_prefactor'] = xr.DataArray(evap_prefactor, dims="depth", coords={"depth": ds["depth"]})
    ds['hybm'] = ds.hybm.isel(depth=0)
    ds.attrs['drag_coef'] = namelist['surface_flux_nml']['drag_const']  # drag coef is a constant here
    if 'rh_flux_q' in namelist['surface_flux_nml']:
        # Constant RH used in latent heat calculations
        ds.attrs['rh_flux_q'] = namelist['surface_flux_nml']['rh_flux_q']
    # Rename temp vars to used in surface flux functions
    ds = ds.rename_vars({'temp': 'temp_atm', 't_surf': 'temp_surf', 'ps': 'p_surf',
                         'hybm': 'sigma_atm'})

    # Get optical depth at surface - assume same for both experiments
    odp_info = {'odp': 1, 'ir_tau_eq': 6, 'ir_tau_pole': 1.5, 'linear_tau': 0.1, 'wv_exponent': 4}  # default vals
    for key in odp_info:  # If provided, update
        if key in namelist['two_stream_gray_rad_nml']:
            odp_info[key] = namelist['two_stream_gray_rad_nml'][key]
    ds['odp_surf'] = opd_lw_gray(ds.lat, kappa=odp_info['odp'], tau_eq=odp_info['ir_tau_eq'],
                                 tau_pole=odp_info['ir_tau_pole'], frac_linear=odp_info['linear_tau'],
                                 k_exponent=odp_info['wv_exponent'])  # optical depth as function of latitude

    # Compute variables required for flux breakdown
    ds['temp_diseqb'] = ds.temp_surf - ds.temp_atm
    ds['p_atm'] = ds.p_surf * ds.sigma_atm
    ds['rh_atm'] = ds.q_atm / sphum_sat(ds.temp_atm, ds.p_atm)
    ds['lw_sfc'] = ds.lwup_sfc - ds.lwdn_sfc
    ds['flux_net'] = ds['lw_sfc'] + ds['flux_lhe'] + ds['flux_t']
    ds['temp_rad'] = get_temp_rad(ds.lwdn_sfc, ds.odp_surf)
    ds['temp_diseqb_r'] = ds.temp_atm - ds.temp_rad
    return ds


def get_annual_zonal_mean(ds, combine_abs_lat=False, lat_name='lat', smooth_n_days=smooth_n_days,
                          smooth_center=True, keep_attrs: bool = True):
    """Compute annual-mean zonal mean, optionally combining ±latitudes.

    This function:
    1) Computes the annual mean via `annual_mean(ds)`,
    2) Takes the zonal mean over longitude,
    3) Optionally averages fields at latitudes with the same absolute value
       (e.g., $(+30^\circ)$ and $(-30^\circ)$) using a groupby on $(|\mathrm{lat}|)$,
    4) Resets the time coordinate to start at 0 (integer years since the first).

    Args:
        ds: An xarray Dataset or DataArray with dimensions including `lon` and
            typically `time` and `lat`.
        combine_abs_lat: If True, combine values at +lat and -lat by averaging
            them together into a single latitude coordinate \(|\mathrm{lat}|\).
            The equator (0) remains unchanged. Defaults to False.
        lat_name: Name of the latitude dimension/coordinate. Defaults to 'lat'.
        smooth_n_days: Optional integer window length for time smoothing (in
            number of time steps, e.g. days). If None or <= 1, no smoothing.
        smooth_center: If True, use a centered window for smoothing.
        keep_attrs: Optional boolean flag for keeping attributes. Defaults to True.

    Returns:
        An xarray Dataset or DataArray containing the annual-mean zonal mean.
        If `combine_abs_lat` is True, the latitude coordinate will be nonnegative
        and sorted (e.g., 0, 30, 60, ...).

    Raises:
        ValueError: If `combine_abs_lat` is True but `lat_name` is not a
            dimension of the input after zonal averaging.
    """
    if smooth_n_days is not None and smooth_n_days > 1:
        ds = ds.rolling(time=int(smooth_n_days), center=smooth_center).mean()
    ds_av = annual_mean(ds).mean(dim='lon')
    # ds_av = annual_mean(ds.mean(dim='lon'))           # order does not matter, I checked gives same result

    if combine_abs_lat:
        if lat_name not in ds_av.dims:
            raise ValueError(f"Expected latitude dim '{lat_name}' in {ds_av.dims}")

        abs_lat = ds_av[lat_name].astype(float).copy()
        ds_av = (
            ds_av.assign_coords(abs_lat=abs_lat.abs())
            .groupby('abs_lat')
            .mean(dim=lat_name)
            .rename({'abs_lat': lat_name})
            .sortby(lat_name)
        )

    ds_av = ds_av.assign_coords(time=(ds_av.time - ds_av.time.min()).astype(int))
    for key in ds:
        # Get rid of time dimension of variables that dont have time dimension initially
        if 'time' not in ds[key].dims:
            ds_av[key] = ds_av[key].isel(time=0)
    if keep_attrs:
        ds_av.attrs = ds.attrs
    return ds_av


get_fourier_fit_xr = wrap_with_apply_ufunc(fourier.get_fourier_fit, input_core_dims=[['time'], ['time']],
                                           output_core_dims=[['time'], ['harmonic'], ['harmonic']])

fourier_series_xr = wrap_with_apply_ufunc(fourier.fourier_series,
                                          input_core_dims=[['time'], ['harmonic'], ['harmonic']],
                                          output_core_dims=[['time']])


# Might need to do different version when include fourier coefs, as more outputs
def polyfit_phase_xr(x: xr.DataArray, y: xr.DataArray,
                     deg: int, time: Optional[xr.DataArray] = None, time_start: Optional[float] = None,
                     time_end: Optional[float] = None,
                     deg_phase_calc: int = 10, resample: bool = resample,
                     include_phase: bool = True, include_fourier: bool = False,
                     integ_method: str = 'spline', coef_fix: Optional[List] = None) -> xr.DataArray:
    """
    Applying `polyfit_phase` to xarray.
    Will always return atleast 6 values across `deg` dimension: [phase, cos, sin, 2, 1, 0].
    First value only non zero if include_phase=True.
    Second and third values only non zero if specify `fourier_harmonics`.
    Fourth value only non zero if `deg=2`.
    Returns highest polyfit power first to match how np.polyfit works.

    If `deg>2`, will return more values e.g. for deg=4: [phase, cos, sin, 4, 3, 2, 1, 0].

    Args:
        x:
        y:
        deg:
        time:
        time_start:
        time_end:
        deg_phase_calc:
        resample:
        include_phase:
        fourier_harmonics: Whether to include a fourier residual with frequency of 2nd harmonic.
            Only 2nd harmonic as that is our expression for approx to $\Gamma$. No point fitting 1st harmonic,
            because approx very good. No point going to higher order as not analytic anymore with 2 harmonic temperature
            expression.
        integ_method:
        coef_fix: Option to fix some coefficients. Must be size `deg+2`.

    Returns:
        poly_coefs: The 6 coefficients found with a `deg` dimension.
    """
    # Not required to be the same order for code to work, but think makes neater
    raise_if_common_dims_not_identical(x, y)
    polyfit_phase_wrap = wrap_with_apply_ufunc(numerical.polyfit_phase, input_core_dims=[['time'], ['time']],
                                               output_core_dims=[['deg'], ['harmonic'], ['harmonic']]
                                               if include_fourier else [['deg']])
    var = polyfit_phase_wrap(x, y, deg=deg, time=time, time_start=time_start, time_end=time_end,
                             deg_phase_calc=deg_phase_calc, resample=resample, include_phase=include_phase,
                             fourier_harmonics=np.atleast_1d(2) if include_fourier else None,
                             # Only find 2nd harmonic coef as that is our approx for
                             integ_method=integ_method, pad_coefs_phase=True, coef_fix=coef_fix)
    # Polyfit outputs phase first and then highest poly coef power is first
    deg_in_var = ['phase'] + np.arange(deg + 1)[::-1].tolist()
    if deg <= 2:
        # Always include 2nd harmonic, but just set to zero if deg=1
        deg_vals_use = deg_vals
    else:
        deg_vals_use = xr.DataArray(['phase', 'cos', 'sin'] + np.arange(deg + 1).tolist()[::-1], dims="deg",
                                    name="deg")
    if not include_fourier:
        var = var.assign_coords(deg=deg_in_var)
        # Also output the fourier cos and sin coefs but set to zero
        var = var.reindex(deg=deg_vals_use, fill_value=0)
        return var
    else:
        # For fourier, need to add the cos and sin coefs as well
        coef_cos, coef_sin = fourier.coef_conversion(var[1].sel(harmonic=2), var[2].sel(harmonic=2))
        var = var[0].assign_coords(deg=deg_in_var)
        var = var.reindex(deg=deg_vals_use, fill_value=0)
        var = update_dim_slice(var, 'deg', 'cos', coef_cos)
        var = update_dim_slice(var, 'deg', 'sin', coef_sin)
        return var


def polyval_phase_xr(param_coefs: xr.DataArray, x: xr.DataArray, include_fourier_thresh: float = 1e-4):
    """
    Applying `polyval_phase` to xarray to estimate $y$ from `x`.

    Args:
        param_coefs: The output from `polyfit_phase_xr`
        x: The variable to apply `param_coefs` to along the dimension `time`.
        include_fourier_thresh: If $\Lambda_{cos}$ or $\Lambda_{sin}$ have a max absolute value greater
            than this, will do include these coefficients in obtaining the estimate of $y$.

    Returns:
        x_approx: The approximate value of $y$ obtained using `param_coefs` and `x`.
    """
    # Not required to be the same order for code to work, but think makes neater
    raise_if_common_dims_not_identical(x, param_coefs, name_y='param_coefs')

    def _polyval_phase(poly_coefs: np.ndarray, x: np.ndarray, coef_cos: Optional[float] = None,
                       coef_sin: Optional[float] = None):
        # Simple wrapper so takes in 2nd harmonic fourier coefs
        if (coef_cos is None) and (coef_sin is None):
            coefs_fourier_amp = None
            coefs_fourier_phase = None
        else:
            # Need to convert (cos, sin) form of params to (amplitude, phase) form expected by polyval_phase
            coefs_fourier_amp, coefs_fourier_phase = fourier.coef_conversion(cos_coef=coef_cos, sin_coef=coef_sin)
            # Only given 2nd harmonic coef, so need to pad with zeros
            # Set 0th and 1st harmonic to zero
            coefs_fourier_amp = np.hstack((np.zeros(2), coefs_fourier_amp))
            coefs_fourier_phase = np.hstack((np.zeros(2), coefs_fourier_phase))
        return numerical.polyval_phase(poly_coefs, x, coefs_fourier_amp=coefs_fourier_amp,
                                       coefs_fourier_phase=coefs_fourier_phase, pad_coefs_phase=True)

    polyval_phase_wrap = wrap_with_apply_ufunc(_polyval_phase, input_core_dims=[['deg'], ['time'], [], []],
                                               output_core_dims=[['time']])

    # Decide whether to fit fourier coefs - only if params significant
    include_fourier = np.abs(param_coefs.sel(deg=['cos', 'sin'])).max() > include_fourier_thresh

    if not include_fourier:
        # Remember highest polyfit power first hence [2, 1, 0] after phase
        var = polyval_phase_wrap(
            param_coefs.sel(deg=[key for key in param_coefs.deg.values if key not in ['cos', 'sin']]),
            x, None, None)
    else:
        var = polyval_phase_wrap(
            param_coefs.sel(deg=[key for key in param_coefs.deg.values if key not in ['cos', 'sin']]),
            x, param_coefs.sel(deg='cos'),
            param_coefs.sel(deg='sin'))
    return var


def get_temp_fourier_analytic_xr(time: xr.DataArray, swdn_sfc: xr.DataArray, heat_capacity: xr.DataArray,
                                 param_coefs: xr.DataArray, n_harmonics: int = 2,
                                 func_use: Literal[1, 2] = 2
                                 ) -> Tuple[xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray]:
    """
    Xarray version of `get_temp_fourier_analytic`. Takes output from `polyfit_phase_xr`.
    Runs with `pad_coefs_phase=True`, so there will be `n_harmonics+1` phase coefficients returned, with
    the first as zero.

    Args:
        time:
        swdn_sfc:
        heat_capacity:
        param_coefs: Output from `polyfit_phase_xr` with `deg` dimension.
        n_harmonics:

    Returns:

    """
    # Not required to be the same order for code to work, but think makes neater
    raise_if_common_dims_not_identical(swdn_sfc, param_coefs, name_x='swdn_sfc', name_y='param_coefs')
    if func_use == 1:
        get_temp_fourier_analytic_wrap = wrap_with_apply_ufunc(get_temp_fourier_analytic,
                                                               input_core_dims=[['time'], ['time'], [], [], [], [], [],
                                                                                []],
                                                               output_core_dims=[['time'], ['harmonic'], ['harmonic'],
                                                                                 ['harmonic'], ['harmonic']])
        kwargs = {'n_harmonics_sw': n_harmonics, 'pad_coefs_phase': True}
    else:
        get_temp_fourier_analytic_wrap = wrap_with_apply_ufunc(get_temp_fourier_analytic2,
                                                               input_core_dims=[['time'], ['time'], [], [], [], [], [],
                                                                                []],
                                                               output_core_dims=[['time'], ['harmonic'], ['harmonic'],
                                                                                 ['harmonic'], ['harmonic']])
        kwargs = {'n_harmonics': n_harmonics, 'pad_coefs_phase': True}
    return get_temp_fourier_analytic_wrap(time, swdn_sfc, heat_capacity, param_coefs.sel(deg='1'),
                                          param_coefs.sel(deg='phase'), param_coefs.sel(deg='2'),
                                          param_coefs.sel(deg='cos'), param_coefs.sel(deg='sin'),
                                          **kwargs)


def update_ds_extrema(ds: xr.Dataset, time: xr.DataArray, temp: xr.DataArray, fit_method: str):
    """Update extrema diagnostics for a given fitting method entry in `ds`.

    This function assumes `ds` stores results indexed by a leading `fit_method`
    dimension (e.g., different fitting approaches). It locates the index
    corresponding to `fit_method`, computes temperature extrema diagnostics from
    the provided time series (after removing the mean), and writes those values
    into the appropriate slice of `ds`.

    Args:
        ds: Dataset containing an ordered leading dimension called `fit_method`
            and variables `time_min`, `time_max`, `amp_min`, `amp_max` indexed
            along that dimension.
        time: Time coordinate (1D) corresponding to `temp`.
        temp: Temperature-like DataArray varying along `time`.
        fit_method: Label identifying which entry along `ds.fit_method` to
            update.

    Returns:
        The same Dataset instance with `time_min`, `time_max`, `amp_min`,
        and `amp_max` updated for the selected `fit_method`.

    Raises:
        ValueError: If the first dimension of `ds` is not `fit_method`.
        ValueError: If `fit_method` is not found in `ds.fit_method` (via the
            index lookup).
    """
    var = get_temp_extrema_numerical_xr(time, temp - temp.mean(dim='time'))
    for i, key in enumerate(['time_min', 'time_max', 'amp_min', 'amp_max']):
        ds = update_dim_slice(ds, 'fit_method', fit_method, var[i], key)
    return ds


def get_weights(ds: Union[xr.DataArray, xr.Dataset]) -> xr.DataArray:
    return np.cos(np.deg2rad(ds.lat))


def get_error(x: xr.DataArray, x_approx: xr.DataArray, kind: Literal['mean', 'median', 'max'] = "mean",
              norm: bool = True, dim: Union[str, list] = "time",
              norm_dim: Optional[Union[str, list]] = 'lat',
              norm_weight: Optional[Union[Callable, xr.DataArray]] = None) -> xr.DataArray:
    """Compute an absolute-error summary between two DataArrays.

    Args:
        x: Reference DataArray.
        x_approx: Approximation DataArray (broadcastable to `x`).
        kind: Error reduction to apply over `dim`. One of {"mean", "median", "max"}.
            Defaults to "mean".
        norm: If True, normalize by 0.01*(max-min) of `x` over `dim`; then takes average of this over `norm_dim.
            I.e., convert to a percentage.
        dim: Dimension name(s) to reduce over. Defaults to "time".
        norm_dim: Dimension name(s) to average normalizing factor over.
            By default, `norm_dim=lat` so that have a single normalization factor for each mixed layer depth.
        norm_weight: Weighting for normalization over `norm_dim`. If function, only argument must be `x`.
            By default will do area weighted average if `norm_dim` is latitude.

    Returns:
        DataArray of the reduced absolute error (and optionally normalized), with
        remaining dimensions preserved.

    Raises:
        ValueError: If `kind` is not one of {"mean", "median", "max"}.
    """
    # raise_if_common_dims_not_identical(x, x_approx, name_y='x_approx')
    err = np.abs(x - x_approx)

    if kind == "mean":
        out = err.mean(dim=dim)
    elif kind == "median":
        out = err.median(dim=dim)
    elif kind == "max":
        out = err.max(dim=dim)
    else:
        raise ValueError(f'Unknown kind="{kind}". Expected "mean", "median", or "max".')

    if norm:
        scale = (x.max(dim=dim) - x.min(dim=dim)) / 100  # /100 so turn to percentage
        if norm_dim is not None:
            if norm_weight is None:
                scale = scale.mean(dim=norm_dim)
            else:
                if isinstance(norm_weight, Callable):
                    norm_weight_use = norm_weight(x)
                else:
                    norm_weight_use = norm_weight
                scale = scale.weighted(norm_weight_use).mean(dim=norm_dim)
        out = out / scale

    return out


def get_flux(ds: xr.Dataset, flux_name: Literal['lh', 'sh', 'lw'] = 'lh',
             calc: bool = False) -> xr.DataArray:
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
        flux_func = {'lh': get_latent_heat, 'sh': get_sensible_heat, 'lw': get_lwup_sfc_net}[flux_name]
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
        return {'lh': ds.flux_lhe, 'sh': ds.flux_t, 'lw': ds.lwup_sfc - ds.lwdn_sfc}[flux_name]


def get_flux_sensitivity(ds: xr.Dataset, flux_name: Literal['lh', 'sh', 'lw'] = 'lh') -> xr.Dataset:
    """
    Return flux sensitivity (Taylor-series coefficients) for a chosen flux decomposition.

    This is a thin wrapper that selects the appropriate sensitivity routine
    (`get_sensitivity_lh`, `get_sensitivity_sh`, or `get_sensitivity_lw`), gathers
    its required inputs from `ds` variables (or `ds.attrs`), and returns the
    resulting coefficients as an `xr.Dataset`.

    Args:
        ds: Dataset containing the required inputs as variables and/or attributes.
        flux_name: Flux decomposition to use: `'lh'`, `'sh'`, or `'lw'`.

    Returns:
        Dataset of Taylor-series coefficients returned by the selected sensitivity function.

    Raises:
        ValueError: If any required input is missing from both `ds` and `ds.attrs`.
    """
    func_use = {'lh': get_sensitivity_lh, 'sh': get_sensitivity_sh, 'lw': get_sensitivity_lw}[flux_name]
    arg_names = inspect.signature(func_use).parameters.keys()
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
                        flux_name: Literal['lh', 'sh', 'lw'] = 'lh',
                        numerical: bool = False,
                        time_dim: str = 'time', ) -> Tuple[xr.DataArray, xr.DataArray, xr.DataArray, xr.Dataset]:
    reconstruct_flux = {'lh': reconstruct_lh, 'sh': reconstruct_sh, 'lw': reconstruct_lw}[flux_name]
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
        flux_ref, flux_anom_linear, flux_anom_nl, info_cont = \
            reconstruct_flux_wrap(ds_ref.temp_surf, ds_ref.temp_diseqb, ds_ref.rh_atm, ds_ref.w_atm, ds_ref.drag_coef,
                                  ds_ref.p_surf, ds_ref.sigma_atm, ds_ref.evap_prefactor,
                                  ds.temp_surf, ds.temp_diseqb, ds.rh_atm, ds.w_atm, drag_coef, ds.p_surf,
                                  evap_prefactor, numerical=numerical)
    elif flux_name == 'sh':
        # Add time dimension to drag, as don't have initially
        drag_coef = ds.temp_surf * 0 + ds_ref.drag_coef
        flux_ref, flux_anom_linear, flux_anom_nl, info_cont = \
            reconstruct_flux_wrap(ds_ref.temp_surf, ds_ref.temp_diseqb, ds_ref.w_atm, ds_ref.drag_coef,
                                  ds_ref.p_surf, ds_ref.sigma_atm, ds.temp_surf, ds.temp_diseqb,
                                  ds.w_atm, drag_coef, ds.p_surf, numerical=numerical)
    elif flux_name == 'lw':
        # Add time dimension to odp_surf, as don't have initially
        odp_surf = ds.temp_surf * 0 + ds_ref.odp_surf
        flux_ref, flux_anom_linear, flux_anom_nl, info_cont = \
            reconstruct_flux_wrap(ds_ref.temp_surf, ds_ref.temp_diseqb, ds_ref.temp_diseqb_r, ds_ref.odp_surf,
                                  ds.temp_surf, ds.temp_diseqb, ds.temp_diseqb_r, odp_surf, numerical=numerical)
    else:
        raise ValueError(f'Unknown flux_name="{flux_name}". Must be one of "lh", "sh", "lw".')
    info_cont = xr.Dataset(convert_ds_of_dicts(info_cont, ds.time, 'time'))
    return flux_ref, flux_anom_linear, flux_anom_nl, info_cont


def get_empirical_var_fit(ds: xr.Dataset, key_use: str = 'temp_surf',
                          time_dim: str = 'time', deg=2,
                          include_phase=True, include_fourier=True,
                          get_nl: bool = True,
                          error_kind: Literal['mean', 'median', 'max'] = "mean",
                          error_norm: bool = True,
                          error_norm_dim: Optional[Union[str, list]] = 'lat'
                          ) -> Tuple[xr.Dataset, xr.Dataset, xr.Dataset, xr.Dataset]:
    """Fit an empirical relationship between a reference driver and all time-varying variables in `ds`.

    This routine builds an empirical (polynomial-in-amplitude) model for each variable's
    *anomaly* as a function of a chosen driver variable (by default, surface temperature).
    The driver is centered by removing its time mean, and each target variable is regressed
    onto this centered driver using `polyfit_phase_xr`. The fitted parameters are then used
    to reconstruct an "empirical" version of each variable via `polyval_phase_xr`, and an
    error metric is computed between the true and empirical fields.

    Optionally, it also fits non-linear "mechanism" terms:
    - Square terms for each variable: $(X - \\bar{X})^2$
    - Pairwise product terms between variables: $(X_1 - \\bar{X}_1)(X_2 - \\bar{X}_2)$
    where the reference state is taken from the fitted degree-0 coefficient (and for the
    driver, from its time mean).

    Args:
        ds:
            Dataset containing the driver and the variables to be fitted.
            Variables to be fitted must include `time_dim` in their dimensions.
        key_use:
            Name of the driver variable used as the predictor (default `'temp_surf'`).
            The predictor used in fitting is
            `x = ds[key_use] - ds[key_use].mean(dim=time_dim)`.
        time_dim:
            Name of the time dimension (default `'time'`).
        deg:
            Polynomial degree used in `polyfit_phase_xr` (default `2`).
            This is passed through to `polyfit_phase_xr` and controls the number of
            fitted coefficients/terms.
        include_phase:
            Passed to `polyfit_phase_xr`/`polyval_phase_xr`. Intended for fits that
            separate in-phase and out-of-phase (or similar) components.
        include_fourier:
            Passed to `polyfit_phase_xr`/`polyval_phase_xr`. Intended for augmenting
            the fit with Fourier components (e.g., seasonal harmonics).
        get_nl:
            If True, additionally fit non-linear mechanism terms (squares and pairwise
            products) using the same predictor `x`.
        error_kind:
            Summary statistic used by `get_error` to reduce the mismatch over `time_dim`.
            One of `'mean'`, `'median'`, or `'max'`.
        error_norm:
            Whether `get_error` returns a normalized error (implementation defined in
            `get_error`).
        error_norm_dim:
            Dimension(s) used for normalization inside `get_error` (implementation defined
            in `get_error`).

    Returns:
        var_ref:
            Dataset of reference (baseline) states for each variable.
            For fitted variables, this is taken as the degree-0 coefficient from the fit
            (`params[key].sel(deg='0')`); for the driver (`key_use`) it is the time mean;
            and for variables without `time_dim` it is copied directly from `ds`.
        params:
            Dataset of fitted parameters for each variable produced by `polyfit_phase_xr`.
            Includes additional keys for non-linear mechanisms when `get_nl=True`:
            `name_square(key)` and `name_nl(var1, var2)`.
        var_empirical:
            Dataset of empirical reconstructions for each fitted variable (and mechanism term),
            evaluated at the full time series of the driver anomaly `x`.
        error:
            Dataset of per-variable error metrics comparing truth and empirical reconstructions,
            computed with `get_error`.

    Notes:
        - Variables are only fitted if they are not `key_use` and they include `time_dim`
          in their dimensions.
        - The list/order of variables used in the pairwise non-linear fits is controlled by
          `style_map_var` (assumed to be defined externally); combinations are computed with
          `itertools.combinations(style_map_var, 2)`.
        - For non-linear mechanism fits, the constant coefficient is forced to zero
          (`coef0=0`) to match a Taylor-series-style interpretation around the reference state.

    """
    # Check has expected keys
    if not all(key in ds for key in style_map_var):
        print(f"Not all keys in {list(style_map_var.keys())} are present in the dataset.")

    params = {}
    var_ref = {}
    var_empirical = {}
    error = {}
    x = ds[key_use] - ds[key_use].mean(dim=time_dim)
    for key in ds:
        if (key == key_use) or (time_dim not in ds[key].dims):
            continue
        params[key] = polyfit_phase_xr(x, ds[key], deg=deg,
                                       include_phase=include_phase, include_fourier=include_fourier)
        var_empirical[key] = polyval_phase_xr(params[key], x)
        error[key] = get_error(ds[key], var_empirical[key], error_kind, error_norm, time_dim, error_norm_dim)
        var_ref[key] = params[key].sel(deg='0', drop=True)

    # Reference state is where all empirical params are zero and temp_surf=temp_surf_av; equivalent to deg=0 value here.
    var_ref[key_use] = ds[key_use].mean(dim=time_dim)
    # Add variables in ds with no time dimension
    for key in ds:
        if time_dim not in ds[key].dims:
            var_ref[key] = ds[key]
    var_ref = xr.Dataset(var_ref)

    if get_nl:
        # Constrained fitting for nl terms to keep const coef as zero
        coef_fix = [None for _ in range(deg + 2)]
        coef_fix[-1] = 0  # enforce const (deg=0) of 0
        # Square mechanism
        for key in ds:
            if (key == key_use) or (time_dim not in ds[key].dims):
                continue
            # Do fitting for square cont separately - more complicated that just squaring the individual params
            # Force const to be 0, as expected from taylor series point of view
            # Note that not using ds_av but the deg=0 coef found which is ds_empirical_ref
            params[name_square(key)] = polyfit_phase_xr(x, (ds[key] - var_ref[key]) ** 2, deg=deg,
                                                        include_phase=include_phase,
                                                        include_fourier=include_fourier,
                                                        coef_fix=coef_fix)
            var_empirical[name_square(key)] = polyval_phase_xr(params[name_square(key)], x)
            error[name_square(key)] = get_error((ds[key] - var_ref[key]) ** 2, var_empirical[name_square(key)],
                                                error_kind, error_norm, time_dim, error_norm_dim)

        # Combination of mechanisms
        # Order important hence use style_map_var
        for var1, var2 in itertools.combinations(style_map_var, 2):
            # Force const to be 0, as expected from taylor series point of view
            var = (ds[var1] - var_ref[var1]) * (ds[var2] - var_ref[var2])
            params[name_nl(var1, var2)] = polyfit_phase_xr(x, var,
                                                           deg=deg, include_phase=include_phase,
                                                           include_fourier=include_fourier, coef_fix=coef_fix)
            var_empirical[name_nl(var1, var2)] = polyval_phase_xr(params[name_nl(var1, var2)], x)
            error[name_nl(var1, var2)] = get_error(var, var_empirical[name_nl(var1, var2)],
                                                   error_kind, error_norm, time_dim, error_norm_dim)
    params = xr.Dataset(params)
    var_empirical = xr.Dataset(var_empirical)
    error = xr.Dataset(error)
    return var_ref, params, var_empirical, error


### Extrema Stuff

get_temp_extrema_numerical_xr = wrap_with_apply_ufunc(get_temp_extrema_numerical, input_core_dims=[['time'], ['time']],
                                                      output_core_dims=[[], [], [], []])


def get_temp_extrema_theory_xr(sw_amp: xr.DataArray, heat_capacity: xr.DataArray,
                               param_coefs: xr.DataArray, numerical: bool = False,
                               n_year_days: int = 360
                               ) -> Tuple[xr.DataArray, xr.Dataset, xr.Dataset]:
    """
    Xarray wrapper for `get_temp_extrema_theory` that computes the timing and amplitude of temperature extrema
    from harmonic shortwave forcing and fitted feedback parameters on arbitrary xarray dimensions.

    This function operates on gridpoint-wise inputs produced by `fourier_series_xr` and `polyfit_phase_xr`,
    calls `get_temp_extrema_theory` via `apply_ufunc`, and then concatenates the results for the first and
    second extrema (e.g. max and min) along a new `type` dimension of length two.

    Args:
        sw_amp:
            Amplitude of the shortwave Fourier series, output of `fourier_series_xr` with a `harmonic`
            dimension. Must contain exactly three entries `[0, 1, 2]` corresponding to the mean and the
            first two harmonics of $SW^{\\downarrow}$; only harmonics 1 and 2 are used.
        heat_capacity:
            Surface heat capacity $C$ as an xarray DataArray, broadcastable to `sw_amp` over all
            non-`harmonic` dimensions. Units are typically $JK^{-1}m^{-2}$.
        param_coefs:
            Polynomial-fit coefficients for the feedback parameters, output of `polyfit_phase_xr` with a
            `deg` dimension. Expected degrees are `1`, `phase`, `2`, `cos`, and `sin`, which are mapped to
            the dimensional parameters $\\lambda$, $\\lambda_{phase}$, $\\lambda_{sq}$, $\\Lambda_{cos}$,
            and $\\Lambda_{sin}$ respectively.
        numerical:
            If `False`, `get_temp_extrema_theory` is called in analytic mode to use pre-derived expressions
            for the timing and amplitude coefficients of the extrema. If `True`, the extrema are obtained
            numerically by solving $\\partial T/\\partial\\Delta = 0$ at each gridpoint.
        n_year_days:
            Number of days in one period $\\mathcal{T}$ (e.g. 360), used to define the annual frequency
            $f = 1/\\mathcal{T}$ inside `get_temp_extrema_theory`.

    Returns:
        answer:
            DataArray with dimensions `approx`, `metric`, and `type` plus the non-core dimensions of `sw_amp`.
            The `approx` coordinate labels the approximation level:
            for `numerical=False`, it contains two slices,
            `approx='linear'` for the linear-order contribution and `approx='nl'` for the total
            (linear + nonlinear) contribution; for `numerical=True`, it additionally includes
            `approx=None`, which holds the numerically evaluated “exact” extrema
            (i.e. including the residual nonlinear contribution stored in `cont['nl_residual']`).
        cont:
            Dataset with dimensions `metric` and `type` plus the non-core dimensions of `sw_amp`, holding the
            decomposition of both extremum timing and amplitude into contributions from individual mechanisms
            and their nonlinear combinations. For `metric='phase'`, variables mirror `info_cont` from
            `get_temp_extrema_theory`; for `metric='amplitude'`, variables mirror `info_cont_amp`.
        coef:
            Dataset with the same dimensions as `cont`, containing the analytic coefficients that multiply
            the dimensionless parameters in `cont`. For `metric='phase'`, variables correspond to the `coef`
            dictionary from `get_temp_extrema_theory`; for `metric='amplitude'`, they correspond to `coef_amp`.
    """
    if not np.array_equal(sw_amp.harmonic, np.arange(3)):
        raise ValueError('sw_amp.harmonic must be [0, 1, 2] not {}'.format(sw_amp.harmonic))
    sw_amp1_sign = np.unique(np.sign(sw_amp.sel(harmonic=1)))
    if sw_amp1_sign.size != 1:
        raise ValueError('More than one sign of sw_amp.sel(harmonic=1) provided.')
    type = xr.DataArray(['min', 'max'] if sw_amp1_sign[0] == -1 else ['max', 'min'],
                        name='type', dims='type')       # Northern Hemisphere - minima occurs first
    _get_temp_extrema_theory = wrap_with_apply_ufunc(get_temp_extrema_theory,
                                                     input_core_dims=[[]] * 8,
                                                     output_core_dims=[[]] * 8)
    phase_linear = []
    phase_nl = []
    cont_phase = []
    coef_phase = []
    amp_linear = []
    amp_nl = []
    cont_amp = []
    coef_amp = []
    for i in range(2):
        var = _get_temp_extrema_theory(heat_capacity, sw_amp.sel(harmonic=1), sw_amp.sel(harmonic=2),
                                       param_coefs.sel(deg='1'), param_coefs.sel(deg='phase'), param_coefs.sel(deg='2'),
                                       param_coefs.sel(deg='cos'), param_coefs.sel(deg='sin'), numerical=numerical,
                                       n_year_days=n_year_days, extrema_ind=i+1)
        phase_linear.append(var[0])
        phase_nl.append(var[1])
        cont_phase.append(xr.Dataset(convert_ds_of_dicts(var[2], [0], 'dim_name')
                            ).isel(dim_name=0, drop=True))
        coef_phase.append(xr.Dataset(convert_ds_of_dicts(var[3], [0], 'dim_name')
                            ).isel(dim_name=0, drop=True))
        amp_linear.append(var[4])
        amp_nl.append(var[5])
        cont_amp.append(xr.Dataset(convert_ds_of_dicts(var[6], [0], 'dim_name')
                            ).isel(dim_name=0, drop=True))
        coef_amp.append(xr.Dataset(convert_ds_of_dicts(var[7], [0], 'dim_name')
                            ).isel(dim_name=0, drop=True))
    phase_linear = xr.concat(phase_linear, dim=type)
    phase_nl = xr.concat(phase_nl, dim=type)
    cont_phase = xr.concat(cont_phase, dim=type)
    coef_phase = xr.concat(coef_phase, dim=type)
    amp_linear = xr.concat(amp_linear, dim=type)
    amp_nl = xr.concat(amp_nl, dim=type)
    cont_amp = xr.concat(cont_amp, dim=type)
    coef_amp = xr.concat(coef_amp, dim=type)

    # Concatenate phase and amplitude together
    metric = xr.DataArray(['phase', 'amplitude'],
                          name='metric', dims='metric')
    answer_linear = xr.concat([phase_linear, amp_linear], dim=metric)
    answer_nl = xr.concat([phase_nl, amp_nl], dim=metric)
    cont = xr.concat([cont_phase, cont_amp], dim=metric)
    coef = xr.concat([coef_phase, coef_amp], dim=metric)

    # Concat answer in approx_level dimension
    if numerical:
        approx_lev = xr.DataArray(['linear', 'nl', None],
                                  name='approx', dims='approx')
        answer_exact = answer_nl + cont['nl_residual']
        answer = xr.concat([answer_linear, answer_nl, answer_exact], dim=approx_lev)
    else:
        approx_lev = xr.DataArray(['linear', 'nl'], name='approx', dims='approx')
        answer = xr.concat([answer_linear, answer_nl], dim=approx_lev)
    return answer, cont, coef


def get_phase_amp_relative_harmonic1(time: xr.DataArray, temp_anom: xr.DataArray, sw_amp1: xr.DataArray,
                                     heat_capacity: xr.DataArray, phase_h1: Optional[xr.DataArray] = None,
                                     amp_h1: Optional[xr.DataArray] = None,
                                     lambda_const: Optional[xr.DataArray]=None,
                                     lambda_phase: Optional[xr.DataArray]=None, extrema_type: Literal['min', 'max']='min',
                                     day_seconds=86400, n_year_days=360):
    """
    Compute the phase and amplitude of temperature extrema relative to the first harmonic extremum.

    This returns the dimensionless phase $y = \\sin(2\\pi f\\Delta)$, where
    $\\Delta = t_{extrema} - t_{extrema,1}$ is the time difference between the full-solution extremum
    and the extremum of the first harmonic, and the relative amplitude
    $A = T_{extrema}/T_{extrema,1}$, both evaluated pointwise in xarray space.

    Args:
        time:
            Time coordinate in days as an xarray DataArray, typically something like
            `time = np.arange(n_year_days)`. Assumed periodic with period one year.
        temp_anom:
            Surface temperature anomaly $T_s - \\overline{T}_s$ as an xarray DataArray, evaluated at the
            extremum of interest and broadcastable over `time`. This is the numerator of the amplitude
            ratio $A = T_{extrema}/T_{extrema,1}$.
        sw_amp1:
            Amplitude of the first harmonic of downward shortwave radiation at the surface, $F_1$, as
            an xarray DataArray, with units of $Wm^{-2}$. Used together with `heat_capacity`,
            `lambda_const`, and `lambda_phase` to reconstruct the first-harmonic temperature response
            if `phase_h1`/`amp_h1` are not supplied.
        heat_capacity:
            Surface heat capacity $C$ as an xarray DataArray, with units of $JK^{-1}m^{-2}$.
        phase_h1:
            Optional precomputed phase of the first harmonic extremum $\\phi_1$ as an xarray DataArray.
            If provided, the time of the first harmonic extremum is taken as
            $t_{extrema,1} = \\phi_1 / (2\\pi f)$. If `None`, it is diagnosed from
            $(C, \\lambda, \\lambda_{phase})$ using the analytic expression.
        amp_h1:
            Optional precomputed amplitude of the first harmonic temperature response $T_1$ as an
            xarray DataArray. If provided, it is used directly in the amplitude ratio
            $A = T_{extrema}/T_{extrema,1}$. If `None`, it is diagnosed from `get_temp_shift_params`
            with all higher-order and empirical parameters set to zero.
        lambda_const:
            Optional linear feedback parameter $\\lambda$ as an xarray DataArray, used to diagnose
            the first-harmonic phase and amplitude when `phase_h1` and `amp_h1` are not provided.
            Required if `phase_h1` is `None`.
        lambda_phase:
            Optional phase-lag feedback parameter $\\lambda_{phase}$ as an xarray DataArray, used
            when diagnosing the first-harmonic phase and amplitude internally. Required if
            `phase_h1` is `None`.
        extrema_type:
            Specifies which extremum of the first harmonic to treat as the reference. Use `'min'`
            for the minimum (default) or `'max'` for the maximum; for `'max'` the reference time
            is shifted by half a period, $1/(2f)$.
        day_seconds:
            Length of one day in seconds, used to convert `time` (in days) to seconds and to define
            the frequency $f = 1/(\\mathcal{T})$ with $\\mathcal{T} = n_{year\\_days} \\times day\\_seconds$.
        n_year_days:
            Number of days in one period $\\mathcal{T}$ (e.g. 360), used together with `day_seconds`
            to define the annual frequency $f$.

    Returns:
        y:
            Dimensionless phase shift $y = \\sin(2\\pi f\\Delta)$ as an xarray DataArray, where
            $\\Delta = t_{extrema} - t_{extrema,1}$ is the time offset of the full-solution extremum
            relative to the first-harmonic extremum.
        amp_harmonic1:
            Amplitude ratio $A = T_{extrema}/T_{extrema,1}$ as an xarray DataArray, giving the
            extremum temperature relative to the first-harmonic extremum amplitude at each point.
    """
    # Returns y=sin(2\pi f\Delta) where \Delta=t_extrema - t_extrema_1 and A=T_extrema/T_extrema_1
    # I.e. the time and amplitude of extrema relative to first harmonic
    f = 1/(n_year_days * day_seconds)
    if phase_h1 is not None:
        time_harmonic1 = phase_h1 / (2 * np.pi * f)
    else:
        x = 2*np.pi*f*heat_capacity/lambda_const
        lambda_phase_dim = get_param_dimensionless(lambda_phase, heat_capacity=heat_capacity, n_year_days=n_year_days)
        x1 = x * (1-lambda_phase_dim)
        time_harmonic1 = np.arctan(x1) / (2 * np.pi * f)
        sw_amp2 = 10  # can be anything, not used, but zero gives error
        amp_h1 = get_temp_shift_params(heat_capacity, sw_amp1, sw_amp2, lambda_const, lambda_phase,
                                       0, 0, 0, n_year_days, day_seconds)[0]
    if np.max(sw_amp1) > 0:
        raise ValueError('sw_amp1>0 so Southern Hemisphere but this only works for Northern')
    if extrema_type == 'max':
        time_harmonic1 = time_harmonic1 + 1/(2*f)
    time_shift = time * day_seconds - time_harmonic1
    y = np.sin(2*np.pi*f*time_shift)
    amp_harmonic1 = np.abs(temp_anom / amp_h1)
    return y, amp_harmonic1


def add_time_shift_second_ax(ax: plt.Axes, n_year_days: int, ex_label: Literal['ex', 'min', 'max']='ex'):
    """
    Add a secondary y-axis showing extremum time shift in days corresponding to $y = \\sin(2\\pi f\\Delta)$.

    The primary y-axis is assumed to show the dimensionless phase variable $y$. This function adds a
    right-hand y-axis whose tick labels are the associated time shifts $\\Delta$ in days, computed as
    $\\Delta = \\arcsin(y)/(2\\pi f)$ with $f = 1/\\mathcal{T}$ and $\\mathcal{T} = n_{year\\_days}$ days.

    Args:
        ax:
            Matplotlib Axes on which the primary y-axis represents the dimensionless phase variable $y$.
        n_year_days:
            Number of days in one period $\\mathcal{T}$ (e.g. 360). Used to convert $y$ to a time shift
            in days via $\\Delta = \\arcsin(y)/(2\\pi) \\times n_{year\\_days}$.
        ex_label:
            Label string inserted into the LaTeX subscript of $\\Delta$ on the secondary y-axis. Use
            `'ex'`, `'min'`, or `'max'` to get labels like $\\Delta_{\\text{ex}}$, $\\Delta_{\\text{min}}$,
            or $\\Delta_{\\text{max}}$.

    Returns:
        ax1_right:
            The newly created secondary y-axis (right-hand side) with tick labels in units of days.
    """
    ax1_right = ax.secondary_yaxis(
        "right",
        functions=(lambda y: y, lambda y: y),
    )
    ax1_right.yaxis.set_major_formatter(
        FuncFormatter(
            lambda y, pos: f"{np.arcsin(y) / 2 / np.pi * n_year_days:.0f}"
        )
    )
    ax1_right.set_ylabel('$\Delta_{\\text{' + ex_label + '}}$ [days]')
    return ax1_right
