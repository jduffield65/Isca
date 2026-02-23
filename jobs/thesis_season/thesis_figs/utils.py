import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import inspect
from typing import Union, Literal, Optional, Tuple, List

from isca_tools.thesis.surface_flux_taylor import get_temp_rad, reconstruct_lh, reconstruct_sh, reconstruct_lw
from isca_tools.utils import numerical
from tqdm.notebook import tqdm

from isca_tools.thesis.surface_energy_budget import get_temp_extrema_numerical, get_temp_fourier_analytic, \
    get_temp_fourier_analytic2
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
    "linear_phase": ("C0", "--", "Linear phase"),
    "square_phase": ("C1", "-.", "Square phase"),
    "square_phase+": ("C1", "-", "Square phase+"),
    "poly10_phase+": ("C2", "-", "Poly10 phase+"),
    "lw": ("C1", "-", "$\\text{LW}^{\\uparrow}_{\\text{net}}$"),
    "lh": ("C0", ":", "LH$^{\\uparrow}$"),
    "sh": ("C2", ":", "SH$^{\\uparrow}$"),
    "net": ("k", "-", "Sum"),
}

style_map_var = {'temp_surf': ("C1", "-", "$T_s$", "K"),
                 'temp_diseqb': ("C3", "-", "$T_{dq}$", "K"),
                 'temp_diseqb_r': ("C0", "-", "$T_{dq_r}$", "K"),
                 'rh_atm': ('C0', "-", "$r_a$", "Unitless"),
                 'w_atm': ('C2', '-', '$U_a$', "ms$^{-1}$"),
                 'p_surf': ('C4', '-', '$p_s$', "Pa")}

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
            lat_min: float = lat_min, lat_max: float = lat_max) -> xr.Dataset:
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
    if depth == 5:
        exp_name = [exp_dir(5, reduced_evap)]
    elif depth == 20:
        exp_name = [exp_dir(20)]
    elif depth == 'both':
        exp_name = [exp_dir(5, reduced_evap), exp_dir(20)]
    else:
        raise ValueError('Depth must be either 5 or 20 or "both"')


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

get_temp_extrema_numerical_xr = wrap_with_apply_ufunc(get_temp_extrema_numerical, input_core_dims=[['time'], ['time']],
                                                      output_core_dims=[[], [], [], []])


# Might need to do different version when include fourier coefs, as more outputs
def polyfit_phase_xr(x: xr.DataArray, y: xr.DataArray,
                     deg: int, time: Optional[xr.DataArray] = None, time_start: Optional[float] = None,
                     time_end: Optional[float] = None,
                     deg_phase_calc: int = 10, resample: bool = resample,
                     include_phase: bool = True, include_fourier: bool = False,
                     integ_method: str = 'spline', coef0: Optional[float]=None) -> xr. DataArray:
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
        coef0: Option to fix the constant i.e. `deg=0` coefficient to be this value.

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
                             integ_method=integ_method, pad_coefs_phase=True, coef0=coef0)
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


def get_error(x: xr.DataArray, x_approx: xr.DataArray, kind: Literal['mean', 'median', 'max'] = "mean",
              norm: bool = True, dim: Union[str, list] = "time",
              norm_dim: Optional[Union[str, list]] = 'lat') -> xr.DataArray:
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

    Returns:
        DataArray of the reduced absolute error (and optionally normalized), with
        remaining dimensions preserved.

    Raises:
        ValueError: If `kind` is not one of {"mean", "median", "max"}.
    """
    raise_if_common_dims_not_identical(x, x_approx, name_y='x_approx')
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
            scale = scale.mean(dim=norm_dim)
        out = out / scale

    return out


def reconstruct_flux_xr(ds: xr.Dataset, ds_ref: xr.Dataset,
                        flux_name: Literal['lh', 'sh', 'lw'] = 'lh',
                        numerical: bool = False,
                        time_dim: str = 'time', ) -> Tuple[xr.DataArray, xr.DataArray, xr.DataArray, xr.Dataset]:
    reconstruct_flux = {'lh': reconstruct_lh, 'sh': reconstruct_sh, 'lw': reconstruct_lw}[flux_name]
    arg_names = list(inspect.signature(reconstruct_flux).parameters.keys())
    arg_names = [key for key in arg_names if key!='numerical']      # treat numerical as kwarg
    input_core_dims = [[] if (('_ref' in arg) or (arg in ['sigma_atm'])) else [time_dim] for arg in
                       arg_names]
    output_core_dims = [[], [time_dim], [time_dim], []]
    reconstruct_flux_wrap = wrap_with_apply_ufunc(reconstruct_flux, input_core_dims=input_core_dims,
                                                  output_core_dims=output_core_dims)

    if flux_name == 'lh':
        # Add time dimension to drag and evap, as don't have initially
        drag_coef = ds.temp_surf*0 + ds_ref.drag_coef
        evap_prefactor = ds.temp_surf*0 + ds_ref.evap_prefactor
        flux_ref, flux_anom_linear, flux_anom_nl, info_cont = \
            reconstruct_flux_wrap(ds_ref.temp_surf, ds_ref.temp_diseqb, ds_ref.rh_atm, ds_ref.w_atm, ds_ref.drag_coef,
                                  ds_ref.p_surf, ds_ref.sigma_atm, ds_ref.evap_prefactor,
                                  ds.temp_surf, ds.temp_diseqb, ds.rh_atm, ds.w_atm, drag_coef, ds.p_surf,
                                  evap_prefactor, numerical=numerical)
    elif flux_name == 'sh':
        # Add time dimension to drag, as don't have initially
        drag_coef = ds.temp_surf*0 + ds_ref.drag_coef
        flux_ref, flux_anom_linear, flux_anom_nl, info_cont = \
            reconstruct_flux_wrap(ds_ref.temp_surf, ds_ref.temp_diseqb, ds_ref.w_atm, ds_ref.drag_coef,
                                  ds_ref.p_surf, ds_ref.sigma_atm, ds.temp_surf, ds.temp_diseqb,
                                  ds.w_atm, drag_coef, ds.p_surf, numerical=numerical)
    elif flux_name == 'lw':
        # Add time dimension to odp_surf, as don't have initially
        odp_surf = ds.temp_surf*0 + ds_ref.odp_surf
        flux_ref, flux_anom_linear, flux_anom_nl, info_cont = \
            reconstruct_flux_wrap(ds_ref.temp_surf, ds_ref.temp_diseqb, ds_ref.temp_diseqb_r, ds_ref.odp_surf,
                                  ds.temp_surf, ds.temp_diseqb, ds.temp_diseqb_r, odp_surf, numerical=numerical)
    else:
        raise ValueError(f'Unknown flux_name="{flux_name}". Must be one of "lh", "sh", "lw".')
    info_cont = xr.Dataset(convert_ds_of_dicts(info_cont, ds.time, 'time'))
    return flux_ref, flux_anom_linear, flux_anom_nl, info_cont
