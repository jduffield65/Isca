import xarray as xr
import numpy as np
import os
from typing import List, Optional, Literal
from tqdm import tqdm
import warnings

from isca_tools.thesis.season_annual_harmonic import get_phase_amp
from isca_tools.thesis.surface_flux_taylor_2layer import get_p_eff, get_sensible_heat
from isca_tools.utils.base import mass_weighted_vertical_integral
from isca_tools.utils.moist_physics import sphum_sat
from isca_tools.utils.radiation import get_heat_capacity, opd_lw_gray, frierson_atmospheric_heating, get_frierson_sw_abs
from isca_tools import load_dataset, load_namelist
from isca_tools.utils.constants import c_p_ocean, rho_ocean, c_p, L_v, g
from isca_tools.utils.xarray import select_coord_window, periodic_rolling_mean
from isca_tools.utils import annual_mean
from .xr_funcs import get_sw_abs_amp_xr, get_temp_from_sphum_sat_xr, spline_deriv_periodic_xr, get_fourier_fit_xr


# Info for saving directories
save_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'processed_data')
complevel = 4


var_keep = ['temp', 'ps', 'sphum', 'olr', 'swdn_toa', 'swdn_sfc', 'lwdn_sfc', 'lwup_sfc', 'flux_t',
            'flux_lhe', 't_surf', 'precipitation', 'convflag']  # just the fluxes, no variables
lat_target = 43
smooth_n_days = 49  # default smoothing window in days
day_seconds = 86400
width = {'one_col': 3.2, 'two_col': 5.5}  # width in inches
label_temp = r'Mean surface temperature, $\overline{T}_s$ [K]'

# Params required for empirical fit, and those not fit for advect and single column simulations
# Note sw_abs is not fit empirically so unlikely to change with warming
params_all = ['lambda_const', 'B', 'coef_phase_olr', 'lambda_a', 'coef_phase_a', 'lambda_adv', 'coef_phase_adv',
              'mu', 'sw_abs', 'coef_amp_col', 'coef_phase_col']     # Order matters for error.ipynb figures
exclude_params = {'advect': ['coef_phase_col'],
                  'column': ['lambda_adv', 'coef_phase_adv', 'coef_phase_olr', 'coef_phase_a']}


def load_ds(exp_name: str, exp_dir: str, var_keep: List = var_keep,
            lat_min: float = 30, lat_max: float = 90,
            first_month_file: Optional[int] = None,
            verbose: bool = False, lat_target: Optional[float] = None,
            n_lat: int = 0) -> xr.Dataset:
    r"""Load experiment data and derive two-layer energy-budget quantities.

    Selects the requested variables and latitudes, loads the data into memory,
    and reads experiment parameters from the namelist. Column quantities are
    calculated before retaining only the lowest atmospheric model level.

    Args:
        exp_name: Experiment directory name within `exp_dir`.
        exp_dir: Parent directory containing the experiment.
        var_keep: Variables to retain from the loaded dataset. Must include
            `temp`, `sphum`, `ps`, `t_surf`, `precipitation`, and `flux_lhe`.
        lat_min: Lower latitude bound in degrees, used when `lat_target` is
            None.
        lat_max: Upper latitude bound in degrees, used when `lat_target` is
            None.
        first_month_file: First monthly file to load. Passed directly to
            `load_dataset`; None uses that function's default behavior.
        verbose: Whether to display progress while computing column
            temperature, specific humidity, and relative humidity.
        lat_target: Target latitude in degrees for selection with
            `select_coord_window`. If provided, overrides `lat_min` and
            `lat_max`.
        n_lat: Latitude-window parameter passed to `select_coord_window`
            when `lat_target` is provided.

    Returns:
        Dataset containing the selected data at the lowest model level,
        together with the following derived variables:

        - `odp_surf`: Latitude-dependent surface longwave optical depth.
        - `temp_col`: Mass-weighted vertical integral of temperature.
        - `sphum_col`: Mass-weighted vertical integral of specific humidity.
        - `rh_col`: Ratio of the specific-humidity integral to the
          saturation-specific-humidity integral.
        - `p_integ_calc`: Pressure span between the first and last full
          sigma levels, in Pa.
        - `rh_atm`: Lowest-level relative humidity expressed as a fraction.
        - `precip_minus_evap`: Precipitation minus evaporation, with
          evaporation calculated as `flux_lhe / L_v`.

        The variables `temp`, `t_surf`, `ps`, `lev_sigma`, and `sphum` are
        renamed to `temp_atm`, `temp_surf`, `p_surf`, `sigma_atm`, and
        `q_atm`, respectively. Attributes include `albedo`, `depth`,
        `p_ref`, `heat_cap_surf`, `odp`, and `atm_abs`.

    Notes:
        Uses the helpers `load_dataset`, `select_coord_window`,
        `load_namelist`, `get_heat_capacity`, `opd_lw_gray`,
        `mass_weighted_vertical_integral`, and `sphum_sat`.

        Full sigma levels are calculated as adjacent averages of the
        namelist's half-level `bk` values. Model-level pressure is then
        calculated as surface pressure multiplied by full-level sigma.

        All vertical integrals explicitly use Simpson's method. The
        column relative humidity is a ratio of integrals, not a vertical
        average of local relative humidity.

        This function supplies quantities for the empirical two-layer
        energy-budget approximation. Additional quantities, such as
        `temp_col_sphum`, `temp_rad_surf`, and `temp_rad_atm`, would be
        needed for the analytical approximation.
    """
    # Load all info required for empirical approximation of 2 layer energy budget
    # Will need to update later to add that required analytically such as
    # temp_col_sphum, temp_rad_surf, temp_rad_atm
    exp_path = os.path.join(exp_dir, exp_name)

    ds = load_dataset(exp_path, first_month_file=first_month_file)[var_keep]
    if lat_target is None:
        ds = ds.sel(lat=slice(lat_min, lat_max))
    else:
        ds = select_coord_window(ds, lat_target, 'lat', n_lat)
    ds = ds.load()

    # Load info from namelist and add to attributes
    namelist = load_namelist(exp_path)
    sigma_levels_half = np.asarray(namelist['vert_coordinate_nml']['bk'])
    sigma_levels_full = np.convolve(sigma_levels_half, np.ones(2) / 2, 'valid')
    ds['lev_sigma'] = (ds.pfull * 0 + sigma_levels_full).squeeze()
    ds.attrs['albedo'] = namelist['mixed_layer_nml']['albedo_value']
    ds.attrs['depth'] = namelist['mixed_layer_nml']['depth']
    try:
        # Ref pressure used for optical depth calculations
        ds.attrs['p_ref'] = namelist['constants_nml']['pstd_mks']
    except KeyError:
        ds.attrs['p_ref'] = 101325.0  # default value
    ds.attrs['heat_cap_surf'] = get_heat_capacity(c_p_ocean, rho_ocean, ds.attrs['depth'])

    # Get longwave optical depth at surface - is a function of latitude
    odp_info = {'ir_tau_eq': 6, 'ir_tau_pole': 1.5, 'linear_tau': 0.1, 'wv_exponent': 4,
                'odp': 1, 'atm_abs': 0}  # default vals
    for key in odp_info:  # If provided, update
        if key in namelist['two_stream_gray_rad_nml']:
            odp_info[key] = namelist['two_stream_gray_rad_nml'][key]
    ds.attrs['odp'] = odp_info['odp']
    ds.attrs['atm_abs'] = odp_info['atm_abs']
    ds['odp_surf'] = opd_lw_gray(ds.lat, kappa=ds.odp, tau_eq=odp_info['ir_tau_eq'],
                                 tau_pole=odp_info['ir_tau_pole'], frac_linear=odp_info['linear_tau'],
                                 k_exponent=odp_info['wv_exponent'])  # optical depth as function of latitude

    # Get column quantities - very important to use simpson integral method
    if verbose:
        pbar = tqdm(total=3, desc="Computing column temp, sphum, and rh")
    p_lev = ds.ps * ds.lev_sigma
    ds['temp_col'] = mass_weighted_vertical_integral(ds.temp, p_lev, 'pfull', simpson_method=True)
    if verbose:
        pbar.update()
    ds['sphum_col'] = mass_weighted_vertical_integral(ds.sphum, p_lev, 'pfull', simpson_method=True)
    if verbose:
        pbar.update()
    ds['rh_col'] = ds['sphum_col'] / mass_weighted_vertical_integral(sphum_sat(ds.temp, p_lev),
                                                                     p_lev, 'pfull', simpson_method=True)
    if verbose:
        pbar.update()

    # Only keep the lowest model level
    ds = ds.sel(pfull=np.inf, method='nearest')
    ds['p_integ_calc'] = ds.ps * (sigma_levels_full[-1] - sigma_levels_full[0])  # keep track of p range for integration

    # Rename temp vars to used in surface flux functions
    ds = ds.rename_vars({'temp': 'temp_atm', 't_surf': 'temp_surf', 'ps': 'p_surf',
                         'lev_sigma': 'sigma_atm', 'sphum': 'q_atm'})
    ds['rh_atm'] = ds.q_atm / sphum_sat(ds.temp_atm, ds.p_surf * ds.sigma_atm)
    ds['precip_minus_evap'] = ds.precipitation - ds.flux_lhe / L_v
    return ds


def process_ds(ds: xr.Dataset, smooth_n_days: int = smooth_n_days,
               smooth_time: Literal['end', 'start'] = 'end') -> xr.Dataset:
    r"""Process simulation output for annual-harmonic energy-budget analysis.

    The dataset is zonally and annually averaged, then supplemented with
    thermodynamic, radiative, and energy-budget variables. In particular,
    `mse_tend_atmos` is the diagnosed atmospheric moist-static-energy tendency.

    `flux_atmos` contains the explicitly diagnosed right-hand-side flux terms
    in the atmospheric energy budget, while `adv_atmos` is the residual
    required to close the budget:

    $$
    \mathrm{mse\_tend\_atmos} =
    \mathrm{flux\_atmos} + \mathrm{adv\_atmos}.
    $$

    Args:
        ds: Dataset containing the raw model output over longitude and time.
        smooth_n_days: Number of days used to smooth the time series before
            annual averaging.
        smooth_time: Whether smoothing is aligned to the `'start'` or `'end'`
            of each averaging window.

    Returns:
        Processed dataset containing zonal and annual means, effective pressure,
        column thermodynamic variables, absorbed shortwave radiation,
        atmospheric energy-budget terms, and first-harmonic temperature and
        shortwave coefficients.
    """
    # Add wind multiplied by drag coef extracted from sensible heat - use to find lambda_const
    flux_t_norm = get_sensible_heat(ds.temp_surf, ds.temp_atm, 1, 1, ds.p_surf,
                                    ds.p_surf * ds.sigma_atm)
    # take av over time as want single value for each sim/location - median to avoid outliers.
    ds['wind_drag_av'] = (ds.flux_t / flux_t_norm).median(dim='time')

    ds = get_annual_zonal_mean(ds, smooth_n_days=smooth_n_days, smooth_time=smooth_time)
    ds['p_eff'] = get_p_eff(ds.p_surf.mean(dim='time'))
    ds['temp_col_sphum'] = get_temp_from_sphum_sat_xr(ds.sphum_col / ds.rh_col, ds.p_eff)
    ds['sw_abs_harmonic'] = get_sw_abs_amp_xr(ds.swdn_sfc, ds.swdn_toa, ds.time, albedo=ds.albedo)
    ds['sw_abs_analytic'] = get_frierson_sw_abs(ds.atm_abs, ds.p_surf.mean(dim='time'), p_ref=ds.p_ref,
                                                albedo=ds.albedo)
    ds['sw_abs'] = ds['sw_abs_analytic']  # use analytic one as simpler, even though difference if p_surf not constant
    # p_surf smaller in summer so sw_abs_analytic > sw_abs_harmonic

    # Atmospheric energy budget components: mse_tend = flux + adv
    ds['mse_tend_atmos'] = spline_deriv_periodic_xr(ds.time * day_seconds,
                                                    (c_p * ds.temp_col + L_v * ds.sphum_col) * ds.p_integ_calc / g)
    ds['sphum_col_tend'] = spline_deriv_periodic_xr(ds.time * day_seconds,
                                                    ds.sphum_col * ds.p_integ_calc / g)
    ds['flux_atmos'] = frierson_atmospheric_heating(ds, ds.albedo) + ds.flux_t + ds.flux_lhe
    ds['adv_atmos'] = ds['mse_tend_atmos'] - ds['flux_atmos']
    # Advection term such that adv_atmos_moist + L_v(E-P) equals moisture component of MSE tendency
    ds['adv_atmos_moist'] = L_v * (ds['sphum_col_tend'] + ds['precip_minus_evap'])
    ds['adv_atmos_dry'] = ds['adv_atmos'] - ds['adv_atmos_moist']

    # Surface fluxes excluding SW
    ds['flux_surf'] = ds.flux_t + ds.flux_lhe - ds.lwdn_sfc + ds.lwup_sfc

    # Compute annual harmonic components - use surface not toa for solar as incorporates albedo and sw_abs automatically
    _, coef_amp, coef_phase = get_fourier_fit_xr(ds.time, ds.temp_surf, n_harmonics=1, pad_coefs_phase=True)
    _, coef_sw_amp_sl, _ = get_fourier_fit_xr(ds.time, ds.swdn_sfc, n_harmonics=1, pad_coefs_phase=True)
    ds['coef_sw_amp'] = np.abs(coef_sw_amp_sl.sel(harmonic=1))
    ds['coef_amp'] = np.abs(coef_amp.sel(harmonic=1))
    ds['coef_phase'] = coef_phase.sel(harmonic=1)

    ds.attrs['omega'] = 2 * np.pi / (day_seconds * ds.time.size)
    ds['heat_cap_eff'], ds['lambda_eff'] = get_phase_amp(ds.coef_sw_amp, ds.omega, coef_phase=ds.coef_phase,
                                                         coef_amp=ds.coef_amp)
    return ds


def get_annual_zonal_mean(ds: xr.Dataset, combine_abs_lat: bool = False, lat_name: str = 'lat',
                          smooth_n_days: int = smooth_n_days,
                          smooth_center: bool = True, keep_attrs: bool = True,
                          smooth_time: Literal['start', 'end'] = 'end') -> xr.Dataset:
    r"""Compute annual-mean zonal mean, optionally combining ±latitudes.

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
        smooth_time: If 'start', will do smoothing before taking annual mean, otherwise will do it after.
            Get more smoothed result if do at the end.

    Returns:
        An xarray Dataset or DataArray containing the annual-mean zonal mean.
        If `combine_abs_lat` is True, the latitude coordinate will be nonnegative
        and sorted (e.g., 0, 30, 60, ...).

    Raises:
        ValueError: If `combine_abs_lat` is True but `lat_name` is not a
            dimension of the input after zonal averaging.
    """
    attrs = ds.attrs.copy()
    if 'lon' in ds.dims:
        ds = ds.mean(dim='lon')
    else:
        print('no lon dimension')
    if (smooth_n_days is not None) and (smooth_n_days > 1) and (smooth_time == 'start'):
        ds = ds.rolling(time=int(smooth_n_days), center=smooth_center).mean()
    ds_av = annual_mean(ds)
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
    if (smooth_n_days is not None) and (smooth_n_days > 1) and (smooth_time == 'end'):
        ds_av = periodic_rolling_mean(ds_av, int(smooth_n_days), 'time')
    for key in ds:
        # Get rid of time dimension of variables that dont have time dimension initially
        if 'time' not in ds[key].dims:
            ds_av[key] = ds_av[key].isel(time=0)
    if keep_attrs:
        ds_av.attrs = attrs
    return ds_av


def load_ds_all(exp_name: str, exp_dir: str = 'thesis_season/publish_exp',
                var_keep: List = var_keep, verbose: bool = False,
                sw_abs_method: Literal['analytic', 'harmonic'] = 'harmonic',
                save: bool = False) -> xr.Dataset:
    """Load and process an experiment across its optical-depth simulations.

    Loads an existing cached dataset if available. Otherwise, loads each
    simulation subdirectory using `load_ds`, concatenates the datasets along
    their optical-depth coordinate, and processes them using `process_ds`.

    Args:
        exp_name: Name of the parent experiment directory containing the
            individual simulation subdirectories.
        exp_dir: Directory containing the parent experiment, relative to
            the directory specified by the `GFDL_DATA` environment variable.
        var_keep: Variables to retain when loading each simulation. Passed
            directly to `load_ds`.
        verbose: Whether to display progress for column calculations within
            `load_ds`. The progress bar over simulations is always shown.
        sw_abs_method: Method for computing shortwave absorption. 'analytic' means the theoretical value,
            'harmonic' means that computed from annual harmonic of insolation and surface shortwave.
            Should be the same for single column simulations, but will differ if surface pressure varies with time.
        save: Whether to save newly processed data as a compressed NetCDF4
            file. Does not control whether an existing cached file is loaded.

    Returns:
        Cached dataset if available; otherwise, the dataset returned by
        `process_ds` after concatenating the individual simulations along
        `odp` and squeezing singleton dimensions.

    Raises:
        ValueError: If no cached dataset exists and `exp_name` is not a
            subdirectory of the specified experiment directory.
        KeyError: If no cached dataset exists and the `GFDL_DATA`
            environment variable is not defined.

    Notes:
        Uses the module-level variables `save_dir`, `lat_target`, and
        `complevel` for the cache directory, target latitude, and NetCDF
        compression level, respectively.

        The cache filename depends only on `exp_name`. An existing cache
        is returned without checking `exp_dir`, `var_keep`, or other
        processing settings.

        Each simulation is loaded at `lat_target`, and its first selected
        latitude is retained. Loading starts at monthly file 25 when
        `'advect'` occurs in the parent experiment name, and file 2
        otherwise.

        Simulation subdirectories are processed in alphabetical order.
        Their `odp` attributes supply the concatenation coordinate; the
        resulting coordinate is not explicitly sorted by optical depth.
    """
    # If saved, load data in
    out_path = os.path.join(save_dir, f"ds_{exp_name}.nc")
    if os.path.exists(out_path):
        ds = xr.load_dataset(out_path)
        ds['sw_abs'] = ds[f"sw_abs_{sw_abs_method}"]
        print(f"Loaded dataset from:\n{out_path}")
        return ds

    # If not saved, create dataset
    with os.scandir(os.path.join(os.environ['GFDL_DATA'], exp_dir)) as entries:
        exp_possible = sorted(entry.name for entry in entries if entry.is_dir())
    if exp_name not in exp_possible:
        raise ValueError(f"Exp name '{exp_name}' not found in\n{os.path.join(os.environ['GFDL_DATA'], exp_dir)}.\n"
                         f"Must be one of\n{exp_possible}.")
    path = os.path.join(os.environ['GFDL_DATA'], exp_dir, exp_name)
    with os.scandir(path) as entries:
        exp_names = sorted(entry.name for entry in entries if entry.is_dir())

    # Load over all optical depth values of this simulation
    odp_vals = []
    ds_base = []
    for i, exp in tqdm(enumerate(exp_names), total=len(exp_names)):
        ds_use = load_ds(exp, os.path.join(exp_dir, exp_name), var_keep, lat_target=lat_target, verbose=verbose,
                         first_month_file=25 if 'advect' in exp_name else 2).isel(lat=0)
        ds_base.append(ds_use)
        odp_vals.append(ds_use.attrs['odp'])
    odp_vals_xr = xr.DataArray(odp_vals, dims="odp", name='odp')
    ds_base = xr.concat(ds_base, dim=odp_vals_xr).squeeze()
    ds = process_ds(ds_base)

    if save:
        # Save processed dataset
        out_path = os.path.join(save_dir, f"ds_{exp_name}.nc")
        if os.path.exists(out_path):
            warnings.warn(f"Output file already exists:\n{out_path}\nNot saving.")
        else:
            ds.to_netcdf(out_path, format="NETCDF4",
                         encoding={var: {"zlib": True, "complevel": complevel} for var in
                                   ds.data_vars})
            print(f"Processed dataset save at:\n{out_path}")
    ds['sw_abs'] = ds[f"sw_abs_{sw_abs_method}"]
    return ds
