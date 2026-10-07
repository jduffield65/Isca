import xarray as xr
import numpy as np
import os
from typing import List, Optional, Union, Literal, Tuple
from tqdm import tqdm

from isca_tools.thesis.surface_flux_taylor_2layer import get_p_eff, get_sensible_heat, get_sensitivity_lh, \
    get_sensitivity_sh, get_sensitivity_lw_surf
from isca_tools.utils.base import mass_weighted_vertical_integral, validate_params
from isca_tools.utils.fourier import coef_conversion
from isca_tools.utils.moist_physics import sphum_sat
from isca_tools.utils.numerical import get_var_shift
from isca_tools.utils.radiation import get_heat_capacity, opd_lw_gray, frierson_atmospheric_heating, get_frierson_sw_abs
from isca_tools import load_dataset, load_namelist
from isca_tools.utils.constants import c_p_ocean, rho_ocean, c_p, L_v, g, Stefan_Boltzmann
from isca_tools.utils.xarray import wrap_with_apply_ufunc, select_coord_window

from jobs.thesis_season.column.utils import get_fit_coef_complex_xr, lat_min, lat_max, get_annual_zonal_mean, \
    get_temp_from_sphum_sat_xr, get_sw_abs_amp_xr, spline_deriv_periodic_xr, day_seconds, get_fourier_fit_xr, \
    fit_linear_zero_mean_xr, apply_linear_zero_mean_xr, apply_fit_complex_xr, get_phase_amp, width, month_ticks
from jobs.thesis_season.thesis_figs.utils import smooth_n_days

var_keep = ['temp', 'ps', 'sphum', 'olr', 'swdn_toa', 'swdn_sfc', 'lwdn_sfc', 'lwup_sfc', 'flux_t',
            'flux_lhe', 't_surf', 'precipitation']  # just the fluxes, no variables


def load_ds(exp_name: str, exp_dir: str, var_keep: List = var_keep,
            lat_min: float = lat_min, lat_max: float = lat_max,
            first_month_file: Optional[int] = None,
            verbose: bool = False, lat_target: Optional[float] = None,
            n_lat: int = 0) -> xr.Dataset:
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
        ds.attrs['p_ref'] = 101325.0        # default value
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
    ds['sw_abs_analytic'] = get_frierson_sw_abs(ds.atm_abs, ds.p_surf.mean(dim='time'), p_ref=ds.p_ref, albedo=ds.albedo)
    ds['sw_abs'] = ds['sw_abs_analytic']        # use analytic one as simpler, even though difference if p_surf not constant
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


def get_fit_complex_xr(x: xr.DataArray, y: xr.DataArray, time: Optional[xr.DataArray] = None,
                       include_phase: bool = False,
                       x2: Optional[xr.DataArray] = None) -> Tuple[xr.DataArray, xr.DataArray]:
    r"""Fit one or two coefficients relating predictor and response DataArrays.

    When ``include_phase`` is True, fit amplitude and phase coefficients using
    ``get_fit_coef_complex_xr``. When ``include_phase`` is False and ``x2`` is
    not provided, fit a single zero-mean linear amplitude coefficient and
    return a zero-valued phase coefficient. When ``x2`` is provided, fit
    separate zero-mean linear amplitude coefficients for ``x`` and ``x2``.

    ``include_phase`` and ``x2`` are mutually exclusive.

    Args:
        x (xr.DataArray): Primary predictor data array.
        y (xr.DataArray): Response data array.
        time (Optional[xr.DataArray]): Time coordinate or array used for the
            complex fit. Ignored unless ``include_phase`` is True. Defaults to
            None.
        include_phase (bool): Whether to fit an amplitude and phase
            coefficient using a complex fit. Cannot be True when ``x2`` is
            provided. Defaults to False.
        x2 (Optional[xr.DataArray]): Optional second predictor data array. If
            provided, the second returned value is its fitted amplitude
            coefficient rather than a phase coefficient. Defaults to None.

    Returns:
        ``coef_amp``: Amplitude coefficient associated with ``x``.
        ``coef_phase``: Phase coefficient associated with ``x`` when
            ``include_phase`` is True; zero when neither ``include_phase``
            nor ``x2`` is supplied; otherwise, the amplitude coefficient
            associated with ``x2``.

    Raises:
        ValueError: If both ``include_phase`` is True and ``x2`` is provided.
    """
    if include_phase and x2 is not None:
        raise ValueError('Not valid for include_phase and x2 provided.')
    if include_phase:
        coef_amp, coef_phase = get_fit_coef_complex_xr(y, x, time)
    else:
        if x2 is None:
            coef_amp = fit_linear_zero_mean_xr(x, y)
            coef_phase = coef_amp * 0
        else:
            coef_amp, coef_phase = fit_linear_zero_mean_xr(x, y, x2)
    return coef_amp, coef_phase


def get_empirical_params(ds: xr.Dataset, const_p: bool = False,
                         empirical_lambda_const: bool = False,
                         include_params: Optional[List] = None,
                         exclude_params: Optional[List] = None,
                         include_olr_surf_cont: bool = False) -> dict:
    r"""Fit empirical parameters for the seasonal surface--atmosphere model.

    The fitted parameters correspond to the coupled surface and atmospheric
    temperature-budget equations

    $$
    C_s \frac{\partial T_s}{\partial t}
    = (1 - \alpha)(1 - \xi)F(t)
    + \lambda(T_a - T_s)
    - \lambda_a\exp(-i\phi_a)T_a,
    $$

    $$
    C_a\left[\beta_{\mathrm{col}}\exp(-i\phi_{\mathrm{col}}) + \mu \right]
    \frac{\partial T_a}{\partial t}
    = \xi F(t)
    + \lambda(T_s - T_a)
    + \lambda_a\exp(-i\phi_a)T_a
    - B\exp(-i\phi_{\mathrm{olr}})
    (T_a - \chi_{\mathrm{olr}}T_s)
    - \lambda_{\mathrm{adv}}\exp(-i\phi_{\mathrm{adv}})T_a.
    $$

    The atmospheric heat-capacity correction is represented by $\mu$, while
    $\beta_{\mathrm{col}}$ and $\phi_{\mathrm{col}}$ account for amplitude
    and phase differences between column-mean and near-surface atmospheric
    temperature tendencies.

    Args:
        ds: Processed dataset containing time-varying surface and atmospheric
            temperatures, column specific humidity, pressure integral, surface
            turbulent and radiative fluxes, outgoing longwave radiation, and
            atmospheric advection. It must have been processed by
            `process_ds`.
        const_p: If `True`, fit $\mu$, $\beta_{\mathrm{col}}$, and
            $\phi_{\mathrm{col}}$ without accounting for seasonal variation
            in the atmospheric pressure integral. If `False`, pressure-weight
            column quantities are used before fitting.
        empirical_lambda_const: Whether to estimate the coefficient
            multiplying $T_s - T_a$, $\lambda$, directly from the seasonal
            flux data. If `True`, the latent-, sensible-, and surface-longwave
            flux contributions are jointly fitted as linear functions of
            $T_s - T_a$ and, where requested, $T_a$. If `False`, the
            $T_s - T_a$ sensitivities are instead calculated from the
            mean-state physical parameterisations; the residual
            atmospheric-temperature dependence is then fitted empirically.
        include_params: Optional list of parameter names to fit and return.
            Parameters not included are set to their default values: zero for
            most coefficients and one for `coef_amp_col`. Specifying a
            composite parameter automatically includes the component
            parameters required to calculate it; for example,
            `lambda_const` includes its latent-, sensible-, and longwave-flux
            components.
        exclude_params: Optional list of parameter names to omit after
            applying `include_params`. Excluded parameters are set to their
            default values: zero for most coefficients and one for
            `coef_amp_col`. This can be used to suppress individual
            components of an otherwise included composite parameter.
        include_olr_surf_cont: Whether to explicitly include the direct
            surface-emitted contribution to outgoing longwave radiation (OLR).
            If `True`, OLR is decomposed into a surface contribution,
            $\sigma \exp(-\tau_{\mathrm{sfc}}) T_s^4$, and a residual
            atmospheric contribution. The linearised surface-temperature
            sensitivity is returned as `lambda_lw`, while `B` and
            `coef_phase_olr` describe the residual atmospheric component. If
            `False`, all OLR variability is assumed to arise from the
            atmospheric component; `lambda_lw` is set to zero and `B` is
            fitted directly to total OLR.

    Returns:
        Dictionary containing the fitted empirical model parameters:

        - `mu`: Moisture-related correction to atmospheric heat capacity,
          $\mu$.
        - `coef_amp_col`: Amplitude factor relating column-mean and
          near-surface atmospheric temperature tendencies,
          $\beta_{\mathrm{col}}$.
        - `coef_phase_col`: Phase correction associated with the column
          temperature tendency, $\phi_{\mathrm{col}}$.
        - `lambda_const_lh`: Latent-heat-flux contribution to the coefficient
          multiplying $T_s - T_a$.
        - `lambda_const_sh`: Sensible-heat-flux contribution to the
          coefficient multiplying $T_s - T_a$.
        - `lambda_const_lw`: Net surface-longwave-flux contribution to the
          coefficient multiplying $T_s - T_a$.
        - `lambda_const`: Total coefficient multiplying $T_s - T_a$,
          $\lambda$.
        - `lambda_a_lh`: Amplitude of the atmospheric-temperature-dependent
          latent-heat-flux contribution.
        - `lambda_a_sh`: Atmospheric-temperature-dependent sensible-heat-flux
          contribution.
        - `lambda_a_lw`: Atmospheric-temperature-dependent net surface
          longwave-flux contribution.
        - `coef_phase_a_lh`: Phase correction of the
          atmospheric-temperature-dependent latent-heat-flux contribution.
          This is zero when `include_phase_lh` is `False`.
        - `lambda_a`: Amplitude of the combined
          atmospheric-temperature-dependent surface-flux term, $\Lambda$.
        - `coef_phase_a`: Phase correction of the combined $\Lambda$ term,
          $\phi_a$, such that the temperature dependence is represented by
          $\Lambda[1 + i\phi_a]$. This is zero when `include_phase_lh` is
          `False`.
        - `lambda_lw`: Linearised surface-temperature dependence of the
          surface-emitted longwave component of outgoing longwave radiation.
        - `B`: Amplitude of the atmospheric contribution to outgoing
          longwave radiation.
        - `coef_phase_olr`: Phase correction for the atmospheric outgoing
          longwave-radiation contribution, $\phi_{\mathrm{olr}}$.
        - `lambda_adv`: Amplitude of the atmospheric advection response.
        - `coef_phase_adv`: Phase correction for atmospheric advection,
          $\phi_{\mathrm{adv}}$.

    Notes:
        The coefficients are estimated from zero-mean linear fits or complex
        harmonic fits. Phase coefficients represent quadrature components of
        seasonal relationships and are implemented as time shifts when
        reconstructing budget terms.
    """
    # Get a list of all parameters to find
    _allowed_params = ['mu', 'coef_amp_col', 'coef_phase_col',
                       'lambda_const', 'lambda_a', 'coef_phase_a', 'B',
                       'coef_phase_olr', 'lambda_lw', 'lambda_adv', 'coef_phase_adv',
                       'lambda_const_lh', 'lambda_const_sh', 'lambda_const_lw',
                       'lambda_a_lh', 'lambda_a_sh', 'lambda_a_lw', 'coef_phase_a_lh', 'coef_amp_col_sphum',
                       'lambda_adv_dry', 'coef_phase_adv_dry', 'lambda_adv_moist', 'coef_phase_adv_moist']

    if include_params is not None:
        # Add intermediate variables if specify include_params
        if 'mu' in include_params:
            include_params += ['coef_amp_col_sphum']
        for key2 in ['lambda_const', 'lambda_a', 'coef_phase_a', 'lambda_adv', 'coef_phase_adv']:
            if key2 in include_params:
                include_params += [key for key in _allowed_params if f"{key2}_" in key]
        include_params = list(set(include_params))      # remove duplicates
    include_params = _allowed_params.copy() if include_params is None else include_params
    validate_params(include_params, _allowed_params, "include_params")

    if exclude_params is not None:
        validate_params(exclude_params, _allowed_params, "exclude_params")
        include_params = [x for x in include_params if x not in exclude_params]

    params = {}
    if const_p:
        # mu accounts for atmospheric heat capacity dependence on sphum
        params['mu'] = \
            get_fit_complex_xr(spline_deriv_periodic_xr(ds.time * day_seconds, ds.temp_atm),
                               spline_deriv_periodic_xr(ds.time * day_seconds, ds.sphum_col))[0] * L_v / c_p
        params['coef_amp_col_sphum'] = get_fit_complex_xr(ds.temp_atm, ds.temp_col_sphum)[0]
        # Account for column mean temp differing from lowest model level
        params['coef_amp_col'], params['coef_phase_col'] = \
            get_fit_complex_xr(ds.temp_atm, ds.temp_col, ds.time, 'coef_phase_col' in include_params)
    else:
        params['mu'] = \
            get_fit_complex_xr(spline_deriv_periodic_xr(ds.time * day_seconds, ds.temp_atm),
                               spline_deriv_periodic_xr(ds.time * day_seconds, ds.sphum_col * ds.p_integ_calc)
                               )[0] * L_v / c_p / ds.p_integ_calc.mean(dim='time')
        params['coef_amp_col_sphum'] = get_fit_complex_xr(
            spline_deriv_periodic_xr(ds.time * day_seconds, ds.temp_atm),
            spline_deriv_periodic_xr(ds.time * day_seconds, ds.temp_col_sphum * ds.p_integ_calc))[0]
        params['coef_amp_col_sphum'] /= ds.p_integ_calc.mean(dim='time')
        params['coef_amp_col'], params['coef_phase_col'] = \
            get_fit_complex_xr(spline_deriv_periodic_xr(ds.time * day_seconds, ds.temp_atm),
                               spline_deriv_periodic_xr(ds.time * day_seconds, ds.temp_col * ds.p_integ_calc),
                               ds.time, 'coef_phase_col' in include_params)
        params['coef_amp_col'] /= ds.p_integ_calc.mean(dim='time')

    # LH, SH, LW params
    if empirical_lambda_const:
        params['lambda_const_lh'], params['lambda_a_lh'] = \
            get_fit_complex_xr(ds.temp_surf - ds.temp_atm, ds.flux_lhe,
                               x2=ds.temp_atm if 'lambda_a' in include_params else None)
        params['coef_phase_a_lh'] = params['lambda_a_lh'] * 0
        params['lambda_const_sh'], params['lambda_a_sh'] = \
            get_fit_complex_xr(ds.temp_surf - ds.temp_atm, ds.flux_t,
                               x2=-ds.temp_atm if 'lambda_a' in include_params else None)
        params['lambda_const_lw'], params['lambda_a_lw'] = \
            get_fit_complex_xr(ds.temp_surf - ds.temp_atm, ds.lwup_sfc - ds.lwdn_sfc,
                               x2=ds.temp_atm if 'lambda_a' in include_params else None)
    else:
        # dont need RH for this
        ds_use = ds.mean(dim='time')
        params['lambda_const_lh'] = get_sensitivity_lh(ds_use.temp_surf, ds_use.temp_atm, 0, ds_use.wind_drag_av, 1,
                                                       ds_use.p_surf, ds_use.sigma_atm)['temp_surf']
        params['lambda_const_sh'] = get_sensitivity_sh(ds_use.temp_surf, ds_use.temp_atm, ds_use.wind_drag_av, 1,
                                                       ds_use.p_surf, ds_use.sigma_atm)['temp_surf']
        # dont need radiative temp for this
        params['lambda_const_lw'] = get_sensitivity_lw_surf(ds_use.temp_surf, 0, 0)['temp_surf']

        flux_t_resid = ds.flux_t - apply_linear_zero_mean_xr(ds.temp_surf - ds.temp_atm, params['lambda_const_sh'])
        params['lambda_a_sh'] = get_fit_complex_xr(-ds.temp_atm, flux_t_resid)[0]       # sign is so lambda_a_sh is positive
        flux_lw_resid = ds.lwup_sfc - ds.lwdn_sfc - apply_linear_zero_mean_xr(ds.temp_surf - ds.temp_atm,
                                                                              params['lambda_const_lw'])
        params['lambda_a_lw'] = get_fit_complex_xr(ds.temp_atm, flux_lw_resid)[0]
    params['lambda_const'] = params['lambda_const_lh'] + params['lambda_const_sh'] + params[
        'lambda_const_lw']  # for temp_s - temp_a     # for temp_a

    # Deal with phase delay of LH, and combine the temp_a fitting into single lambda_a coefficient
    if ('coef_phase_a' in include_params) or ('lambda_a_lh' not in params):
        # Get what is left of LH after the temp_surf-temp_atm fit
        flux_lhe_resid = ds.flux_lhe - apply_linear_zero_mean_xr(ds.temp_surf - ds.temp_atm, params['lambda_const_lh'])
        # Refind the residual atmospheric effect taking into account of phase delay
        params['lambda_a_lh'], params['coef_phase_a_lh'] = get_fit_complex_xr(ds.temp_atm, flux_lhe_resid, ds.time,
                                                                              'coef_phase_a' in include_params)
    params['lambda_a'], params['coef_phase_a'] = \
        coef_conversion(cos_coef=params['lambda_a_lh'] * np.cos(params['coef_phase_a_lh']) + params['lambda_a_lw'] -
                                 params['lambda_a_sh'],
                        sin_coef=params['lambda_a_lh'] * np.sin(params['coef_phase_a_lh']), take_cos_sign=True)

    # OLR params
    if include_olr_surf_cont:
        olr_surf_cont = Stefan_Boltzmann * np.exp(-ds.odp_surf) * ds.temp_surf ** 4
        params['lambda_lw'] = get_fit_complex_xr(ds.temp_surf, olr_surf_cont)[0]
        params['B'], params['coef_phase_olr'] = get_fit_complex_xr(ds.temp_atm, ds.olr - olr_surf_cont, ds.time,
                                                                   'coef_phase_olr' in include_params)
    else:
        params['B'], params['coef_phase_olr'] = get_fit_complex_xr(ds.temp_atm, ds.olr, ds.time,
                                                                   'coef_phase_olr' in include_params)
        params['lambda_lw'] = params['B'] * 0

    # Advection params
    params['lambda_adv'], params['coef_phase_adv'] = \
        get_fit_complex_xr(-ds.temp_atm, ds.adv_atmos, ds.time, 'coef_phase_adv' in include_params)
    params['lambda_adv_dry'], params['coef_phase_adv_dry'] = \
        get_fit_complex_xr(-ds.temp_atm, ds.adv_atmos_dry, ds.time, 'coef_phase_adv' in include_params)
    params['lambda_adv_moist'], params['coef_phase_adv_moist'] = \
        get_fit_complex_xr(-ds.temp_atm, ds.adv_atmos_moist, ds.time, 'coef_phase_adv' in include_params)
    for key in _allowed_params:
        if key not in include_params:
            params[key] *= 0
            if key == 'coef_amp_col':
                params[key] += 1        # default value for this param is 1
    return params


def get_approx_mse_tend(temp_atm: xr.DataArray, coef_amp_col: xr.DataArray,
                        coef_phase_col: xr.DataArray, mu: xr.DataArray,
                        p_integ_calc: xr.DataArray,
                        time: xr.DataArray) -> xr.DataArray:
    r"""Approximate the atmospheric moist-static-energy tendency.

    Reconstructs the reduced-model approximation to the atmospheric
    moist-static-energy tendency,

    $$
    C_a\left[\beta_{\mathrm{col}} + \mu
    - i\beta_{\mathrm{col}}\phi_{\mathrm{col}}\right]
    \frac{\partial T_a}{\partial t},
    $$

    using the near-surface atmospheric temperature tendency. The column
    temperature tendency is scaled by $\beta_{\mathrm{col}}$ and shifted in
    time according to $\phi_{\mathrm{col}}$, while the specific-humidity
    contribution is represented by $\mu \partial T_a / \partial t$.

    Args:
        temp_atm: Near-surface atmospheric temperature, $T_a$.
        coef_amp_col: Amplitude factor relating column-mean and near-surface
            atmospheric temperature tendencies, $\beta_{\mathrm{col}}$.
        coef_phase_col: Phase correction for the column-temperature tendency,
            $\phi_{\mathrm{col}}$.
        mu: Moisture-related atmospheric heat-capacity correction, $\mu$.
        p_integ_calc: Pressure thickness of the atmospheric column used in the
            energy-budget calculation. Should be the mean over the `time` dimension.
        time: Time coordinate, in days, used to calculate the periodic
            temperature tendency and apply the phase shift.

    Returns:
        Approximation to the atmospheric moist-static-energy tendency in units
        of energy flux, $C_a(\partial T_{\mathrm{col}}/\partial t +
        \mu\partial T_a/\partial t)$.
    """
    if 'time' in p_integ_calc.dims:
        raise ValueError('p_integ_calc should be an average over time')
    temp_atm_deriv = spline_deriv_periodic_xr(time * day_seconds, temp_atm)
    temp_col_tend = apply_fit_complex_xr(temp_atm_deriv, coef_amp_col, coef_phase_col)
    sphum_tend = mu * temp_atm_deriv
    c_a = c_p * p_integ_calc / g
    return c_a * (temp_col_tend + sphum_tend)


def get_approx_flux_atmos(temp_atm: xr.DataArray, temp_surf: xr.DataArray, swdn_toa: xr.DataArray,
                          sw_abs: xr.DataArray, lambda_const: xr.DataArray, lambda_a: xr.DataArray,
                          B: xr.DataArray, lambda_lw: xr.DataArray, coef_phase_olr: xr.DataArray,
                          coef_phase_a: Optional[xr.DataArray] = None) -> xr.DataArray:
    r"""Approximate non-advective atmospheric energy-budget fluxes.

    Reconstructs the explicitly diagnosed terms on the right-hand side of the
    atmospheric energy budget, excluding atmospheric advection:

    $$
    \mathrm{flux}_{\mathrm{atmos}} =
    \mathrm{SW}_{\mathrm{abs}}(t)
    + \lambda(T_s - T_a)
    + \Lambda\left[1 + i\phi_a\right]T_a
    - \lambda_{\mathrm{lw1}} T_s
    - B\left[1 - i\phi_{\mathrm{olr}}\right]T_a.
    $$

    All temperature and incoming solar-radiation anomalies are calculated
    relative to their time means. The phase coefficients $\phi_a$ and
    $\phi_{\mathrm{olr}}$ are implemented as time shifts of the relevant
    atmospheric-temperature contributions.

    Args:
        temp_atm: Near-surface atmospheric temperature, $T_a$.
        temp_surf: Surface temperature, $T_s$.
        swdn_toa: Downward shortwave radiation at the top of the atmosphere.
        sw_abs: Fraction of top-of-atmosphere shortwave radiation absorbed by
            the atmosphere.
        lambda_const: Coefficient multiplying the surface--atmosphere
            temperature contrast, $\lambda$.
        lambda_a: Amplitude of the atmospheric-temperature-dependent
            surface-flux term, $\Lambda$.
        B: Amplitude of the atmospheric contribution to outgoing longwave
            radiation.
        lambda_lw: Coefficient for the surface-temperature-dependent
            longwave contribution to outgoing longwave radiation.
        coef_phase_olr: Phase correction for the atmospheric outgoing
            longwave-radiation contribution, $\phi_{\mathrm{olr}}$.
        coef_phase_a: Optional phase correction for the combined
            atmospheric-temperature-dependent surface-flux term, $\phi_a$.
            If `None`, this term is assumed to have no phase shift.

    Returns:
        Approximate atmospheric energy-budget flux convergence excluding
        advection, comprising absorbed shortwave radiation, surface--atmosphere
        exchange, the potentially phase-shifted $\Lambda$ contribution, and the
        phase-shifted atmospheric outgoing-longwave-radiation contribution.
    """
    temp_atm = temp_atm - temp_atm.mean(dim='time')
    temp_surf = temp_surf - temp_surf.mean(dim='time')
    flux_abs = sw_abs * (swdn_toa - swdn_toa.mean(dim='time'))

    flux_linear = apply_linear_zero_mean_xr(temp_surf - temp_atm, lambda_const, temp_atm, lambda_a, coef_phase_a) \
                  - lambda_lw * temp_surf

    flux_shift = apply_fit_complex_xr(temp_atm, -B, coef_phase_olr)
    return flux_abs + flux_linear + flux_shift


def get_approx_adv_atmos(temp_atm: xr.DataArray, lambda_adv: xr.DataArray,
                         coef_phase_adv: xr.DataArray) -> xr.DataArray:
    r"""Approximate the atmospheric advection term.

    Reconstructs the residual atmospheric energy-budget contribution from
    advection using a phase-shifted atmospheric temperature anomaly:

    $$
    \mathrm{adv}_{\mathrm{atmos}} =
    -\lambda_{\mathrm{adv}}
    \left[1 - i\phi_{\mathrm{adv}}\right]T_a.
    $$

    The phase correction $\phi_{\mathrm{adv}}$ is implemented as a time shift
    of the demeaned near-surface atmospheric temperature.

    Args:
        temp_atm: Near-surface atmospheric temperature, $T_a$.
        lambda_adv: Amplitude of the atmospheric advection response,
            $\lambda_{\mathrm{adv}}$.
        coef_phase_adv: Phase correction for atmospheric advection,
            $\phi_{\mathrm{adv}}$.

    Returns:
        Approximate atmospheric advection term, with the same units as the
        atmospheric energy-budget fluxes.
    """
    temp_atm = temp_atm - temp_atm.mean(dim='time')
    return apply_fit_complex_xr(temp_atm, -lambda_adv, coef_phase_adv)
