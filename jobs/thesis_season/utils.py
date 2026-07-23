import xarray as xr
import numpy as np
import os
from typing import List, Optional, Union, Literal
from tqdm import tqdm

from isca_tools.thesis.surface_flux_taylor_2layer import get_p_eff
from isca_tools.utils.base import mass_weighted_vertical_integral
from isca_tools.utils.fourier import coef_conversion
from isca_tools.utils.moist_physics import sphum_sat
from isca_tools.utils.numerical import get_var_shift
from isca_tools.utils.radiation import get_heat_capacity, opd_lw_gray, frierson_atmospheric_heating
from isca_tools import load_dataset, load_namelist
from isca_tools.utils.constants import c_p_ocean, rho_ocean, c_p, L_v, g, Stefan_Boltzmann
from isca_tools.utils.xarray import wrap_with_apply_ufunc

from jobs.thesis_season.column.utils import get_fit_coef_complex_xr, lat_min, lat_max, get_annual_zonal_mean, \
    get_temp_from_sphum_sat_xr, get_sw_abs_amp_xr, spline_deriv_periodic_xr, day_seconds, get_fourier_fit_xr, \
    fit_linear_zero_mean_xr, apply_linear_zero_mean_xr, apply_fit_complex_xr, width, month_ticks
from jobs.thesis_season.thesis_figs.utils import smooth_n_days

var_keep = ['temp', 'ps', 'sphum', 'olr', 'swdn_toa', 'swdn_sfc', 'lwdn_sfc', 'lwup_sfc', 'flux_t',
            'flux_lhe', 't_surf']  # just the fluxes, no variables


def load_ds(exp_name: str, exp_dir: str, var_keep: List = var_keep,
            lat_min: float = lat_min, lat_max: float = lat_max,
            first_month_file: Optional[int] = None,
            verbose: bool = False) -> xr.Dataset:
    # Load all info required for empirical approximation of 2 layer energy budget
    # Will need to update later to add that required analytically such as
    # temp_col_sphum, temp_rad_surf, temp_rad_atm
    exp_path = os.path.join(exp_dir, exp_name)

    ds = load_dataset(exp_path, first_month_file=first_month_file).sel(lat=slice(lat_min, lat_max))[var_keep]
    ds = ds.load()

    # Load info from namelist and add to attributes
    namelist = load_namelist(exp_path)
    sigma_levels_half = np.asarray(namelist['vert_coordinate_nml']['bk'])
    sigma_levels_full = np.convolve(sigma_levels_half, np.ones(2) / 2, 'valid')
    ds['lev_sigma'] = (ds.pfull * 0 + sigma_levels_full).squeeze()
    ds.attrs['albedo'] = namelist['mixed_layer_nml']['albedo_value']
    ds.attrs['depth'] = namelist['mixed_layer_nml']['depth']
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
    ds = get_annual_zonal_mean(ds, smooth_n_days=smooth_n_days, smooth_time=smooth_time)
    ds['p_eff'] = get_p_eff(ds.p_surf.mean(dim='time'))
    ds['temp_col_sphum'] = get_temp_from_sphum_sat_xr(ds.sphum_col / ds.rh_col, ds.p_eff)
    ds['sw_abs'] = get_sw_abs_amp_xr(ds.swdn_sfc, ds.swdn_toa, ds.time, albedo=ds.albedo)

    # Atmospheric energy budget components: mse_tend = flux + adv
    ds['mse_tend_atmos'] = spline_deriv_periodic_xr(ds.time * day_seconds,
                                                    (c_p * ds.temp_col + L_v * ds.sphum_col) * ds.p_integ_calc / g)
    ds['flux_atmos'] = frierson_atmospheric_heating(ds, ds.albedo) + ds.flux_t + ds.flux_lhe
    ds['adv_atmos'] = ds['mse_tend_atmos'] - ds['flux_atmos']

    # Surface fluxes excluding SW
    ds['flux_surf'] = ds.flux_t + ds.flux_lhe - ds.lwdn_sfc + ds.lwup_sfc

    # Compute annual harmonic components - use surface not toa for solar as incorporates albedo and sw_abs automatically
    _, coef_amp, coef_phase = get_fourier_fit_xr(ds.time, ds.temp_surf, n_harmonics=1, pad_coefs_phase=True)
    _, coef_sw_amp_sl, _ = get_fourier_fit_xr(ds.time, ds.swdn_sfc, n_harmonics=1, pad_coefs_phase=True)
    ds['coef_sw_amp'] = np.abs(coef_sw_amp_sl.sel(harmonic=1))
    ds['coef_amp'] = np.abs(coef_amp.sel(harmonic=1))
    ds['coef_phase'] = coef_phase.sel(harmonic=1)
    return ds


def get_empirical_params(ds: xr.Dataset, const_p: bool = False,
                         include_phase_lh: bool = False) -> dict:
    r"""Fit empirical parameters for the seasonal surface--atmosphere model.

    The fitted parameters correspond to the coupled surface and atmospheric
    temperature-budget equations

    $$
    C_s \frac{\partial T_s}{\partial t}
    = (1 - \alpha)(1 - \xi)F(t)
    + \lambda(T_a - T_s)
    - \Lambda T_a,
    $$

    $$
    C_a\left[\beta_{\mathrm{col}} + \mu - i\beta_{\mathrm{col}}\phi_{\mathrm{col}}\right]
    \frac{\partial T_a}{\partial t}
    = \xi F(t)
    + \lambda(T_s - T_a)
    + \Lambda T_a
    - B\left[1 - i\phi_{\mathrm{olr}}\right](T_a - \chi_{\mathrm{olr}}T_s)
    - \lambda_{\mathrm{adv}}\left[1 - i\phi_{\mathrm{adv}}\right]T_a.
    $$

    The atmospheric heat-capacity correction is represented by $\mu$, while
    differences between column-mean and near-surface atmospheric temperature
    are represented by $\beta_{\mathrm{col}}$ and
    $\phi_{\mathrm{col}}$. Flux regressions are used to estimate the
    surface--atmosphere exchange and longwave parameters.

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

    Returns:
        Dictionary containing the fitted empirical model parameters:

        - `mu`: Moisture-related correction to atmospheric heat capacity,
          $\mu$.
        - `coef_amp_col`: Amplitude factor relating column-mean and
          near-surface atmospheric temperature tendencies,
          $\beta_{\mathrm{col}}$.
        - `coef_phase_col`: Phase correction associated with the column
          temperature factor, $\phi_{\mathrm{col}}$.
        - `lambda_const`: Component of surface--atmosphere exchange
          proportional to $T_s - T_a$, combining latent heat, sensible heat,
          and longwave contributions.
        - `lambda_a`: Component of the surface energy budget proportional to
          atmospheric temperature, $\Lambda$.
        - `lambda_lh`: Atmospheric-temperature dependence of the latent heat
          flux contribution.
        - `lambda_sh`: Atmospheric-temperature dependence of the sensible heat
          flux contribution.
        - `lambda_lw2`: Atmospheric-temperature dependence of the net
          surface longwave flux contribution.
        - `lambda_lw1`: Linearised surface-temperature dependence of the
          surface-emitted longwave component of OLR.
        - `B`: Amplitude of the atmospheric contribution to outgoing
          longwave radiation.
        - `coef_phase_olr`: Phase correction for the atmospheric OLR
          contribution, $\phi_{\mathrm{olr}}$.
        - `lambda_adv`: Amplitude of atmospheric temperature damping by
          advection.
        - `coef_phase_adv`: Phase correction for atmospheric advection,
          $\phi_{\mathrm{adv}}$.

    Notes:
        The returned coefficients are estimated from zero-mean linear fits or
        complex harmonic fits. Phase coefficients describe the imaginary,
        quadrature component of the relevant seasonal relationship.
    """
    params = {}
    if const_p:
        # mu accounts for atmospheric heat capacity dependence on sphum
        params['mu'] = \
            fit_linear_zero_mean_xr(spline_deriv_periodic_xr(ds.time * day_seconds, ds.temp_atm),
                                    spline_deriv_periodic_xr(ds.time * day_seconds, ds.sphum_col)) * L_v / c_p
        # Account for column mean temp differing from lowest model level
        params['coef_amp_col'], params['coef_phase_col'] = \
            get_fit_coef_complex_xr(ds["temp_col"], ds.temp_atm, ds.time)
    else:
        params['mu'] = \
            fit_linear_zero_mean_xr(spline_deriv_periodic_xr(ds.time * day_seconds, ds.temp_atm),
                                    spline_deriv_periodic_xr(ds.time * day_seconds, ds.sphum_col * ds.p_integ_calc)
                                    ) * L_v / c_p / ds.p_integ_calc.mean(dim='time')
        params['coef_amp_col'], params['coef_phase_col'] = \
            get_fit_coef_complex_xr(spline_deriv_periodic_xr(ds.time * day_seconds, ds.temp_col * ds.p_integ_calc),
                                    spline_deriv_periodic_xr(ds.time * day_seconds, ds.temp_atm), ds.time)
        params['coef_amp_col'] /= ds.p_integ_calc.mean(dim='time')

    # LH, SH, LW params
    params['lambda_const_lh'], params['lambda_a_lh'] = fit_linear_zero_mean_xr(ds.temp_surf - ds.temp_atm, ds.flux_lhe, ds.temp_atm)
    params['lambda_const_sh'], params['lambda_a_sh'] = fit_linear_zero_mean_xr(ds.temp_surf - ds.temp_atm, ds.flux_t, -ds.temp_atm)
    params['lambda_const_lw'], params['lambda_a_lw'] = fit_linear_zero_mean_xr(ds.temp_surf - ds.temp_atm,
                                                                   ds.lwup_sfc - ds.lwdn_sfc, ds.temp_atm)
    params['lambda_const'] = params['lambda_const_lh'] + params['lambda_const_sh'] + params['lambda_const_lw']  # for temp_s - temp_a     # for temp_a

    # Deal with phase delay of LH, and combine the temp_a fitting into single lambda_a coefficient
    if include_phase_lh:
        # Get what is left of LH after the temp_surf-temp_atm fit
        flux_lhe_resid = ds.flux_lhe - apply_linear_zero_mean_xr(ds.temp_surf - ds.temp_atm, params['lambda_const_lh'])
        # Refind the residual atmospheric effect taking into account of phase delay
        params['lambda_a_lh'], params['coef_phase_a_lh'] = get_fit_coef_complex_xr(flux_lhe_resid, ds.temp_atm, ds.time)
        params['lambda_a'], params['coef_phase_a'] = \
            coef_conversion(cos_coef=params['lambda_a_lh']*np.cos(params['coef_phase_a_lh']) + params['lambda_a_lw'] -
                                     params['lambda_a_sh'],
                            sin_coef=params['lambda_a_lh']*np.sin(params['coef_phase_a_lh']), take_cos_sign=True)
    else:
        params['coef_phase_a_lh'] = 0
        params['coef_phase_a'] = 0
        params['lambda_a'] = params['lambda_a_lh'] + params['lambda_a_lw'] - params['lambda_a_sh']

    # OLR params
    olr_surf_cont = Stefan_Boltzmann * np.exp(-ds.odp_surf) * ds.temp_surf ** 4
    params['lambda_lw1'] = fit_linear_zero_mean_xr(ds.temp_surf, olr_surf_cont)
    params['B'], params['coef_phase_olr'] = get_fit_coef_complex_xr(ds.olr - olr_surf_cont, ds.temp_atm, ds.time)

    # Advection params
    params['lambda_adv'], params['coef_phase_adv'] = get_fit_coef_complex_xr(ds.adv_atmos, -ds.temp_atm, ds.time)

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
    temp_col_tend = apply_fit_complex_xr(temp_atm_deriv, coef_amp_col, coef_phase_col / 2 / np.pi)
    sphum_tend = mu * temp_atm_deriv
    c_a = c_p * p_integ_calc / g
    return c_a * (temp_col_tend + sphum_tend)


def get_approx_flux_atmos(temp_atm: xr.DataArray, temp_surf: xr. DataArray, swdn_toa: xr.DataArray,
                          sw_abs: xr.DataArray, lambda_const: xr.DataArray, lambda_a: xr.DataArray,
                          B: xr.DataArray, lambda_lw1: xr.DataArray, coef_phase_olr: xr.DataArray):
    r"""Approximate non-advective atmospheric energy-budget fluxes.

    Reconstructs the explicitly diagnosed terms on the right-hand side of the
    atmospheric energy budget, excluding atmospheric advection:

    $$
    \mathrm{flux}_{\mathrm{atmos}} =
    \mathrm{SW}_{\mathrm{abs}}(t)
    + \lambda(T_s - T_a)
    + \Lambda T_a
    - \lambda_{\mathrm{lw1}} T_s
    - B\left[1 - i\phi_{\mathrm{olr}}\right]T_a.
    $$

    All temperature and incoming solar-radiation anomalies are calculated
    relative to their time means. The phase correction
    $\phi_{\mathrm{olr}}$ is applied as a time shift to the atmospheric
    temperature contribution to outgoing longwave radiation.

    Args:
        temp_atm: Near-surface atmospheric temperature, $T_a$.
        temp_surf: Surface temperature, $T_s$.
        swdn_toa: Downward shortwave radiation at the top of the atmosphere.
        sw_abs: Fraction of top-of-atmosphere shortwave radiation absorbed by
            the atmosphere.
        lambda_const: Coefficient multiplying the surface--atmosphere
            temperature contrast, $\lambda$.
        lambda_a: Coefficient multiplying atmospheric temperature,
            $\Lambda$.
        B: Amplitude of the atmospheric contribution to outgoing longwave
            radiation.
        lambda_lw1: Coefficient for the surface-temperature-dependent
            longwave contribution to outgoing longwave radiation.
        coef_phase_olr: Phase correction for the atmospheric outgoing
            longwave-radiation contribution, $\phi_{\mathrm{olr}}$.

    Returns:
        Approximate atmospheric energy-budget flux convergence excluding
        advection, comprising absorbed shortwave radiation, turbulent and
        longwave surface-exchange terms, and the phase-shifted atmospheric
        outgoing-longwave-radiation term.
    """
    temp_atm = temp_atm - temp_atm.mean(dim='time')
    temp_surf = temp_surf - temp_surf.mean(dim='time')
    flux_abs = sw_abs * (swdn_toa - swdn_toa.mean(dim='time'))

    flux_linear = lambda_const * (temp_surf - temp_atm) + lambda_a * temp_atm - lambda_lw1 * temp_surf
    # Need to divide coef_phase by 2*np.pi for fraction of period to shift by
    flux_shift = apply_fit_complex_xr(temp_atm, -B, coef_phase_olr/2/np.pi)
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
    return apply_fit_complex_xr(temp_atm, -lambda_adv, coef_phase_adv / 2 / np.pi)


def get_approx_flux_surf(temp_atm: xr.DataArray, temp_surf: xr. DataArray,
                         lambda_const: xr.DataArray, lambda_a: xr.DataArray):
    temp_atm = temp_atm - temp_atm.mean(dim='time')
    temp_surf = temp_surf - temp_surf.mean(dim='time')
    return lambda_const * (temp_atm - temp_surf) - lambda_a * temp_atm

get_var_shift_xr = wrap_with_apply_ufunc(get_var_shift, input_core_dims=[['time'], [], [], ['time']],
                                         output_core_dims=[['time']])
