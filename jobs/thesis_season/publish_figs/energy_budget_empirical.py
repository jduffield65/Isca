import xarray as xr
import numpy as np
from typing import Optional, List

from isca_tools.thesis.surface_flux_taylor_2layer import get_sensitivity_lh, \
    get_sensitivity_sh, get_sensitivity_lw_surf
from isca_tools.utils.base import validate_params
from isca_tools.utils.constants import L_v, c_p, g

from .xr_funcs import apply_linear_zero_mean_xr, apply_fit_complex_xr, get_fit_complex_xr, spline_deriv_periodic_xr
from .load import day_seconds

def get_approx_flux_atmos(temp_atm: xr.DataArray, temp_surf: xr.DataArray,
                          swdn_toa: Optional[xr.DataArray]=None,
                          sw_abs: Optional[xr.DataArray]=None,
                          lambda_const: Optional[xr.DataArray]=None,
                          lambda_a: Optional[xr.DataArray]=None,
                          coef_phase_a: Optional[xr.DataArray] = None,
                          B: Optional[xr.DataArray]=None,
                          coef_phase_olr: Optional[xr.DataArray]=None,
                          lambda_adv: Optional[xr.DataArray] = None,
                          coef_phase_adv: Optional[xr.DataArray] = None) -> xr.DataArray:
    r"""Approximate non-advective atmospheric energy-budget fluxes.

    Reconstructs the explicitly diagnosed terms on the right-hand side of the
    atmospheric energy budget, excluding atmospheric advection:

    This approximates four separate terms in the atmospheric energy budget:

    * Shortwave absorbed: $F_{\mathrm{abs}} \approx f\mathrm{S}_{\mathrm{TOA}}$
    * Surface flux: $F_s = LH^{\uparrow} + SH^{\uparrow} + LW^{\uparrow} - LW^{\downarrow} \approx
    \lambda (T_s-T_a) + \Lambda\exp(- i\phi_{\Lambda})T_a$
    * OLR: $F_{\mathrm{OLR}} \approx -B\exp(- i\phi_B)T_a$
    * Meridional advection: $F_{\mathrm{adv}} \approx -\lambda_{\mathrm{adv}}\exp(- i\phi_{\mathrm{adv}})T_a$

    These can then be summed to give the net energy into the atmosphere (flux convergence):
    $F_{\mathrm{atmos}} = F_{\mathrm{abs}} + F_s + F_{\mathrm{OLR}} + F_{\mathrm{adv}}$

    Only certain terms can be returned depending on the inputs provided.

    All temperature and incoming solar-radiation anomalies are calculated
    relative to their time means. The phase coefficients $\phi_a$ and
    $\phi_{\mathrm{olr}}$ are implemented as time shifts of the relevant
    atmospheric-temperature contributions.

    Args:
        temp_atm: Near-surface atmospheric temperature, $T_a$.
        temp_surf: Surface temperature, $T_s$.
        swdn_toa: Downward shortwave radiation at the top of the atmosphere.
        sw_abs: Fraction of top-of-atmosphere shortwave radiation absorbed by
            the atmosphere, $f$.
        lambda_const: Coefficient multiplying the surface--atmosphere
            temperature contrast, $\lambda$.
        lambda_a: Amplitude of the atmospheric-temperature-dependent
            surface-flux term, $\Lambda$.
        coef_phase_a: Optional phase correction for the combined
            atmospheric-temperature-dependent surface-flux term, $\phi_{\Lambda}$.
            If `None`, this term is assumed to have no phase shift.
        B: Amplitude of the atmospheric contribution to outgoing longwave
            radiation.
        coef_phase_olr: Phase correction for the atmospheric outgoing
            longwave-radiation contribution, $\phi_B$.
        lambda_adv: Amplitude of the atmospheric advection response,
            $\lambda_{\mathrm{adv}}$.
        coef_phase_adv: Phase correction for atmospheric advection,
            $\phi_{\mathrm{adv}}$.

    Returns:
        Approximate atmospheric energy-budget flux convergence.
    """
    temp_atm = temp_atm - temp_atm.mean(dim='time')
    temp_surf = temp_surf - temp_surf.mean(dim='time')

    # SW absorbed
    if (swdn_toa is None) or (sw_abs is None):
        flux_abs = 0 * temp_surf
    else:
        flux_abs = sw_abs * (swdn_toa - swdn_toa.mean(dim='time'))

    # Surface flux
    if (lambda_const is None) and (lambda_a is None):
        flux_surf = 0 * temp_surf
    else:
        if lambda_const is None:
            lambda_const = 0 * lambda_a
        elif lambda_a is None:
            lambda_a = 0 * lambda_const
        if coef_phase_a is None:
            coef_phase_a = 0 * lambda_a
        flux_surf = apply_linear_zero_mean_xr(temp_surf - temp_atm, lambda_const, temp_atm, lambda_a, coef_phase_a)

    # OLR
    if B is not None:
        if coef_phase_olr is None:
            coef_phase_olr = 0 * B
        flux_olr = apply_fit_complex_xr(temp_atm, -B, coef_phase_olr)
    else:
        flux_olr = 0 * temp_surf

    # Advection
    if lambda_adv is not None:
        if coef_phase_adv is None:
            coef_phase_adv = 0 * lambda_adv
        flux_adv = apply_fit_complex_xr(temp_atm, -lambda_adv, coef_phase_adv)
    else:
        flux_adv = 0 * temp_surf
    return flux_abs + flux_surf + flux_olr + flux_adv



def get_approx_mse_tend(temp_atm: xr.DataArray, coef_amp_col: xr.DataArray,
                        coef_phase_col: xr.DataArray, mu: xr.DataArray,
                        coef_phase_mu: xr.DataArray,
                        p_integ_calc: xr.DataArray,
                        time: xr.DataArray) -> xr.DataArray:
    r"""Approximate the atmospheric moist-static-energy tendency.

    Reconstructs the reduced-model approximation to the atmospheric
    moist-static-energy tendency,

    $$
    C_a\left[\beta\exp(-i\phi_{\beta}) +
    \mu\exp(-i\phi_{\mu}) \right]
    \frac{\partial T_a}{\partial t},
    $$

    using the near-surface atmospheric temperature tendency. The column
    temperature tendency is scaled by $\beta$ and shifted in
    time according to $\phi_{\beta}$, while the specific-humidity
    contribution is represented by $\mu \partial T_a / \partial t$.

    Args:
        temp_atm: Near-surface atmospheric temperature, $T_a$.
        coef_amp_col: Amplitude factor relating column-mean and near-surface
            atmospheric temperature tendencies, $\beta$.
        coef_phase_col: Phase correction for the column-temperature tendency,
            $\phi_{\beta}$.
        mu: Moisture-related atmospheric heat-capacity correction, $\mu$, $\phi_{\mu}$.
        coef_phase_mu: Phase correction for the column-temperature tendency,
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
    sphum_tend = apply_fit_complex_xr(temp_atm_deriv, mu, coef_phase_mu)
    c_a = c_p * p_integ_calc / g
    return c_a * (temp_col_tend + sphum_tend)



def get_empirical_params(ds: xr.Dataset, const_p: bool = False,
                         include_params: Optional[List] = None,
                         exclude_params: Optional[List] = None) -> dict:
    r"""Fit empirical parameters for the seasonal surface--atmosphere model.

    The fitted parameters correspond to the coupled surface and atmospheric
    temperature-budget equations

    $$
    C_s \frac{\partial T_s}{\partial t}
    = (1 - \alpha)(1 - f)\mathrm{S}_{\mathrm{TOA}}
    + \lambda(T_a - T_s)
    - \lambda_a\exp(-i\phi_a)T_a,
    $$

    $$
    C_a\left[\beta_{\mathrm{col}}\exp(-i\phi_{\mathrm{col}}) + \mu \right]
    \frac{\partial T_a}{\partial t}
    = f\mathrm{S}_{\mathrm{TOA}}
    + \lambda(T_s - T_a)
    + \lambda_a\exp(-i\phi_a)T_a
    - B\exp(-i\phi_{B}) T_a
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

    Returns:
        Dictionary containing the fitted empirical model parameters:

        - `mu`: Moisture-related correction to atmospheric heat capacity,
          $\mu$.
        - `coef_phase_mu`: Phase correction associated with the column
          moisture tendency, $\phi_{\mu}$.
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
    _allowed_params = ['mu', 'coef_phase_mu', 'coef_amp_col', 'coef_phase_col',
                       'lambda_const', 'lambda_a', 'coef_phase_a', 'B',
                       'coef_phase_olr', 'lambda_adv', 'coef_phase_adv',
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
        params['mu'], params['coef_phase_mu'] = \
            get_fit_complex_xr(spline_deriv_periodic_xr(ds.time * day_seconds, ds.temp_atm),
                               spline_deriv_periodic_xr(ds.time * day_seconds, ds.sphum_col))
        params['mu'] *= L_v / c_p
        params['coef_amp_col_sphum'] = get_fit_complex_xr(ds.temp_atm, ds.temp_col_sphum)[0]
        # Account for column mean temp differing from lowest model level
        params['coef_amp_col'], params['coef_phase_col'] = \
            get_fit_complex_xr(ds.temp_atm, ds.temp_col, ds.time, 'coef_phase_col' in include_params)
    else:
        params['mu'], params['coef_phase_mu'] = \
            get_fit_complex_xr(spline_deriv_periodic_xr(ds.time * day_seconds, ds.temp_atm),
                               spline_deriv_periodic_xr(ds.time * day_seconds, ds.sphum_col * ds.p_integ_calc)
                               )
        params['mu'] *= L_v / c_p / ds.p_integ_calc.mean(dim='time')
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

    # Deal with phase delay of LH
    # Get what is left of LH after the temp_surf-temp_atm fit
    flux_lhe_resid = ds.flux_lhe - apply_linear_zero_mean_xr(ds.temp_surf - ds.temp_atm, params['lambda_const_lh'])
    # Refind the residual atmospheric effect taking into account of phase delay
    params['lambda_a_lh'], params['coef_phase_a_lh'] = get_fit_complex_xr(ds.temp_atm, flux_lhe_resid, ds.time,
                                                                          'coef_phase_a' in include_params)

    # Compute total lambda_a from combined residual
    flux_surf_resid = flux_t_resid + flux_lw_resid + flux_lhe_resid
    params['lambda_a'], params['coef_phase_a'] = get_fit_complex_xr(ds.temp_atm, flux_surf_resid, ds.time,
                                                                          'coef_phase_a' in include_params)
    # Compute coef_phase_a using only latent heat contribution to phase. Note negative in sin_coef
    # because coef is
    # params['lambda_a'], params['coef_phase_a'] = \
    #     coef_conversion(cos_coef=params['lambda_a_lh'] * np.cos(params['coef_phase_a_lh']) + params['lambda_a_lw'] -
    #                              params['lambda_a_sh'],
    #                     sin_coef=params['lambda_a_lh'] * np.sin(params['coef_phase_a_lh']), take_cos_sign=True)

    # OLR params
    params['B'], params['coef_phase_olr'] = get_fit_complex_xr(ds.temp_atm, ds.olr, ds.time,
                                                               'coef_phase_olr' in include_params)

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
