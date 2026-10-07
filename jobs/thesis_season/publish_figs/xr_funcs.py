# Xarray wrapped versions of isca_tools functions
# Must print __doc__ rather than hover over to see documentation e.g., `print(get_sw_abs_amp_xr.__doc__)`
# or `help(get_sw_abs_amp_xr)`
import xarray as xr
from typing import Optional, Tuple
from isca_tools.thesis.surface_flux_taylor_2layer import get_temp_from_sphum_sat
import isca_tools.utils.numerical as numerical
from isca_tools.utils.radiation import get_sw_abs_amp
import isca_tools.utils.fourier as fourier
from isca_tools.utils.xarray import wrap_with_apply_ufunc

# Physical variable functions
get_sw_abs_amp_xr = wrap_with_apply_ufunc(
    get_sw_abs_amp,
    input_core_dims=[['time'], ['time'], ['time']],
    output_core_dims=[[]])

get_temp_from_sphum_sat_xr = wrap_with_apply_ufunc(
    get_temp_from_sphum_sat,
    input_core_dims=[[], []],
    output_core_dims=[[]])


# Numerical functions - getting fit
spline_deriv_periodic_xr = wrap_with_apply_ufunc(
    numerical.spline_deriv_periodic,
    input_core_dims=[['time'], ['time']],
    output_core_dims=[['time']])

fit_linear_zero_mean_xr_1 = wrap_with_apply_ufunc(
    lambda x1, y: numerical.fit_linear_zero_mean(x1, y, x2=None)[0],
    input_core_dims=[['time'], ['time']],
    output_core_dims=[[]],
)

fit_linear_zero_mean_xr_2 = wrap_with_apply_ufunc(
    lambda x1, y, x2: numerical.fit_linear_zero_mean(x1, y, x2=x2),
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

get_fit_coef_complex_xr = wrap_with_apply_ufunc(
    numerical.get_fit_coef_complex,
    input_core_dims=[['time'], ['time'], ['time']],
    output_core_dims=[[], []])

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


# Numerical - applying fit
apply_linear_zero_mean_xr_1 = wrap_with_apply_ufunc(
    numerical.apply_linear_zero_mean,
    input_core_dims=[['time'], []],
    output_core_dims=[['time']],
)

apply_linear_zero_mean_xr_2 = wrap_with_apply_ufunc(
    numerical.apply_linear_zero_mean,
    input_core_dims=[['time'], [], ['time'], []],
    output_core_dims=[['time']],
)

apply_fit_complex_xr = wrap_with_apply_ufunc(numerical.apply_fit_complex, input_core_dims=[['time'], [], []],
                                             output_core_dims=[['time']])

def apply_linear_zero_mean_xr(x1: xr.DataArray, a: xr.DataArray, x2: Optional[xr.DataArray]=None,
                              b: Optional[xr.DataArray]=None, b_phase: Optional[xr.DataArray] = None) -> xr.DataArray:
    r"""Apply a linear model to one or two mean-centred predictors.

    Removes the temporal mean from each supplied predictor before applying a
    no-intercept linear model. With one predictor, computes

    $$
    \hat{y}' = a x_1'.
    $$

    With two unshifted predictors, computes

    $$
    \hat{y}' = a x_1' + b x_2'.
    $$

    If `b_phase` is supplied, applies a phase shift to the second predictor
    before scaling it by $b$:

    $$
    \hat{y}' = a x_1' + b \, x_{2, \mathrm{shift}}',
    $$

    where $x_{2, \mathrm{shift}}'$ is the mean-centred version of $x_2$
    shifted by `b_phase` using `apply_fit_complex_xr`.

    Args:
        x1: First predictor. Must contain a `time` dimension and be
            broadcast-compatible with `a`.
        a: Fitted coefficient for `x1`. Typically the first output from
            `fit_linear_zero_mean_xr`.
        x2: Optional second predictor. Must contain a `time` dimension and be
            broadcast-compatible with `x1`, `a`, and `b`.
        b: Fitted coefficient for `x2`. Required when `x2` is supplied.
            Typically the second output from `fit_linear_zero_mean_xr`.
        b_phase: Optional phase shift, in radians, applied to `x2` before
            multiplying it by `b`. Must be broadcast-compatible with `x2`
            excluding its `time` dimension. If `None`, `x2` is used without
            a phase shift.

    Returns:
        Predicted response anomaly as an `xarray.DataArray`, with the
        broadcast dimensions of the supplied predictors and coefficients.

    Notes:
        Each predictor is centred independently over `time`:

        $$
        x_i' = x_i - \overline{x_i}.
        $$

        When `b_phase` is supplied, the second contribution is evaluated
        separately because it uses a temporally shifted version of `x2`.
        This function does not add back the mean of the response variable.
    """
    if x2 is None:
        return apply_linear_zero_mean_xr_1(x1, a)
    elif b_phase is None:
        return apply_linear_zero_mean_xr_2(x1, a, x2, b)
    else:
        # If have a phase delay for fitting to x2 then must do separately
        return apply_linear_zero_mean_xr_1(x1, a) + apply_fit_complex_xr(x2 - x2.mean(dim='time'), b, b_phase)



# Fourier Functions
get_fourier_fit_xr = wrap_with_apply_ufunc(
    fourier.get_fourier_fit,
    input_core_dims=[['time'], ['time']],
    output_core_dims=[['time'], ['harmonic'], ['harmonic']])

fourier_series_xr = wrap_with_apply_ufunc(
    fourier.fourier_series,
    input_core_dims=[['time'], ['harmonic'], ['harmonic']],
    output_core_dims=[['time']])
