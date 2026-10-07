# Xarray wrapped versions of isca_tools functions
# Must print __doc__ rather than hover over to see documentation e.g., `print(get_sw_abs_amp_xr.__doc__)`
# or `help(get_sw_abs_amp_xr)`
from isca_tools.thesis.surface_flux_taylor_2layer import get_temp_from_sphum_sat
import isca_tools.utils.numerical as numerical
from isca_tools.utils.radiation import get_sw_abs_amp
import isca_tools.utils.fourier as fourier
from isca_tools.utils.xarray import wrap_with_apply_ufunc

get_sw_abs_amp_xr = wrap_with_apply_ufunc(
    get_sw_abs_amp,
    input_core_dims=[['time'], ['time'], ['time']],
    output_core_dims=[[]])

get_temp_from_sphum_sat_xr = wrap_with_apply_ufunc(
    get_temp_from_sphum_sat,
    input_core_dims=[[], []],
    output_core_dims=[[]])


# Numerical functions
spline_deriv_periodic_xr = wrap_with_apply_ufunc(
    numerical.spline_deriv_periodic,
    input_core_dims=[['time'], ['time']],
    output_core_dims=[['time']])

get_fit_coef_complex_xr = wrap_with_apply_ufunc(
    numerical.get_fit_coef_complex,
    input_core_dims=[['time'], ['time'], ['time']],
    output_core_dims=[[], []])


# Fourier Functions
get_fourier_fit_xr = wrap_with_apply_ufunc(
    fourier.get_fourier_fit,
    input_core_dims=[['time'], ['time']],
    output_core_dims=[['time'], ['harmonic'], ['harmonic']])

fourier_series_xr = wrap_with_apply_ufunc(
    fourier.fourier_series,
    input_core_dims=[['time'], ['harmonic'], ['harmonic']],
    output_core_dims=[['time']])
