import numpy as np
from scipy.interpolate import CubicSpline
from typing import Optional


def apply_polyfit(x: np.ndarray, poly_coefs: np.ndarray, time: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Given the polynomial coefficients found by `np.polyfit` for fitting a polynomial
    of degree `len(polyfit)-1` of $x$ to $y$, this will return the approximation of $y$:

    $y_{approx} = \sum_{n=0}^{n_{deg}} \lambda_n x^n$

    where $\lambda_n=$`poly_coefs[-1-n]`.

    If $t=$`time` is provided, will assume `poly_coefs` is the output of `polyfit_with_phase` instead, so:

    $y_{approx} = \lambda_{phase} x(t-T/4) + \sum_{n=0}^{n_{deg}} \lambda_n x^n$

    where $\lambda_{phase}=$`poly_coefs[0]` and
    $x$ is assumed periodic with period $T=$`time[-1]-time[0]+time_spacing`.

    Args:
        x: `float [n_x]`</br>
            $x$ coordinates used to approximate $y$.
        poly_coefs: `float [n_deg+1]` or `float [n_deg+2]` if `time` given.</br>
            Polynomial coefficients as output by `np.polyfit`, lowest power last.</br>
            If `time` is provided, will assume output of `polyfit_with_phase` instead, so will have the additional
            coefficient $\lambda_{phase}=$`poly_coefs[0]`.
        time: `float [n_x]`</br>
            $x$ has the value `x[i]` at time `time[i]`. If `time` is provided, assumes $x$ is periodic
            such that $x$ at $t=$`time[-1]+time_spacing` is equal to `x[0]`.
            Also assumes `time_spacing` is a constant.

    Returns:
        y_approx: `float [n_x]`</br>
            Polynomial approximation to $y$.
    """
    if time is None:
        return np.sum([poly_coefs[-i - 1] * x ** i for i in range(len(poly_coefs))], axis=0)
    else:
        # In this case, poly_coefs are output of polyfit_with_phase so last coefficient is the phase coefficient
        y_approx = np.sum([poly_coefs[-i - 2] * x ** i for i in range(len(poly_coefs - 1))], axis=0)
        time_spacing = np.median(np.ediff1d(time))
        x_spline_fit = CubicSpline(np.append(time, time[-1] + time_spacing), np.append(x, x[0]),
                                   bc_type='periodic')
        period_length = time[-1] - time[0] + time_spacing
        return y_approx + poly_coefs[0] * x_spline_fit(time - period_length / 4)


def polyfit_with_phase(x: np.ndarray, y: np.ndarray,
                       deg: int, time: np.ndarray,
                       deg_phase_calc: int = 10) -> np.ndarray:
    """
    This fits a polynomial `y_approx(x) = p[0] * x**deg + ... + p[deg]` of degree `deg` to points (x, y) as `np.polyfit`
    but also includes additional phase shift term such that the total approximation for y is:

    $y_{approx} = \lambda_{phase} x(t-T/4) + \sum_{n=0}^{n_{deg}} \lambda_n x^n$

    where $\lambda_n=$`poly_coefs[-1-n]` and $\lambda_{phase}=$`poly_coefs[0]`.
    $x$ is assumed periodic with period $T=$`time[-1]-time[0]+time_spacing`.

    The phase component, `y_phase=`$\lambda_{phase} x(t-T/4)$, is found first from the residual of $y-y_{best}$,
    where $y_{best}$ is the polynomial approximation of degree `deg_phase_calc`.

    $\sum_{n=0}^{n_{deg}} \lambda_n x^n$ is then found by doing the normal polynomial approximation of degree
    `deg` to the residual $y-y_{phase}$.

    Args:
        x: `float [n_x]`</br>
            $x$ coordinates used to approximate $y$.
        y: `float [n_x]`</br>
            $y$ coordinate correesponding to each $x$.
        deg: Degree of the fitting polynomial.
        time: `float [n_x]`</br>
            $x$ has the value `x[i]` at time `time[i]`. If `time` is provided, assumes $x$ is periodic
            such that $x$ at $t=$`time[-1]+time_spacing` is equal to `x[0]`.
            Also assumes `time_spacing` is a constant.
        deg_phase_calc: Degree of the fitting polynomial to use in the phase term calculation.
            Should be a large integer.

    Returns:
        poly_coefs: `float [n_deg+2]`
            Polynomial coefficients, phase first and then normal output of `np.polyfit` with lowest power last.
    """
    coefs = np.zeros(deg + 2)  # last coef is phase coef
    time_spacing = np.median(np.ediff1d(time))
    x_spline_fit = CubicSpline(np.append(time, time[-1] + time_spacing), np.append(x, x[0]),
                               bc_type='periodic')
    gamma_best_polyfit = apply_polyfit(x, np.polyfit(x, y, deg_phase_calc)[0])
    period_length = time[-1] - time[0] + time_spacing
    # Use linalg to find coefficient not polyfit as know 0th order coefficient is 0 i.e. want y=mx not y=mx+c
    coefs[0] = np.linalg.lstsq(x_spline_fit(time - period_length / 4)[:, np.newaxis], y - gamma_best_polyfit)
    y_no_phase = y - apply_polyfit(x, coefs, time)      # residual after removing phase dependent term
    coefs[1:] = np.polyfit(x, y_no_phase, deg)[0]
    return coefs
