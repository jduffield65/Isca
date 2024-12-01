import numpy as np
from scipy.interpolate import CubicSpline
from typing import Optional
import warnings


def apply_polyfit(x: np.ndarray, poly_coefs: np.ndarray) -> np.ndarray:
    """
    Given the polynomial coefficients found by `np.polyfit` for fitting a polynomial
    of degree `len(polyfit)-1` of $x$ to $y$, this will return the approximation of $y$:

    $y_{approx} = \sum_{n=0}^{n_{deg}} \lambda_n x^n$

    where $\lambda_n=$`poly_coefs[-1-n]`.

    Args:
        x: `float [n_x]`</br>
            $x$ coordinates used to approximate $y$.
        poly_coefs: `float [n_deg+1]`</br>
            Polynomial coefficients as output by `np.polyfit`, lowest power last.</br>

    Returns:
        y_approx: `float [n_x]`</br>
            Polynomial approximation to $y$.
    """
    return np.sum([poly_coefs[-i - 1] * x ** i for i in range(len(poly_coefs))], axis=0)


def apply_polyfit_phase(x: np.ndarray, poly_coefs: np.ndarray) -> np.ndarray:
    """
    Given the polynomial coefficients found by `polyfit_phase` for fitting a polynomial
    of degree `len(polyfit)-2` of $x$ to $y$, this will return the approximation of $y$:

    $y_{approx} = \lambda_{phase} x(t-T/4) + \sum_{n=0}^{n_{deg}} \lambda_n x^n$

    where $\lambda_n=$`poly_coefs[-1-n]`, $\lambda_{phase}=$`poly_coefs[0]` and
    $x$ is assumed periodic with period $T=$.

    Args:
        x: `float [n_x]`</br>
            $x$ coordinates used to approximate $y$.
        poly_coefs: `float [n_deg+2]`</br>
            Polynomial coefficients as output by `polyfit_phase`, lowest power last.</br>
            $\lambda_{phase}=$`poly_coefs[0]` and $\lambda_n=$`poly_coefs[-1-n]`.

    Returns:
        y_approx: `float [n_x]`</br>
            Polynomial approximation to $y$.
    """
    # In this case, poly_coefs are output of polyfit_with_phase so first coefficient is the phase coefficient
    y_approx = np.sum([poly_coefs[-i - 1] * x ** i for i in range(len(poly_coefs) - 1)], axis=0)
    # time_spacing = np.median(np.ediff1d(time))
    # x_spline_fit = CubicSpline(np.append(time, time[-1] + time_spacing), np.append(x, x[0]),
    #                            bc_type='periodic')
    # period_length = time[-1] - time[0] + time_spacing
    shift_n_elements = int(np.round(x.size / 4))
    if shift_n_elements != x.size / 4:
        warnings.warn('Cannot shift by whole number of elements - may be better using spline')
    return y_approx + poly_coefs[0] * np.roll(x, shift_n_elements)


def polyfit_phase(x: np.ndarray, y: np.ndarray,
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
    # time_spacing = np.median(np.ediff1d(time))
    # x_spline_fit = CubicSpline(np.append(time, time[-1] + time_spacing), np.append(x, x[0]),
    #                            bc_type='periodic')
    # y_best_polyfit = apply_polyfit(x, np.polyfit(x, y, deg_phase_calc))
    # period_length = time[-1] - time[0] + time_spacing
    # # Use linalg to find coefficient not polyfit as know 0th order coefficient is 0 i.e. want y=mx not y=mx+c
    # coefs[0] = np.linalg.lstsq(x_spline_fit(time - period_length / 4)[:, np.newaxis], y - y_best_polyfit,
    #                            rcond=-1)[0][0]
    shift_n_elements = int(np.round(x.size / 4))
    if shift_n_elements != x.size / 4:
        warnings.warn('Cannot shift by whole number of elements - may be better using spline')
    y_best_polyfit = apply_polyfit(x, np.polyfit(x, y, deg_phase_calc))
    coefs[0] = np.linalg.lstsq(np.roll(x, shift_n_elements)[:, np.newaxis], y - y_best_polyfit, rcond=-1)[0][0]
    y_no_phase = y - apply_polyfit_phase(x, coefs)  # residual after removing phase dependent term
    coefs[1:] = np.polyfit(x, y_no_phase, deg)
    return coefs


def spline_integral(x: np.ndarray, dy_dx: np.ndarray, y0: float = 0, x0: Optional[float] = None,
                    x_return: Optional[np.ndarray] = None,
                    periodic: bool = False) -> np.ndarray:
    """
    Uses spline integration to solve for $y$ given $\\frac{dy}{dx}$ such that $y=y_0$ at $x=x_0$.

    Args:
        x: `float [n_x]`</br>
            Values of $x$ where $\\frac{dy}{dx}$ given, and used to fit the spline.
        dy_dx: `float [n_x]`</br>
            Values of $\\frac{dy}{dx}$ corresponding to $x$, and used to fit the spline.
        y0: Boundary condition, $y(x_0)=y_0$.
        x0: Boundary condition, $y(x_0)=y_0$.</br>
            If not given, will assume `x0=x_return[0]`.
        x_return: `float [n_x_return]`</br>
            Values of $y$ are returned for these $x$ values. If not given, will set to `x`.
        periodic: Whether to use periodic boundary condition.</br>
            If periodic expect $\\frac{dy}{dx}$ at $x=$`x[-1]+x_spacing` is equal to `dy_dx[0]`.

    Returns:
        y_return: `float [n_x_return]`</br>
            Values of $y$ corresponding to `x_return.
    """
    if periodic:
        x_spacing = np.median(np.ediff1d(x))
        spline_use = CubicSpline(np.append(x, x[-1] + x_spacing), np.append(dy_dx, dy_dx[0]), bc_type='periodic')
    else:
        spline_use = CubicSpline(x, dy_dx)
    if x_return is None:
        x_return = x
    if x0 is None:
        x0 = x_return[0]
    y = np.full_like(x_return, y0, dtype=float)
    for i in range(x_return.size):
        y[i] += spline_use.integrate(x0, x_return[i], extrapolate='periodic' if periodic else None)
    return y
