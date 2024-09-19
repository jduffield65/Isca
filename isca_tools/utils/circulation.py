import xarray as xr
import numpy as np
from scipy import integrate
from .constants import radius_earth, g, rot_earth, R


def get_stream(v: np.ndarray, p: np.ndarray, lat: np.ndarray, ax_p: int = 0) -> np.ndarray:
    """
    Computes the streamfunction $\psi$ at latitude $\phi$ and pressure $p$
    [according to](https://sandrolubis.wordpress.com/2012/06/17/mass-streamfunction/):

    $\psi(\phi, p) = \int_0^z v\\rho dx dz = 2\pi a \cos \phi\int_0^z v\\rho dz =
    -\\frac{2\pi a \cos \phi}{g} \int_{p_{surf}}^p vdp$

    Args:
        v: `float [n_p_levels, n_lat]`. </br>
            Meridional velocity at each pressure and latitude. Units: *m/s*.
        p: `float [n_p_levels]`.</br>
            Pressure levels corresponding to meridional velocity.</br>
            Must be descending so first value is closest to surface. Units: *Pa*.
        lat: `float [n_lat]`.</br>
            Latitude corresponding to meridional velocity. Units: *degrees*.
        ax_p: Axis corresponding to pressure levels in $v$.

    Returns:
        `float [n_lat]`.</br>
            Streamfunction at pressure level given by `p[-1]`. Units: *kg/s*.

    """
    if len(p) > 1:
        if p[1] > p[0]:
            raise ValueError(f'Pressure is not in correct order, expect p[0] to be surface value but it is {p[0]}Pa')
    cos_lat = np.cos(np.deg2rad(lat))
    stream = -2 * np.pi * radius_earth * cos_lat / g * integrate.simpson(v, p, axis=ax_p)
    return stream


def get_u_thermal(temp: np.ndarray, p: np.ndarray, lat: np.ndarray, ax_p: int = 0, ax_lat: int = 1) -> np.ndarray:
    """
    Computes thermal wind at pressure $p$ and latitude $\phi$ according to Equation 1
    in [shaw_2023](https://www.nature.com/articles/s41558-023-01884-1) paper:

    $u_T(p, \phi) = \int_{p_s}^{p}\\frac{R}{fap'}\\frac{\partial T}{\partial \phi} dp'$

    where $p_s$ is near-surface pressure, $f$ is the coriolis parameter, $R$ is the gas constant for dry air,
    $a$ is the radius of the Earth and $T$ is temperature.

    Args:
        temp: `float [n_p_levels, n_lat, ...]`. </br>
            Temperature at each pressure and latitude. Units: *K*.
        p: `float [n_p_levels]`.</br>
            Pressure levels corresponding to meridional velocity.</br>
            Must be descending so first value is closest to surface. Units: *Pa*.
        lat: `float [n_lat]`.</br>
            Latitude corresponding to meridional velocity. Units: *degrees*.
        ax_p: Axis corresponding to pressure levels in `temp`.</br>
            Must be $0$ or $1$.
        ax_lat: Axis corresponding to latitude in `temp`.</br>
            Must be $0$ or $1$.

    Returns:
         `float [n_lat, ...]`.</br>
            Thermal wind at pressure level given by `p[-1]`. Units: *m/s*.
    """
    if len(p) > 1:
        if p[1] > p[0]:
            raise ValueError(f'Pressure is not in correct order, expect p[0] to be surface value but it is {p[0]}Pa')
    n_ax = len(temp.shape)  # all axis 2 or higher are not p or lat
    ax_expand_dims = list(np.append(np.asarray([ax_lat]), np.arange(2, n_ax)))
    integrand = np.gradient(temp, np.deg2rad(lat), axis=ax_lat) / np.expand_dims(p, axis=ax_expand_dims)
    f_coriolis = 2 * rot_earth * np.sin(np.deg2rad(lat).to_numpy())
    if n_ax > 2:
        f_coriolis = np.expand_dims(f_coriolis, list(np.arange(2, n_ax) - 1))
    return integrate.simpson(integrand, p, axis=ax_p) * R / (radius_earth * f_coriolis)
