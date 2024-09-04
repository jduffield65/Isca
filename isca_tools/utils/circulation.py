import xarray as xr
import numpy as np
from scipy import integrate
from .constants import radius_earth, g


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
            Must be ascending so first value is closest to surface. Units: *Pa*.
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
