import numpy as np
from .constants import radius_earth


def divergence_2d(var_x: np.ndarray, var_y: np.ndarray, lat: np.ndarray, lon: np.ndarray, lat_axis: int = -2,
                  lon_axis: int = -1) -> np.ndarray:
    """
    Calculate the 2D divergence of a vector field with the given x and y components.

    Args:
        var_x: x component of the vector e.g. u for wind. Typical shape is `[n_time x n_pressure x n_lat x n_lon]`,
            but can accept all different shapes as long as lat and lon are included.
        var_y: y component of the vector e.g. v for wind. Typical shape is `[n_time x n_pressure x n_lat x n_lon]`,
            but can accept all different shapes as long as lat and lon are included.
        lat: `float [n_lat]`
            Latitude coordinates in degrees.
        lon: `float [n_lon]`
            Longitude coordinates in degrees.
        lat_axis: Axis of var_x and var_y which corresponds to latitude.
        lon_axis: Axis of var_x and var_y which corresponds to longitude.

    Returns:
        Divergence of var with same shape as `var_x` and `var_y`. Units are those of `var` multiplied by $m^{-1}$.
    """
    if var_x.shape != var_y.shape:
        raise ValueError('var_x and var_y should have the same shapes')
    var_shape = np.ones(len(var_x.shape), dtype=int)
    var_shape[lat_axis] = len(lat)
    cos_lat = np.asarray(np.cos(np.deg2rad(lat))).reshape(var_shape)
    div_x = np.gradient(var_x, np.deg2rad(lon), axis=lon_axis) / (radius_earth * cos_lat)
    div_y = np.gradient(var_y * cos_lat, np.deg2rad(lat), axis=lat_axis) / (radius_earth * cos_lat)
    return div_x + div_y


def grad_x(var: np.ndarray, lat: np.ndarray, lon: np.ndarray, lat_axis: int = -2, lon_axis: int = -1) -> np.ndarray:
    """
    Finds the gradient in the x direction of a scalar field.

    Args:
        var: Scalar field to find gradient of. Typical shape is `[n_time x n_pressure x n_lat x n_lon]`,
            but can accept all different shapes as long as lat and lon are included.
        lat: `float [n_lat]`
            Latitude coordinates in degrees.
        lon: `float [n_lon]`
            Longitude coordinates in degrees.
        lat_axis: Axis of var_x and var_y which corresponds to latitude.
        lon_axis: Axis of var_x and var_y which corresponds to longitude.

    Returns:
        Gradient of `var` in x direction with same shape as `var`. Units are those of `var` multiplied by $m^{-1}$.
    """
    var_shape = np.ones(len(var.shape), dtype=int)
    var_shape[lat_axis] = len(lat)
    cos_lat = np.asarray(np.cos(np.deg2rad(lat))).reshape(var_shape)
    return np.gradient(var, np.asarray(np.deg2rad(lon)), axis=lon_axis) / (radius_earth * cos_lat)


def grad_y(var: np.ndarray, lat: np.ndarray, lat_axis: int = -2) -> np.ndarray:
    """
    Finds the gradient in the y direction of a scalar field.

    Args:
        var: Scalar field to find gradient of. Typical shape is `[n_time x n_pressure x n_lat x n_lon]`,
            but can accept all different shapes as long as lat and lon are included.
        lat: `float [n_lat]`
            Latitude coordinates in degrees.
        lat_axis: Axis of var_x and var_y which corresponds to latitude.

    Returns:
        Gradient of `var` in y direction with same shape as `var`. Units are those of `var` multiplied by $m^{-1}$.
    """
    return np.gradient(var, np.asarray(np.deg2rad(lat)), axis=lat_axis) / radius_earth
