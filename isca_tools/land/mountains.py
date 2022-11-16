import numpy as np


def mountain_range_height(lon_array: np.ndarray, lat_array: np.ndarray, mountain: str = 'rockys') -> np.ndarray:
    """
    Returns the height of the land at each latitude, longitude coordinate for a given mountain range.

    Args:
        lon_array: `float [n_lat x n_lon]`.</br>
            Array indicating the longitude at each (latitude, longitude) coordinate in the grid used for the experiment.
        lat_array: `float [n_lat x n_lon]`.</br>
            Array indicating the latitude at each (latitude, longitude) coordinate in the grid used for the experiment.
        mountain: There are 2 options indicating different mountain ranges:

            * `rockys`
            * `tibet`

    Returns:
        `h_arr`: `float [n_lat x n_lon]`</br>
            `h_arr[lat, lon]` is the height of the land at the coordinate (`lat`, `lon`) in meters.
            Most coordinates will be 0.

    """
    if mountain.lower() == 'rockys':
        # Rockys from Sauliere 2012
        h_0 = 2670
        central_lon = 247.5
        central_lat = 40
        L_1 = 7.5
        L_2 = 20
        gamma_1 = 42
        gamma_2 = 42
        delta_1 = ((lon_array - central_lon) * np.cos(np.radians(gamma_1)) + (lat_array - central_lat) *
                   np.sin(np.radians(gamma_1))) / L_1
        delta_2 = (-(lon_array - central_lon) * np.sin(np.radians(gamma_2)) + (lat_array - central_lat) *
                   np.cos(np.radians(gamma_2))) / L_2
        h_arr = h_0 * np.exp(-(delta_1 ** 2. + delta_2 ** 2.))
    elif mountain.lower() == 'tibet':
        # Tibet from Sauliere 2012
        h_0 = 5700.
        central_lon = 82.5
        central_lat = 28
        L_1 = 12.5
        L_2 = 12.5
        gamma_1 = -49.5
        gamma_2 = -18
        delta_1 = ((lon_array - central_lon) * np.cos(np.radians(gamma_1)) + (lat_array - central_lat) *
                   np.sin(np.radians(gamma_1))) / L_1
        delta_2 = (-(lon_array - central_lon) * np.sin(np.radians(gamma_2)) + (lat_array - central_lat) *
                   np.cos(np.radians(gamma_2))) / L_2
        h_arr_tibet_no_amp = np.exp(-(delta_1 ** 2)) * (1 / delta_2) * np.exp(-0.5 * (np.log(delta_2)) ** 2)
        # For some reason my maximum value of h_arr_tibet_no_amp > 1. Renormalise so h_0 sets amplitude.
        maxval = np.nanmax(h_arr_tibet_no_amp)
        h_arr = (h_arr_tibet_no_amp / maxval) * h_0
    else:
        raise ValueError(f"mountain was given as {mountain} but it must be 'rockys' or 'tibet'")
    # make sure exponentials are cut at some point - use the value from p70 of Brayshaw's thesis.
    set_to_zero_height_ind = (h_arr / h_0 <= 0.05)
    h_arr[set_to_zero_height_ind] = 0
    return h_arr


def gaussian_mountain(lon_array: np.ndarray, lat_array: np.ndarray, central_lat: float, central_lon: float,
                      radius_degrees: float, std_dev: float, height: float) -> np.ndarray:
    """
    Returns the height of the land at each latitude, longitude coordinate for a single gaussian mountain.

    Args:
        lon_array: `float [n_lat x n_lon]`.</br>
            Array indicating the longitude at each (latitude, longitude) coordinate in the grid used for the experiment.
        lat_array: `float [n_lat x n_lon]`.</br>
            Array indicating the latitude at each (latitude, longitude) coordinate in the grid used for the experiment.
        central_lat: Latitude coordinate of mountain in degrees.
        central_lon: Longitude coordinate of mountain in degrees.
        radius_degrees: Radius of mountain in degrees. Altitude at a distance from the center greater than this will
            be set to 0, so set to very high number to ignore this functionality. Typical: 20.
        std_dev: Standard deviation indicating how steep the mountain is. The smaller the value, the steeper the
            mountain. Units are degrees and typical value would be 10.
        height: Height of mountain peak in meters.

    Returns:
        `h_arr`: `float [n_lat x n_lon]`</br>
            `h_arr[lat, lon]` is the height of the land at the coordinate (`lat`, `lon`) in meters.
            Most coordinates will be 0.

    """
    rsqd_array = np.sqrt((lon_array - central_lon) ** 2. + (lat_array - central_lat) ** 2.)
    h_arr = height * np.exp(-(rsqd_array**2.)/(2.*std_dev**2.))
    h_arr[rsqd_array > radius_degrees] = 0  # Make sure height goes to 0 at some point
    return h_arr
