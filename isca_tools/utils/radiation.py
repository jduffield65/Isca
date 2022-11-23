import xarray as xr
import numpy as np


def frierson_sw_optical_depth(surface_pressure: xr.DataArray, tau_equator: float = 0, tau_lat_var: float = 0,
                              pressure_exponent: float = 4, ref_pressure: float = 101325) -> xr.DataArray:
    """
    Function to calculate shortwave surface optical depth, $\\tau_s$, as a function of latitude, $\\phi$, as performed in
    [Isca](https://execlim.github.io/Isca/modules/two_stream_gray_rad.html#frierson-byrne-schemes) for the *Frierson*
    and *Byrne* radiation schemes:

    $$
    \\tau_s(\\phi) = (1-\Delta \\tau^* \sin^2 \\phi)\\tau_{0e}^*\\big(\\frac{p_s}{p_0}\\big)^{\\kappa^*}
    $$

    All default values are the default values in *Isca* if the option is not specified in the relavent `namelist`.

    Args:
        surface_pressure: Surface pressure in *Pa* with dimensions of latitude, longitude and time, $p_s$.
            This is saved by *Isca* if the variable `ps` in the `dynamics` module is specified in the diagnostic table.
        tau_equator: Surface optical depth at the equator, $\\tau_{0e}^*$.
            It is specified through the option `atm_abs` in the `two_stream_gray_rad_nml` namelist.
        tau_lat_var: Variation of optical depth with latitude, $\Delta \\tau^*$.
            It is specified through the option `sw_diff` in the `two_stream_gray_rad_nml` namelist.
        pressure_exponent: Determines the variation of optical depth with pressure, $\\kappa^*$.
            It is specified through the option `solar_exponent` in the `two_stream_gray_rad_nml` namelist.
        ref_pressure: Reference pressure used by Isca in *Pa*.
            It is specified through the option `pstd_mks` in the `constants_nml` namelist.

    Returns:
        Shortwave surface optical depth with dimensions of latitude, longitude and time.
    """
    tau_surface = (1-tau_lat_var * np.sin(np.deg2rad(surface_pressure.lat))**2) * tau_equator
    return tau_surface * (surface_pressure/ref_pressure)**pressure_exponent


def frierson_net_toa_sw_dwn(insolation: xr.DataArray, surface_pressure: xr.DataArray, albedo: float = 0,
                            tau_equator: float = 0, tau_lat_var: float = 0, pressure_exponent: float = 4,
                            ref_pressure: float = 101325) -> xr.DataArray:
    """
    Function to calculate the net downward shortwave radiation at the top of atmosphere for the *Frierson*
    and *Byrne* radiation schemes.

    This takes into account any radiation that is absorbed by the atmosphere on its way down from space to the surface,
    as specified through `tau_equator` and the amount reflected at the surface through the `albedo`.

    In *Isca*, there is no absorption of the shortwave radiation as it moves back up through the atmosphere to space
    after being reflected at the surface.

    All default values are the default values in *Isca* if the option is not specified in the relavent `namelist`.

    Args:
        insolation: Incident shortwave radiation at the top of atmosphere
            with dimensions of latitude, longitude and time.
            This is saved by *Isca* if the variable `swdn_toa` in the `two_stream` module is specified in the
            diagnostic table.
        surface_pressure: Surface pressure in *Pa* with dimensions of latitude, longitude and time, $p_s$.
            This is saved by *Isca* if the variable `ps` in the `dynamics` module is specified in the diagnostic table.
        albedo: Fraction of incident shortwave radiation reflected by the surface.
            It is specified through the option `albedo_value` in the `mixed_layer_nml` namelist.
        tau_equator: Surface optical depth at the equator, $\\tau_{0e}^*$.
            It is specified through the option `atm_abs` in the `two_stream_gray_rad_nml` namelist.
        tau_lat_var: Variation of optical depth with latitude, $\Delta \\tau^*$.
            It is specified through the option `sw_diff` in the `two_stream_gray_rad_nml` namelist.
        pressure_exponent: Determines the variation of optical depth with pressure, $\\kappa^*$.
            It is specified through the option `solar_exponent` in the `two_stream_gray_rad_nml` namelist.
        ref_pressure: Reference pressure used by Isca in *Pa*.
            It is specified through the option `pstd_mks` in the `constants_nml` namelist.

    Returns:
        Net downward shortwave radiation at the top of atmosphere with dimensions of latitude, longitude and time.
    """
    tau = frierson_sw_optical_depth(surface_pressure, tau_equator, tau_lat_var, pressure_exponent, ref_pressure)
    return insolation*(1-albedo*np.exp(-tau))
