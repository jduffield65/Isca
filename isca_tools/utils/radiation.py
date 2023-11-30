import xarray as xr
import numpy as np
from netCDF4 import Dataset


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


def frierson_atmospheric_heating(ds: Dataset, albedo: float = 0) -> xr.DataArray:
    """
    Returns the atmospheric radiative heating rate from the surface and top of atmosphere energy fluxes. A negative
    value indicates that the atmosphere is cooling.

    This takes into account any radiation that is absorbed by the atmosphere on its way down from space to the surface,
    as specified through `tau_equator` and the amount reflected at the surface through the `albedo`.

    In *Isca*, there is no absorption of the shortwave radiation as it moves back up through the atmosphere to space
    after being reflected at the surface.

    Args:
        ds: Dataset for particular experiment, must contain:

            * `swdn_toa` - Incident shortwave radiation at the top of atmosphere.
                This is saved by *Isca* if the variable `swdn_toa` in the `two_stream` module is specified in the
                diagnostic table.
            * `swdn_sfc` - Net shortwave radiation absorbed at the surface i.e. incident - reflected.
                This is the negative of net upward shortwave radiation at the surface.
                This is saved by *Isca* if the variable `swdn_sfc` in the `two_stream` module is specified in the
                diagnostic table.
            * `lwup_sfc` - Upward longwave flux at the surface.
                This is saved by *Isca* if the variable `lwdn_sfc` in the `two_stream` module is specified in the
                diagnostic table.
            * `lwdn_sfc` - Downward longwave flux at the surface.
                This is saved by *Isca* if the variable `lwdn_sfc` in the `two_stream` module is specified in the
                diagnostic table.
            * `olr` - Outgoing longwave radiation at the top of atmosphere.
                This is saved by *Isca* if the variable `lwdn_sfc` in the `two_stream` module is specified in the
                diagnostic table.
        albedo: Fraction of incident shortwave radiation reflected by the surface.
            It is specified through the option `albedo_value` in the `mixed_layer_nml` namelist.

    Returns:
        Atmospheric radiative heating rate in $W/m^2$.

    """
    # return ds.swdn_toa - ds.swdn_sfc / (1 - albedo) + ds.lwup_sfc - ds.lwdn_sfc - ds.olr
    # do not need albedo as net up - net down shortwave at surface equals -ds.swdn_sfc
    return ds.swdn_toa - ds.swdn_sfc + ds.lwup_sfc - ds.lwdn_sfc - ds.olr


def get_heat_capacity(c_p: float, density: float, layer_depth: float):
    """
    Given heat capacity un units of $JK^{-1}kg^{-1}$, this returns heat capacity in units of $JK^{-1}m^{-2}$.

    Args:
        c_p: Specific heat at constant pressure.</br>
            Units: $JK^{-1}kg^{-1}$
        density: Density of substance (usually air or water).</br>
            Units: $kgm^{-3}$
        layer_depth: Depth of layer.</br>
            Units: $m$

    Returns:
        Heat capacity in units of $JK^{-1}m^{-2}$.
    """
    return c_p * density * layer_depth
