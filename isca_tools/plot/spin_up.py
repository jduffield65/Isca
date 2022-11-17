import matplotlib.pyplot as plt
import xarray as xr
from ..utils import area_weighting


def plot_spin_up(olr: xr.DataArray, sw_net_down: xr.DataArray, t_surf: xr.DataArray, ax: plt.Axes):
    """
    Function to plot how net TOA flux and mean global surface temperature evolve with time.
    Net flux should converge to zero, once spin up has finished.

    `olr`, `sw_net_down` and `t_surf` are just variables saved in the dataset *.nc* file produced by *Isca*
    e.g. `t_surf = ds.t_surf` if `ds` is the dataset for the experiment.

    Args:
        olr: Outgoing Longwave Radiation with time, longitude and latitude dimensions.
        sw_net_down: Net downward shortwave radiation at top of atmosphere, accounting for any reflected due to
            the `albedo` or absorbed in the atmosphere due to shortwave optical depth.
            Dimensions are time, longitude, latitude.
        t_surf: Surface temperature with time, longitude and latitude dimensions.
        ax: Axes to plot results on.
    """
    # When area weighted summed over whole globe, TOA OLR-SW should converge to zero
    olr_sum = area_weighting(olr).sum(dim=['lon', 'lat'])
    short_wave_sum = area_weighting(sw_net_down).sum(dim=['lon', 'lat'])
    net_flux = olr_sum - short_wave_sum
    net_flux.plot.line(ax=ax, color='b')
    ax.set_ylabel('TOA Net Outgoing FLux / $Wm^{-2}$\n$OLR - SW_{net}$', color='b')
    # Add second axes to show how the global average surface temperature evolves
    ax2 = ax.twinx()
    t_surf_mean = area_weighting(t_surf).mean(dim=['lon', 'lat']) - 273.15  # In Celsius
    t_surf_mean.plot.line(ax=ax2, color='r')
    ax2.set_ylabel('Surface Temperature / $°C$', color='r')
    ax.set_xlabel(t_surf.time.units)
