# Outdatated version of above_quant_2m_500hPa/save_info
# This considers vertical coupling between two model levels rather than REFHT and 500hPa
from isca_tools import cesm
from isca_tools.papers.byrne_2021 import get_quant_ind
from isca_tools.utils.moist_physics import moist_static_energy, sphum_sat
from isca_tools.utils import get_memory_usage, len_safe
import sys
import os
import numpy as np
import f90nml
import logging
import time
import xarray as xr
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'lat_quant')))
from save_quant import has_out_of_range, get_exp_info_dict, get_ds
# Set up logging configuration to output to console and don't output milliseconds, and stdout so saved to out file
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S',
                    stream=sys.stdout)


def main(input_file_path: str):
    logger = logging.getLogger()    # for printing to console time info
    logger.info("Start")
    exp_info = get_exp_info_dict(input_file_path)
    ds, ds_land = get_ds(exp_info['exp_name'], exp_info['archive_dir'],
                         exp_info['chunks_time'], exp_info['chunks_lat'], exp_info['chunks_lon'],
                         exp_info['load_parallel'], exp_info['p_ft_approx_guess'],
                         exp_info['p_surf_approx_guess'],
                         exp_info['year_first'], exp_info['year_last'],
                         exp_info['month_nh'], exp_info['month_sh'],
                         exp_info['lat_min'], exp_info['lat_max'],
                         exp_info['lon_min'], exp_info['lon_max'],
                         logger=logger)
    logger.info(f"Finished lazy-loading datasets | Memory used {get_memory_usage()/1000:.1f}GB")

    # Do chunking after open - for checking timings
    # ds_land = ds_land.chunk({"time": exp_info['chunks_time'], "lat": exp_info['chunks_lat'],
    #                            "lon": exp_info['chunks_lon']})  # Do chunking after open
    # logger.info(f"Finished chunking land data | Memory used {get_memory_usage() / 1000:.1f}GB")
    # ds = ds.chunk({"time": exp_info['chunks_time'], "lat": exp_info['chunks_lat'],
    #                "lon": exp_info['chunks_lon']})
    # logger.info(f"Finished chunking atmospheric data | Memory used {get_memory_usage()/1000:.1f}GB")

    # Fully load data if chose to do that
    if exp_info['load_all_at_start']:
        ds_land.load()
        logger.info(f"Fully loaded land data | Memory used {get_memory_usage() / 1000:.1f}GB")
        ds.load()
        logger.info(f"Fully loaded atmospheric data | Memory used {get_memory_usage() / 1000:.1f}GB")

    # Deal with variables that are constant in time - weighting and landmask
    # Use max rather than isel(time=0, drop=True) in case NH and SH have different times
    ds['gw'] = ds['gw'].max(dim='time')
    ds_land['landmask'] = ds_land['landmask'].max(dim='time') > 0

    # Get pressure info
    ind_surf = 0
    ind_ft = 1
    # use max not isel(time=0) in case have different months for NH and SH
    p_ref = float(ds.P0.max())
    hybrid_a_coef_ft = float(ds.hyam.isel(lev=ind_ft).max())
    hybrid_b_coef_ft = float(ds.hybm.isel(lev=ind_ft).max())
    p_ft_approx = float(ds.lev[ind_ft]) * 100
    p_surf_approx = float(ds.lev[ind_surf]) * 100


    # Initialize dict to save output data - n_surf (land or ocean) x n_lat x n_quant
    # Initialize as nan, so remains as Nan if not enough data to perform calculation
    n_lat = ds.lat.size
    n_lon = ds.lon.size
    n_lev = 2  # find quantile by looking for all values in quantile range between quant_use-quant_range to quant_use+quant_range
    n_quant = len_safe(exp_info['quant'])
    output_info = {var: np.full((n_quant, n_lat, n_lon), np.nan, dtype=float) for var in
                   ['rh', 'mse', 'mse_sat_ft', 'mse_lapse',
                    'mse_sat_ft_p_approx', 'mse_lapse_p_approx', 'pressure_ft', 'SOILLIQ']}
    for var in ['T', 'Q', 'Z3']:
        output_info[var] = np.full((n_quant, n_lat, n_lon, n_lev), np.nan, dtype=float)         # have pressure dim as well
    var_keys = [key for key in output_info.keys()]
    for var in var_keys:
        output_info[var + '_std'] = np.full_like(output_info[var], np.nan, dtype=float)
    output_info['use_in_calc'] = np.zeros((n_quant, n_lat, ds.lon.size, ds.time.size), dtype=bool)
    # output_info['lon_most_common'] = np.zeros((n_surf, n_quant, n_lat))
    # output_info['lon_most_common_freq'] = np.zeros((n_surf, n_quant, n_lat), dtype=int)
    # output_info['n_grid_points'] = np.zeros((n_surf, n_lat), dtype=int)  # number of grid points used at each location
    # Record approx number of days used in quantile calculation. If quant_range=0.5 and 1 year used, this is just 0.01*365=3.65
    # n_days_quant = get_quant_ind(np.arange(ds.time.size * n_lat), quant_use[0], quant_range,
    #                                             quant_range).size / n_lat

    coords = {'quant': np.array(exp_info['quant'], ndmin=1),
              'lev': ds.lev, 'lat': ds.lat, 'lon': ds.lon, 'time': ds.time}

    # Coordinate info to convert to xarray datasets
    output_dims = {var: ['quant', 'lat', 'lon'] for var in output_info}
    for var in ['T', 'Q', 'Z3']:
        output_dims[var] = ['quant', 'lat', 'lon', 'lev']
        output_dims[var+'_std'] = ['quant', 'lat', 'lon', 'lev']
    output_dims['use_in_calc'] = ['quant', 'lat', 'lon', 'time']

    logger.info(f"Starting iteration over {n_lat} latitudes, {n_lon} longitudes, and {n_quant} quantiles")

    # Loop through and get quantile info at each latitude and surface
    for i in range(n_lat):
        for k in range(n_lon):
            time_log = {'load': 0, 'calc': 0, 'start': time.time()}
            ds_latlon = ds.isel(lat=i, lon=k, drop=True)
            ds_latlon_land = ds_land.isel(lat=i, lon=k, drop=True)
            if not exp_info['load_all_at_start']:
                ds_latlon.load()
                ds_latlon_land.load()
            time_log['load'] += time.time() - time_log['start']

            time_log['start'] = time.time()
            is_land = ds_latlon_land.landmask.sum() > 0

            # if is_surf.sum() == 0:
            #     # If surface not at this latitude, record no data
            #     continue
            #
            # # if surf == 'land':
            # #     soil_liq_use = ds_land_lat.SOILLIQ.sel(lon=is_surf)
            # #     if not exp_info['load_all_at_start']:
            # #         soil_liq_use.load()
            # # output_info['n_grid_points'][k, i] = ds_use.lon.size
            # time_log['load'] += time.time() - time_log['start']
            time_log['start'] = time.time()
            for j in range(n_quant):
                # get indices corresponding to given near-surface temp quantile
                quant_mask = get_quant_ind(ds_latlon.T.isel(lev=ind_surf), coords['quant'][j],
                                           exp_info['quant_range_below'], exp_info['quant_range_above'],
                                           return_mask=True, av_dim='time')
                if quant_mask.sum() == 0:
                    logger.info(f"Latitude {i + 1}/{n_lat} | Longitude {k+1}/{n_lon}: no data found for quant={coords['quant'][j]}")
                    continue
                ds_use = ds_latlon.where(quant_mask)
                output_info['use_in_calc'][j, i, k] = quant_mask
                var_use = {}
                var_use['T'] = ds_use.T
                var_use['Q'] = ds_use.Q
                var_use['Z3'] = ds_use.Z3
                var_use['rh'] = ds_use.Q.isel(lev=ind_surf) / sphum_sat(ds_use.T.isel(lev=ind_surf), p_surf_approx)
                var_use['mse'] = moist_static_energy(ds_use.T.isel(lev=ind_surf), ds_use.Q.isel(lev=ind_surf),
                                                     ds_use.Z3.isel(lev=ind_surf))
                var_use['mse_sat_ft_p_approx'] = moist_static_energy(ds_use.T.isel(lev=ind_ft),
                                                                     sphum_sat(ds_use.T.isel(lev=ind_ft), p_ft_approx),
                                                                     ds_use.Z3.isel(lev=ind_ft))
                var_use['mse_lapse_p_approx'] = var_use['mse'] - var_use['mse_sat_ft_p_approx']
                var_use['pressure_ft'] = cesm.get_pressure(ds_use.PS, p_ref, hybrid_a_coef_ft, hybrid_b_coef_ft)
                var_use['mse_sat_ft'] = moist_static_energy(ds_use.T.isel(lev=ind_ft),
                                                            sphum_sat(ds_use.T.isel(lev=ind_ft), var_use['pressure_ft']),
                                                            ds_use.Z3.isel(lev=ind_ft))
                var_use['mse_lapse'] = var_use['mse'] - var_use['mse_sat_ft']
                if is_land:
                    var_use['SOILLIQ'] = ds_latlon_land.SOILLIQ.where(quant_mask)
                for key in var_use:
                    output_info[key][j, i, k] = var_use[key].mean(dim='time')
                    output_info[key + '_std'][j, i, k] = var_use[key].std(dim='time')
                # lon_use = np.unique(ds_use.lon[use_ind], return_counts=True)
                #
                # # Record most common specific coordinate within grid to see if most of days are at a given location
                # output_info['lon_most_common'][k, j, i] = lon_use[0][lon_use[1].argmax()]
                # output_info['lon_most_common_freq'][k, j, i] = lon_use[1][lon_use[1].argmax()]
            time_log['calc'] += time.time() - time_log['start']
            # if (i+1) == 1 or (i+1) == n_lat or (i+1) % 10 == 0:
            # # Log info on 1st, last and every 10th latitude
            logger.info(f"Latitude {i + 1}/{n_lat} | Longitude {k+1}/{n_lon}: Loading took {time_log['load']:.1f}s |"
                        f" Calculation took {time_log['calc']:.1f}s | "
                        f"Memory used {get_memory_usage()/1000:.1f}GB")

    # Convert output dict into xarray dataset
    # Convert individual arrays
    for var in output_info:
        output_info[var] = xr.DataArray(output_info[var], dims=output_dims[var],
                                        coords={key: coords[key] for key in output_dims[var]})
    # Add pressure level info to variables at single level
    for var in ['rh', 'mse', 'mse_sat_ft', 'mse_sat_ft_p_approx', 'pressure_ft']:
        if '_ft' in var:
            output_info[var].coords['lev'] = coords['lev'][ind_ft]
            output_info[var+'_std'].coords['lev'] = coords['lev'][ind_ft]
        else:
            output_info[var].coords['lev'] = coords['lev'][ind_surf]
            output_info[var+'_std'].coords['lev'] = coords['lev'][ind_surf]
    # Convert dict to dataset
    ds_out = xr.Dataset(output_info)
    # Add basic info to dataset
    # ds_out['time'] = ds.time
    # ds_out['lon'] = ds.lon
    ds_out['landmask'] = ds_land.landmask
    ds_out['gw'] = ds.gw    # add weighting to output directory


    # Save output to nd2 file with compression - reduces size of file by factor of 10
    # Compression makes saving step slower
    ds_out.to_netcdf(os.path.join(exp_info['out_dir'], exp_info['out_name']), format="NETCDF4",
                     encoding={var: {"zlib": True, "complevel": 4} for var in ds_out.data_vars})
    logger.info("End")

if __name__ == "__main__":
    main(sys.argv[1])
