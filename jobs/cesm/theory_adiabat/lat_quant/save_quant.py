from isca_tools import cesm
from isca_tools.papers.byrne_2021 import get_quant_ind
from isca_tools.utils.moist_physics import moist_static_energy, sphum_sat
from isca_tools.utils import get_memory_usage
import sys
import os
import numpy as np
import f90nml
import logging
import time
import xarray as xr
# Set up logging configuration to output to console and don't output milliseconds, and stdout so saved to out file
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S',
                    stream=sys.stdout)


def get_exp_info_dict(input_file_path: str) -> dict:
    # From input nml file, this returns dict containing info required for getting npz file
    exp_info = f90nml.read(input_file_path)['script_info']
    if exp_info['out_dir'] is None:
        # Default directory to save output is the directory containing input nml file
        exp_info['out_dir'] = os.path.dirname(input_file_path)
    if exp_info['out_name'] is None:
        # Default name of saved data is output.npz
        exp_info['out_name'] = 'output.nd2'
    elif exp_info['out_name'][-4:] != '.nd2':
        # Ensure output file ends in npz
        exp_info['out_name'] = exp_info['out_name'] + '.nd2'
    if exp_info['year_first'] is None:
        # Default first year of data is year 1
        exp_info['year_first'] = 1
    if exp_info['year_last'] is None:
        # Default last year of data is last year in cesm data
        exp_info['year_last'] = -1
    out_file = os.path.join(exp_info['out_dir'], exp_info['out_name'])
    if not exp_info['exist_ok'] and os.path.exists(out_file):
        # Raise error if output data already exists
        raise ValueError('Output file already exists at:\n{}'.format(out_file))
    return exp_info


def get_ds(exp_name, archive_dir, chunks_time, chunks_lat, chunks_lon, parallel,
           p_ft_approx_guess, p_surf_approx_guess,
           year_first, year_last, logger):
    # Load in datasets - one from atm and lnd components
    chunks={"time": chunks_time, "lat": chunks_lat, "lon": chunks_lon}

    def preprocess_land(ds):
        # Only 2 variables, and sum over all soil levels
        soil_liq_sum = ds['SOILLIQ'].sum(dim='levsoi')  # Sum over 'levsoi'
        return xr.Dataset({'SOILLIQ': soil_liq_sum, 'landmask': ds['landmask']})
    ds_land = cesm.load_dataset(exp_name, archive_dir=archive_dir, hist_file=1, comp='lnd',
                               year_first=year_first, year_last=year_last,chunks=chunks, parallel=parallel,
                               preprocess=preprocess_land,
                               logger=logger)
    logger.info('Loaded land data')

    var_atm = ['T', 'Q', 'Z3', 'PS', 'P0', 'hyam', 'hybm']
    def preprocess_atm(ds):
        # Preprocessing so don't load in entire dataset
        ds = ds[var_atm]
        return ds.sel(lev=xr.DataArray([p_surf_approx_guess, p_ft_approx_guess], dims='lev'), method='nearest')
    ds_atm = cesm.load_dataset(exp_name, archive_dir=archive_dir, hist_file=1,
                               year_first=year_first, year_last=year_last,chunks=chunks, parallel=parallel,
                               preprocess=preprocess_atm, logger=logger)
    logger.info('Loaded atmospheric data')

    # Reduce size of datasets
    # ind_ft = int(np.argmin(np.abs(ds.T.lev - p_ft_approx_guess).to_numpy()))
    # ind_surf = int(np.argmin(np.abs(ds.T.lev - p_surf_approx_guess).to_numpy()))
    # ds_atm = ds.isel(lev=[ind_surf, ind_ft])[var_atm]  # For atm dataset, only need a few variables and 2 pressure levels

    # ds_lnd = ds_lnd.reindex_like(ds['PS'], method="nearest", tolerance=0.01)

    # soil_liq = ds_lnd.SOILLIQ.sum(dim='levsoi')    # for lnd dataset, only need soil moisture
    #
    # is_land = ds_lnd.landfrac.isel(time=0, drop=True) > landfrac_thresh     # whether a coordinate is land or not

    # re-index so lat align - otherwise get issues because lat slightly different
    ds_land = ds_land.reindex_like(ds_atm['PS'], method="nearest", tolerance=0.01)
    # is_land = is_land.reindex_like(ds['PS'].isel(time=0), method="nearest", tolerance=0.01)  # re-index so lat align

    return ds_atm, ds_land


def main(input_file_path: str):
    logger = logging.getLogger()    # for printing to console time info
    logger.info("Start")
    exp_info = get_exp_info_dict(input_file_path)
    ds, ds_land = get_ds(exp_info['exp_name'], exp_info['archive_dir'],
                                   exp_info['chunks_time'], exp_info['chunks_lat'], exp_info['chunks_lon'],
                                   exp_info['load_parallel'], exp_info['p_ft_approx_guess'],
                                   exp_info['p_surf_approx_guess'],
                                   exp_info['year_first'], exp_info['year_last'], logger=logger)
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

    ds_land['landmask'] = ds_land['landmask'].isel(time=0, drop=True) > 0

    # Get pressure info
    ind_surf = 0
    ind_ft = 1
    p_ref = float(ds.P0.isel(time=0))
    hybrid_a_coef_ft = float(ds.hyam.isel(time=0, lev=ind_ft))
    hybrid_b_coef_ft = float(ds.hybm.isel(time=0, lev=ind_ft))
    p_ft_approx = float(ds.lev[ind_ft]) * 100
    p_surf_approx = float(ds.lev[ind_surf]) * 100


    # Initialize dict to save output data - n_surf (land or ocean) x n_lat x n_quant
    n_lat = ds.lat.size
    n_surf = 2
    n_lev = 2  # find quantile by looking for all values in quantile range between quant_use-quant_range to quant_use+quant_range
    n_quant = len(exp_info['quant'])
    output_info = {var: np.zeros((n_surf, n_quant, n_lat)) for var in
                   ['rh', 'mse', 'mse_sat_ft', 'mse_lapse',
                    'mse_sat_ft_p_approx', 'mse_lapse_p_approx', 'pressure_ft', 'SOILLIQ']}
    for var in ['T', 'Q', 'Z3']:
        output_info[var] = np.zeros((n_surf, n_quant, n_lat, n_lev))         # have pressure dim as well
    var_keys = [key for key in output_info.keys()]
    for var in var_keys:
        output_info[var + '_std'] = np.zeros_like(output_info[var])
    output_info['use_in_calc'] = np.zeros((n_surf, n_quant, n_lat, ds.lon.size, ds.time.size), dtype=bool)
    # output_info['lon_most_common'] = np.zeros((n_surf, n_quant, n_lat))
    # output_info['lon_most_common_freq'] = np.zeros((n_surf, n_quant, n_lat), dtype=int)
    # output_info['n_grid_points'] = np.zeros((n_surf, n_lat), dtype=int)  # number of grid points used at each location
    # Record approx number of days used in quantile calculation. If quant_range=0.5 and 1 year used, this is just 0.01*365=3.65
    # n_days_quant = get_quant_ind(np.arange(ds.time.size * n_lat), quant_use[0], quant_range,
    #                                             quant_range).size / n_lat

    coords = {'surface': ['land', 'ocean'], 'quant': exp_info['quant'],
              'lev': ds.lev, 'lat': ds.lat, 'lon': ds.lon, 'time': ds.time}

    # Coordinate info to convert to xarray datasets
    output_dims = {var: ['surface', 'quant', 'lat'] for var in output_info}
    for var in ['T', 'Q', 'Z3']:
        output_dims[var] = ['surface', 'quant', 'lat', 'lev']
        output_dims[var+'_std'] = ['surface', 'quant', 'lat', 'lev']
    output_dims['use_in_calc'] = ['surface', 'quant', 'lat', 'lon', 'time']

    logger.info(f"Starting iteration over {n_lat} latitudes, 2 surfaces, and {n_quant} quantiles")

    # Loop through and get quantile info at each latitude and surface
    for i in range(n_lat):
        time_log = {'load': 0, 'calc': 0, 'start': time.time()}
        ds_lat = ds.isel(lat=i, drop=True)
        ds_land_lat = ds_land.isel(lat=i, drop=True)
        if not exp_info['load_all_at_start']:
            ds_lat.load()
            ds_land_lat.load()
        time_log['load'] += time.time() - time_log['start']
        for k, surf in enumerate(coords['surface']):
            time_log['start'] = time.time()
            if surf == 'land':
                is_surf = ds_land_lat.landmask
            else:
                is_surf = ~ds_land_lat.landmask

            if is_surf.sum() == 0:
                # If surface not at this latitude, record no data
                continue

            # if surf == 'land':
            #     soil_liq_use = ds_land_lat.SOILLIQ.sel(lon=is_surf)
            #     if not exp_info['load_all_at_start']:
            #         soil_liq_use.load()
            # output_info['n_grid_points'][k, i] = ds_use.lon.size
            time_log['load'] += time.time() - time_log['start']
            time_log['start'] = time.time()
            for j in range(n_quant):
                # get indices corresponding to given near-surface temp quantile
                quant_mask = get_quant_ind(ds_lat.T.isel(lev=ind_surf).where(is_surf), coords['quant'][j],
                                           exp_info['quant_range'], exp_info['quant_range'], return_mask=True,
                                           av_dim=['lon', 'time'])
                if quant_mask.sum() == 0:
                    logger.info(f"Latitude {i + 1}/{n_lat}: no data found for {surf} and quant={coords['quant'][j]}")
                    continue
                ds_use = ds_lat.where(quant_mask)
                output_info['use_in_calc'][k, j, i] = quant_mask.transpose("lon", "time")
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
                if surf == 'land':
                    var_use['SOILLIQ'] = ds_land_lat.SOILLIQ.where(quant_mask)
                for key in var_use:
                    output_info[key][k, j, i] = var_use[key].mean(dim=['lon', 'time'])
                    output_info[key + '_std'][k, j, i] = var_use[key].std(dim=['lon', 'time'])
                # lon_use = np.unique(ds_use.lon[use_ind], return_counts=True)
                #
                # # Record most common specific coordinate within grid to see if most of days are at a given location
                # output_info['lon_most_common'][k, j, i] = lon_use[0][lon_use[1].argmax()]
                # output_info['lon_most_common_freq'][k, j, i] = lon_use[1][lon_use[1].argmax()]
                time_log['calc'] += time.time() - time_log['start']
        # if (i+1) == 1 or (i+1) == n_lat or (i+1) % 10 == 0:
        # # Log info on 1st, last and every 10th latitude
        logger.info(f"Latitude {i + 1}/{n_lat}: Loading took {time_log['load']:.1f}s | Calculation took {time_log['calc']:.1f}s | "
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


    # Save output to nd2 file
    ds_out.to_netcdf(os.path.join(exp_info['out_dir'], exp_info['out_name']))
    logger.info("End")

if __name__ == "__main__":
    main(sys.argv[1])
