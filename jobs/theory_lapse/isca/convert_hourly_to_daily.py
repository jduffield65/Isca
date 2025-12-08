import numpy as np
import os
from tqdm import tqdm
import xarray as xr
import logging
import sys
sys.path.append('/Users/joshduffield/Documents/StAndrews/Isca')
import isca_tools
from isca_tools.utils.base import print_log
from isca_tools.utils.base import top_n_peaks_ind
from isca_tools.utils.xarray import wrap_with_apply_ufunc


# -- Specific info for running the script --
do_test = False                         # use small dataset
hour_spacing = 13                      # Data points must be further than this apart in time - want 1 value per day
complevel = 4                          # Compression level for saving
sort_xr = wrap_with_apply_ufunc(np.sort, input_core_dims=[['sample']], output_core_dims=[['sample']])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S',
                        stream=sys.stdout)
    logger = logging.getLogger()  # for printing to console time info
    #  -- Info for loading in data --
    # exp_path = 'tau_sweep/aquaplanet/depth=1/k=1_3hour'
    exp_path = sys.argv[1]
    ds_out_path = os.path.join(os.environ['GFDL_DATA'], exp_path).replace('3hour', 'daily')
    if not os.path.exists(ds_out_path):
        os.makedirs(ds_out_path)
        # print_log(f'Data already exists at {ds_out_path}', logger)
        # sys.exit()

    print_log(f'Loading data for {exp_path}', logger)
    ds = isca_tools.load_dataset(exp_path, decode_times=True)
    ds = ds.drop_vars(['average_T1', 'average_T2', 'average_DT', 'time_bounds'])
    time_days = np.unique(ds.time.dt.floor('D'))
    n_days = time_days.size          # number of days
    time_dt = (ds.time[1] - ds.time[0]) / np.timedelta64(1, 'h')
    ind_spacing = int(np.ceil(hour_spacing / time_dt))
    get_idx_time = wrap_with_apply_ufunc(top_n_peaks_ind, input_core_dims=[["time"]],
                                         output_core_dims=[["sample"]],
                                         output_dtypes=[int])

    if do_test:
        # small subset and remove pfull dimension
        ds = ds.isel(lat=slice(0, 3), lon=slice(20, 22))[['temp', 'sphum', 'temp_2m']].isel(pfull=-1)
    ds_out = []
    print_log('Looping over each coordinate to get hottest hour for each day', logger)
    for i in range(ds.lat.size):
        print_log(f'Lat {i + 1}/{ds.lat.size} | Start', logger)
        path_out_lat = os.path.join(ds_out_path, f'lat{i}.nc')
        if os.path.exists(path_out_lat):
            print_log(f'Lat {i+1}/{ds.lat.size} | File already exists', logger)
            continue
        ds_use = ds.isel(lat=i)
        temp_use = ds_use.temp_2m.load()
        print_log(f'Lat {i + 1}/{ds.lat.size} | Loaded temp_2m', logger)
        time_idx = get_idx_time(ds_use.temp_2m, n=n_days, min_ind_spacing=ind_spacing)
        time_idx = sort_xr(time_idx)
        ds_use = ds_use.isel(time=time_idx).rename({"time": "sample"})      # select time indices, rename coord to sample
        ds_use = ds_use.assign_coords(sample=time_days) # Set sample value to daily time
        ds_use = ds_use.rename({"sample": "time"})      # rename sample to time
        ds_use = ds_use.assign_coords(time=time_days)
        print_log(f'Lat {i + 1}/{ds.lat.size} | Found daily data', logger)
        ds_use.to_netcdf(path_out_lat, format="NETCDF4",
                         encoding={var: {"zlib": True, "complevel": complevel} for var in ds_use.data_vars})
        print_log(f'Lat {i + 1}/{ds.lat.size} | Saved', logger)
            # for j in range(ds.lon.size):
            #     ds_use = ds.isel(lat=i, lon=j)
            #     ds_use = ds_use.load()
            #     # Find indices of hottest n_days times such that none closer than hour_spacing in time
            #     # I.e. get 1 value per day - daily data
            #     time_idx = get_idx_time(ds_use.temp_2m, n=n_days, min_ind_spacing=ind_spacing)
            #     time_idx = np.sort(time_idx)   # sort so in time order
            #     ds_use = ds_use.isel(time=time_idx)
            #     ds_use['time'] = time_days     # Change time dimension so just the day for each location
            #     ds_lat.append(ds_use)
            #     pbar.update(1)
            # ds_lat = xr.concat(ds_lat, dim=ds.lon)
            # ds_lat.to_netcdf(path_out_lat, format="NETCDF4",
            #                  encoding={var: {"zlib": True, "complevel": complevel} for var in ds_lat.data_vars})

    # ds_out = xr.concat(ds_out, dim=ds.lat)
    # print_log('Finished processing data', logger)
    #
    # if not os.path.exists(ds_out_path):
    #     ds.to_netcdf(os.path.join(ds_out_path), format="NETCDF4",
    #                  encoding={var: {"zlib": True, "complevel": complevel} for var in ds.data_vars})
    #     ds_out.to_netcdf(ds_out_path)
    #     print_log('Saved data to {}'.format(ds_out_path), logger)
