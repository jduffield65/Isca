# ## Get times for the hottest 50% of days
# Simple script to load and save the times and TREFHT of the hottest 50% of days at each grid point for each location. Must be run on JASMIN.
# Only consider limited latitudes and longitudes as only interested in Canada
#
# Must first run `cesm/3_hour/hottest_quant/exp/REFHT_quant50/T_Q_PS_all_lat/input.nml` to create the output.nc files.

import xarray as xr
import os
import sys
import logging
from pathlib import Path

from isca_tools.utils.base import print_log
from isca_tools.utils.xarray import convert_ds_dtypes

co2_vals = xr.DataArray([1, 2], dims="co2", name='co2')
path_input = lambda x: f'/home/users/jamd1/Isca/jobs/cesm/3_hour/hottest_quant/{x}/REFHT_quant50/T_Q_PS_all_lat/output.nc'
exp_names = ['pre_industrial', 'co2_2x']
lev_REFHT = -1
comp_level = 4
# Only need Canada hence a limited range
lat_sel = slice(30, 75)
lon_sel = slice(200, 360)
dir_script = Path(__file__).resolve().parent



def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S',
                        stream=sys.stdout)
    logger = logging.getLogger()  # for printing to console time info
    dir_out = os.path.join(dir_script.parent, 'ds_processed')
    path_out = os.path.join(dir_out, 'ds_quant50_times.nc')
    if os.path.exists(path_out):
        print_log(f'Dataset already exists at {path_out}', logger=logger)
        sys.exit(0)

    ds = []
    print_log(f'Start', logger)
    for exp_name in exp_names:
        ds_use = xr.open_dataset(path_input(exp_name))[['time', 'T']].isel(lev=lev_REFHT).sel(lat=lat_sel, lon=lon_sel)
        print_log(f'{exp_name} | Lazy Loaded', logger)
        ds.append(ds_use.load())
        print_log(f'{exp_name} | Fully Loaded', logger)
    ds = xr.concat(ds, dim=co2_vals)
    ds = ds.load()
    print_log(f'Concat | Fully Loaded', logger)

    ds.attrs['lev_REFHT'] = lev_REFHT
    ds = convert_ds_dtypes(ds)

    if not os.path.exists(path_out):
        ds.to_netcdf(path_out, format="NETCDF4",
                     encoding={var: {"zlib": True, "complevel": comp_level} for var in ds.data_vars})
        print_log(f'Saved to {path_out}', logger)

if __name__ == '__main__':
    # Need function called main so can use run_script_jasmin through just:
    # from isca_tools.jasmin import run_script
    # run_script('/home/users/jamd1/Isca/jobs/theory_lapse/cesm/thesis_figs/scripts/load_ds_quant50_times.py',
    #            slurm=False)
    main()
