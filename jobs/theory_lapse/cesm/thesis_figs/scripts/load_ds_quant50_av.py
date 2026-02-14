# Must be run on JASMIN with run_jasmin_script.py:
# from isca_tools.jasmin import run_script
# But have issue with args not working - just ends up with default values
# run_script('/home/users/jamd1/Isca/jobs/theory_lapse/cesm/thesis_figs/scripts/load_ds_quant50_av.py',
#            slurm=True, time = '24:00:00', mem=100)
#
# Loads in daily max data at the hottest 50% of days at each location.
# Computes the average value of TREFHT, QREFHT, rh_REFHT, PREFHT, PS, T_ft_env at each location across these days
# Takes output from Isca/jobs/cesm/3_hour/hottest_quant/exp/REFHT_quant50/T_Q_PS_all_lat
# for exp=pre_industrial and exp=co2_2x.
import xarray as xr
import os
import sys
import logging
import numpy as np
from geocat.comp import interp_hybrid_to_pressure
from xarray_einstats.stats import circmean as xr_circmean
from pathlib import Path

from isca_tools.utils import get_memory_usage
from isca_tools.utils.base import print_log
from isca_tools.cesm import get_pressure

from isca_tools.utils.moist_physics import sphum_sat
from isca_tools.utils.xarray import convert_ds_dtypes

comp_level = 4
p_ft = 400 * 100
test_mode = False        # uses a small subset of samples
lev_REFHT = -1
temp_ft_pos_anom_thresh = 2     # Record fraction of days where anomaly larger than this
dir_script = Path(__file__).resolve().parent  # directory of this script
exp_names = ['pre_industrial', 'co2_2x']
co2_vals = xr.DataArray([1, 2], dims="co2", name='co2')


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S',
                        stream=sys.stdout)
    logger = logging.getLogger()  # for printing to console time info
    dir_out = os.path.join(dir_script.parent, 'ds_processed')
    path_out = lambda x: os.path.join(dir_out, f'ds_quant50_av_{x}.nc')
    path_out_combined = os.path.join(dir_out, f'ds_quant50_av.nc')
    path_input = lambda x: f'/home/users/jamd1/Isca/jobs/cesm/3_hour/hottest_quant/{x}/REFHT_quant50/T_Q_PS_all_lat/output.nc'


    if os.path.exists(path_out_combined):
        print_log(f'Dataset already exists at {path_out_combined}', logger=logger)
        sys.exit(0)

    for exp_name in exp_names:
        if os.path.exists(path_out(exp_name)):
            print_log(f'Dataset already exists at {path_out(exp_name)}', logger=logger)
            continue

        print_log(f'{exp_name} | Start | Test' if test_mode else f'{exp_name} | Start', logger=logger)

        ds = xr.open_dataset(path_input(exp_name))
        if test_mode:
            ds = ds.isel(sample=slice(0, 5))
        print_log(f'Lazy loaded dataset | Memory used {get_memory_usage() / 1000:.1f}GB', logger=logger)
        ds['T_ft_env'] = interp_hybrid_to_pressure(ds.T, ds.PS, ds.hyam, ds.hybm, float(ds.P0), np.atleast_1d(p_ft))
        print_log(f'Computed T_ft_env | Memory used {get_memory_usage() / 1000:.1f}GB', logger=logger)
        ds = ds.isel(lev=lev_REFHT)
        print_log(f'Selected lev_REFHT | Memory used {get_memory_usage() / 1000:.1f}GB', logger=logger)
        ds = ds.load()
        print_log(f'Fully loaded | Memory used {get_memory_usage() / 1000:.1f}GB', logger=logger)
        ds['PREFHT'] = get_pressure(ds.PS, float(ds.P0), ds.hyam, ds.hybm)
        ds = ds.rename_vars({'Q': 'QREFHT', 'T': 'TREFHT'})
        ds['rh_REFHT'] = ds.QREFHT / sphum_sat(ds.TREFHT, ds.PREFHT)
        print_log(f'Computed rh_REFHT | Memory used {get_memory_usage() / 1000:.1f}GB', logger=logger)

        # Decompose FT into zonal mean on that day, and anomaly to this
        temp_ft_zm = xr.load_dataset(os.path.join(dir_out, f"ds_t400_climatology_{exp_name}.nc")
                                     ).T.sel(plev=p_ft, drop=True)
        ds['T_ft_env_zm'] = temp_ft_zm.sel(dayofyear=ds.time.dt.dayofyear).drop_vars('dayofyear')
        ds['T_ft_env_anom'] = ds.T_ft_env - ds['T_ft_env_zm']
        # Record average day of the year at each grid point
        ds['doy'] = xr_circmean(ds.time.dt.dayofyear, dims="sample", low=1, high=365,
                                     nan_policy="omit")  # av day of year on which hot days occur
        ds.attrs['temp_ft_pos_anom_thresh'] = temp_ft_pos_anom_thresh  # Frac of days anomalously hot at FT
        n_sample = ds.sample.size
        # Record number of days at each grid point with a significant positive anomaly
        ds['n_temp_ft_pos_anom'] = (ds.T_ft_env_anom > ds.temp_ft_pos_anom_thresh).sum(dim='sample')
        print_log(f'Computed T_ft zonal mean and anom | Memory used {get_memory_usage() / 1000:.1f}GB', logger=logger)
        ds = ds.mean(dim='sample')
        print_log(f'Computed mean | Memory used {get_memory_usage() / 1000:.1f}GB', logger=logger)

        ds.attrs['temp_ft_pos_anom_thresh'] = temp_ft_pos_anom_thresh  # Frac of days anomalously hot at FT
        ds.attrs['n_sample'] = n_sample       # number of days at each grid point
        ds.attrs['lev_REFHT'] = lev_REFHT
        ds = convert_ds_dtypes(ds)
        if not os.path.exists(path_out(exp_name)):
            ds.to_netcdf(path_out(exp_name), format="NETCDF4",
                         encoding={var: {"zlib": True, "complevel": comp_level} for var in ds.data_vars})
            print_log(f'{exp_name} | Saved', logger=logger)

    if not os.path.exists(path_out_combined):
        # Combine all sample files into a single file for each experiment
        ds = xr.concat([xr.load_dataset(path_out(exp_name)) for exp_name in exp_names], dim=co2_vals)
        # Need to make hyam and hybm a single latitude
        ds['hyam'] = ds['hyam'].isel(co2=0, drop=True)
        ds['hybm'] = ds['hybm'].isel(co2=0, drop=True)
        ds['P0'] = ds['P0'].isel(co2=0, drop=True)
        ds = ds.transpose('co2', 'lat', 'lon', 'plev')
        ds.to_netcdf(path_out_combined, format="NETCDF4",
                     encoding={var: {"zlib": True, "complevel": comp_level} for var in ds.data_vars})
        print_log(f'Combined samples into one {path_out_combined} File', logger)


if __name__ == '__main__':
    # Need function called main so can use run_script_jasmin through just:
    # from isca_tools.jasmin import run_script
    # run_script('/home/users/jamd1/Isca/jobs/theory_lapse/cesm/thesis_figs/scripts/load_july_hottest_day.py',
    #            slurm=False)
    main()
