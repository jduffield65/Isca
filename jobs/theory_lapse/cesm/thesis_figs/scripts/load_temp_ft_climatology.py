# Must be run on JASMIN with run_jasmin_script.py:
# from isca_tools.jasmin import run_script
# But have issue with args not working - just ends up with default values
# run_script('/home/users/jamd1/Isca/jobs/theory_lapse/cesm/thesis_figs/scripts/load_temp_ft_climatology.py',
#            script_args=['co2_2x', 400*100, 3, False], slurm=True, time = '24:00:00', mem=100)
import xarray as xr
import os
import sys
import logging
import numpy as np
from typing import Literal
from geocat.comp import interp_hybrid_to_pressure

from isca_tools.utils import get_memory_usage
from isca_tools.utils.base import print_log
from isca_tools.cesm import load_dataset
from pathlib import Path

comp_level = 4
dir_script = Path(__file__).resolve().parent  # directory of this script


def main(exp_name: Literal['pre_industrial', 'co2_2x'] = 'co2_2x', p_ft: float = 400 * 100,
         n_smooth: int = 3, test_mode: bool = False):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S',
                        stream=sys.stdout)
    logger = logging.getLogger()  # for printing to console time info
    path_out = os.path.join(dir_script.parent, 'ds_processed',
                            f'ds_t{p_ft/100:.0f}_climatology_{exp_name}.nc')
    if test_mode:
        path_out = path_out.replace('.nc', '_test.nc')
    if os.path.exists(path_out):
        print_log(f'Dataset already exists at {path_out}', logger=logger)
        sys.exit(0)

    print_log(f'Start | {exp_name}', logger=logger)
    if exp_name == 'pre_industrial':
        exp_name = 'e.e20.E1850TEST.f09_g17.3hour_output'
    elif exp_name == 'co2_2x':
        exp_name = 'e.e20.E1850TEST.f09_g17.co2_2x_3hour_output'
    else:
        raise ValueError(f'Unknown exp_name: {exp_name}')

    print_log(f'Test | 1 file only from {exp_name}' if test_mode
              else f'Loading all {exp_name}', logger=logger)
    var_keep = ['T', 'PS', 'hyam', 'hybm', 'P0']
    ds = load_dataset(exp_name, hist_file=1, ind_files=0 if test_mode else None)[var_keep]
    print_log(f'Lazy loaded dataset | Memory used {get_memory_usage() / 1000:.1f}GB', logger=logger)

    # Get info for computing pressure - as same for all days
    hyam = ds.hyam.isel(time=0).load()
    hybm = ds.hybm.isel(time=0).load()
    p0 = float(ds.P0.isel(time=0))
    ds = ds.drop_vars(['hyam', 'hybm', 'P0'])
    print_log(f'Fully loaded hyam, hybm, p0 | Memory used {get_memory_usage() / 1000:.1f}GB', logger=logger)

    T_d = ds.T.resample(time="1D").mean()
    PS_d = ds.PS.resample(time="1D").mean()
    print_log(f'Computed daily average T and PS | Memory used {get_memory_usage() / 1000:.1f}GB', logger=logger)

    T_d_zm = T_d.mean("lon")  # (time, lev, lat)
    PS_d_zm = PS_d.mean("lon")  # (time, lat)
    print_log(f'Computed zonal mean T and PS | Memory used {get_memory_usage() / 1000:.1f}GB', logger=logger)

    T_d_zm = T_d_zm.load()
    print_log(f'Fully loaded zonal mean T | Memory used {get_memory_usage() / 1000:.1f}GB', logger=logger)
    PS_d_zm = PS_d_zm.load()
    print_log(f'Fully loaded zonal mean PS | Memory used {get_memory_usage() / 1000:.1f}GB', logger=logger)

    # Interpolate to 400 hPa (now only (time, lev, lat) size)
    T400_d_zm = interp_hybrid_to_pressure(T_d_zm, PS_d_zm, hyam, hybm, p0, np.atleast_1d(p_ft))
    print_log(f'Interpolated T to {p_ft/100:.0f}hPa | Memory used {get_memory_usage() / 1000:.1f}GB', logger=logger)

    # Day-of-year climatology across all years
    doy = T400_d_zm["time"].dt.dayofyear
    T400_doy = T400_d_zm.groupby(doy).mean("time")  # (dayofyear, lat)
    print_log(f'Found average T to {p_ft/100:.0f}hPa for each day of year | '
              f'Memory used {get_memory_usage() / 1000:.1f}GB', logger=logger)

    T400_doy_sm = (
        T400_doy.pad(dayofyear=n_smooth, mode="wrap")
        .rolling(dayofyear=2 * n_smooth + 1, center=True).mean()
        .isel(dayofyear=slice(n_smooth, -n_smooth))
    )
    print_log(f'Found average T to {p_ft / 100:.0f}hPa for each day of year | '
              f'Memory used {get_memory_usage() / 1000:.1f}GB', logger=logger)

    encoding = {T400_doy_sm.name: {"zlib": True, "complevel": 4}}
    T400_doy_sm.attrs['n_smooth'] = n_smooth
    T400_doy_sm.attrs['exp_name'] = exp_name
    T400_doy_sm.to_netcdf(path_out, format="NETCDF4", encoding=encoding)
    print_log(f'Saved dataset to {path_out}', logger=logger)


if __name__ == '__main__':
    # Need function called main so can use run_script_jasmin through just:
    # from isca_tools.jasmin import run_script
    # run_script('/home/users/jamd1/Isca/jobs/theory_lapse/cesm/thesis_figs/scripts/load_july_hottest_day.py',
    #            slurm=False)
    main()
