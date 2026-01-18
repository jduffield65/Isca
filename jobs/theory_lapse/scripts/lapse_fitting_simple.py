# Uses physical LCL to compute the mod_parcel1 and const1 lapse rate fitting
# Also estimates the LNB where parcel no longer buoyant
# Does lapse analysis for three separate p_ft values
import os
import numpy as np
import xarray as xr
import logging
import sys

sys.path.append('/Users/joshduffield/Documents/StAndrews/Isca')

from isca_tools.utils import get_memory_usage
from isca_tools.utils.base import print_log
from isca_tools.utils.moist_physics import sphum_sat
from isca_tools.utils.xarray import print_ds_var_list, convert_ds_dtypes, wrap_with_apply_ufunc
from isca_tools.cesm import get_pressure
from isca_tools.papers.miyawaki_2022 import get_lapse_dev
from isca_tools.utils.constants import lapse_dry
from isca_tools.convection.base import lcl_metpy
from isca_tools.thesis.lapse_theory import get_var_at_plev
from geocat.comp.interpolation import interp_hybrid_to_pressure
from isca_tools.thesis.lapse_integral_simple import fitting_2_layer_xr, get_lnb_ind

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S',
                    stream=sys.stdout)
get_lnb_ind_xr = wrap_with_apply_ufunc(get_lnb_ind, input_core_dims=[['lev'], ['lev'], [], []])

# File location Info
from jobs.theory_lapse.scripts.lcl import load_ds_quant, data_dir, var_keep

var_initial_load = [item for item in var_keep if
                    item not in ['Z3', 'CAPE', 'FREQZM']]  # get rid of variables don't need
test_mode = False            # for testing on a small dataset
try:
    # ideally get quant_type from terminal
    quant_type = sys.argv[1]  # e.g. 'REFHT_quant50'
    exp_name = sys.argv[2]    # 'pre_industrial' or 'co2_2x'
    lat_ind_start = int(sys.argv[3])
    try:
        lat_ind_end = int(sys.argv[4])
    except IndexError:
        lat_ind_end = None
except IndexError:
    # Default values of don't call from terminal - for testing
    exp_name = ['pre_industrial', 'co2_2x']
    quant_type = 'REFHT_quant50'
    lat_ind_start = 0
    lat_ind_end = None
exp_name = np.atleast_1d(exp_name)
processed_dir = [os.path.join(data_dir, exp_name[i], quant_type, 'lapse_fitting') for i in range(len(exp_name))]
processed_file_name = 'ds_lapse_simple.nc'  # combined file from all samples
p_ft = [400 * 100, 500 * 100, 700 * 100]  # Free tropospheric levels - for each one, will obtain lapse fitting info
p_ft = np.atleast_1d(p_ft)
comp_level = 4
n_lev_above_integral = 3
temp_surf_lcl_calc = 300  # Compute LCL from RH using 300K as surf temp
lev_REFHT = -1  # Model level used to compute rh_REFHT, and subsequent lapse fitting. If None will use actual REFHT


def get_ds_quant_lat(co2_ind, lat_ind):
    if test_mode:
        lon_ind = [0, 1, 2]
        sample_ind = [0, 1]
    else:
        lon_ind = None
        sample_ind = None
    ds = load_ds_quant([exp_name[co2_ind]], quant_type, data_dir, var_initial_load, compute_p_diff=False,
                       lat_ind=lat_ind, lon_ind=lon_ind, sample_ind=sample_ind, lev_REFHT=lev_REFHT)
    # Compute rh at REFHT
    ds['rh_REFHT'] = ds.QREFHT / sphum_sat(ds.TREFHT, ds.PREFHT)

    # Interpolate data onto FT level
    ds['T_ft_env'] = interp_hybrid_to_pressure(ds.T, ds.PS, ds.hyam, ds.hybm, ds.P0, p_ft,
                                               lev_dim='lev')
    ds['T_ft_env'].load()
    ds = ds.rename({'plev': 'p_ft'})  # change to p_ft
    ds.attrs['temp_surf_lcl_calc'] = temp_surf_lcl_calc
    ds.attrs['n_lev_above_integral'] = n_lev_above_integral
    ds.attrs['lev_REFHT'] = lev_REFHT
    return ds


if __name__ == '__main__':
    logger = logging.getLogger()  # for printing to console time info

    # ds = load_ds_quant(exp_name, quant_type, data_dir, var_keep, small_ds=small_ds, compute_p_diff=False,
    #                    load_fully=False)
    print_log(f'Quantile data loaded | Memory used {get_memory_usage() / 1000:.1f}GB',
              logger)

    load_processed = [os.path.exists(os.path.join(processed_dir[i], processed_file_name)) for i in range(len(exp_name))]

    # Compute empirical estimate of LCL
    lat_vals = load_ds_quant([exp_name[0]], quant_type, data_dir, ['TREFHT'], compute_p_diff=False,
                             sample_ind=0, lon_ind=0).lat
    n_lat = lat_vals.size
    if lat_ind_end is None:
        lat_ind_end = n_lat-1
    n_files = len(exp_name) * n_lat  # One file for each co2 conc and sample due to speed
    print_log(f'Empirical lapse fitting for {n_files} Files | Start | Lat start ind = {lat_ind_start} | '
              f'Lat end ind = {lat_ind_end} | Max possible lat ind = {n_lat-1}', logger)

    var_names = ['lapse', 'integral', 'error']
    n_digit = len(str(n_lat))
    # One file for each latitude and co2
    for i in range(len(exp_name)):
        if load_processed[i]:
            print_log(f'Files already exist for {exp_name[i]}', logger)
            continue
        path_use = [os.path.join(processed_dir[i], f'lat{j:0{n_digit}d}.nc') for j in range(n_lat)]
        for j in range(n_lat):
            if (j < lat_ind_start) or (j > lat_ind_end):
                # Don't get data for these latitudes as outside range selected
                print_log(f'File {i * n_lat + j + 1}/{n_files} | '
                          f'Skipped - outside lat ind range of {lat_ind_start}-{lat_ind_end} | '
                          f'Memory used {get_memory_usage() / 1000:.1f}GB', logger)
                continue
            if os.path.exists(path_use[j]):
                print_log(f'File {i * n_lat + j + 1}/{n_files} Exists Already', logger)
                continue
            print_log(f'File {i * n_lat + j + 1}/{n_files} | Start | Memory used {get_memory_usage() / 1000:.1f}GB',
                      logger)
            ds_use = get_ds_quant_lat(i, j)
            print_log(
                f'File {i * n_lat + j + 1}/{n_files} | Data loaded | Memory used {get_memory_usage() / 1000:.1f}GB',
                logger)
            for key in ['const', 'mod_parcel']:
                # Compute for multiple p_ft and T_ft_env at the same time
                var = fitting_2_layer_xr(ds_use.T, ds_use.P, ds_use.TREFHT, ds_use.PREFHT, ds_use.rh_REFHT,
                                         ds_use.T_ft_env, ds_use.p_ft,
                                         n_lev_above_upper2_integral=ds_use.n_lev_above_integral,
                                         temp_surf_lcl_calc=ds_use.temp_surf_lcl_calc, method_layer2=key)
                for k, key2 in enumerate(var_names):
                    ds_use[f'{key}1_{key2}'] = var[k]
                print_log(
                    f'File {i * n_lat + j + 1}/{n_files} | {key}1 Lapse Fitting Complete | Memory used {get_memory_usage() / 1000:.1f}GB',
                    logger)

            # boundary layer lapse rate is same for all methods and p_ft so just select one
            # 2 different estimates of LNB depending on where parcel rising from
            # New dimension parcel type, surf means parcel rising from surface so lapse_mod_D=0
            # lcl means parcel rising from lcl means take actual value of lapse_mod_D
            lapse_mod_D = ds_use.mod_parcel1_lapse.isel(layer=0, p_ft=0, drop=True) / 1000 - lapse_dry
            lapse_mod_D = lapse_mod_D.fillna(0)  # if nan set to exactly dry adiabat for lnb computation
            lapse_mod_D = lapse_mod_D.expand_dims({'parcel_type': ['surf', 'lcl']})
            lapse_mod_D = lapse_mod_D.assign_coords(parcel_type=['surf', 'lcl'])
            lapse_mod_D = lapse_mod_D.where(lapse_mod_D.parcel_type == 'lcl', 0.)  # where parcel_type=surf, set to 0
            # ds_use['lnb_ind'] = get_lnb_ind_xr(ds_use.T, ds_use.P, ds_use.rh_REFHT, lapse_mod_D,
            #                                    temp_surf=None, p_surf=None, temp_surf_lcl_calc=temp_surf_lcl_calc)
            lnb = []  # loop so can output data progress more often
            for q in range(lapse_mod_D.parcel_type.size):
                lnb.append(get_lnb_ind_xr(ds_use.T, ds_use.P, ds_use.rh_REFHT,
                                          lapse_mod_D.isel(parcel_type=q),
                                          temp_surf=None, p_surf=None, temp_surf_lcl_calc=temp_surf_lcl_calc))
                print_log(
                    f'File {i * n_lat + j + 1}/{n_files} | Computed LNB {q+1}/2 | Memory used {get_memory_usage() / 1000:.1f}GB',
                    logger)
            ds_use['lnb_ind'] = xr.concat(lnb, dim=lapse_mod_D.parcel_type)

            ds_use['lapse_miy2022_M'], ds_use['lapse_miy2022_D'] = \
                get_lapse_dev(ds_use.T, ds_use.P, ds_use.PS)
            print_log(
                f'File {i * n_lat + j + 1}/{n_files} | Computed Miyawaki 2022 params | Memory used {get_memory_usage() / 1000:.1f}GB',
                logger)

            # Save data
            ds_use['layer'] = xr.DataArray(['below lcl', 'above lcl'], name='layer', dims='layer')
            ds_use = ds_use.drop_vars(['T', 'P'])  # drop lat-lon-lev vars as have that data anyway
            ds_use = convert_ds_dtypes(ds_use)
            if not os.path.exists(path_use[j]):
                ds_use.to_netcdf(path_use[j], format="NETCDF4",
                                 encoding={var: {"zlib": True, "complevel": comp_level} for var in ds_use.data_vars})
            print_log(f'File {i * n_lat + j + 1}/{n_files} | Saved | Memory used {get_memory_usage() / 1000:.1f}GB',
                      logger)

        if not os.path.exists(os.path.join(processed_dir[i], processed_file_name)):
            # Combine all sample files into a single file for each experiment
            ds_lapse = xr.concat([xr.load_dataset(path_use[j]) for j in range(n_lat)], dim=lat_vals)
            # Need to make hyam and hybm a single latitude
            ds_lapse['hyam'] = ds_lapse['hyam'].isel(lat=0, drop=True)
            ds_lapse['hybm'] = ds_lapse['hybm'].isel(lat=0, drop=True)
            # Reorder
            ds_lapse = ds_lapse.transpose('lat', 'lon', 'sample', 'p_ft', 'layer', 'parcel_type', 'lev')
            ds_lapse.to_netcdf(os.path.join(processed_dir[i], processed_file_name), format="NETCDF4",
                               encoding={var: {"zlib": True, "complevel": comp_level} for var in ds_lapse.data_vars})
            print_log(f'{exp_name[i]} | Combined samples into one {processed_file_name} File', logger)

    hi = 5
