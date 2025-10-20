import numpy as np
import os
import xarray as xr
import re
import logging
import sys
from isca_tools import cesm
from isca_tools.convection import potential_temp, dry_profile_temp, lcl_metpy
from isca_tools.thesis.lapse_theory import get_var_at_plev
from isca_tools.thesis.profile_fitting import get_mse_env, get_lnb_lev_ind
from isca_tools.utils.constants import g
from isca_tools.utils.base import weighted_RMS, dp_from_pressure, print_log

def get_co2_multiplier(name):
    match = re.match(r'co2_([\d_]+)x', name)
    if match:
        # Replace underscore with decimal point and convert to float
        return float(match.group(1).replace('_', '.'))
    elif name == 'pre_industrial':
        return 1  # for pre_industrial or other defaults
    else:
        raise ValueError(f'Not valid name = {name}')

def find_lcl_empirical2(temp_env, p_env, z_env, temp_start=None, p_start=None, temp_pot_thresh=2,
                        temp_pot_thresh_lapse=0.5, lapse_thresh=8.5):
    # Find the lowest model layer with lapse rate less than lapse_thresh
    # LCL is the level at which the pot temp drops by 0.5K within this layer
    if temp_start is None:
        temp_start = temp_env.isel(lev=-1)
    if p_start is None:
        p_start = p_env.isel(lev=-1)
    temp_pot_env = potential_temp(temp_env, p_env)

    # First mask is pot temp close to surface pot temperature
    temp_pot_start = potential_temp(temp_start, p_start)
    mask_temp = np.abs(temp_pot_env - temp_pot_start) <= temp_pot_thresh

    # Second mask is lapse rate close to dry adiabat
    # lower is so append high value at surface
    lapse = -temp_env.diff(dim='lev', label='lower') / z_env.diff(dim='lev', label='lower') * 1000
    lapse = lapse.reindex_like(temp_env)  # make same shape
    lapse = lapse.fillna(lapse_thresh + 5)  # ensure final value satisfies lapse criteria
    mask_lapse = lapse > lapse_thresh
    mask = (mask_temp & mask_lapse)
    lcl_ind = ((mask.where(mask, other=np.nan) * np.arange(lapse.lev.size)).min(dim='lev')).astype(int)

    p_low = np.log10(p_env.isel(lev=lcl_ind))  # use log as better for interpolation - gradient is approx constant
    p_high = np.log10(p_env.isel(lev=lcl_ind - 1))  # further from surface

    # print(p_low)
    # print(p_high)
    # print(np.log10(p_env))

    temp_pot_low = temp_pot_env.isel(lev=lcl_ind)
    temp_pot_high = temp_pot_env.isel(lev=lcl_ind - 1)
    gradient = (temp_pot_high - temp_pot_low) / (p_high - p_low)
    temp_pot_target = temp_pot_low + temp_pot_thresh_lapse
    p_target = p_low + (temp_pot_target - temp_pot_low) / gradient
    p_target = p_target.clip(min=p_high)
    p_lcl = 10 ** p_target
    # print(dry_profile_temp(temp_start, p_start, p_lcl))
    # print(lapse.isel(lev=lcl_ind-1))
    return p_lcl, dry_profile_temp(temp_start, p_start, p_lcl)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S',
                    stream=sys.stdout)
logger = logging.getLogger()  # for printing to console time info

print_log('Start', logger)
data_dir = '/Users/joshduffield/Documents/StAndrews/Isca/jobs/cesm/3_hour/hottest_quant'
quant_type = 'REFHT_quant99'
exp_name = 'pre_industrial'
var_keep = ['T', 'Q', 'Z3', 'PS', 'P0', 'hyam', 'hybm']
ds = xr.open_dataset(os.path.join(data_dir, exp_name, quant_type, 'output.nc'))
ds = ds[var_keep].load()
ds.attrs['co2'] = get_co2_multiplier(exp_name)
p0 = float(ds.P0)
ds = ds.drop_vars('P0')
ds.attrs['P0'] = p0
print_log('Loaded in Data', logger)

ds['P'] = cesm.get_pressure(ds.PS, ds.P0, ds.hyam, ds.hybm)
ds['P_diff'] = dp_from_pressure(ds.P)
ds['TREFHT'] = ds.T.isel(lev=-1)
ds['QREFHT'] = ds.Q.isel(lev=-1)
ds['ZREFHT'] = ds.Z3.isel(lev=-1)
ds['PREFHT'] = ds.P.isel(lev=-1)
ds = ds.drop_vars('Q')
print_log('Computed pressure difference', logger)

ds['p_lcl'], ds['T_lcl'] = lcl_metpy(ds.TREFHT, ds.QREFHT, ds.PREFHT)
ds['lnb_ind'] = get_lnb_lev_ind(ds.T, ds.Z3, ds.P)
print_log('Computed physical LCL', logger)
small = 1
# ds = ds.where(ds.P >= ds.P.isel(lev=ds.lnb_ind) - small)  # set to nan above LNB
# a = weighted_RMS(ds.T, weight=ds.P_diff/g, dim='lev')
hi = 5