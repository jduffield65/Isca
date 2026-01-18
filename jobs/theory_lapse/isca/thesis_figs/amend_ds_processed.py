# Script to save data to ds that forgot about before, or recently added not in jobs.theory_lapse.isca.thesis_figs.load_ds_quant
# Somdon't have to run whole script again
import xarray as xr
import sys
import logging
sys.path.append('/Users/joshduffield/Documents/StAndrews/Isca')
from isca_tools.utils.base import print_log
from isca_tools.utils.constants import lapse_dry
# from jobs.theory_lapse.isca.thesis_figs.load_ds_quant import get_ds_out_path, get_lnb_ind_xr, get_P, comp_level
from jobs.theory_lapse.cesm.thesis_figs.scripts.utils import path_lapse_data, path_all_data
from jobs.theory_lapse.scripts.lapse_fitting_simple import get_lnb_ind_xr, get_pressure, comp_level, get_lapse_dev

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S',
                    stream=sys.stdout)
logger = logging.getLogger()  # for printing to console time info

surf = 'aquaplanet'
test_mode = False

try:
    key = sys.argv[1]
    key2 = None
    # key2 = sys.argv[2]
except IndexError:
    key = 'pre_industrial'
    key2 = None
    # key = 'k=1'
    # key2 = 'north'
out_path = path_lapse_data(key, 50)
# out_path = get_ds_out_path(key, surf, hemisphere=key2, dailymax=True)
print_log(f'{key} | {key2} | Start', logger=logger)
ds_quant = xr.load_dataset(out_path)
temp_env = xr.open_dataset(path_all_data(key, 50))['T']
temp_env = temp_env.T.load()
# temp_env = ds_quant.T
if test_mode:
    ds_quant = ds_quant.sel(sample=slice(0, 4), lon=slice(0, 2))
    temp_env = temp_env.sel(sample=slice(0, 4), lon=slice(0, 2))
ds_quant['hyam'] = ds_quant['hyam'].isel(lat=0, drop=True)
ds_quant['hybm'] = ds_quant['hybm'].isel(lat=0, drop=True)
print_log(f'{key} | {key2} | Loaded', logger=logger)
lapse_mod_D = ds_quant.mod_parcel1_lapse.isel(layer=0, p_ft=0, drop=True) / 1000 - lapse_dry
# 2 different estimates of LNB depending on where parcel rising from
# New dimension parcel type, surf means parcel rising from surface so lapse_mod_D=0
# lcl means parcel rising from lcl means take actual value of lapse_mod_D
lapse_mod_D = lapse_mod_D.expand_dims({'parcel_type': ['surf', 'lcl']})
lapse_mod_D = lapse_mod_D.assign_coords(parcel_type=['surf', 'lcl'])
lapse_mod_D = lapse_mod_D.where(lapse_mod_D.parcel_type == 'lcl', 0.)  # where parcel_type=surf, set to 0
print_log(f'{key} | {key2} | Starting LNB computation', logger=logger)
p_use = get_pressure(ds_quant.PS, ds_quant.P0, ds_quant.hyam, ds_quant.hybm)
# p_use = get_P(ds_quant)
lnb = []
for i in range(lapse_mod_D.parcel_type.size):
    lnb.append(get_lnb_ind_xr(temp_env, p_use, ds_quant.rh_REFHT,
                              lapse_mod_D.isel(parcel_type=i), temp_surf_lcl_calc=ds_quant.temp_surf_lcl_calc))
    print_log(f'{key} | {key2} | Finished LNB Calc {i+1}/2', logger=logger)
ds_quant['lnb_ind'] = xr.concat(lnb, dim=lapse_mod_D.parcel_type)
print_log(f'{key} | {key2} | Starting Miyawaki 2022 Computation', logger=logger)
ds_quant['lapse_miy2022_M'], ds_quant['lapse_miy2022_D'] = \
    get_lapse_dev(temp_env, p_use, ds_quant.PS)
out_path_new = out_path.replace('.nc', '_new.nc')
# Reorder
ds_quant = ds_quant.transpose('lat', 'lon', 'sample', 'p_ft', 'layer', 'parcel_type', 'lev')
ds_quant.to_netcdf(out_path_new, format="NETCDF4",
                   encoding={var: {"zlib": True, "complevel": comp_level} for var in ds_quant.data_vars})
print_log(f'{key} | {key2} | Saved', logger=logger)