# Script to save data to ds that forgot about before, or recently added not in jobs.theory_lapse.isca.thesis_figs.load_ds_quant
# Somdon't have to run whole script again
import xarray as xr
import sys
import logging
sys.path.append('/Users/joshduffield/Documents/StAndrews/Isca')
from isca_tools.utils.base import print_log
from tqdm.notebook import tqdm
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
    ds_quant = ds_quant.sel(sample=slice(0, 40), lon=slice(0, 2))
    temp_env = temp_env.sel(sample=slice(0, 40), lon=slice(0, 2))
ds_quant['hyam'] = ds_quant['hyam'].isel(lat=0, drop=True)
ds_quant['hybm'] = ds_quant['hybm'].isel(lat=0, drop=True)
print_log(f'{key} | {key2} | Loaded', logger=logger)
lapse_mod_D = ds_quant.mod_parcel1_lapse.isel(layer=0, p_ft=0, drop=True) / 1000 - lapse_dry
# 2 different estimates of LNB depending on where parcel rising from
# New dimension parcel type, surf means parcel rising from surface so lapse_mod_D=0
# lcl means parcel rising from lcl means take actual value of lapse_mod_D
lapse_mod_D = lapse_mod_D.fillna(0)  # if nan set to exactly dry adiabat for LCL parcel type lnb computation
lapse_mod_D = lapse_mod_D.expand_dims({'parcel_type': ['surf', 'lcl']})
lapse_mod_D = lapse_mod_D.assign_coords(parcel_type=['surf', 'lcl'])
lapse_mod_D = lapse_mod_D.where(lapse_mod_D.parcel_type == 'lcl', 0.)  # where parcel_type=surf, set to 0
print_log(f'{key} | {key2} | Starting Computation', logger=logger)
p_use = get_pressure(ds_quant.PS, ds_quant.P0, ds_quant.hyam, ds_quant.hybm)
# p_use = get_P(ds_quant)
lnb = []
miy2022_M = []
miy2022_D = []
for i in range(ds_quant.sample.size):
    lnb.append(get_lnb_ind_xr(temp_env.isel(sample=i), p_use.isel(sample=i), ds_quant.rh_REFHT.isel(sample=i),
                              lapse_mod_D.isel(sample=i), temp_surf_lcl_calc=ds_quant.temp_surf_lcl_calc))
    var = get_lapse_dev(temp_env.isel(sample=i), p_use.isel(sample=i), ds_quant.PS.isel(sample=i))
    miy2022_M.append(var[0])
    miy2022_D.append(var[1])
    print_log(f'Completed Sample {i+1}/{ds_quant.sample.size}', logger=logger)
ds_quant['lnb_ind'] = xr.concat(lnb, dim=ds_quant.sample)
ds_quant['lapse_miy2022_M'] = xr.concat(miy2022_M, dim=ds_quant.sample)
ds_quant['lapse_miy2022_D'] = xr.concat(miy2022_D, dim=ds_quant.sample)
print_log(f'{key} | {key2} | Finished Computation', logger=logger)
out_path_new = out_path.replace('.nc', '_new.nc')
# Reorder
ds_quant = ds_quant.transpose('lat', 'lon', 'sample', 'p_ft', 'layer', 'parcel_type', 'lev')
ds_quant.to_netcdf(out_path_new, format="NETCDF4",
                   encoding={var: {"zlib": True, "complevel": comp_level} for var in ds_quant.data_vars})
print_log(f'{key} | {key2} | Saved to {out_path_new}', logger=logger)