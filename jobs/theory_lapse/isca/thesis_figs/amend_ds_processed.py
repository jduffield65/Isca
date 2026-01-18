# Script to save data to ds that forgot about before, or recently added not in jobs.theory_lapse.isca.thesis_figs.load_ds_quant
# Somdon't have to run whole script again
import xarray as xr
import sys
import logging
sys.path.append('/Users/joshduffield/Documents/StAndrews/Isca')
from isca_tools.utils.base import print_log
from isca_tools.utils.constants import lapse_dry
from jobs.theory_lapse.isca.thesis_figs.load_ds_quant import get_ds_out_path, get_lnb_ind_xr, get_P, temp_surf_lcl_calc, \
    comp_level

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S',
                    stream=sys.stdout)
logger = logging.getLogger()  # for printing to console time info

surf = 'aquaplanet'
# with tqdm(total=4, position=0, leave=True) as pbar:
#     for key in ['k=1', 'k=1_5']:
#         for key2 in ['north', 'south']:

try:
    key = sys.argv[1]
    key2 = sys.argv[2]
except IndexError:
    key = 'k=1'
    key2 = 'north'
out_path = get_ds_out_path(key, surf, hemisphere=key2)
print_log(f'{key} | {key2} | Start', logger=logger)
ds_quant = xr.load_dataset(out_path)
print_log(f'{key} | {key2} | Loaded', logger=logger)
lapse_mod_D = ds_quant.mod_parcel1_lapse.isel(layer=0, p_ft=0, drop=True) / 1000 - lapse_dry
# 2 different estimates of LNB depending on where parcel rising from
# New dimension parcel type, surf means parcel rising from surface so lapse_mod_D=0
# lcl means parcel rising from lcl means take actual value of lapse_mod_D
lapse_mod_D = lapse_mod_D.expand_dims({'parcel_type': ['surf', 'lcl']})
lapse_mod_D = lapse_mod_D.assign_coords(parcel_type=['surf', 'lcl'])
lapse_mod_D = lapse_mod_D.where(lapse_mod_D.parcel_type == 'lcl', 0.)  # where parcel_type=surf, set to 0
print_log(f'{key} | {key2} | Starting computation', logger=logger)
ds_quant['lnb_ind'] = get_lnb_ind_xr(ds_quant.T, get_P(ds_quant), ds_quant.rh_REFHT,
                                     lapse_mod_D, temp_surf_lcl_calc=ds_quant.temp_surf_lcl_calc)
print_log(f'{key} | {key2} | Computed', logger=logger)
out_path_new = out_path.replace('.nc', '_new.nc')
ds_quant.to_netcdf(out_path_new, format="NETCDF4",
                   encoding={var: {"zlib": True, "complevel": comp_level} for var in ds_quant.data_vars})
print_log(f'{key} | {key2} | Saved', logger=logger)