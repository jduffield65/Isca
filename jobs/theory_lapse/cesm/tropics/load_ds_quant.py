# Takes lapse fitting info dataset for hottest 50% of days at each coordinate i.e. summer.
# Then does quantile analysis of this over dimensions of `sample` and `time`.
import xarray as xr
import os
import numpy as np
from tqdm import tqdm
import logging
import sys
from isca_tools.papers.byrne_2021 import get_quant_ind
from isca_tools.utils.base import print_log

from jobs.theory_lapse.scripts.lcl import get_co2_multiplier

surf_use = 'ocean'
land_frac_thresh = 0.1
n_sample = None
quant_range = 0.5
quant_all = np.arange(1, 100)
ds_out_path = f'/Users/joshduffield/Desktop/ds_cesm_tropics_quant_{surf_use}.nc'
if surf_use == 'land':
    ds_out_path = ds_out_path.replace('_land', f'_land_{land_frac_thresh:.1f}'.replace('.', '_'))

if os.path.exists(ds_out_path):
    ds_quant = xr.load_dataset(ds_out_path)
    print(f"Loaded {ds_out_path}")
else:
    ds_quant = None


def get_ds_quant_single_coord(ds, quant=90, range_below=0.5, range_above=0.5):
    quant_mask = get_quant_ind(ds.TREFHT.squeeze(), quant, range_below, range_above, av_dim=['lon', 'time'],
                               return_mask=True)
    ds_use = ds.where(quant_mask).stack(sample=("lon", "time"), create_index=False).chunk(dict(sample=-1))
    ds_use = ds_use.load()
    ds_use = ds_use.where(ds_use.TREFHT > 0, drop=True)
    return ds_use


def get_ds_quant(ds, quant=90, range_below=0.5, range_above=0.5, n_keep=None):
    quant_mask = get_quant_ind(ds.TREFHT, quant, range_below, range_above, av_dim=['lon', 'time'], return_mask=True)
    # n_keep is so can concat ds of different quantiles. A given quant range will give slightly different numbers of samples at each location.
    # Through providing n_keep, you can ensure the number is always the same.
    n_keep_max = int(quant_mask.sum(dim=['lon', 'time']).min())
    if n_keep is None:
        n_keep = n_keep_max
    if n_keep > n_keep_max:
        raise ValueError(f'n_keep={n_keep} > n_keep_max={n_keep_max}')
    ds_out = []
    for i in range(ds.co2.size):
        ds_use_k = []
        for k in range(ds.lat.size):
            ds_use_k.append(
                get_ds_quant_single_coord(ds.isel(co2=i, lat=k, drop=True), quant, range_below,
                                          range_above).isel(sample=slice(0, n_keep)))
        ds_use_k = xr.concat(ds_use_k, dim=ds.lat)
        ds_out.append(ds_use_k)
    ds_out = xr.concat(ds_out, dim=ds.co2)
    return ds_out


if __name__ == '__main__':
    if os.path.exists(ds_out_path):
        sys.exit('File already exists')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S',
                        stream=sys.stdout)
    logger = logging.getLogger()  # for printing to console time info
    print_log('Loading data', logger)
    exp_names = ['pre_industrial', 'co2_2x']
    file_path = lambda x: f'/Users/joshduffield/Documents/StAndrews/Isca/jobs/cesm/3_hour/hottest_quant/{x}/REFHT_quant50/lapse_fitting/ds_lapse_simple.nc'
    var_keep = ['TREFHT', 'PREFHT', 'CAPE', 'FREQZM', 'rh_REFHT', 'T_ft_env']
    for key2 in ['const1', 'mod_parcel1']:
        for key in ['lapse', 'integral', 'error']:
            var_keep.append(f"{key2}_{key}")
    ds = []
    for i in range(len(exp_names)):
        ds.append(xr.open_dataset(file_path(exp_names[i]))[var_keep])
    ds = xr.concat(ds, dim=xr.DataArray([get_co2_multiplier(exp_names[i]) for i in range(len(exp_names))], dims="co2", name='co2'))
    ds = ds.rename({'sample': 'time'})              # so matches isca stuff - sample goes from hottest to coldest times
    print_log('Finished loading data', logger)


    # Land masks
    invariant_data_path = '/Users/joshduffield/Documents/StAndrews/Isca/jobs/cesm/input_data/fv_0.9x1.25_nc3000_Nsw042_Nrs008_Co060_Fi001_ZR_sgh30_24km_GRNL_c170103.nc'
    land_frac = xr.open_dataset(invariant_data_path).LANDFRAC
    land_frac = land_frac.reindex_like(ds, method="nearest", tolerance=0.01)
    lsm = (land_frac > land_frac_thresh)
    surf_mask = {'land': land_frac > land_frac_thresh, 'ocean': land_frac == 0}

    ds_quant = []
    for i in tqdm(quant_all):
        ds_quant.append(get_ds_quant(ds.where(surf_mask[surf_use]), i, quant_range, quant_range, n_sample))
    if n_sample is None:
        n_sample = np.min([i.sample.size for i in ds_quant])
        # Ensure all have same size
        ds_quant = [i.isel(sample=slice(0, n_sample)) for i in ds_quant]
    ds_quant = xr.concat(ds_quant, dim=xr.DataArray(quant_all, dims="quant", name='quant'))
    if not os.path.exists(ds_out_path):
        ds_quant.to_netcdf(ds_out_path)
        print_log('Saved ds_quant to {}'.format(ds_out_path), logger)