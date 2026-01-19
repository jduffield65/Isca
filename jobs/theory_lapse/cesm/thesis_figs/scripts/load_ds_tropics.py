# Takes lapse fitting info dataset for hottest 50% of days at each coordinate i.e. summer.
# Then does quantile analysis of this over dimensions of `lon` and `sample`
# Really is just flattening the lon and sample dimensions together - does not take average over this dimension
# Takes around 15 minutes for land, 1 hour for ocean
# First need to get processed datasets from raw 3 hourly data on JASMIN before running this:
# Run `utils.data_dir/save_quant_ind.py` on JASMIN to create `time_ind.nc` file,
# Followed by `utils.data_dir/save_info.py` on JASMIN to create `output.nc` file
# Followed by `jobs/theory_lapse/scripts/lapse_fitting_simple.py` locally to create `lapse_fitting/ds_lapse.nc` file
import xarray as xr
import numpy as np
import os
import logging
import sys
from tqdm import tqdm
sys.path.append('/Users/joshduffield/Documents/StAndrews/Isca')
import jobs.theory_lapse.cesm.thesis_figs.scripts.utils as utils
from isca_tools.utils.base import print_log
from isca_tools.cesm import get_pressure
from isca_tools.utils.xarray import convert_ds_dtypes

ds_out_path = lambda x: os.path.join(utils.out_dir, f'ds_tropics_{x}.nc')

# Metadata detailing how data obtained
test_mode = False       # try on small dataset
comp_level = 4
var_keep = utils.vars_lapse_data + ['CAPE', 'FREQZM']
land_frac_thresh = 0.5
n_sample = None  # for testing set this to a small number
quant_range = 0.5
quant_all = np.arange(1, 100)
try:
    # ideally get quant_type from terminal
    surf = sys.argv[1]  # e.g. 'ocean'
except IndexError:
    # Default values
    surf = 'land'

if test_mode:
    n_sample = 5
    quant_all = np.arange(30, 32)


def get_ds_quant(ds: xr.Dataset, quant: int, range_below: float=quant_range, range_above: float=quant_range,
                 n_keep=None, av_dim=['lat', 'lon', 'sample'], av_dim_out='lat_lon_sample'):
    # Get ds conditioned on quant for all co2 and lat values
    quant_mask = utils.get_quant_ind(ds.TREFHT, quant, range_below, range_above,
                                     av_dim=av_dim, return_mask=True)
    # n_keep is so can concat ds of different quantiles.
    # A given quant range will give slightly different numbers of samples at each location.
    # Through providing n_keep, you can ensure the number is always the same.
    n_keep_max = int(quant_mask.sum(dim=av_dim).min())
    if n_keep is None:
        n_keep = n_keep_max
    if n_keep > n_keep_max:
        raise ValueError(f'n_keep={n_keep} > n_keep_max={n_keep_max}')
    ds_out = []
    for i in range(ds.co2.size):
        ds_out.append(
            utils.get_ds_quant_single_coord(ds.isel(co2=i, drop=True), quant, range_below,
                                            range_above, av_dim=av_dim, av_dim_out=av_dim_out))
        ds_out[-1] = ds_out[-1].isel(**{av_dim_out: slice(0, n_keep)})
    ds_out = xr.concat(ds_out, dim=ds.co2)
    ds_out['quant_mask'] = quant_mask           # keep track of locations used
    return ds_out


if __name__ == '__main__':
    if os.path.exists(ds_out_path(surf)):
        sys.exit('File already exists')

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S',
                        stream=sys.stdout)
    logger = logging.getLogger()  # for printing to console time info

    print_log('Loading data', logger)
    ds = []
    for i in range(len(utils.exp_names)):
        ds.append(utils.load_ds(utils.exp_names[i], 50, var_keep))  # quant=50 means kept hottest 6 months
    ds = xr.concat(ds, dim=xr.DataArray([utils.get_co2_multiplier(utils.exp_names[i])
                                         for i in range(len(ds))], dims="co2", name='co2'))

    hyam = ds['hyam'].isel(co2=0, lat=0, drop=True)  # these variables are the same for all CO2 and lat so just keep one
    hybm = ds['hybm'].isel(co2=0, lat=0, drop=True)
    ds['LANDFRAC'] = ds.LANDFRAC.isel(co2=0, drop=True)  # same for all co2 so keep one
    print_log('Finished loading data', logger)

    # ds['p_lnb'] = get_pressure(ds.PS, ds.P0, ds.hyam.isel(lev=ds.lnb_ind), ds.hybm.isel(lev=ds.lnb_ind))

    surf_mask = {'land': ds.LANDFRAC > land_frac_thresh, 'ocean': ds.LANDFRAC == 0}
    ds = ds.drop_vars(['LANDFRAC', 'hyam', 'hybm'])           # Adding lev dim makes for loop about 2x longer
    ds_quant = []
    for i in tqdm(quant_all):
        ds_quant.append(get_ds_quant(ds.where(surf_mask[surf]), i, quant_range, quant_range, n_sample))
    if n_sample is None:
        # Ensure all have quantiles have same size
        n_sample = np.min([i.lat_lon_sample.size for i in ds_quant])
        ds_quant = [i.isel(lat_lon_sample=slice(0, n_sample)) for i in ds_quant]
    ds_quant = xr.concat(ds_quant, dim=xr.DataArray(quant_all, dims="quant", name='quant'))

    # Record attributes
    ds_quant.attrs['land_frac_thresh'] = land_frac_thresh
    ds_quant.attrs['quant_range'] = quant_range
    ds_quant.attrs['surf'] = surf

    ds_quant = ds_quant.assign_coords(lat_lon_sample=('lat_lon_sample', range(len(ds_quant.lat_lon_sample))))
    ds_quant['hyam'] = hyam
    ds_quant['hybm'] = hybm

    ds_quant['lnb_ind'] = ds_quant['lnb_ind'].astype(int)       # for some reason no longer int at this stage
    ds_quant = convert_ds_dtypes(ds_quant)                      # make sure lower memory

    if not os.path.exists(ds_out_path(surf)):
        ds_quant.to_netcdf(ds_out_path(surf), format="NETCDF4",
                           encoding={var: {"zlib": True, "complevel": comp_level} for var in ds_quant.data_vars})
        print_log('Saved ds_quant to {}'.format(ds_out_path(surf)), logger)
