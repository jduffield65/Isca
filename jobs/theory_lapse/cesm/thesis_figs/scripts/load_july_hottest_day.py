# Must be run on JASMIN with run_jasmin_script.py:
# from isca_tools.jasmin import run_script
# run_script('/home/users/jamd1/Isca/jobs/theory_lapse/cesm/thesis_figs/scripts/load_july_hottest_day.py',
#            slurm=False)
#
# loads in data from July for a single year, and finds hottest day at each grid point. Saves data to compute lapse rate
# Code from Isca/jobs/theory_lapse/cesm_3hour_diurnal.ipynb notebook but changed from
# ds_loc.time.dt.floor('D') == hottest_day_loc as that won't work at all longitudes. Also quicker
import xarray as xr
import os
import sys
import logging
import numpy as np
from isca_tools.cesm.load import jasmin_archive_dir
from isca_tools.utils.base import print_log
from pathlib import Path


# Specify experiment
exp_name = 'e.e20.E1850TEST.f09_g17.3hour_output'                               # 1xCO2 dataset with 3 hourly output
file_name = 'e.e20.E1850TEST.f09_g17.3hour_output.cam.h1.0061-07-02-54000.nc'   # July of 1 year - one file for speed
var_keep = ['T', 'TREFHT', 'Q', 'QREFHT', 'PS']
lev_REFHT = None          # temperature to use to compute hottest day (None to use REFHT)
comp_level = 4
dir_script = Path(__file__).resolve().parent         # directory of this script
path_out = os.path.join(dir_script.parent, 'ds_processed', 'ds_july_hottest.nc')

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S',
                        stream=sys.stdout)
    logger = logging.getLogger()  # for printing to console time info
    if os.path.exists(path_out):
        print_log(f'Dataset already exists at {path_out}', logger=logger)
        sys.exit(0)

    file_path = os.path.join(jasmin_archive_dir, exp_name, 'atm', 'hist', file_name)
    ds = xr.open_dataset(file_path)
    print_log('Lazy loaded dataset', logger=logger)

    # Get info for computing pressure - as same for all days
    gw = ds.gw.load()               # grid weighting - not for pressure but for area average weighting
    hyam = ds.hyam.load()
    hybm = ds.hybm.load()
    p0 = float(ds.P0)

    ds = ds[var_keep].load()
    print_log('Fully loaded dataset', logger=logger)

    # Find time of max temperature in the month, then take 4 times below and 4 above so have 8 times (8x3=24 hours)
    # for each location corresponding to the day of max temperature, plus the hottest time. Spans 24 hours
    if lev_REFHT is None:
        i_max = ds.TREFHT.argmax(dim="time")  # (lat, lon)
    else:
        i_max = ds.T.isel(lev=lev_REFHT, drop=True).argmax(dim="time")
    offset_ind = xr.DataArray(
        [-4, -3, -2, -1, 0, 1, 2, 3, 4],
        dims=("hour_offset",),
        coords={"hour_offset": np.asarray([-4, -3, -2, -1, 0, 1, 2, 3, 4]) * 3},
    )
    i_window_raw = i_max.expand_dims(dim={"hour_offset": offset_ind.sizes["hour_offset"]}) + offset_ind
    valid = (i_window_raw >= 0) & (i_window_raw < ds.sizes["time"])         # mask which is False for times outside of month i.e. edge cases
    i_window = i_window_raw.clip(min=0, max=ds.sizes["time"] - 1)           # clip so all time indices within month dataset
    ds_hot = ds.isel(time=i_window).where(valid)                      # select 9 time values for each location, set to nan if outside monthly dataset
    print_log('Computed hottest time', logger=logger)

    # Need to convert time to string to save dataset
    def safe_iso(t):
        try:
            # check for numeric NaN
            if isinstance(t, float) and np.isnan(t):
                return np.nan
            # otherwise assume cftime object
            return t.isoformat()
        except:
            return np.nan  # fallback

    vec_iso = np.vectorize(safe_iso)
    # only save time for hottest hour i.e. hour_offset=0 as rest obvious
    ds_hot['time_str'] = (('lat', 'lon'), vec_iso(ds_hot['time'].sel(hour_offset=0).values))
    ds_hot = ds_hot.drop_vars('time')

    # Add attributes
    ds_hot['gw'] = gw
    ds_hot['hyam'] = hyam
    ds_hot['hybm'] = hybm
    ds_hot['p0'] = p0
    ds_hot.attrs['lev_REFHT'] = str(lev_REFHT)
    ds_hot.attrs['file_name'] = file_name

    # Save to this directory in jasmin
    Path(path_out).resolve().parent.mkdir(parents=True, exist_ok=True)     # make directory if does not exist

    comp = dict(zlib=True, complevel=comp_level)
    encoding = {}
    for var in ds_hot.data_vars:
        if var == "time_str":
            # no compression for variable-length strings
            encoding[var] = {}
        else:
            encoding[var] = comp
    ds_hot.to_netcdf(path_out, format="NETCDF4", encoding=encoding)
    print_log(f'Saved dataset to {path_out}', logger=logger)

if __name__ == '__main__':
    # Need function called main so can use run_script_jasmin through just:
    # from isca_tools.jasmin import run_script
    # run_script('/home/users/jamd1/Isca/jobs/theory_lapse/cesm/thesis_figs/scripts/load_july_hottest_day.py',
    #            slurm=False)
    main()
