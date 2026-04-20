import os.path

import xarray as xr
import numpy as np
import pandas as pd
from datatree import DataTree, open_datatree
import os
import warnings
import logging
import sys
from isca_tools.utils.moist_physics import sphum_sat
import jobs.thesis_season.thesis_figs.utils as utils

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S',
                    stream=sys.stdout)
test = False     # will run on small dataset
if test:
    logging.info(f"test = True so doing quick small dataset")

data_dir = '/Users/joshduffield/Documents/StAndrews/Isca/jobs/era5/surface_flux/av_annual_cycle/output_5years/'
out_path = os.path.join(data_dir, f"ds_processed_test.nc" if test else "ds_processed.nc")
if os.path.exists(out_path):
    logging.info(f"File already exists at {out_path}")
    sys.exit()
else:
    logging.info(f"Start - data to be saved to {out_path}")

ds = xr.open_mfdataset(f'{data_dir}/*.nc')
if test:
    ds = ds.sel(latitude=slice(60, 50), longitude=slice(330, 360))
logging.info(f"Lazy loaded data")

net_up_flux = {'simulated': (-ds.mslhf - ds.msshf - ds.msnlwrf).load()}
logging.info(f"Loaded net-up flux")
temp_surf = {'simulated': ds.skt.load()}
logging.info(f"Loaded skin temperature")
# w_atm = np.sqrt(ds.u ** 2 + ds.v ** 2)
w_atm = np.sqrt(ds.u10 ** 2 + ds.v10 ** 2).load()
logging.info(f"Loaded w_atm")


net_up_flux['direct'] = \
    utils.get_fourier_fit_xr(np.arange(ds.time.size), net_up_flux['simulated'], n_harmonics=1, pad_coefs_phase=True)[0]
logging.info(f"Performed Fourier fit on net up flux")
temp_surf['direct'] = \
    utils.get_fourier_fit_xr(np.arange(ds.time.size), temp_surf['simulated'], n_harmonics=1, pad_coefs_phase=True)[0]
logging.info(f"Performed Fourier fit on skin temperature")

net_up_flux_params = {}
w_atm_params = {}
for key in ['linear', 'linear_phase']:
    # Compute params with simulated flux
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", np.RankWarning)
        net_up_flux_params[key] = utils.polyfit_phase_xr(temp_surf['simulated'], net_up_flux['simulated'], deg=1,
                                                         include_phase='phase' in key, include_fourier=False,
                                                         deg_phase_calc=10)
        logging.info(f"{key} | net up flux | Obtained fit parameters")
        w_atm_params[key] = utils.polyfit_phase_xr(temp_surf['simulated'], w_atm, deg=1,
                                                   include_phase='phase' in key, include_fourier=False,
                                                   deg_phase_calc=10)
        logging.info(f"{key} | w_atm | Obtained fit parameters")
    # Apply params with single harmonic temp
    net_up_flux[key] = utils.polyval_phase_xr(net_up_flux_params[key], temp_surf['direct'])
    logging.info(f"{key} | net up flux | Obtained fit")

ds_out = xr.concat(net_up_flux.values(), dim=pd.Index(net_up_flux.keys(), name="fit_method")).to_dataset(
    name="net_up_flux")
ds_out['temp_surf'] = xr.concat(temp_surf.values(), dim=pd.Index(temp_surf.keys(), name="fit_method"))
fit_params = xr.concat(net_up_flux_params.values(),
                       dim=pd.Index(net_up_flux_params.keys(), name="fit_method")).to_dataset(name="net_up_flux")
fit_params['w_atm'] = xr.concat(w_atm_params.values(),
                                dim=pd.Index(net_up_flux_params.keys(), name="fit_method"))
tree_out = DataTree.from_dict({'ds_out': ds_out, 'fit_params': fit_params})
logging.info(f"Set up data tree to save to file")

if not os.path.exists(out_path):
    tree_out.to_netcdf(out_path)
    logging.info(f"Saved data to {out_path}")