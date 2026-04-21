from typing import Literal

import xarray as xr
import numpy as np
import pandas as pd
import os
import warnings
import logging
import sys
sys.path.append('/Users/joshduffield/Documents/StAndrews/Isca')
from metpy.calc import specific_humidity_from_dewpoint
from metpy.units import units
from isca_tools.utils import get_memory_usage
from isca_tools.utils.moist_physics import sphum_sat
import jobs.thesis_season.thesis_figs.utils as utils
from isca_tools.utils.xarray import convert_ds_dtypes

complevel = 4

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S',
                    stream=sys.stdout)


def get_rh(dew_temp_2m: xr.DataArray, temp_2m: xr.DataArray, pressure_surf: xr.DataArray) -> xr.DataArray:
    """Calculate near-surface relative humidity from dewpoint, temperature, and surface pressure.

    This function computes the \(2\ \mathrm{m}\) specific humidity from dewpoint and
    surface pressure, then divides it by the saturation specific humidity at the
    \(2\ \mathrm{m}\) air temperature and surface pressure. Any MetPy units attached
    to the diagnosed specific humidity are removed before broadcasting to match the
    saturation humidity field.

    Args:
        dew_temp_2m: \(2\ \mathrm{m}\) dewpoint temperature in kelvin.
        temp_2m: \(2\ \mathrm{m}\) air temperature in kelvin.
        pressure_surf: Surface pressure in pascals.

    Returns:
        Relative humidity as a dimensionless `xr.DataArray`, typically on the
        interval \(0\) to \(1\).
    """
    q2m = specific_humidity_from_dewpoint(pressure_surf * units.pascal, dew_temp_2m * units.kelvin)
    q2m_sat = sphum_sat(temp_2m, pressure_surf)
    q2m = q2m.metpy.dequantify().broadcast_like(q2m_sat)  # get rid of metpy units stuff
    return q2m / q2m_sat


def main(var: str, test: bool = False, data_duration: Literal[5, 20] = 20) -> None:
    """Compute and save annual-cycle diagnostics for a selected surface variable.

    This function loads ERA5 surface data, derives the requested variable where
    needed, and then computes two classes of diagnostics:
    1. Annual Fourier coefficients for selected variables.
    2. Empirical fit parameters relating the variable to surface temperature.

    The Fourier output stores the amplitude and phase of the first annual
    harmonic. The empirical fitting output stores regression-based parameters for
    relationships between $T_s$ and the requested variable, with and without
    phase information.

    Args:
        var: Variable name to process. This may be a directly available dataset
            variable or a derived quantity such as `'w_atm'`, `'rh_atm'`, or
            `'net_up_flux'`.
        test: Whether to run on a reduced spatial subset for quicker testing.
        data_duration: Number of years included in the input dataset. Must be
            either $5$ or $20$.

    Returns:
        None. Results are written to NetCDF files on disk.

    Raises:
        OSError: If input files cannot be opened or output files cannot be
            written.
        KeyError: If `var` or one of its required input fields is missing from
            the dataset.
        ValueError: If downstream fitting or Fourier utilities fail because of
            incompatible data.
    """
    if test:
        logging.info(f"test = True so doing quick small dataset")

    # List of variables to obtain Fourier coefficients and Empirical fitting for
    var_fourier = ['net_up_flux', 'mslhf', 'msshf', 'msnlwrf', 'msnswrf', 'skt']
    var_params = [x for x in var_fourier if x != 'skt'] + ['w_atm', 'rh_atm']

    # Where to save data
    data_dir = f'/Users/joshduffield/Documents/StAndrews/Isca/jobs/era5/surface_flux/av_annual_cycle/output_{data_duration}years/'
    out_path_fourier = os.path.join(data_dir, 'fourier_coef', f"{var}_test.nc" if test else f"{var}.nc")
    out_path_params = os.path.join(data_dir, 'empirical_fitting', f"{var}_test.nc" if test else f"{var}.nc")

    # Open dataset with all variables
    ds = xr.open_mfdataset(f'{data_dir}/*.nc')
    if test:
        ds = ds.sel(latitude=slice(60, 50), longitude=slice(330, 360))
    logging.info(f"Lazy loaded data | Memory used {get_memory_usage() / 1000:.1f}GB")

    # Obtain relavent variable and load in
    if var == 'w_atm':
        ds_var = np.sqrt(ds['u10'] ** 2 + ds['v10'] ** 2)
    elif var == 'rh_atm':
        ds_var = get_rh(ds['d2m'], ds['t2m'], ds['sp'])
    elif var == 'net_up_flux':
        ds_var = -ds.mslhf - ds.msshf - ds.msnlwrf
    else:
        ds_var = ds[var]
    logging.info(f"Computed {var} | Memory used {get_memory_usage() / 1000:.1f}GB")
    ds_var = ds_var.load()
    logging.info(f"Fully loaded {var} | Memory used {get_memory_usage() / 1000:.1f}GB")

    # Obtain annual harmonic of the variable
    if os.path.exists(out_path_fourier):
        logging.info(f"Fourier data already exists for {var} | Memory used {get_memory_usage() / 1000:.1f}GB")
    elif (var in var_fourier) and not os.path.exists(out_path_fourier):
        _, coef_amp, coef_phase = utils.get_fourier_fit_xr(np.arange(ds.time.size), ds_var,
                                                           n_harmonics=1, pad_coefs_phase=True)
        logging.info(f"Obtained Fourier coefficients for {var} | Memory used {get_memory_usage() / 1000:.1f}GB")
        ds_out = xr.Dataset({'amp': coef_amp, 'phase': coef_phase})
        encoding = {var: {'zlib': True, 'complevel': complevel} for var in ds_out.data_vars}
        ds_out = convert_ds_dtypes(ds_out)
        ds_out.attrs['n_time'] = ds.time.size
        if not os.path.exists(out_path_fourier):
            ds_out.to_netcdf(out_path_fourier, encoding=encoding)
            logging.info(
                f"Saved fourier coefficients to {out_path_fourier} | Memory used {get_memory_usage() / 1000:.1f}GB")

    # Obtain empirical fitting between the surface temperature and variable
    if os.path.exists(out_path_params):
        logging.info(f"Empirical data already exists for {var} | Memory used {get_memory_usage() / 1000:.1f}GB")
    elif var in var_params:
        fit_params = {}
        skt = ds['skt'].load()
        logging.info(f"Fully loaded skt | Memory used {get_memory_usage() / 1000:.1f}GB")
        for key in ['linear', 'linear_phase']:
            # Compute params with simulated flux
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", np.RankWarning)
                fit_params[key] = utils.polyfit_phase_xr(skt, ds_var, deg=1,
                                                         include_phase='phase' in key, include_fourier=False,
                                                         deg_phase_calc=10)
                logging.info(
                    f"{key} | {var} | Obtained fit parameters | Memory used {get_memory_usage() / 1000:.1f}GB")
        fit_params = xr.concat(fit_params.values(),
                               dim=pd.Index(fit_params.keys(), name="fit_method")).to_dataset(name=var)
        fit_params = convert_ds_dtypes(fit_params)
        fit_params.attrs['n_time'] = ds.time.size
        if not os.path.exists(out_path_params):
            encoding = {var: {'zlib': True, 'complevel': complevel} for var in fit_params.data_vars}
            fit_params.to_netcdf(out_path_params, encoding=encoding)
            logging.info(
                f"Saved empirical parameters to {out_path_params} | Memory used {get_memory_usage() / 1000:.1f}GB")

if __name__ == '__main__':
    try:
        var = sys.argv[1]
    except IndexError:
        var = 'mslhf'
    try:
        test = sys.argv[2]
    except IndexError:
        test = False
    try:
        data_duration = sys.argv[3]
    except IndexError:
        data_duration = 20
    main(var, test, data_duration)
