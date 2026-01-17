# Veery similar to load_ds_quant in folder above this, but different output folder, and allow for land
# Also outputs convflag info
# Gets lapse rate fitting info for each day at each quantile
from typing import Literal
import numpy as np
import os
from tqdm import tqdm
import xarray as xr
import logging
import sys

sys.path.append('/Users/joshduffield/Documents/StAndrews/Isca')
import isca_tools
from isca_tools.utils.base import print_log
from isca_tools.utils.moist_physics import sphum_sat, moist_static_energy
from isca_tools.thesis.lapse_theory import get_var_at_plev
from isca_tools.papers.byrne_2021 import get_quant_ind
from isca_tools.thesis.lapse_integral_simple import fitting_2_layer_xr
from isca_tools.utils.xarray import convert_ds_dtypes


def get_ds_out_path(kappa_name: str, surf: Literal['aquaplanet', 'land'] = 'aquaplanet',
                    region: Literal['tropics', 'not_tropics'] = 'tropics',
                    hemisphere: Literal['north', 'south'] = 'north', dailymax: bool = False):
    """
    Returns path name in ./ds_processed directory

    Args:
        kappa_name: Name of a particular experiment e.g. 'k=1'
        surf:
        region:
        hemisphere:
        dailymax: If `True`, will load in ds conditioned on hour of max surface temperature for each day.
            If `False`, will use daily average.

    Returns:
        ds_out_path:
    """
    dir_script = os.path.dirname(os.path.abspath(__file__))
    ds_out_path = os.path.join(dir_script, 'ds_processed', surf, region, hemisphere)
    if dailymax:
        ds_out_path = os.path.join(ds_out_path, 'dailymax')
    ds_out_path = os.path.join(ds_out_path, f"{kappa_name}.nc")
    return ds_out_path


def amend_sample_dim(ds_list, n_sample_max_range=2):
    # Hack so all ds_list values have same number of samples, should only differ by one or two (less than n_sample_max_range)
    n_ds = len(ds_list)
    n_sample = [ds_list[i].sample.size for i in range(n_ds)]
    if np.max(n_sample) - np.min(n_sample) > n_sample_max_range:
        raise ValueError(f"n_sample={n_sample} has too wide a range of values")
    n_sample = np.min(n_sample)
    ds_list = [ds_list[i].isel(sample=slice(0, n_sample)) for i in range(n_ds)]
    return ds_list


def get_P(ds):
    return ds.PS * ds.hybm

def get_ds(surf=['aquaplanet', 'land'], kappa_names=['k=1', 'k=1_5'], hemisphere=['south', 'north'],
           dailymax=False):
    """
    Function to load in datasets from ds_processed folder in this directory and combine them

    Args:
        surf:
        kappa_names:
        hemisphere:
        dailymax: If `True`, will load in ds conditioned on hour of max surface temperature for each day.
            If `False`, will use daily average.

    Returns:

    """
    n_exp = len(kappa_names)
    ds = {}
    for key in surf:
        ds[key] = [0, 0]  # initialize as empty list
        for i in range(n_exp):
            # Load both hemispheres and combine into single ds
            ds[key][i] = [xr.open_dataset(get_ds_out_path(kappa_names[i], surf=key, hemisphere=hemisphere[j],
                                                          dailymax=dailymax)).isel(surf=0)
                          for j in range(len(hemisphere))]
            ds[key][i] = amend_sample_dim(ds[key][i])
            ds[key][i] = xr.concat(ds[key][i], dim='lat')

        ds[key] = amend_sample_dim(ds[key])
        # print('Selecting subsection of quant')
        # ds[key] = [ds[key][i].sel(quant=[1, 50, 99]) for i in range(n_exp)]
        ds[key] = xr.concat(ds[key], dim="tau_lw")
        ds[key]['hybm'] = ds[key].hybm.isel(quant=0, tau_lw=0, lat=0, sample=0)
        ds[key]['rh_REFHT'] = ds[key]['QREFHT'] / sphum_sat(ds[key].TREFHT, ds[key].PREFHT)
        ds[key]['ZREFHT'] = ds[key].Z3.isel(lev=-1)
        ds[key]['mse_REFHT'] = moist_static_energy(ds[key].TREFHT, ds[key].QREFHT, ds[key].ZREFHT)
        ds[key]['Z_ft_env'] = get_var_at_plev(ds[key].Z3, get_P(ds[key]), ds[key].p_ft)
        ds[key]['mse_ft_sat_env'] = moist_static_energy(ds[key].T_ft_env, sphum_sat(ds[key].T_ft_env, ds[key].p_ft),
                                                        ds[key].Z_ft_env)
        ds[key]['epsilon'] = ds[key]['mse_REFHT'] - ds[key]['mse_ft_sat_env']
        for key2 in ds[key]:
            if 'mod_parcel' in key2:
                ds[key] = ds[key].rename_vars({key2: key2.replace('mod_parcel', 'modParc')})
    return ds


try:
    ds = get_ds()
except Exception as e:
    print(f"No ds loaded because of the following Exception:\n{e}")
    ds = None

if __name__ == '__main__':
    # -- Specific info for running the script --
    p_ft = [500 * 100, 700*100]  # FT pressure level to use
    n_sample = None  # How many data points for each quantile, x to use. None to get all.
    quant_all = np.arange(1, 100, 1)  # Quantiles, x, to get data for.
    # quant_all = np.array([1, 50, 99])    # Small test run
    temp_surf_lcl_calc = 300  # Temperature to use to calculate the LCL. 'median' to compute from data.
    n_lev_above_integral = 3  # Used to compute error in lapse rate integral
    try:
        # ideally get these from terminal
        surf = sys.argv[1]  # 'land' or 'aquaplanet'
        hemisphere = sys.argv[2]  # 'north' or 'south'
        kappa_names = sys.argv[3]  # 'k=1' or 'k=1_5'
    except IndexError:
        # Default values of don't call from terminal - for testing
        surf = 'aquaplanet'
        hemisphere = 'north'
        kappa_names = 'k=1'
    region = 'tropics'
    season = 'summer'
    dailymax = False  # Take daily max data
    # ds_out_path = '/Users/joshduffield/Desktop/ds_isca_quant_dailymax.nc'
    ds_out_path = get_ds_out_path(kappa_names, surf, region, hemisphere, dailymax)

    if os.path.exists(ds_out_path):
        ds_quant = xr.load_dataset(ds_out_path)
        print(f"Loaded {ds_out_path}")
    else:
        print(f"Creating {ds_out_path}")
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S',
                            stream=sys.stdout)
        logger = logging.getLogger()  # for printing to console time info
        #  -- Info for loading in data --
        exp_dir = {'land': 'tau_sweep/land/meridional_band/depth=1/bucket_evap/'} if surf == 'land' else \
            {'aquaplanet': 'tau_sweep/aquaplanet/depth=1/'}  # add a surface dimension
        kappa_names = [kappa_names]  # make a list so add dimension
        n_kappa = len(kappa_names)

        lat_min = {'tropics': {'north': 0, 'south': -20},
                   'not_tropics': {'north': 20, 'south': -90}}
        lat_max = {'tropics': {'north': 20, 'south': 0},
                   'not_tropics': {'north': 90, 'south': -20}}

        lat_min = lat_min[region][hemisphere]
        lat_max = lat_max[region][hemisphere]

        season_months = {'summer': {'tropics': {'north': [5, 6, 7, 8, 9, 10],
                                                'south': [11, 12, 1, 2, 3, 4]},
                                    'not_tropics': {'north': [6, 7, 8],
                                                    'south': [12, 1, 2]}},
                         'winter': {}}
        for key in ['tropics', 'not_tropics']:
            season_months['winter'][key] = {'north': season_months['summer'][key]['south'],
                                            'south': season_months['summer'][key]['north']}

        # Load dataset - one at surface and one in free troposphere
        var_keep = ['temp', 'sphum', 'height', 'cape',
                    'ps', 'convflag', 'klzbs']  # only keep variables required to compute relative humidity and MSE
        use_time_start = 360 * 2

        #  -- Load Data --
        print_log('Loading data', logger)
        ds = {key: [] for key in exp_dir}
        albedo = {key: [] for key in exp_dir}
        tau_lw = {key: [] for key in exp_dir}
        with tqdm(total=n_kappa, position=0, leave=True) as pbar:
            for key in exp_dir:
                for j in range(n_kappa):
                    if dailymax:
                        exp_path = os.path.join(os.environ['GFDL_DATA'], exp_dir[key] + kappa_names[j] + '_dailymax')
                        ds_use = xr.open_mfdataset(f"{exp_path}/lat*.nc", combine='nested',
                                                   concat_dim='lat', decode_times=False)[var_keep]
                        # Convert time so matches that expected from daily average data
                        ds_use['time'] = ds_use['time'] + use_time_start + 0.5
                    else:
                        ds_use = isca_tools.load_dataset(exp_dir[key] + kappa_names[j]
                                                         ).sel(time=slice(use_time_start, np.inf))[var_keep]
                    ds_use['sphum'] = ds_use.sphum.isel(pfull=-1)  # only keep surface SPHUM

                    ds_use = ds_use.sel(lat=slice(lat_min, lat_max))

                    # Only keep land longitudes - for aquaplanet, does not matter which we keep
                    if surf == 'land':
                        job_dir = os.path.join(os.path.dirname(os.environ['GFDL_DATA']), 'jobs')
                        land_file_name = os.path.join(job_dir, exp_dir[key], kappa_names[0], 'land.nc')
                        lon_land = isca_tools.utils.land.get_land_coords(land_file=land_file_name)[1]
                        ds_use = ds_use.isel(lon=np.where(np.isin(ds_use.lon, np.unique(lon_land)))[0])

                    # Only load in months of interest for season and hemisphere
                    ds_use = isca_tools.utils.annual_time_slice(ds_use, season_months[season][region][hemisphere])
                    # Stack longitude and time into new sample dimension to match CESM
                    # ds_use = ds_use.stack(sample=("lon", "time"), create_index=False).chunk(dict(sample=-1))
                    ds[key] += [ds_use.load()]

                    namelist = isca_tools.load_namelist(exp_dir[key] + kappa_names[j])  # Need this for albedo_value
                    tau_lw[key] += [namelist['two_stream_gray_rad_nml']['odp']]
                    pbar.update(1)
                ds[key] = xr.concat(ds[key], dim=xr.DataArray(tau_lw[key], dims="tau_lw", name='tau_lw'))

        # Concatenate ds along surf dimension
        ds = xr.concat([ds[key] for key in ds],
                       dim=xr.DataArray([key for key in ds], dims="surf", name='surf'))
        print_log('Finished loading data', logger)

        # -- Convert ds to CESM like data --
        ds = ds.rename({'temp': 'T', 'sphum': 'QREFHT',
                        'height': 'Z3', 'cape': 'CAPE', 'ps': 'PS',
                        'pfull': 'lev'})

        # sigma_half reflects ds_us.pfull (starts with 0 - space, ends with 1 - surface)
        sigma_levels_half = np.asarray(namelist['vert_coordinate_nml']['bk'])
        # hybm are sigma full levels
        hybm = np.convolve(sigma_levels_half, np.ones(2) / 2,
                           'valid')  # sigma levels corresponding to pressure levels
        ds['hybm'] = ds.lev * 0 + hybm  # convert to xarray

        # choose lowest model level as REFHT
        ds['TREFHT'] = ds.T.isel(lev=-1)
        ds['PREFHT'] = ds.PS * ds.hybm.isel(lev=-1)
        if temp_surf_lcl_calc == 'median':
            temp_surf_lcl_calc = float(np.ceil(ds.TREFHT.median()))


        ds['rh_REFHT'] = ds.QREFHT / sphum_sat(ds.TREFHT, ds.PREFHT)
        p_ft_xr = xr.DataArray(p_ft, dims=["p_ft"], name="p_ft")
        var = [get_var_at_plev(ds.T, get_P(ds), p_ft[i]) for i in range(len(p_ft))]
        ds['T_ft_env'] = xr.concat(var,p_ft_xr)
        ds['T_ft_env_zonal_av'] = ds['T_ft_env'].mean(dim='lon')


        ## -- Get Data Conditioned on Quantile of TREFHT
        def get_ds_quant_single_coord(ds, quant=90, range_below=0.5, range_above=0.5):
            quant_mask = get_quant_ind(ds.TREFHT.squeeze(), quant, range_below, range_above, av_dim=['lon', 'time'],
                                       return_mask=True)
            ds_use = ds.where(quant_mask).stack(sample=("lon", "time"), create_index=False).chunk(dict(sample=-1))
            ds_use = ds_use.load()
            ds_use = ds_use.where(ds_use.TREFHT > 0, drop=True)
            return ds_use


        def get_ds_quant(ds, quant=90, range_below=0.5, range_above=0.5, n_keep=None):
            quant_mask = get_quant_ind(ds.TREFHT, quant, range_below, range_above, av_dim=['lon', 'time'],
                                       return_mask=True)
            # n_keep is so can concat ds of different quantiles. A given quant range will give slightly different numbers of samples at each location.
            # Through providing n_keep, you can ensure the number is always the same.
            n_keep_max = int(quant_mask.sum(dim=['lon', 'time']).min())
            if n_keep is None:
                n_keep = n_keep_max
            if n_keep > n_keep_max:
                raise ValueError(f'n_keep={n_keep} > n_keep_max={n_keep_max}')
            ds_out = []
            for i in range(ds.tau_lw.size):
                ds_use_j = []
                for j in range(ds.surf.size):
                    ds_use_k = []
                    for k in range(ds.lat.size):
                        ds_use_k.append(
                            get_ds_quant_single_coord(ds.isel(tau_lw=i, surf=j, lat=k, drop=True), quant, range_below,
                                                      range_above).isel(sample=slice(0, n_keep)))
                    ds_use_k = xr.concat(ds_use_k, dim=ds.lat)
                    ds_use_j.append(ds_use_k)
                ds_use_j = xr.concat(ds_use_j, dim=ds.surf)
                ds_out.append(ds_use_j)
            ds_out = xr.concat(ds_out, dim=ds.tau_lw)
            return ds_out


        print_log('Computing data conditioned on each quantile', logger)
        ds_quant = []
        for i in tqdm(quant_all):
            ds_quant.append(get_ds_quant(ds, i, n_keep=n_sample))
        if n_sample is None:
            n_sample = np.min([i.sample.size for i in ds_quant])
            # Ensure all have same size
            ds_quant = [i.isel(sample=slice(0, n_sample)) for i in ds_quant]
        ds_quant = xr.concat(ds_quant, dim=xr.DataArray(quant_all, dims="quant", name='quant'))


        ## -- Get Error Parameters for lapse rate fitting --

        def get_lapse_fitting_info(ds: xr.Dataset):
            """Compute lapse fitting diagnostics for each quant, returning ds with added vars.

            Args:
                ds (xr.Dataset): Input dataset with dimension 'quant'.

            Returns:
                xr.Dataset: Dataset with computed lapse-fitting fields for each quant.
            """
            ds.attrs["n_lev_above_integral"] = n_lev_above_integral
            var_names = ["lapse", "integral", "error"]
            keys = ["const", "mod_parcel"]

            quants = ds["quant"].values
            out_list_p = []         # contains one dataarray for each p_ft

            with tqdm(total=len(quants)*len(p_ft), position=0, leave=True) as pbar:
                for p_ft_use in p_ft:
                    out_list_q = []             # contains one dataarray for each quant
                    for q in quants:
                        small = {}

                        for key in keys:
                            var = fitting_2_layer_xr(
                                ds.T.sel(quant=q),
                                get_P(ds).sel(quant=q),
                                ds.TREFHT.sel(quant=q),
                                ds.PREFHT.sel(quant=q),
                                ds.rh_REFHT.sel(quant=q),
                                ds.T_ft_env.sel(quant=q, p_ft=p_ft_use),
                                p_ft_use,
                                n_lev_above_upper2_integral=ds.n_lev_above_integral,
                                method_layer2=key,
                                temp_surf_lcl_calc=temp_surf_lcl_calc,
                            )

                            for k, name in enumerate(var_names):
                                # Build variable name such as "const1_lapse"
                                vname = f"{key}1_{name}"
                                small[vname] = var[k].expand_dims(quant=[q])

                        # One small dataset per q
                        out_list_q.append(xr.Dataset(small))

                        pbar.update(1)
                    # Concatenate all per-quant datasets
                    out_list_p.append(xr.concat(out_list_q, dim="quant"))
            new_vars = xr.concat(out_list_p, dim=p_ft_xr)       # concatenate p_ft values
            # Merge into original ds
            return xr.merge([ds, new_vars])


        print_log('Obtaining lapse rate fitting data', logger)
        ds_quant = get_lapse_fitting_info(ds_quant)
        # Add metadata to save
        ds_quant.attrs['temp_surf_lcl_calc'] = temp_surf_lcl_calc
        ds_quant.attrs["region"] = region
        ds_quant.attrs["hemisphere"] = hemisphere
        ds_quant.attrs["season"] = season
        ds_quant.attrs["exp_dir"] = str(exp_dir)
        ds_quant.attrs["dailymax"] = str(dailymax)
        ds_quant = convert_ds_dtypes(ds_quant)
        if not os.path.exists(ds_out_path):
            ds_quant.to_netcdf(ds_out_path)
            print_log('Saved ds_quant to {}'.format(ds_out_path), logger)
