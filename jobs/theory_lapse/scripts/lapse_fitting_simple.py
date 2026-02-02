# Uses physical LCL to compute the mod_parcel1 and const1 lapse rate fitting
# Also estimates the LNB where parcel no longer buoyant
# Does lapse analysis for three separate p_ft values
import os
import numpy as np
import xarray as xr
import logging
import sys

sys.path.append('/Users/joshduffield/Documents/StAndrews/Isca')

from isca_tools.utils import get_memory_usage
from isca_tools.utils.base import print_log
from isca_tools.utils.moist_physics import sphum_sat
from isca_tools.utils.xarray import print_ds_var_list, convert_ds_dtypes, wrap_with_apply_ufunc
from isca_tools.cesm import get_pressure
from isca_tools.papers.miyawaki_2022 import get_lapse_dev
from isca_tools.utils.constants import lapse_dry
from isca_tools.convection.base import lcl_metpy
from isca_tools.thesis.lapse_theory import get_var_at_plev
from geocat.comp.interpolation import interp_hybrid_to_pressure
from isca_tools.thesis.lapse_integral_simple import fitting_2_layer_xr, get_lnb_ind
from tqdm import tqdm
from typing import Literal, Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S',
                    stream=sys.stdout)
get_lnb_ind_xr = wrap_with_apply_ufunc(get_lnb_ind, input_core_dims=[['lev'], ['lev'], [], []])

# File location Info
from jobs.theory_lapse.scripts.lcl import load_ds_quant, data_dir, var_keep

var_initial_load = [item for item in var_keep if
                    item not in ['Z3', 'CAPE', 'FREQZM']]  # get rid of variables don't need
test_mode = False            # for testing on a small dataset
try:
    # ideally get quant_type from terminal
    quant_type = sys.argv[1]  # e.g. 'REFHT_quant50'
    exp_name = sys.argv[2]    # 'pre_industrial' or 'co2_2x'
    try:
        lat_ind_start = int(sys.argv[3])
    except (IndexError, ValueError):
        lat_ind_start = 1       # starts at 1
    try:
        lat_ind_end = int(sys.argv[4])
    except (IndexError, ValueError):
        lat_ind_end = None
except (IndexError, ValueError):
    # Default values of don't call from terminal - for testing
    exp_name = ['pre_industrial', 'co2_2x']
    quant_type = 'REFHT_quant50'
    lat_ind_start = 1
    lat_ind_end = None
exp_name = np.atleast_1d(exp_name)
processed_dir = [os.path.join(data_dir, exp_name[i], quant_type, 'lapse_fitting') for i in range(len(exp_name))]
processed_file_name = 'ds_lapse_simple.nc'  # combined file from all samples
p_ft = [400 * 100, 500 * 100]  # Free tropospheric levels - for each one, will obtain lapse fitting info
rh_mod = [-0.1, -0.05, 0, 0.05, 0.1]        # modifications to rh and thus LCL to consider
p_ft = np.atleast_1d(p_ft)
comp_level = 4
n_lev_above_integral = 3
temp_surf_lcl_calc = 300  # Compute LCL from RH using 300K as surf temp
lev_REFHT = -1  # Model level used to compute rh_REFHT, and subsequent lapse fitting. If None will use actual REFHT
const_layer1_method = 'optimal'


def get_ds_quant_lat(co2_ind, lat_ind):
    if test_mode:
        lon_ind = [0, 1, 2]
        sample_ind = [0, 1]
    else:
        lon_ind = None
        sample_ind = None
    ds = load_ds_quant([exp_name[co2_ind]], quant_type, data_dir, var_initial_load, compute_p_diff=False,
                       lat_ind=lat_ind, lon_ind=lon_ind, sample_ind=sample_ind, lev_REFHT=lev_REFHT)
    # Compute rh at REFHT
    ds['rh_REFHT'] = ds.QREFHT / sphum_sat(ds.TREFHT, ds.PREFHT)

    # Interpolate data onto FT level
    ds['T_ft_env'] = interp_hybrid_to_pressure(ds.T, ds.PS, ds.hyam, ds.hybm, ds.P0, p_ft,
                                               lev_dim='lev')
    ds['T_ft_env'].load()
    ds = ds.rename({'plev': 'p_ft'})  # change to p_ft
    ds.attrs['temp_surf_lcl_calc'] = temp_surf_lcl_calc
    ds.attrs['n_lev_above_integral'] = n_lev_above_integral
    ds.attrs['lev_REFHT'] = lev_REFHT
    ds.attrs['const_layer1_method'] = const_layer1_method
    return ds


def get_lapse_fitting_info(
        ds: xr.Dataset,
        n_lev_above_integral: int,
        rh_mod: np.ndarray,
        var_loop: Optional[str],
        p_lev: Optional[float] = None,
        temp_surf_lcl_calc: float = 300,
        const_layer1_method: Literal['bulk', 'optimal'] = 'optimal') -> xr.Dataset:
    """Compute lapse fitting diagnostics for each value of `var_loop`, returning ds with added vars.

    This loops over `p_ft` and over a chosen dataset dimension/coordinate given by `var_loop`
    (e.g. "quant"), and computes lapse-fitting fields for each combination.

    Args:
        ds (xr.Dataset): Input dataset containing `p_ft` and the loop coordinate `var_loop`,
            plus the variables used in fitting (e.g. T, TREFHT, PREFHT, rh_REFHT, T_ft_env, PS).
        n_lev_above_integral (int): Number of levels above the upper layer boundary used in
            the integral calculation; saved to `ds.attrs["n_lev_above_integral"]`.
        rh_mod (np.ndarray): Relative-humidity perturbations to apply for the modified-parcel
            calculation; mapped onto an xarray coordinate called "rh_mod".
        var_loop (str): Name of the coordinate/dimension to loop over;
            must be indexable via `.sel({var_loop: value})` on required variables.
        p_lev (float, optional): Pressure levels to use in fitting. If None, uses `ds.P`.
        temp_surf_lcl_calc:
        const_layer1_method:

    Returns:
        xr.Dataset: Dataset with computed lapse-fitting fields added.
    """
    if p_lev is None:
        p_lev = ds.P

    ds.attrs["n_lev_above_integral"] = n_lev_above_integral
    ds.attrs['temp_surf_lcl_calc'] = temp_surf_lcl_calc
    ds.attrs["const_layer1_method"] = const_layer1_method

    rh_mod_xr = xr.DataArray(
        rh_mod,
        dims=["rh_mod"],
        name="rh_mod",
        coords={"rh_mod": rh_mod},
        attrs={"long_name": "RH perturbations applied to REFHT RH for modified parcel"},
    )

    var_names = ["lapse", "integral", "error"]
    keys = ["const", "mod_parcel", "parcel"]  # add parcel for perfect parcel error

    # Keep the same structure: always loop over *some* dimension, but make it temporary if var_loop is None.
    tmp_dim = None
    if var_loop is None:
        tmp_dim = "var_loop_tmp"
        var_loop_use = tmp_dim
        loop_vals = np.array([0])
    else:
        var_loop_use = var_loop
        loop_vals = ds[var_loop_use].values

    out_list_p = []  # contains one dataset for each p_ft

    with tqdm(total=len(loop_vals) * len(ds.p_ft), position=0, leave=True) as pbar:
        for p_ft_ind, p_ft_use in enumerate(ds.p_ft):
            out_list_l = []  # contains one dataset for each var_loop value

            for v in loop_vals:
                small = {}

                # Helper selector: either select along var_loop, or do nothing if tmp dim.
                if tmp_dim is None:
                    sel_loop = {var_loop_use: v}
                else:
                    sel_loop = {}

                # --- fitting outputs for each method ---
                for key in keys:
                    var = fitting_2_layer_xr(
                        ds.T.sel(sel_loop),
                        p_lev.sel(sel_loop),
                        ds.TREFHT.sel(sel_loop),
                        ds.PREFHT.sel(sel_loop),
                        ds.rh_REFHT.sel(sel_loop)
                        if key == "parcel"
                        else ds.rh_REFHT.sel(sel_loop) + rh_mod_xr,
                        ds.T_ft_env.sel({**sel_loop, "p_ft": p_ft_use}),
                        p_ft_use,
                        n_lev_above_upper2_integral=ds.n_lev_above_integral,
                        method_layer1="const",
                        method_layer2=key,
                        const_layer1_method=const_layer1_method,
                        mod_parcel_method="add",
                        force_parcel=key == "parcel",
                        temp_surf_lcl_calc=temp_surf_lcl_calc,
                    )

                    for k, name in enumerate(var_names):
                        vname = f"{key}_{name}"
                        small[vname] = var[k]

                # --- diagnostics that only need one p_ft ---
                if p_ft_ind == 0:
                    lapse_mod_D = (
                            small["mod_parcel_lapse"].isel(layer=0, drop=True) / 1000 - lapse_dry
                    )

                    small["lnb1_ind"] = get_lnb_ind_xr(
                        ds.T.sel(sel_loop),
                        p_lev.sel(sel_loop),
                        ds.rh_REFHT.sel(sel_loop),
                        0,
                        temp_surf_lcl_calc=temp_surf_lcl_calc,
                    )
                    # small["lnb2_ind"] = get_lnb_ind_xr(
                    #     ds.T.sel(sel_loop),
                    #     p_lev.sel(sel_loop),
                    #     ds.rh_REFHT.sel(sel_loop) + rh_mod_xr,
                    #     lapse_mod_D,
                    #     temp_surf_lcl_calc=temp_surf_lcl_calc,
                    # )
                    small["lapse_miy2022_M"], small["lapse_miy2022_D"] = get_lapse_dev(
                        ds.T.sel(sel_loop),
                        p_lev.sel(sel_loop),
                        ds.PS.sel(sel_loop),
                    )

                # Add variable-level attrs (strings) for everything created in `small`
                for vn, da in list(small.items()):
                    if hasattr(da, "attrs"):
                        da.attrs.setdefault("long_name", vn)

                # One small dataset per loop value
                out_list_l.append(xr.Dataset(small).expand_dims({var_loop_use: [v]}))

                pbar.update(1)

            # Concatenate all per-var_loop datasets at fixed p_ft
            out_list_p.append(xr.concat(out_list_l, dim=var_loop_use))

    # Concatenate across p_ft values
    new_vars = xr.concat(out_list_p, dim=ds.p_ft)

    # For these variables, only need a single p_ft
    for key in ["lnb1_ind", "lapse_miy2022_M", "lapse_miy2022_D"]:
        new_vars[key] = new_vars[key].isel(p_ft=0, drop=True)

    # Drop the temporary loop dimension if we created it (size-1).
    # squeeze(..., drop=True) removes the length-1 dimension and drops its coordinate. [web:21]
    if tmp_dim is not None:
        new_vars = new_vars.squeeze(tmp_dim, drop=True)

    # Merge into original ds
    return xr.merge([ds, new_vars])


if __name__ == '__main__':
    logger = logging.getLogger()  # for printing to console time info

    # ds = load_ds_quant(exp_name, quant_type, data_dir, var_keep, small_ds=small_ds, compute_p_diff=False,
    #                    load_fully=False)
    print_log(f'Quantile data loaded | Memory used {get_memory_usage() / 1000:.1f}GB',
              logger)

    load_processed = [os.path.exists(os.path.join(processed_dir[i], processed_file_name)) for i in range(len(exp_name))]

    # Compute empirical estimate of LCL
    lat_vals = load_ds_quant([exp_name[0]], quant_type, data_dir, ['TREFHT'], compute_p_diff=False,
                             sample_ind=0, lon_ind=0).lat
    n_lat = lat_vals.size
    if lat_ind_end is None:
        lat_ind_end = n_lat
    n_files = len(exp_name) * n_lat  # One file for each co2 conc and sample due to speed
    print_log(f'Empirical lapse fitting for {n_files} Files | Start | Lat start ind = {lat_ind_start} | '
              f'Lat end ind = {lat_ind_end} | Max possible lat ind = {n_lat}', logger)

    var_names = ['lapse', 'integral', 'error']
    n_digit = len(str(n_lat))
    # One file for each latitude and co2
    for i in range(len(exp_name)):
        if load_processed[i]:
            print_log(f'Files already exist for {exp_name[i]}', logger)
            continue
        path_use = [os.path.join(processed_dir[i], f'lat{j:0{n_digit}d}.nc') for j in range(n_lat)]
        for j in range(n_lat):
            if (j+1 < lat_ind_start) or (j+1 > lat_ind_end):
                # Don't get data for these latitudes as outside range selected
                print_log(f'File {i * n_lat + j + 1}/{n_files} | '
                          f'Skipped - outside lat ind range of {lat_ind_start}-{lat_ind_end} | '
                          f'Memory used {get_memory_usage() / 1000:.1f}GB', logger)
                continue
            if os.path.exists(path_use[j]):
                print_log(f'File {i * n_lat + j + 1}/{n_files} Exists Already', logger)
                continue
            print_log(f'File {i * n_lat + j + 1}/{n_files} | Start | Memory used {get_memory_usage() / 1000:.1f}GB',
                      logger)
            ds_use = get_ds_quant_lat(i, j)
            print_log(
                f'File {i * n_lat + j + 1}/{n_files} | Data loaded | Memory used {get_memory_usage() / 1000:.1f}GB',
                logger)
            ds_use = get_lapse_fitting_info(ds_use, n_lev_above_integral, rh_mod, var_loop='lon',
                                            temp_surf_lcl_calc=temp_surf_lcl_calc,
                                            const_layer1_method=const_layer1_method)
            print_log(
                f'File {i * n_lat + j + 1}/{n_files} | Obtained lapse info | Memory used {get_memory_usage() / 1000:.1f}GB',
                logger)
            # Save lnb indices as integers
            ds_use['lnb1_ind'] = ds_use['lnb1_ind'].fillna(-1).astype(int)
            # ds_use['lnb2_ind'] = ds_use['lnb2_ind'].fillna(-1).astype(int)

            # Save data
            ds_use['layer'] = xr.DataArray(['below lcl', 'above lcl'], name='layer', dims='layer')
            ds_use = ds_use.drop_vars(['T', 'P'])  # drop lat-lon-lev vars as have that data anyway
            ds_use = convert_ds_dtypes(ds_use)
            if not os.path.exists(path_use[j]):
                ds_use.to_netcdf(path_use[j], format="NETCDF4",
                                 encoding={var: {"zlib": True, "complevel": comp_level} for var in ds_use.data_vars})
            print_log(f'File {i * n_lat + j + 1}/{n_files} | Saved | Memory used {get_memory_usage() / 1000:.1f}GB',
                      logger)

        if not os.path.exists(os.path.join(processed_dir[i], processed_file_name)):
            # Combine all sample files into a single file for each experiment
            ds_lapse = xr.concat([xr.load_dataset(path_use[j]) for j in range(n_lat)], dim=lat_vals)
            # Need to make hyam and hybm a single latitude
            ds_lapse['hyam'] = ds_lapse['hyam'].isel(lat=0, drop=True)
            ds_lapse['hybm'] = ds_lapse['hybm'].isel(lat=0, drop=True)
            # Reorder
            ds_lapse = ds_lapse.transpose('lat', 'lon', 'sample', 'p_ft', 'rh_mod', 'layer', 'lev')
            ds_lapse.to_netcdf(os.path.join(processed_dir[i], processed_file_name), format="NETCDF4",
                               encoding={var: {"zlib": True, "complevel": comp_level} for var in ds_lapse.data_vars})
            print_log(f'{exp_name[i]} | Combined samples into one {processed_file_name} File', logger)

    hi = 5
