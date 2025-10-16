# Test functions in profile_fitting
import os
import xarray as xr
import numpy as np
from isca_tools import cesm
from isca_tools.convection.base import lcl_metpy
from isca_tools.thesis.lapse_theory import get_var_at_plev
from isca_tools.thesis.profile_fitting import get_lnb_lev_ind, get_mse_env, interp_var_to_pnorm
from isca_tools.utils.moist_physics import moist_static_energy, sphum_sat
from isca_tools.utils.xarray import flatten_to_numpy, unflatten_from_numpy
from isca_tools.utils.decomposition import pca_on_xarray, scaled_k_means
from isca_tools.utils.debug import save_workspace, load_workspace
from isca_tools.utils.base import print_log
import matplotlib
import logging
import sys

show_plot = True
if show_plot:
    matplotlib.use("TkAgg")  # To show plots in debugging mode
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S',
                    stream=sys.stdout)
logger = logging.getLogger()  # for printing to console time info

workspace_path = '/Users/joshduffield/Desktop/workspace_shelf.joblib'
if os.path.exists(workspace_path):
    load_workspace(workspace_path, globals())
    print_log(f'Workspace loaded in mse_env_pnorm', logger)
else:
    print_log(f'No workspace | Loading Data', logger)
    # Load in example dataset
    data_dir = '/Users/joshduffield/Documents/StAndrews/Isca/jobs/cesm/3_hour/hottest_quant'
    quant_type = 'REFHT_quant99'
    exp_name = ['pre_industrial', 'co2_2x']
    n_exp = len(exp_name)
    co2_vals = np.arange(1, n_exp + 1)
    lat_min = 0
    lat_max = 50
    lon_min = 0
    lon_max = 20
    ds = [xr.open_dataset(os.path.join(data_dir, exp_name[i], quant_type, 'output.nc')) for i in range(n_exp)]
    ds = xr.concat(ds, dim=xr.DataArray(co2_vals, dims="co2", coords={"exp_name": ("co2", exp_name)}))
    # ds = ds.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))
    ds = ds.load()
    print_log(f'Loaded Data', logger)

    # Set pressure and refht to lowest model level
    ds['P'] = cesm.get_pressure(ds.PS, ds.P0.isel(co2=0), ds.hyam.isel(co2=0), ds.hybm.isel(co2=0))
    ds['TREFHT'] = ds.T.isel(lev=-1)
    ds['QREFHT'] = ds.Q.isel(lev=-1)
    ds['ZREFHT'] = ds.Z3.isel(lev=-1)
    ds['PREFHT'] = ds.P.isel(lev=-1)

    # Compute LCL and LNB
    ds['p_lcl'], ds['T_lcl'] = lcl_metpy(ds.T.isel(lev=-1), ds.Q.isel(lev=-1), ds.P.isel(lev=-1))
    ds['T_at_lcl'] = get_var_at_plev(ds.T, ds.P, ds.p_lcl)
    ds['Z_at_lcl'] = get_var_at_plev(ds.Z3, ds.P, ds.p_lcl)
    ds['mse_sat_at_lcl'] = moist_static_energy(ds['T_at_lcl'], sphum_sat(ds['T_at_lcl'], ds['p_lcl']), ds['Z_at_lcl'])
    ds['lnb_ind'] = get_lnb_lev_ind(ds.T, ds.Z3, ds.P)
    print_log(f'Computed LCL', logger)

    # Set FT info
    p_ft = 500 * 100
    ds['T_ft'] = get_var_at_plev(ds.T, ds.P, p_ft)
    ds['Z_ft'] = get_var_at_plev(ds.Z3, ds.P, p_ft)
    ds['mse_sat_ft'] = moist_static_energy(ds['T_ft'], sphum_sat(ds['T_ft'], p_ft), ds['Z_ft'])

    # Compute MSE environmental profile (separate above and below LCL), then innterpolate
    for key in ['below', 'above']:
        ds[f'mse_env_{key}'] = get_mse_env(ds.T, ds.P, ds.Z3, ds.T_at_lcl, ds.p_lcl, prof_type=f'{key}_lcl')

    mse_env_pnorm = {}
    mse_env_pnorm['below'] = interp_var_to_pnorm(ds.mse_env_below, ds.P, ds.mse_env_below.isel(lev=-1),
                                                 ds.P.isel(lev=-1), ds.mse_sat_at_lcl, ds.p_lcl, ds.lnb_ind)
    print_log(f'Computed below LCL MSE profile', logger)
    mse_env_pnorm['above'] = interp_var_to_pnorm(ds.mse_env_above, ds.P, ds.mse_sat_at_lcl, ds.p_lcl,
                                                 ds.mse_sat_ft, p_ft, ds.lnb_ind)
    print_log(f'Computed above LCL MSE profile', logger)
    save_workspace(workspace_path, {'mse_env_pnorm': mse_env_pnorm, 'p_ft': p_ft})
    print_log(f'Saved mse_env_pnorm to workspace', logger)

# Land masks
invariant_data_path = '/Users/joshduffield/Documents/StAndrews/Isca/jobs/cesm/input_data/fv_0.9x1.25_nc3000_Nsw042_Nrs008_Co060_Fi001_ZR_sgh30_24km_GRNL_c170103.nc'
lsm = (xr.open_dataset(invariant_data_path).LANDFRAC > 0)
lsm = lsm.reindex_like(mse_env_pnorm['below'][0], method="nearest", tolerance=0.01)

# Do PCA on mse on pnorm grid, to get starting point for scaled k-means
lat = mse_env_pnorm['below'][0].lat
extrap_valid_thresh = 2  # no more than this many points above LNB
mse_valid_thresh = 20  # only use points where MSE deviates from conv neutral by this much
lat_valid_lims = [-60, 75]
valid = {key: (lsm > 0) & (mse_env_pnorm[key][1] <= extrap_valid_thresh) & (
        np.abs(mse_env_pnorm[key][0]).max(dim='pnorm') <= mse_valid_thresh) &
              (lat > lat_valid_lims[0]) & (lat < lat_valid_lims[1]) for key in mse_env_pnorm}

# Do scaled k-means using PC output as starting point
n_modes = 4
dot_product_thresh = 0.7  # for initializing 2-atom scaled k-means, use as starting point atoms with dot product less than this
pca_output = {}
k_output = {}
pca_output_residual = {}
k_output_residual = {}
k_output_final = {}
mse_norm_thresh = 1  # for profiles with L2 norm less than this, don't use to update clusters
min_cluster_size = 200  # need more than this many samples to compute cluster mean
clusters_final = {}
for key in valid:
    pca_output[key] = list(pca_on_xarray(mse_env_pnorm[key][0].isel(pnorm=slice(1, 9999)),
                                         valid=valid[key], feature_dim_name='pnorm',
                                         standardize=True, n_modes=n_modes,
                                         reference_mean=False))
    print_log(f'{key} LCL | Computed PCA on MSE profiles', logger)
    x_use = flatten_to_numpy(mse_env_pnorm[key][0].isel(pnorm=slice(1, 9999)), 'pnorm')
    k_output[key] = list(scaled_k_means(x_use, pca_output[key][0].to_numpy(), valid=flatten_to_numpy(valid[key]),
                                        norm_thresh=mse_norm_thresh, min_cluster_size=min_cluster_size))
    residual = x_use - k_output[key][-2] * k_output[key][0][k_output[key][2]]
    residual = unflatten_from_numpy(residual, mse_env_pnorm[key][0].isel(pnorm=slice(1, 9999)), 'pnorm')
    k_output[key].append(residual)  # add residual to k_output dict
    print_log(f'{key} LCL | Computed k-means from PCs', logger)

    # Do PC on residual to see what best explains what is left after fitting 1 PC
    pca_output_residual[key] = list(pca_on_xarray(residual, valid=valid[key], feature_dim_name='pnorm',
                                                  standardize=True, n_modes=n_modes, reference_mean=False))
    print_log(f'{key} LCL | Computed PCA on residual profiles', logger)

    x_use = flatten_to_numpy(residual, 'pnorm')
    k_output_residual[key] = list(scaled_k_means(x_use, pca_output_residual[key][0].to_numpy(), valid=flatten_to_numpy(valid[key]),
                                                 norm_thresh=mse_norm_thresh, min_cluster_size=min_cluster_size))

    # Sanity check that clusters of residual differ from clusters of main profile
    for i in range(n_modes):
        print(f'{key} Residual cluster {i} dot product with each k-means cluster: '
              f'{np.abs(np.round(k_output[key][0] @ k_output_residual[key][0][i], 2))}')
    clusters_final[key] = k_output[key][0][np.linalg.norm(k_output[key][0], axis=1) > 0]
    for i in range(n_modes):
        if np.linalg.norm(k_output_residual[key][0][i]) == 0:
            continue
        cluster_dot_product = np.abs(np.round(clusters_final[key] @ k_output_residual[key][0][i], 2))
        print(i, cluster_dot_product, np.max(cluster_dot_product) < dot_product_thresh)
        if np.max(cluster_dot_product) < dot_product_thresh:
            clusters_final[key] = np.vstack([clusters_final[key], k_output_residual[key][0][i]])
    x_use = flatten_to_numpy(mse_env_pnorm[key][0].isel(pnorm=slice(1, 9999)), 'pnorm')
    k_output_final[key] = list(scaled_k_means(x_use, clusters_final[key], norm_thresh=mse_norm_thresh,
                                              n_atom_select=2, n_iter=0))       # 0 iterations so does not recompute clusters

hi = 5

# Updating 2 clusters at same time - too complicated I think
# For final k-means, initialize with the residual PCs as well as the initial clusters
# clusters_init[key] = k_output[key][0][np.linalg.norm(k_output[key][0], axis=1) > 0]
# for i in range(n_modes):
#     cluster_dot_product = np.abs(np.round(clusters_init[key] @ pca_output_residual[key][0][i].to_numpy(), 2))
#     print(i, cluster_dot_product)
#     if np.max(cluster_dot_product) < dot_product_thresh:
#         clusters_init[key] = np.vstack([clusters_init[key], pca_output_residual[key][0][i].to_numpy()])
# x_use = flatten_to_numpy(mse_env_pnorm[key][0].isel(pnorm=slice(1, 9999)), 'pnorm')
# # a = list(scaled_k_means2(x_use, clusters_init, valid=flatten_to_numpy(valid[key]),
# #                                             norm_thresh=mse_norm_thresh, n_atom_select=2))
# a2 = list(scaled_k_means(x_use, clusters_init[key], valid=flatten_to_numpy(valid[key]),
#                          norm_thresh=mse_norm_thresh, n_atom_select=2, n_iter=10,
#                          min_cluster_size=min_cluster_size))

# key = 'above'
# for i in range(n_modes):
#     plt.plot(k_output[key][0][i], pca_output_residual[key][0].pnorm, color=f'C{i}')
#     plt.plot(pca_output[key][0].isel(mode=i), pca_output_residual[key][0].pnorm, color=f'C{i}', linestyle='--')
#     plt.plot(pca_output_residual[key][0].isel(mode=i), pca_output_residual[key][0].pnorm, color=f'C{i}',
#              linestyle=':')

# for i in range(clusters_init[key].shape[0]):
#     plt.plot(clusters_init[key][i], pca_output_residual[key][0].pnorm, color=f'C{i}')
#     plt.plot(a2[0][i], pca_output_residual[key][0].pnorm, color=f'C{i}', linestyle=':')
