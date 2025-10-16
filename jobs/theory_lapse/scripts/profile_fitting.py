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
show_plot = False
if show_plot:
    matplotlib.use("TkAgg")  # To show plots in debugging mode
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S',
                    stream=sys.stdout)
logger = logging.getLogger()  # for printing to console time info

workspace_path = '/Users/joshduffield/Desktop/workspace_shelf'
if os.path.exists(f"{workspace_path.replace('.db', '')}.db"):
    load_workspace(workspace_path.replace('.db', ''), globals())
    print_log(f'Workspace loaded in', logger)
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
    ds = ds.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))
    ds = ds.load()
    print_log(f'Loaded Data', logger)

    # Land masks
    invariant_data_path = '/Users/joshduffield/Documents/StAndrews/Isca/jobs/cesm/input_data/fv_0.9x1.25_nc3000_Nsw042_Nrs008_Co060_Fi001_ZR_sgh30_24km_GRNL_c170103.nc'
    lsm = (xr.open_dataset(invariant_data_path).LANDFRAC > 0)
    lsm = lsm.reindex_like(ds, method="nearest", tolerance=0.01)
    ds['ZREFHT'] = cesm.load.load_z2m(invariant_data_path, var_reindex_like=ds.PS)

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

    # Do PCA on mse on pnorm grid, to get starting point for scaled k-means
    extrap_valid_thresh = 2     # no more than this many points above LNB
    mse_valid_thresh = 20         # only use points where MSE deviates from conv neutral by this much
    lat_valid_lims = [-60, 75]
    valid = {key: (lsm>0) & (mse_env_pnorm[key][1] <= extrap_valid_thresh) & (np.abs(mse_env_pnorm[key][0]).max(dim='pnorm') <= mse_valid_thresh) &
                  (ds.lat > lat_valid_lims[0]) & (ds.lat < lat_valid_lims[1]) for key in mse_env_pnorm}

    n_modes = 4
    # slice so avoid pnorm=0 where var=0
    pca_output = {key: list(pca_on_xarray(mse_env_pnorm[key][0].isel(pnorm=slice(1, 9999)),
                                          valid=valid[key], feature_dim_name='pnorm',
                                          standardize=True, n_modes=n_modes,
                                          reference_mean=False)) for key in mse_env_pnorm}
    print_log(f'Computed PCA on MSE profiles', logger)

    # Do scaled k-means using PC output as starting point
    k_output = {}
    pca_output_residual = {}
    mse_norm_thresh = 1         # for profiles with L2 norm less than this, don't use to update clusters
    for key in pca_output:
        x_use = flatten_to_numpy(mse_env_pnorm[key][0].isel(pnorm=slice(1, 9999)), 'pnorm')
        k_output[key] = list(scaled_k_means(x_use, pca_output[key][0].to_numpy(), valid=flatten_to_numpy(valid[key]),
                                            score_thresh=mse_norm_thresh))
        residual = x_use - k_output[key][-1][:, np.newaxis] * k_output[key][0][k_output[key][2]]
        residual = unflatten_from_numpy(residual, mse_env_pnorm[key][0].isel(pnorm=slice(1, 9999)), 'pnorm')
        k_output[key].append(residual)  # add residual to k_output dict

        # Do PC on residual to see what best explains what is left after fitting 1 PC
        pca_output_residual[key] = list(pca_on_xarray(residual, valid=valid[key], feature_dim_name='pnorm',
                                                      standardize=True, n_modes=n_modes, reference_mean=False))

        # Sanity check that PCs of residual differ from the k-means clusters
        for i in range(n_modes):
            print(f'{key} Residual PC{i} dot product with each k-means cluster: '
                  f'{np.abs(np.round(k_output[key][0] @ pca_output_residual[key][0][i].to_numpy(), 2))}')
    print_log(f'Finished scaled k-means', logger)
    save_workspace(workspace_path.replace('.db', ''))
    print_log(f'Saved workspace', logger)

from typing import Optional
import itertools

def best_score_excluding_atom(norm_reduction: np.ndarray,
                              combinations: np.ndarray,
                              atom: np.ndarray) -> np.ndarray:
    """
    For each sample, find the maximum norm_reduction value
    among all combinations that do NOT contain atom[i].

    Parameters
    ----------
    norm_reduction : (n_sample, n_comb)
        Score or reduction value for each sample–combination pair.
    combinations : (n_comb, n_atom_select)
        Atom indices used in each combination.
    atom : (n_sample,)
        Atom index to exclude for each sample.

    Returns
    -------
    best_score_excl : (n_sample,)
        Max norm_reduction for each sample excluding combinations that contain atom[i].
    """
    # (n_sample, n_comb): True if this combo contains that sample’s excluded atom
    contains_atom = np.any(combinations[None, :, :] == atom[:, None, None], axis=2)

    # Mask those out
    masked_scores = np.where(~contains_atom, norm_reduction, -np.inf)

    # Take max over combinations
    best_score_excl = masked_scores.max(axis=1)

    return best_score_excl

def scaled_k_means2(x: np.ndarray, initial_cluster_mean: np.ndarray, valid: Optional[np.ndarray] = None,
                    n_atom_select: int = 1, norm_thresh: float = 1, score_thresh: float=0.5, score_diff_thresh: float=0.1,
                    score_diff_thresh_test_converge: float = 0.05, min_cluster_size: int = 10,
                    n_iter: int = 100) -> np.ndarray:
    n_sample, n_feature = x.shape
    norm_cluster_mean = initial_cluster_mean / np.linalg.norm(initial_cluster_mean, axis=1).reshape(-1, 1)
    norm_cluster_mean = np.vstack([norm_cluster_mean, np.zeros(n_feature)])      # add an array of zeros
    n_atom = norm_cluster_mean.shape[0]
    cluster_eig_val = np.zeros(n_atom)
    cluster_ind = np.full(x.shape[0], -20, dtype=int)
    x_norm = np.linalg.norm(x, axis=1)
    # n_atom_select = 2
    atom_perm = np.array(list(itertools.combinations(range(n_atom), n_atom_select)))      # all possible permutations of atoms
    atom_perm = np.sort(atom_perm, axis=1)      # ensure larger index later to ensure zeros array always last
    n_perm = len(atom_perm)
    perm_zero_ind = np.where([n_atom-1 in atom_perm[i] for i in range(n_perm)])[0].squeeze()       # all permutations with the last ind which is 0
    atom_perm[perm_zero_ind] = 0
    ignore_perm = np.zeros(n_perm, dtype=bool)
    coef = np.zeros((n_sample, n_perm, n_atom_select))
    for i in range(n_iter):
        for j in range(n_perm):
            if ignore_perm[j]:
                continue        # keep all coefs zero in this case
            if j in perm_zero_ind: # keep zero atom coefficient as zero, compute coefficient for other atoms
                if n_atom_select > 1:
                    # Compute coefficient of other atoms
                    A = norm_cluster_mean[atom_perm[j][:-1]]
                    AAT_inv = np.linalg.inv(A @ A.T)   # (n_atom_select-1, n_atom_select-1)
                    coef[:, j, :-1] = (AAT_inv @ A @ x.T).T  # (n_sample, n_atom_select-1)
            else:
                A = norm_cluster_mean[atom_perm[j]] # (n_atom_select, n_dim)
                AAT_inv = np.linalg.inv(A @ A.T)  # (n_atom_select, n_atom_select)
                coef[:, j] = (AAT_inv @ A @ x.T).T # (n_sample, n_atom_select), repeat for all possible permutations of atoms
        cluster_ind_old = cluster_ind.copy()
        # coef = x @ norm_cluster_mean.transpose()   # because each initial_cluster_mean has norm of 1
        x_residual = x[:, None] - (coef[..., None] * norm_cluster_mean[atom_perm][None]).sum(axis=-2)   # sum over n_atom_select
        x_residual_norm = np.linalg.norm(x_residual, axis=-1)
        # norm_reduction = (x_norm[:, None] - x_residual_norm) / x_norm[:, None]
        cluster_ind = x_residual_norm.argmin(axis=1)
        if n_atom_select > 1:
            # If residual is already small including one of atoms selected as zero, then select as best cluster
            good_with_zero = x_residual_norm[:, perm_zero_ind].min(axis=1) <= norm_thresh
            cluster_ind[good_with_zero] = perm_zero_ind[x_residual_norm[good_with_zero][:, perm_zero_ind].argmin(axis=1)]
        cluster_ind[x_norm <= norm_thresh] = -1             # The case where no atoms at all are needed
        norm_reduction = (x_norm[:, None] - x_residual_norm) / x_norm[:, None]
        score_exclude_atom = [best_score_excluding_atom(norm_reduction, atom_perm, atom_perm[cluster_ind][:, k])
                              for k in range(n_atom_select)]

        top_score = norm_reduction[np.arange(n_sample), cluster_ind]
        top_score[x_norm <= norm_thresh] = 0            # if no atoms fit, residual is same as start
        high_score = [(top_score > score_thresh
                      ) & (top_score - score_exclude_atom[k] > score_diff_thresh) for k in range(n_atom_select)]
        # to help terminate
        low_score = [top_score - score_exclude_atom[k] < score_diff_thresh_test_converge for k in range(n_atom_select)]
        low_score = np.any(low_score, axis=0)
        if valid is not None:
            # Only use valid points to compute the clusters
            high_score = [high_score[k] & valid for k in range(n_atom_select)]
            low_score = low_score | ~valid
        for c in range(n_atom-1):
            my_points = np.zeros((0, n_feature))
            for k in range(n_atom_select):
                samples_use = (cluster_ind >= 0) & (atom_perm[cluster_ind, k]==c) & high_score[k]
                if samples_use.sum() > 0:
                    # Get residual excluding atom currently considering
                    x_use_fit = coef[samples_use, cluster_ind[samples_use], :, None] * norm_cluster_mean[atom_perm[cluster_ind[samples_use]]]
                    x_use_fit = np.delete(x_use_fit, k, axis=1)     # exclude atom currently considering
                    x_use_fit = x_use_fit.sum(axis=1)               # sum over all atoms excluding current one
                    my_points = np.append(my_points, x[samples_use] - x_use_fit, axis=0)
            n_my_points = my_points.shape[0]
            # print(n_my_points)
            if n_my_points < min_cluster_size:
                norm_cluster_mean[c] = 0
                ignore_perm[np.where([c in atom_perm[k] for k in range(n_perm)])[0].squeeze()] = True      # make sure not used in future
                continue
            # print(n_my_points)
            eig_vals, eigs = np.linalg.eig(my_points.transpose() @ my_points / n_my_points)
            best_eig_ind = np.argmax(eig_vals)
            norm_cluster_mean[c] = eigs[:, best_eig_ind] * np.sign(eigs[:, best_eig_ind].mean())  # make them positive
            cluster_eig_val[c] = eig_vals[best_eig_ind]
        print(i+1, (cluster_ind[~low_score] != cluster_ind_old[~low_score]).sum())

        if (cluster_ind[~low_score] == cluster_ind_old[~low_score]).all():
            print(f'Done after {i+1} iter')
            break
    # coef_best = coef[:, np.clip(cluster_ind, 0, n_modes-1)]
    # coef_best[cluster_ind<0] = 0
    return norm_cluster_mean, cluster_eig_val, cluster_ind, top_score, coef[np.arange(x.shape[0]), cluster_ind]

hi = 5
# For final k-means, initialize with the residual PCs as well as the initial clusters
key = 'above'
clusters_init = k_output[key][0][np.linalg.norm(k_output[key][0], axis=1)>0]
for i in range(n_modes):
    cluster_dot_product = np.abs(np.round(clusters_init @ pca_output_residual[key][0][i].to_numpy(), 2))
    print(i, cluster_dot_product)
    if np.max(cluster_dot_product) < 0.7:
        clusters_init = np.vstack([clusters_init, pca_output_residual[key][0][i].to_numpy()])
x_use = flatten_to_numpy(mse_env_pnorm[key][0].isel(pnorm=slice(1, 9999)), 'pnorm')
mse_norm_thresh = 1
a = list(scaled_k_means2(x_use, clusters_init, valid=flatten_to_numpy(valid[key]),
                                            norm_thresh=mse_norm_thresh, n_atom_select=2))
a2 = list(scaled_k_means(x_use, clusters_init, valid=flatten_to_numpy(valid[key]),
                                            norm_thresh=mse_norm_thresh))
hi = 5
# key = 'above'
# for i in range(n_modes):
#     plt.plot(k_output[key][0][i], pca_output_residual[key][0].pnorm, color=f'C{i}')
#     plt.plot(pca_output[key][0].isel(mode=i), pca_output_residual[key][0].pnorm, color=f'C{i}', linestyle='--')
#     plt.plot(pca_output_residual[key][0].isel(mode=i), pca_output_residual[key][0].pnorm, color=f'C{i}',
#              linestyle=':')
