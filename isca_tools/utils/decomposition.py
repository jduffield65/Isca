import xarray as xr
import numpy as np
from typing import Optional, Tuple, Union
from .xarray import flatten_to_numpy, unflatten_from_numpy
import itertools


def pca_on_xarray(data: xr.DataArray, n_modes: int = 4, standardize: bool = True,
                  valid: Optional[xr.DataArray] = None, feature_dim_name: str = "lev",
                  reference_mean: Union[bool, xr.DataArray] = True,
                  ) -> Tuple[xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray, np.ndarray, np.ndarray]:
    """
    Perform PCA (via SVD) on xarray dataset. The PCA is fit only on samples where `valid` is True. The
    components found are then fit to all samples in `data`.

    Args:
        data : DataArray with dims (..., feature_dim_name) (e.g. (co2, lat, lon, lev) with `feature_dim_name=lev`).
        n_modes (int): Number of PCA modes to keep.
        standardize (bool): If True, divide each feature by its std (computed from valid samples)
            *before* SVD so that features with different variances are equalized.
            If False, SVD is performed on raw deviations from `reference_mean`.
        valid: Boolean mask with the same non-feature dims as `data` (e.g. (co2, lat, lon)).
            True indicates the grid cell is used to compute the PCA basis. If None, all
            grid cells with finite values across lev are considered valid.
        feature_dim_name: Name of the dimension containing features of interest in `data`.
        reference_mean: 1-D DataArray (dim `feature_dim_name`) to subtract before SVD.
            If False, a zero reference mean is used (i.e. PCA on deviations from zero).
            If True, a reference mean will be computed from all `valid` samples.

    Returns:
        components: EOFs (modes) with dims (mode, feature_dim_name).
        scores: PC coefficients with same dims as `data` but `mode` replacing `feature_dim_name`.
        mean_profile: The reference_mean actually used (dim `feature_dim_name`).
        std_profile: Std used for scaling (dim `feature_dim_name`). Ones if `standardize=False`.

    Notes:
        - This function uses np.linalg.svd directly so there is NO automatic re-centering:
          the `reference_mean` you supply (or zero) is the baseline from which deviations
          are computed.
        - If standardize=True, std_profile is computed from the valid set and used both
          for the SVD input and for projecting all profiles.
    """
    if feature_dim_name not in data.dims:
        raise ValueError(f"X must have a '{feature_dim_name}' dimension")

    non_feature_dims = [d for d in data.dims if d != feature_dim_name]
    n_feature = data.sizes[feature_dim_name]

    # prepare reference mean (1d array length n_feature)
    if reference_mean == False:
        reference_mean_vals = np.zeros(n_feature)
    elif reference_mean == True:
        if valid is None:
            reference_mean = data.mean(dim=non_feature_dims)
        else:
            reference_mean = data.where(valid).mean(dim=non_feature_dims)
        reference_mean_vals = reference_mean.values
    else:
        if feature_dim_name not in reference_mean.dims:
            raise ValueError(f"reference_mean must have dimension named {feature_dim_name}")
        # align and extract numeric array in feature order of data
        reference_mean_vals = reference_mean.reindex({feature_dim_name: data[feature_dim_name]}).values

    X_all = flatten_to_numpy(data, feature_dim_name)
    if valid is None:
        X_valid = X_all
    else:
        X_valid = X_all[flatten_to_numpy(valid)]
    n_valid = X_valid.shape[0]
    if n_valid < (n_modes + 1):
        raise ValueError("Too few valid samples for PCA; reduce n_modes or check coverage.")

    # subtract reference mean
    Xc_valid = X_valid - reference_mean_vals[None, :]

    # compute std_profile from valid subset if requested
    if standardize:
        std_profile_vals = Xc_valid.std(axis=0, ddof=1)
        # avoid zeros
        std_profile_vals = np.where(std_profile_vals == 0, 1.0, std_profile_vals)
        Xc_valid = Xc_valid / std_profile_vals[None, :]
    else:
        std_profile_vals = np.ones(n_feature)

    # --- SVD on the prepared valid data (no further centering) ---
    # Xc_valid shape: (n_valid, n_feature). compute thin SVD
    U, S, Vt = np.linalg.svd(Xc_valid, full_matrices=False)
    # components (EOFs) are rows of Vt; keep first n_modes
    components_vals = Vt[:n_modes, :]  # (n_modes, n_feature)

    # --- project ALL profiles using same transform ---
    # subtract reference mean and divide by std_profile (if standardize)
    Xc_all = X_all - reference_mean_vals[None, :]
    if standardize:
        Xc_all = Xc_all / std_profile_vals[None, :]

    # scores_all: (n_samples, n_modes)
    scores_all = Xc_all @ components_vals.T

    # reshape back to original non-feature dims + mode
    out_shape = [data.sizes[d] for d in non_feature_dims] + [n_modes]
    scores_da = xr.DataArray(
        scores_all.reshape(*out_shape),
        dims=non_feature_dims + ["mode"],
        coords={**{d: data[d] for d in non_feature_dims}, "mode": np.arange(n_modes)}
    )

    components_da = xr.DataArray(
        components_vals,
        dims=("mode", feature_dim_name),
        coords={"mode": np.arange(n_modes), feature_dim_name: data[feature_dim_name]}
    )

    mean_profile_da = xr.DataArray(reference_mean_vals, dims=(feature_dim_name,),
                                   coords={feature_dim_name: data[feature_dim_name]})
    std_profile_da = xr.DataArray(std_profile_vals, dims=(feature_dim_name,),
                                  coords={feature_dim_name: data[feature_dim_name]})

    # Variance explained by each mode
    var_explained = (S ** 2) / (Xc_valid.shape[0] - 1)

    # Fractional variance explained
    frac_var_explained = var_explained / var_explained.sum()

    return components_da, scores_da, mean_profile_da, std_profile_da, var_explained[:n_modes], frac_var_explained[
        :n_modes]


def best_score_excluding_atom(norm_reduction: np.ndarray,
                              combinations: np.ndarray,
                              atom: np.ndarray) -> np.ndarray:
    """
    For each sample, find the maximum norm_reduction value
    among all combinations that do NOT contain atom[i].

    Parameters
    ----------
    norm_reduction: (n_sample, n_comb)
        Score or reduction value for each sample–combination pair.
    combinations: (n_comb, n_atom_select)
        Atom indices used in each combination.
    atom: (n_sample,)
        Atom index to exclude for each sample.

    Returns
    -------
    best_score_excl: (n_sample,)
        Max norm_reduction for each sample excluding combinations that contain atom[i].
    """
    # (n_sample, n_comb): True if this combo contains that sample’s excluded atom
    contains_atom = np.any(combinations[None, :, :] == atom[:, None, None], axis=2)

    # Mask those out
    masked_scores = np.where(~contains_atom, norm_reduction, -np.inf)

    # Take max over combinations
    best_score_excl = masked_scores.max(axis=1)

    return best_score_excl


def scaled_k_means(x: np.ndarray, initial_cluster_mean: np.ndarray, valid: Optional[np.ndarray] = None,
                   n_atom_select: int = 1, norm_thresh: float = 1, score_thresh: float = 0.5,
                   score_diff_thresh: float = 0.1,
                   score_diff_thresh_test_converge: float = 0.05, score_thresh_multi_atom: float = 0.05,
                   min_cluster_size: int = 10,
                   n_iter: int = 100) -> np.ndarray:
    n_sample, n_feature = x.shape
    norm_cluster_mean = initial_cluster_mean / np.linalg.norm(initial_cluster_mean, axis=1).reshape(-1, 1)
    norm_cluster_mean = np.vstack([norm_cluster_mean, np.zeros(n_feature)])  # add an array of zeros
    n_atom = norm_cluster_mean.shape[0]
    cluster_eig_val = np.zeros(n_atom)
    cluster_ind = np.full(x.shape[0], -20, dtype=int)
    x_norm = np.linalg.norm(x, axis=1)
    small_norm = x_norm <= norm_thresh  # set coef=0 for all atoms for this case; don't recompute each time
    # n_atom_select = 2
    atom_perm = np.array(
        list(itertools.combinations(range(n_atom), n_atom_select)))  # all possible permutations of atoms
    atom_perm = np.sort(atom_perm, axis=1)  # ensure larger index later to ensure zeros array always last
    n_perm = len(atom_perm)
    perm_zero_ind = np.where([n_atom - 1 in atom_perm[i] for i in range(n_perm)])[
        0].squeeze()  # all permutations with the last ind which is 0
    ignore_perm = np.zeros(n_perm, dtype=bool)
    for i in range(np.clip(n_iter, 1, 1000)):
        coef = np.zeros((n_sample, n_perm, n_atom_select))
        for j in range(n_perm):
            if ignore_perm[j]:
                continue  # keep all coefs zero in this case
            if j in perm_zero_ind:  # keep zero atom coefficient as zero, compute coefficient for other atoms
                if n_atom_select > 1:
                    # Compute coefficient of other atoms
                    A = norm_cluster_mean[atom_perm[j][:-1]]
                    AAT_inv = np.linalg.inv(A @ A.T)  # (n_atom_select-1, n_atom_select-1)
                    coef[~small_norm, j, :-1] = (AAT_inv @ A @ x[~small_norm].T).T  # (n_sample, n_atom_select-1)
            else:
                A = norm_cluster_mean[atom_perm[j]]  # (n_atom_select, n_dim)
                AAT_inv = np.linalg.inv(A @ A.T)  # (n_atom_select, n_atom_select)
                coef[~small_norm, j] = (AAT_inv @ A @ x[
                    ~small_norm].T).T  # (n_sample, n_atom_select), repeat for all possible permutations of atoms
        cluster_ind_old = cluster_ind.copy()
        # coef = x @ norm_cluster_mean.transpose()   # because each initial_cluster_mean has norm of 1
        x_residual = x[:, None] - (coef[..., None] * norm_cluster_mean[atom_perm][None]).sum(
            axis=-2)  # sum over n_atom_select
        x_residual_norm = np.linalg.norm(x_residual, axis=-1)
        # norm_reduction = (x_norm[:, None] - x_residual_norm) / x_norm[:, None]
        cluster_ind = x_residual_norm.argmin(axis=1)
        norm_reduction = (x_norm[:, None] - x_residual_norm) / (x_norm[:, None] + 1e-20)
        if n_atom_select > 1:
            # If residual is already small including one of atoms selected as zero, then select as best cluster
            good_with_zero = x_residual_norm[:, perm_zero_ind].min(axis=1) <= norm_thresh
            good_with_zero = good_with_zero | (
                        norm_reduction.max(axis=1) - norm_reduction[:, perm_zero_ind].max(axis=1) < score_thresh_multi_atom)
            cluster_ind[good_with_zero] = perm_zero_ind[
                x_residual_norm[good_with_zero][:, perm_zero_ind].argmin(axis=1)]
        cluster_ind[small_norm] = -1  # The case where no atoms at all are needed

        top_score = norm_reduction[np.arange(n_sample), cluster_ind]
        top_score[x_norm <= norm_thresh] = 0  # if no atoms fit, residual is same as start

        if n_iter == 0:
            print('n_iter=0 so not updating atoms')
            break

        score_exclude_atom = [best_score_excluding_atom(norm_reduction, atom_perm, atom_perm[cluster_ind][:, k])
                              for k in range(n_atom_select)]
        high_score = [(top_score > score_thresh
                       ) & (top_score - score_exclude_atom[k] > score_diff_thresh) for k in range(n_atom_select)]
        # to help terminate
        low_score = [top_score - score_exclude_atom[k] < score_diff_thresh_test_converge for k in range(n_atom_select)]
        low_score = np.any(low_score, axis=0)
        if valid is not None:
            # Only use valid points to compute the clusters
            high_score = [high_score[k] & valid for k in range(n_atom_select)]
            low_score = low_score | ~valid
        for c in range(n_atom - 1):
            my_points = np.zeros((0, n_feature))
            for k in range(n_atom_select):
                samples_use = (cluster_ind >= 0) & (atom_perm[cluster_ind, k] == c) & high_score[k]
                if samples_use.sum() > 0:
                    # Get residual excluding atom currently considering
                    x_use_fit = coef[samples_use, cluster_ind[samples_use], :, None] * norm_cluster_mean[
                        atom_perm[cluster_ind[samples_use]]]
                    x_use_fit = np.delete(x_use_fit, k, axis=1)  # exclude atom currently considering
                    x_use_fit = x_use_fit.sum(axis=1)  # sum over all atoms excluding current one
                    my_points = np.append(my_points, x[samples_use] - x_use_fit, axis=0)
            n_my_points = my_points.shape[0]
            # print(n_my_points)
            if n_my_points < min_cluster_size:
                norm_cluster_mean[c] = 0
                ignore_perm[np.where([c in atom_perm[k] for k in range(n_perm)])[
                    0].squeeze()] = True  # make sure not used in future
                continue
            # print(n_my_points)
            eig_vals, eigs = np.linalg.eig(my_points.transpose() @ my_points / n_my_points)
            best_eig_ind = np.argmax(eig_vals)
            norm_cluster_mean[c] = eigs[:, best_eig_ind] * np.sign(eigs[:, best_eig_ind].mean())  # make them positive
            cluster_eig_val[c] = eig_vals[best_eig_ind]
        print(i + 1, (cluster_ind[~low_score] != cluster_ind_old[~low_score]).sum())

        if (cluster_ind[~low_score] == cluster_ind_old[~low_score]).all():
            print(f'Done after {i + 1} iter')
            break
    # coef_best = coef[:, np.clip(cluster_ind, 0, n_modes-1)]
    # coef_best[cluster_ind<0] = 0
    return norm_cluster_mean, cluster_eig_val, cluster_ind, top_score, coef[
        np.arange(x.shape[0]), cluster_ind], atom_perm


def scaled_k_means_single(x: np.ndarray, initial_cluster_mean: np.ndarray, valid: Optional[np.ndarray] = None,
                          norm_thresh: float = 1, score_thresh: float = 0.5, score_diff_thresh: float = 0.1,
                          score_diff_thresh_test_converge: float = 0.05, min_cluster_size: int = 10,
                          n_iter: int = 100) -> np.ndarray:
    # Only 1 atom fit to each sample
    norm_cluster_mean = initial_cluster_mean / np.linalg.norm(initial_cluster_mean, axis=1).reshape(-1, 1)
    n_modes = initial_cluster_mean.shape[0]
    cluster_eig_val = np.zeros(n_modes)
    cluster_ind = np.full(x.shape[0], -20, dtype=int)
    x_norm = np.linalg.norm(x, axis=1)
    for i in range(n_iter):
        cluster_ind_old = cluster_ind.copy()
        coef = x @ norm_cluster_mean.transpose()  # because each initial_cluster_mean has norm of 1
        x_residual = x[:, :, np.newaxis] - coef[:, np.newaxis] * norm_cluster_mean.transpose()[np.newaxis]
        norm_reduction = (x_norm[:, np.newaxis] - np.linalg.norm(x_residual, axis=1)) / x_norm[:, np.newaxis]
        cluster_ind = norm_reduction.argmax(axis=1)
        cluster_ind[x_norm <= norm_thresh] = -1  # already has low norm, so don't use in updating coefs
        top_score = norm_reduction.max(axis=1)
        top_score[x_norm <= norm_thresh] = 0  # if no atoms fit, residual is same as start
        high_score = (top_score > score_thresh
                      ) & (top_score - np.partition(norm_reduction, -2, axis=1)[:, -2] > score_diff_thresh)
        # to help terminate
        low_score = top_score - np.partition(norm_reduction, -2, axis=1)[:, -2] < score_diff_thresh_test_converge
        if valid is not None:
            # Only use valid points to compute the clusters
            high_score = high_score & valid
            low_score = low_score | ~valid
        for c in range(n_modes):
            my_points = x[(cluster_ind == c) & high_score]
            n_my_points = my_points.shape[0]
            # print(n_my_points)
            if n_my_points < min_cluster_size:
                norm_cluster_mean[c] = 0
                continue
            # print(n_my_points)
            eig_vals, eigs = np.linalg.eig(my_points.transpose() @ my_points / n_my_points)
            best_eig_ind = np.argmax(eig_vals)
            norm_cluster_mean[c] = eigs[:, best_eig_ind] * np.sign(eigs[:, best_eig_ind].mean())  # make them positive
            cluster_eig_val[c] = eig_vals[best_eig_ind]
        print(i + 1, (cluster_ind[~low_score] != cluster_ind_old[~low_score]).sum())

        if (cluster_ind[~low_score] == cluster_ind_old[~low_score]).all():
            print(f'Done after {i + 1} iter')
            break
    # coef_best = coef[:, np.clip(cluster_ind, 0, n_modes-1)]
    # coef_best[cluster_ind<0] = 0
    return norm_cluster_mean, cluster_eig_val, cluster_ind, top_score, coef[np.arange(x.shape[0]), cluster_ind]
