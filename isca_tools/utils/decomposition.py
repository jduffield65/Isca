import xarray as xr
import numpy as np
from typing import Optional, Tuple, Union
from .xarray import flatten_to_numpy, unflatten_from_numpy

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

def scaled_k_means(x: np.ndarray, initial_cluster_mean: np.ndarray, valid: Optional[np.ndarray] = None,
                   norm_thresh: float = 1, score_thresh: float=0.5, score_diff_thresh: float=0.1,
                   score_diff_thresh_test_converge: float = 0.05, min_cluster_size: int = 10,
                   n_iter: int = 100) -> np.ndarray:
    norm_cluster_mean = initial_cluster_mean / np.linalg.norm(initial_cluster_mean, axis=1).reshape(-1, 1)
    n_modes = initial_cluster_mean.shape[0]
    cluster_eig_val = np.zeros(n_modes)
    cluster_ind = np.full(x.shape[0], -20, dtype=int)
    x_norm = np.linalg.norm(x, axis=1)
    for i in range(n_iter):
        # A = norm_cluster_mean[:2] # (2, n_dim)
        # ATA_inv = np.linalg.inv(A @ A.T)  # (2, 2)
        # coef = (ATA_inv @ A @ x[:5].T).T # (n_sample, 2), repeat for all possible permutations of atoms
        cluster_ind_old = cluster_ind.copy()
        coef = x @ norm_cluster_mean.transpose()   # because each initial_cluster_mean has norm of 1
        x_residual = x[:, :, np.newaxis] - coef[:, np.newaxis] * norm_cluster_mean.transpose()[np.newaxis]
        norm_reduction = (x_norm[:, np.newaxis] - np.linalg.norm(x_residual, axis=1)) / x_norm[:, np.newaxis]
        cluster_ind = norm_reduction.argmax(axis=1)
        cluster_ind[x_norm <= norm_thresh] = -1             # already has low norm, so don't use in updating coefs
        top_score = norm_reduction.max(axis=1)
        top_score[x_norm <= norm_thresh] = 0                # if no atoms fit, residual is same as start
        high_score = (top_score > score_thresh
                      ) & (top_score - np.partition(norm_reduction, -2, axis=1)[:, -2] > score_diff_thresh)
        # to help terminate
        low_score = top_score - np.partition(norm_reduction, -2, axis=1)[:, -2] < score_diff_thresh_test_converge
        if valid is not None:
            # Only use valid points to compute the clusters
            high_score = high_score & valid
            low_score = low_score | ~valid
        for c in range(n_modes):
            my_points = x[(cluster_ind==c) & high_score]
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
        print(i+1, (cluster_ind[~low_score] != cluster_ind_old[~low_score]).sum())

        if (cluster_ind[~low_score] == cluster_ind_old[~low_score]).all():
            print(f'Done after {i+1} iter')
            break
    # coef_best = coef[:, np.clip(cluster_ind, 0, n_modes-1)]
    # coef_best[cluster_ind<0] = 0
    return norm_cluster_mean, cluster_eig_val, cluster_ind, top_score, coef[np.arange(x.shape[0]), cluster_ind]