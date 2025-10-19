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
        if list(valid.dims) != non_feature_dims:
            raise ValueError(f"Valid has dims {list(valid.dims)}\nShould have dims {non_feature_dims}\nOrder important too.")
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

    Args:
        norm_reduction: (n_sample, n_comb)
            Score or reduction value for each sample–combination pair.
        combinations: (n_comb, n_atom_select)
            Atom indices used in each combination.
        atom: (n_sample,)
            Atom index to exclude for each sample.

    Returns:
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

import numpy as np
import itertools
from typing import Optional, Tuple

def scaled_k_means(
    x: np.ndarray,
    initial_cluster_mean: np.ndarray,
    valid: Optional[np.ndarray] = None,
    n_atom_select: int = 1,
    norm_thresh: float = 0,
    score_thresh: float = 0.5,
    score_diff_thresh: float = 0.1,
    score_diff_thresh_test_converge: float = 0.05,
    score_thresh_multi_atom: float = 0.05,
    min_cluster_size: int = 10,
    n_iter: int = 100,
    remove_perm: Optional[np.ndarray] = None,
    atom_ind_no_update: Optional[np.ndarray] = None,
    use_norm: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform scaled k-means clustering with optional multi-atom combinations.

    This algorithm generalizes k-means by allowing each data point to be represented
    as a *scaled combination* of a small subset of cluster "atoms" (mean vectors),
    optionally including a zero vector (to allow sparse fits). At each iteration,
    coefficients for all possible atom combinations are computed to minimize
    residual norm, and cluster means are updated as the dominant direction
    of assigned samples’ residuals.

    Args:
        x:
            Input data of shape (n_sample, n_feature).
        initial_cluster_mean:
            Initial cluster centroids of shape (n_cluster, n_feature).
        valid:
            Boolean mask (n_sample,) specifying valid samples for updates.
        n_atom_select:
            Number of atoms combined to represent each sample. Defaults to 1.
        norm_thresh:
            Threshold for treating samples as small-norm (ignored in fitting). Defaults to 0.
        score_thresh:
            Minimum improvement (norm reduction) for a sample to influence cluster update. Defaults to 0.5.
        score_diff_thresh:
            Minimum difference in score between best and next-best atom to be considered distinct. Defaults to 0.1.
        score_diff_thresh_test_converge:
            Tolerance for convergence test (difference between old and new best scores). Defaults to 0.05.
        score_thresh_multi_atom:
            Threshold for assigning multi-atom fits when residual difference is small. Defaults to 0.05.
        min_cluster_size:
            Minimum number of samples required to update a cluster. Defaults to 10.
        n_iter:
            Maximum number of iterations. Defaults to 100.
        remove_perm:
            List of atom combinations (indices) to exclude. Defaults to None.
        atom_ind_no_update:
            Atom indices that should not be updated. Defaults to None.
        use_norm:
            Whether to normalize each residual before updating atoms. Defaults to False.

    Returns:
        norm_cluster_mean: Updated normalized cluster mean vectors (atoms).
        cluster_eig_val: Leading eigenvalues for each cluster.
        cluster_ind: Cluster/combination index assigned to each sample.
        top_score: Norm reduction score of the assigned combination for each sample.
        coef_best: Coefficients of best-fitting atom combination per sample.
        atom_perm: Array of atom index combinations considered.

    Notes:
        - The algorithm can handle multi-atom fits by enumerating all valid atom combinations.
        - A zero vector is appended as an additional atom to allow sparse representations.
        - Clusters with fewer than `min_cluster_size` assigned samples are deactivated.
    """

    n_sample, n_feature = x.shape

    # Normalize initial cluster means (atoms)
    norm_cluster_mean = initial_cluster_mean / np.linalg.norm(initial_cluster_mean, axis=1).reshape(-1, 1)

    # Append a zero vector atom to allow sparse/no-fit representations
    norm_cluster_mean = np.vstack([norm_cluster_mean, np.zeros(n_feature)])
    n_atom = norm_cluster_mean.shape[0]

    # Initialize containers
    cluster_eig_val = np.zeros(n_atom)
    cluster_ind = np.full(x.shape[0], -20, dtype=int)
    x_norm = np.linalg.norm(x, axis=1)

    # Identify samples with very small norms — skip coefficient computation for them
    small_norm = x_norm <= norm_thresh

    # Generate all possible atom index combinations (permutations of n_atom_select atoms)
    atom_perm = np.array(list(itertools.combinations(range(n_atom), n_atom_select)))
    atom_perm = np.sort(atom_perm, axis=1)  # Sort to ensure zero atom appears last consistently

    # Optionally remove forbidden combinations
    if remove_perm is not None:
        remove_perm = np.sort(remove_perm, axis=1)
        mask = ~np.isin(
            atom_perm.view([('', atom_perm.dtype)] * atom_perm.shape[1]),
            remove_perm.view([('', atom_perm.dtype)] * remove_perm.shape[1])
        ).squeeze()
        if (~mask).sum() > 0:
            print(f"Removing the following atom permutations:\n{atom_perm[~mask]}")
        atom_perm = atom_perm[mask]

    if atom_ind_no_update is None:
        atom_ind_no_update = np.zeros(0, dtype=int)

    n_perm = len(atom_perm)

    # Identify all permutations that include the zero atom
    perm_zero_ind = np.where([n_atom - 1 in atom_perm[i] for i in range(n_perm)])[0].squeeze()

    # Track permutations to ignore (e.g., if corresponding atoms become inactive)
    ignore_perm = np.zeros(n_perm, dtype=bool)

    for i in range(np.clip(n_iter, 1, 1000)):
        coef = np.zeros((n_sample, n_perm, n_atom_select))  # coefficients for each permutation

        # --- Step 1: Compute coefficients for all permutations ---
        for j in range(n_perm):
            if ignore_perm[j]:
                continue

            if j in perm_zero_ind:
                if n_atom_select > 1:
                    # Compute coefficients for non-zero atoms in combinations including zero
                    A = norm_cluster_mean[atom_perm[j][:-1]]
                    AAT_inv = np.linalg.inv(A @ A.T)
                    coef[~small_norm, j, :-1] = (AAT_inv @ A @ x[~small_norm].T).T
            else:
                # Compute coefficients for full atom combinations
                A = norm_cluster_mean[atom_perm[j]]
                AAT_inv = np.linalg.inv(A @ A.T)
                coef[~small_norm, j] = (AAT_inv @ A @ x[~small_norm].T).T

        cluster_ind_old = cluster_ind.copy()

        # --- Step 2: Compute residuals and assign each sample to the best combination ---
        x_residual = x[:, None] - (coef[..., None] * norm_cluster_mean[atom_perm][None]).sum(axis=-2)
        x_residual_norm = np.linalg.norm(x_residual, axis=-1)

        # Compute fractional norm reduction
        norm_reduction = (x_norm[:, None] - x_residual_norm) / (x_norm[:, None] + 1e-20)

        # Choose combination with smallest residual
        cluster_ind = x_residual_norm.argmin(axis=1)

        # If multi-atom case, prefer those with near-zero residuals that include zero atom
        if n_atom_select > 1:
            good_with_zero = x_residual_norm[:, perm_zero_ind].min(axis=1) <= norm_thresh
            good_with_zero |= (
                norm_reduction.max(axis=1) - norm_reduction[:, perm_zero_ind].max(axis=1) < score_thresh_multi_atom
            )
            cluster_ind[good_with_zero] = perm_zero_ind[
                x_residual_norm[good_with_zero][:, perm_zero_ind].argmin(axis=1)
            ]

        # Assign -1 for samples below norm threshold
        cluster_ind[small_norm] = -1

        # Top score per sample (how much norm was reduced)
        top_score = norm_reduction[np.arange(n_sample), cluster_ind]
        top_score[x_norm <= norm_thresh] = 0

        if n_iter == 0:
            print('n_iter=0 so not updating atoms')
            break

        # --- Step 3: Identify strong assignments to guide cluster updates ---
        score_exclude_atom = [
            best_score_excluding_atom(norm_reduction, atom_perm, atom_perm[cluster_ind][:, k])
            for k in range(n_atom_select)
        ]
        high_score = [
            (top_score > score_thresh) & (top_score - score_exclude_atom[k] > score_diff_thresh)
            for k in range(n_atom_select)
        ]

        # Convergence test: low score difference means cluster assignment has stabilized
        low_score = [
            top_score - score_exclude_atom[k] < score_diff_thresh_test_converge
            for k in range(n_atom_select)
        ]
        low_score = np.any(low_score, axis=0)

        if valid is not None:
            # Restrict updates to valid samples
            high_score = [high_score[k] & valid for k in range(n_atom_select)]
            low_score = low_score | ~valid

        # --- Step 4: Update cluster means (atoms) ---
        for c in range(n_atom - 1):  # skip zero atom
            if c in atom_ind_no_update:
                continue

            my_points = np.zeros((0, n_feature))

            # Collect residuals corresponding to this atom
            for k in range(n_atom_select):
                samples_use = (cluster_ind >= 0) & (atom_perm[cluster_ind, k] == c) & high_score[k]
                if samples_use.sum() > 0:
                    x_use_fit = coef[samples_use, cluster_ind[samples_use], :, None] * norm_cluster_mean[
                        atom_perm[cluster_ind[samples_use]]
                    ]
                    x_use_fit = np.delete(x_use_fit, k, axis=1)
                    x_use_fit = x_use_fit.sum(axis=1)
                    my_points = np.append(my_points, x[samples_use] - x_use_fit, axis=0)

            n_my_points = my_points.shape[0]

            # Deactivate cluster if too few points assigned
            if n_my_points < min_cluster_size:
                norm_cluster_mean[c] = 0
                ignore_perm[np.where([c in atom_perm[k] for k in range(n_perm)])[0].squeeze()] = True
                continue

            if use_norm:
                # Normalize residuals to equalize influence
                my_points = my_points / (np.linalg.norm(my_points, axis=1)[:, None] + 1e-20)

            # Update atom as the leading eigenvector of covariance matrix
            eig_vals, eigs = np.linalg.eig(my_points.T @ my_points / n_my_points)
            best_eig_ind = np.argmax(eig_vals)
            norm_cluster_mean[c] = eigs[:, best_eig_ind] * np.sign(eigs[:, best_eig_ind].mean())
            cluster_eig_val[c] = eig_vals[best_eig_ind]

        # Print number of reassignments to monitor convergence
        print(i + 1, (cluster_ind[~low_score] != cluster_ind_old[~low_score]).sum())

        # Stop if cluster assignments have stabilized
        if (cluster_ind[~low_score] == cluster_ind_old[~low_score]).all():
            print(f"Done after {i + 1} iter")
            break

    # Return best-fit coefficients for each sample
    coef_best = coef[np.arange(x.shape[0]), cluster_ind]

    return norm_cluster_mean, cluster_eig_val, cluster_ind, top_score, coef_best, atom_perm

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
