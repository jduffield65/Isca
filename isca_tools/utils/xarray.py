from typing import Optional
import xarray as xr
import numpy as np


def print_ds_var_list(ds: xr.Dataset, phrase: Optional[str] = None) -> None:
    """
    Prints all variables in `ds` which contain `phrase` in the variable name or variable `long_name`.

    Args:
        ds: Dataset to investigate variables of.
        phrase: Key phrase to search for in variable info.

    """
    # All the exceptions to deal with case when var does not have a long_name
    var_list = list(ds.keys())
    if phrase is None:
        for var in var_list:
            try:
                print(f'{var}: {ds[var].long_name}')
            except AttributeError:
                print(f'{var}')
    else:
        for var in var_list:
            if phrase.lower() in var.lower():
                try:
                    print(f'{var}: {ds[var].long_name}')
                except AttributeError:
                    print(f'{var}')
                continue
            try:
                if phrase.lower() in ds[var].long_name.lower():
                    print(f'{var}: {ds[var].long_name}')
                    continue
            except AttributeError:
                continue
    return None


def set_attrs(var: xr.DataArray, overwrite: bool = True, **kwargs: str) -> xr.DataArray:
    """
    Set attributes of a given variable.

    Examples:
        `set_attrs(ds.plev, long_name='pressure', units='Pa')`

    Args:
        var: Variable to set attributes of.
        overwrite: If `True`, overwrite existing attributes, otherwise leave unchanged.
        **kwargs: Attributes to set. Common ones include `long_name` and `units`

    Returns:
        `var` with attributes set.
    """
    # Function to set main attributes of given variable
    for key in kwargs:
        if (key in var.attrs) and not overwrite:
            continue
        var.attrs[key] = kwargs[key]
    return var


def flatten_to_numpy(var: xr.DataArray, keep_dim: Optional[str] = None) -> np.ndarray:
    """
    Flattens `var` to a numpy array with at most 2 dimensions.

    Examples:
        If `var` has `dims=(lat, lon, lev)` and `keep_dim=lev`, it will return a numpy array of
            size `[n_lat*n_lon, n_lev]`.

        If `var` has `dims=(lat, lon)` and `keep_dim=None`, it will return a numpy array of
            size `[n_lat*n_lon]`.

    Args:
        var: Variable to flatten.
        keep_dim: Dimension along which not to flatten.

    Returns:
        var_flatten: Numpy array with flattened dimension first, and `keep_dim` dimension second.
    """
    if (keep_dim is not None) and (keep_dim not in var.dims):
        raise ValueError(f"var must have a '{keep_dim}' dimension")

    # dims except vertical
    flatten_dims = [d for d in var.dims if d != keep_dim]

    # stack all flatten_dims into a single "points" axis
    stacked = var.stack(points=flatten_dims)  # dims (..., lev_name) -> (points, lev_name) after transpose
    if keep_dim is not None:
        stacked = stacked.transpose("points", keep_dim)  # ensure order is (points, lev)
    return stacked.values


def unflatten_from_numpy(arr: np.ndarray, var: xr.DataArray, keep_dim: Optional[str] = None) -> xr.DataArray:
    """
    Reconstructs an xarray.DataArray from a flattened NumPy array created by `flatten_to_numpy`.

    Examples:
        If `var` had dims=(lat, lon, lev) and `keep_dim='lev'`, and `arr` has shape (n_lat*n_lon, n_lev),
        this will return a DataArray with dims (lat, lon, lev).

        If `var` had dims=(lat, lon)` and `keep_dim=None`, and `arr` has shape (n_lat*n_lon),
        this will return a DataArray with dims (lat, lon).

    Args:
        arr: Flattened NumPy array from `flatten_to_numpy`.
        var: The original DataArray used to determine dimension order, shape, and coordinates.
        keep_dim: Dimension that was kept unflattened in `flatten_to_numpy`.

    Returns:
        xr.DataArray: DataArray with the original dimensions and coordinates restored.
    """
    # Validate keep_dim
    if (keep_dim is not None) and (keep_dim not in var.dims):
        raise ValueError(f"var must have a '{keep_dim}' dimension")

    # Identify flattened dims
    flatten_dims = [d for d in var.dims if d != keep_dim]

    # Compute target shape
    target_shape = [var.sizes[d] for d in flatten_dims]
    if keep_dim is not None:
        target_shape.append(var.sizes[keep_dim])

    # Reshape numpy array
    reshaped = arr.reshape(target_shape)

    # Reconstruct DataArray with original dimension order
    if keep_dim is not None:
        dims = flatten_dims + [keep_dim]
    else:
        dims = flatten_dims

    # Unstack the flattened dims
    da_flat = xr.DataArray(reshaped, dims=dims, coords={d: var.coords[d] for d in dims if d in var.coords},
                           attrs=var.attrs)

    # Reverse the stacking by unstacking the combined "points" dimension
    if keep_dim is not None:
        da_flat = da_flat.transpose(*var.dims)
    else:
        da_flat = da_flat.transpose(*var.dims)

    return da_flat
