from typing import Optional, Callable, Any, Sequence, Literal, Union
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

def convert_ds_dtypes(ds: xr.Dataset, verbose: bool = False) -> xr.Dataset:
    """
    Convert all float variables to float32 and all int variables to int32 in an xarray Dataset.

    Args:
        ds: Input xarray Dataset.
        verbose: Whether to print out variables converted

    Returns:
        ds_out: Dataset with all float variables converted to float32 and all int variables to int32.
    """
    converted = {}
    float_conv = []
    int_conv = []
    for var_name, da in ds.data_vars.items():
        if np.issubdtype(da.dtype, np.floating) and da.dtype != np.float32:
            converted[var_name] = da.astype(np.float32)
            float_conv.append(var_name)
        elif np.issubdtype(da.dtype, np.integer) and da.dtype != np.int32:
            converted[var_name] = da.astype(np.int32)
            int_conv.append(var_name)
        else:
            converted[var_name] = da
    if verbose:
        if len(float_conv) > 0:
            print(f"Converted the following float variables:\n{float_conv}")
        if len(int_conv) > 0:
            print(f"Converted the following integer variables:\n{int_conv}")
    return ds.assign(**converted)

def wrap_with_apply_ufunc(
    func: Callable,
    input_core_dims: Optional[Sequence[Sequence[str]]] = None,
    output_core_dims: Optional[Sequence[Sequence[str]]] = None,
    dask: Literal["forbidden", "allowed", "parallelized"] = "parallelized",
    vectorize: bool = True,
    **ufunc_kwargs: Any,
) -> Callable:
    """Wrap a function for use with xarray.apply_ufunc.
    Usage:
    `func_xr = wrap_with_apply_ufunc(func_numpy, input_core_dims=[[], []])` # only 2 args, rest are kwarg
    `var_xr = func_xr(arg1, arg2, kwarg1=5, kwarg2=3)` # where kwarg are key word arguments of the function

    Args:
        func: The function to wrap.
        input_core_dims: Core dimensions for input parameters by the function. Defaults
            to ``[[]]`` for a single scalar-like output.
        output_core_dims: Core dimensions produced by the function. Defaults
            to ``[[]]`` for a single scalar-like output.
        dask: Dask handling mode.
        vectorize: Whether to auto-vectorize over non-core dims.
        **ufunc_kwargs: Extra apply_ufunc keyword arguments.

    Returns:
        A callable that behaves like ``func`` but operates on xarray objects.
    """

    def wrapped(*args: Any, **kwargs: Any):
        # Default: one empty core-dim list for each input argument
        if input_core_dims is None:
            input_core = [[] for _ in args]
        else:
            input_core = input_core_dims

        # Default: one empty core-dim list for the function output
        output_core = output_core_dims if output_core_dims is not None else [[]]

        return xr.apply_ufunc(
            func,
            *args,
            input_core_dims=input_core,
            output_core_dims=output_core,
            dask=dask,
            vectorize=vectorize,
            kwargs=kwargs,
            **ufunc_kwargs,
        )

    return wrapped


def isel_float(var: xr.DataArray, ind_float: xr.DataArray, dim='lev') -> xr.DataArray:
    """
    Performs `da.isel(dim=ind)` but for a float index where interpolation is required.

    Args:
        var: Variable to find value of at index `ind_float` along dimension `dim`.
        ind_float: Fractional index to find value of `var` along dimension `dim`.
        dim: Dimension in which to perform `isel`.

    Returns:
        Value of `var` at index `ind_float` along dimension `dim`.
    """
    # Returns lzb pressure in Pa from the kzlbs model index - requires interpolation to non integer index
    # map fractional index k -> physical lev coordinate
    ind_integer = xr.DataArray(
        np.arange(var[dim].size),
        dims=(dim,),
        coords={dim: var[dim]},
    )

    lev_k = xr.apply_ufunc(
        np.interp,
        ind_float,  # x: fractional index
        ind_integer,  # xp: integer indices 0..n_lev-1
        var[dim],  # fp: lev values
        input_core_dims=[[], [dim], [dim]],
        output_core_dims=[[]],
        vectorize=True,
    )

    # now do the "fractional isel" via interp along lev
    return var.interp(lev=lev_k)


def update_dim_slice(obj: Union[xr.DataArray, xr.Dataset], dim: str, dim_val,
                     var: Union[xr.DataArray, np.ndarray, float], var_name: Optional[str] = None):
    """Update values at one coordinate along `dim` for a Dataset or DataArray.

    If `obj` is a Dataset, updates `obj[var_name]` at `dim=dim_val`.
    If `obj` is a DataArray, updates `obj` itself at `dim=dim_val` and ignores
    `var_name` (unless you want to rename the returned DataArray).

    Args:
        obj: xarray Dataset or DataArray to update.
        dim: Dimension name to index (e.g. "fit_method").
        dim_val: Coordinate value along `dim` to update (e.g. "simulated").
        var: New values (DataArray/ndarray/scalar) broadcastable to the target slice.
        var_name: Variable name to update if `obj` is a Dataset. If None, tries
            to infer from `var.name` when `var` is a DataArray.

    Returns:
        Updated object (same type as input). Note: this mutates data in place.

    Raises:
        ValueError: If `obj` is a Dataset and `var_name` cannot be inferred.
        TypeError: If `obj` is neither Dataset nor DataArray.
    """
    if isinstance(obj, xr.Dataset):
        if var_name is None:
            if isinstance(var, xr.DataArray) and (var.name is not None):
                var_name = var.name
            else:
                raise ValueError("Provide var_name (or pass a DataArray with var.name set).")

        obj[var_name].loc[{dim: dim_val}] = var
        return obj

    if isinstance(obj, xr.DataArray):
        try:
            obj.loc[{dim: dim_val}] = var
        except ValueError as e:
            print(f"Encountered ValueError: {e}\nSolution is probably ds=ds.copy(deep=True) so can update ds.")
        return obj

    raise TypeError(f"Expected xarray Dataset or DataArray, got {type(obj)}")


def raise_if_common_dims_not_identical(x, y, name_x="x", name_y="y"):
    """Raise an error if the shared dimensions of two xarray objects differ in order.

    This compares only the dimensions that appear in both `x` and `y`. It
    extracts the subsequence of `x.dims` containing only shared dims and does
    the same for `y.dims`, then checks the two subsequences are identical (same
    dim names in the same order).

    Args:
        x: An xarray object (DataArray or Dataset) with a `.dims` attribute.
        y: An xarray object (DataArray or Dataset) with a `.dims` attribute.
        name_x: Name to use for `x` in the error message. Defaults to "x".
        name_y: Name to use for `y` in the error message. Defaults to "y".

    Raises:
        ValueError: If the order (or names) of the dimensions shared by `x` and
            `y` are not identical.

    Example:
        If `x.dims == ('time', 'lat', 'lon')` and `y.dims == ('lat', 'time')`,
        the shared dims are `('time', 'lat')` for `x` and `('lat', 'time')` for
        `y`, so this function raises.
    """
    common = [d for d in x.dims if d in y.dims]  # preserves x order
    x_common = tuple(d for d in x.dims if d in common)
    y_common = tuple(d for d in y.dims if d in common)

    if x_common != y_common:
        raise ValueError(
            f"{name_x} and {name_y} common-dim order differs.\n"
            f"{name_x} common dims = {x_common}\n"
            f"{name_y} common dims = {y_common}\n"
            f"{name_x}.dims = {x.dims}\n"
            f"{name_y}.dims = {y.dims}\n"
            f"Try using `transpose_common_dims_like` function to fix"
        )


def transpose_common_dims_like(y: xr.DataArray, x: xr.DataArray) -> xr.DataArray:
    """Reorder y so any dims shared with x follow x's dim order.

    Dims that are not in `x.dims` are left in their original relative order and
    appended after the reordered common dims.

    Args:
        y: Target xarray object (DataArray or Dataset) to reorder.
        x: Reference xarray object (DataArray or Dataset). Only `x.dims` is used.

    Returns:
        `y` transposed so that its common dims with `x` match the order in `x`.

    Raises:
        ValueError: If `y` is missing any dim required to complete the transpose
            order (should only happen if `y.dims` changes unexpectedly).
    """
    common_order = [d for d in x.dims if d in y.dims]
    remaining = [d for d in y.dims if d not in common_order]
    order = tuple(common_order + remaining)
    return y.transpose(*order)
