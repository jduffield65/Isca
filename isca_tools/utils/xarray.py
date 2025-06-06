from typing import Optional
import xarray as xr


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
