# Basic functions for loading processed JASMIN data that has been processed and saved locally
# These data sets are all averaged over a given quantile
import re
import numpy as np
import xarray as xr
from typing import Union, Literal, Optional, List

from isca_tools.cesm.load import load_z2m
from isca_tools.papers.byrne_2021 import get_quant_ind
import f90nml

exp_names = ['pre_industrial', 'co2_2x']

# Where topography and land frac data stored - copied from JASMIN to local
invariant_data_path = ('/Users/joshduffield/Documents/StAndrews/Isca/jobs/cesm/input_data/'
                       'fv_0.9x1.25_nc3000_Nsw042_Nrs008_Co060_Fi001_ZR_sgh30_24km_GRNL_c170103.nc')
vars_invariant = ['LANDFRAC', 'ZREFHT']

# Where 3 hourly dataset outputs saved locally, after running `data_dir/save_quant_ind.py`
# on JASMIN to create `time_ind.nc` file,
# followed by `data_dir/save_info.py` on JASMIN to create `output.nc` file
# Followed by `jobs/theory_lapse/scripts/lapse_fitting_simple.py` locally to create `lapse_fitting/ds_lapse.nc` file
jobs_dir = '/Users/joshduffield/Documents/StAndrews/Isca/jobs'
data_dir = f'{jobs_dir}/cesm/3_hour/hottest_quant'
path_all_data = lambda x, q: f"{data_dir}/{x}/REFHT_quant{q}/output.nc"  # Contains all data transferred from JASMIN
path_lapse_data = lambda \
    x, q: f"{data_dir}/{x}/REFHT_quant{q}/lapse_fitting/ds_lapse_simple.nc"  # Processed data to create lapse fitting info
out_dir = f'{jobs_dir}/theory_lapse/cesm/thesis_figs/ds_processed'

# Data saved in path_lapse_data
vars_lapse_data = ['PS', 'hyam', 'hybm', 'TREFHT', 'QREFHT', 'PREFHT', 'rh_REFHT', 'T_ft_env', 'const1_lapse',
                   'const1_integral', 'const1_error', 'mod_parcel1_lapse', 'mod_parcel1_integral',
                   'mod_parcel1_error', 'lnb_ind']
attrs_lapse_data = ['P0', 'temp_surf_lcl_calc', 'n_lev_above_integral', 'lev_REFHT']


def get_co2_multiplier(name: Literal['pre_industrial', 'co2_2x']) -> float:
    """
    Returns the co2 multiplier for the given experiment name, either 1 or 2.
    Args:
        name: Name of the experiment.

    Returns:
        Either 1 for `pre_industrial` or 2 for `co2_2x`.
    """
    match = re.match(r'co2_([\d_]+)x', name)
    if match:
        # Replace underscore with decimal point and convert to float
        return float(match.group(1).replace('_', '.'))
    elif name == 'pre_industrial':
        return 1  # for pre_industrial or other defaults
    else:
        raise ValueError(f'Not valid name = {name}')


def load_ds(exp_name: Literal['pre_industrial', 'co2_2x'], quant: Literal[50, 95, 99],
            var_keep: Optional[List]=vars_lapse_data,
            load_lapse_attrs: bool=True,
            load_invariant: bool=True,
            lev_REFHT_actual: bool = False, reindex_var='PS')->xr.Dataset:
    """
    Loading in processed JASMIN data for a particular experiment and quantile from `data_dir`

    Args:
        exp_name: CO2 concentration corresponding to dataset  in `data_dir` to load.
        quant: Quantile corresponding to dataset  in `data_dir` to load.
        var_keep: Variables to load in
        load_lapse_attrs: If `True`, will copy the `attrs` listed in `attrs_lapse_data` into output dataset
        load_invariant: If `True`, will add `LANDFRAC`. Will also add `ZREFHT` if `lev_REFHT_actual=True`.
        lev_REFHT_actual: How to define `TREFHT`, `QREFHT`, and `PREFHT`. If `False` will load in values
            from `path_lapse_data`. Otherwise will load in actual CESM i.e. 2m REFHT values from path_all_data
        reindex_var: Invariant data will be put on same lat-lon grid as this variable.

    Returns:
        Dataset with desired variables
    """
    var_keep_no_attrs = [var for var in var_keep if var not in attrs_lapse_data]    # so don't try and load attrs as variables
    var_lapse_load = [var for var in var_keep_no_attrs if var in vars_lapse_data]
    if lev_REFHT_actual:
        # Remove REFHT variables
        var_lapse_load = [var_lapse_load for var in var_lapse_load if 'REFHT' not in var]
    if len(var_lapse_load) > 0:
        ds = xr.open_dataset(path_lapse_data(exp_name, quant))[var_lapse_load]
        if not load_lapse_attrs:
            for var in attrs_lapse_data:
                del ds.attrs[var]
    var_invariant_load = [var for var in var_keep_no_attrs if var in vars_invariant]
    var_all_data_load = [var for var in var_keep_no_attrs if var not in (var_lapse_load+var_invariant_load)]
    if len(var_all_data_load) > 0:
        vars_all_data = f90nml.read(path_all_data(exp_name, quant).replace('output.nc', 'input.nml')
                                    )['script_info']['var']
        var_missing = []
        for var in var_all_data_load:
            if var not in vars_all_data:
                var_missing.append(var)
        if len(var_missing) > 0:
            raise ValueError(f'The following variables are missing from\n{path_all_data(exp_name, quant)}:\n'
                             f'{", ".join(var_missing)}')
        ds_full = xr.open_dataset(path_all_data(exp_name, quant))[var_all_data_load]
        if len(var_lapse_load) > 0:
            ds = xr.merge([ds, ds_full])
        else:
            ds = ds_full
    if load_invariant:
        ds['LANDFRAC'] = xr.open_dataset(invariant_data_path
                                         ).LANDFRAC.reindex_like(ds[reindex_var], method="nearest", tolerance=0.01)
        if lev_REFHT_actual:
            ds['ZREFHT'] = load_z2m(invariant_data_path, ds[reindex_var])
    return ds


def get_ds_quant_single_coord(ds: xr.Dataset, quant: int, range_below: float = 0.5, range_above: float = 0.5,
                              av_dim: Union[list, str, int, None] = 'sample',
                              av_dim_out: str = 'sample', var='TREFHT') -> xr.Dataset:
    """
    Wrapper function for `get_quant_ind`, so given a dataset, returns that entire dataset but only with values
    whereby the value of var is between the quant-range_below and quant+range_above

    Args:
        ds: Dataset to be processed
        quant: The quantile around which, you want to find the indices.
        range_below: All indices will var in the quantile range between quant-range_below and quant+range_above will be returned
        range_above: All indices will var in the quantile range between quant-range_below and quant+range_above will be returned
        av_dim: Dimension to find quantile over
        av_dim_out: Only required if av_dim is a list, in which case this is the name of the dimension
            in the output dataset that has been averaged over.
        var: Variable to find quantiles for, usually near-surface temperature.

    Returns:
        Dataset conditioned on quantile
    """
    quant_mask = get_quant_ind(ds[var].squeeze(), quant, range_below, range_above, av_dim=av_dim,
                               return_mask=True)
    ds_use = ds.where(quant_mask)
    if isinstance(av_dim, list):
        ds_use = ds_use.stack(**{av_dim_out: av_dim, 'create_index': False}).chunk({av_dim_out: -1})
    ds_use = ds_use.load()
    ds_use = ds_use.where(~np.isnan(ds_use[var]), drop=True)  # not sure this is necessary, but had it before
    return ds_use
