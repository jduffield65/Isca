# Basic functions for loading processed JASMIN data that has been processed and saved locally
# These data sets are all averaged over a given quantile
import re

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from typing import Union, Literal, Optional, List, Tuple
import cartopy.crs as ccrs
from cartopy.mpl.contour import GeoContourSet
from cartopy.util import add_cyclic_point
from matplotlib.contour import QuadContourSet

from isca_tools.cesm.load import load_z2m
from isca_tools.convection.base import lcl_sigma_bolton_simple
from isca_tools.papers.byrne_2021 import get_quant_ind
from isca_tools.thesis.mod_parcel_theory import get_scale_factor_theory, get_scale_factor_theory_numerical2
import f90nml

exp_names = ['pre_industrial', 'co2_2x']
percentile_label = 'Temperature Percentile, $x$'
sf_label = "Scaling Factor, $\delta T_s(x)/\delta \overline{T}_s$ [KK$^{-1}$]"
labels_cont = {'temp_ft_change': 'FT change', 'rh_change': 'RH change', 'sCAPE_change': 'CAPE change',
               'temp_surf_anom': 'Hot-get-hotter', 'rh_anom': 'Drier-get-hotter', 'lapse_D_change': '$\eta_D$ change',
               'lapse_M_change': '$\eta_M$ change', 'lapse_D_anom': '$\eta_D$ climatological',
               'p_surf_change': '$p_s$ change', 'p_surf_anom': 'Higher-get-hotter'}

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
# Processed data to create lapse fitting info
path_lapse_data = lambda x, q: f"{data_dir}/{x}/REFHT_quant{q}/lapse_fitting/ds_lapse_simple.nc"
out_dir = f'{jobs_dir}/theory_lapse/cesm/thesis_figs/ds_processed'      # Where files directly for thesis figs saved

# Data saved in path_lapse_data
vars_lapse_data = ['PS', 'hyam', 'hybm', 'TREFHT', 'QREFHT', 'PREFHT', 'rh_REFHT', 'T_ft_env', 'const1_lapse',
                   'const1_integral', 'const1_error', 'mod_parcel1_lapse', 'mod_parcel1_integral',
                   'mod_parcel1_error', 'lnb_ind', 'lapse_miy2022_M', 'lapse_miy2022_D']
attrs_lapse_data = ['P0', 'temp_surf_lcl_calc', 'n_lev_above_integral', 'lev_REFHT']

# `gw` parameter from dataset
lat_weights = xr.load_dataset('/Users/joshduffield/Documents/StAndrews/Isca/jobs/cesm/input_data/'
                              'e.e20.E1850TEST.f09_g17.3hour_gw.nc').gw

# Used for masking
mask_error_thresh = 0.25
mask_aloft_p_size_thresh = 100 * 100  # pressure in Pa


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
co2_labels = [f"${get_co2_multiplier(name):.0f} \\times CO_2$" for name in exp_names]


def load_ds(exp_name: Literal['pre_industrial', 'co2_2x'], quant: Literal[50, 95, 99],
            var_keep: Optional[List] = vars_lapse_data,
            load_lapse_attrs: bool = True,
            load_invariant: bool = True,
            lev_REFHT_actual: bool = False, reindex_var='PS') -> xr.Dataset:
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
    var_keep_no_attrs = [var for var in var_keep if
                         var not in attrs_lapse_data]  # so don't try and load attrs as variables
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
    var_all_data_load = [var for var in var_keep_no_attrs if var not in (var_lapse_load + var_invariant_load)]
    if len(var_all_data_load) > 0:
        vars_all_data = f90nml.read(path_all_data(exp_name, quant).replace('output.nc', 'input.nml')
                                    )['script_info']['var'] + ['time']
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


def initialize_ax_projection(ax: plt.Axes, lon_min: float = -180, lon_max: float = 180, lat_min: float = 30,
                             lat_max: float = 80,
                             grid_lon: Union[List, np.ndarray] = np.arange(-180, 180.01, 60),
                             grid_lat: Union[List, np.ndarray] = np.asarray([40, 65]),
                             coastline_color: str = 'k', gridline_color: str = 'k',
                             gridline_lw: float = 1,
                             draw_gridline_labels: bool = True,
                             auto_aspect: bool = False,
                             return_gl: bool = False) -> Union[plt.Axes, Tuple[plt.Axes, plt.Axes]]:
    """
    Function from Zhang Boos 2023 paper used to initialize axis to make nice looking spatial plots.
    Run `initialize_ax_projection(ax)` before doing any plotting.

    Args:
        ax: Axes to initialize
        lon_min: Lowest longitude to show
        lon_max: Highest longitude to show
        lat_min: Lowest latitude to show
        lat_max: Highest latitude to show
        grid_lon: Position of grid vertical lines
        grid_lat: Position of horizontal lines
        coastline_color: Coastline color
        gridline_color: Gridline color
        gridline_lw: Gridline width
        auto_aspect: Automatic aspect ratio, can help keep subplots looking more square
        return_gl: Whether to return gridline object for later editing

    Returns:
        ax: Modified axes

    """
    if auto_aspect:
        ax.set_aspect('auto')
    ax.coastlines(color=coastline_color, linewidth=gridline_lw)
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    gl = ax.gridlines(ccrs.PlateCarree(), xlocs=grid_lon, ylocs=grid_lat, linestyle=':',
                      color=gridline_color, alpha=1, draw_labels=draw_gridline_labels)
    gl.right_labels = 0
    gl.top_labels = 0
    if return_gl:
        return ax, gl
    else:
        return ax


def plot_contour_projection(ax: plt.Axes, var: xr.DataArray, levels: Optional[Union[np.ndarray, List]] = None,
                            n_levels: int = 10, mask: Optional[xr.DataArray] = None,
                            cmap: str = 'viridis') -> Union[GeoContourSet, QuadContourSet]:
    """
    Function from Zhang Boos 2023 paper to perform a contour plot in axes initialized with
    `initialize_ax_projection`.

    Args:
        ax: Axes to perform contour plot on
        var: Variable to plot, must contain `lat` and `lon` coordinates.
        levels: `float [n_levels]`
            Contours of `var` to use
        n_levels: If don't provide `levels, will use this many-contour
        mask: Can provide mask to exclude parts of `var` from the plot. Must match dimensions of `var` if given.
        cmap: Colourmap to use.

    Returns:
        The set of contour lines
    """
    data, lon = add_cyclic_point(var * (1 if mask is None else mask), coord=var.lon.values, axis=1)
    if levels is None:
        levels = np.linspace(var.min(), var.max(), n_levels)
    im = ax.contourf(lon, var.lat.values, data, transform=ccrs.PlateCarree(),
                     levels=levels, extend='both', cmap=cmap)
    return im


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


def pad_with_nans(ds: Union[xr.Dataset, xr.DataArray], n_dim_size: int,
                  dim: str = 'sample') -> Union[xr.Dataset, xr.DataArray]:
    """
    Extend dataset along dimension `dim`, padding with nans

    Args:
        ds: Dataset to be padded
        n_dim_size: Size of desired dimension `dim`, will pad to reach this length.
        dim: Dimension to pad

    Returns:
        Padded dataset
    """
    pad_size = n_dim_size - ds.sizes[dim]
    if pad_size > 0:
        ds = ds.pad(**{dim: (0, pad_size), 'mode': 'constant'})
    return ds


def get_valid_mask(ds: xr.Dataset, error_thresh: float = mask_error_thresh,
                   aloft_p_size_thresh: Optional[float] = mask_aloft_p_size_thresh) -> xr.DataArray:
    """
    Returns mask which is True for convective days - have lower modParc error than const error.
    Also, optionally have to have LCL further than `aloft_p_size_thresh` from `p_FT`.

    Args:
        ds: Dataset with variables needed
        error_thresh: Must have modParc error less than this to be valid.
        aloft_p_size_thresh: Threshold for distance between LCL and FT

    Returns:
        Mask which is True for convective days
    """
    # Take average over all days for which error satisfies convective threshold
    const1_error = np.abs(ds.const1_error.sum(dim='layer') / ds.const1_integral.sum(dim='layer'))
    mod_parcel1_error = np.abs(ds.mod_parcel1_error.sum(dim='layer') / ds.mod_parcel1_integral.sum(dim='layer'))

    mask_fit = (mod_parcel1_error < const1_error) & (mod_parcel1_error < error_thresh)
    if aloft_p_size_thresh is not None:
        p_lcl = lcl_sigma_bolton_simple(ds.rh_REFHT, ds.temp_surf_lcl_calc) * ds.PS
        mask_fit = mask_fit & (p_lcl - ds.p_ft > aloft_p_size_thresh)
    return mask_fit


def convert_ds_of_dicts(ds_of_dicts: xr.Dataset, quant_dim_vals: Union[xr.DataArray, np.ndarray],
                        quant_dim_name: str = 'quant') -> dict:
    """
    Takes in a dataset where each for every dimension other than quant_dim, the item is a dictionary of variables.
    This converts that to a dictionary of DataArrays i.e. a data array for each variable.

    E.g. If initial dims were `co2`, `lat`, `quant`, `ds_of_dicts.isel(co2=0, lat=0)` is a dictionary
    such that `ds_of_dicts.isel(co2=0, lat=0).item()[key1][q]` is the value of key1 for `quant=q, co2=0, lat=0`.

    This converts it to a dictionary of data arrays such that `ds_out[key1].isel(co2=0, lat=0, q=0)` is the value
    of key1 for `quant=q, co2=0, lat=0`.

    Args:
        ds_of_dicts: Dataset of dictionaries
        quant_dim_vals: Values of quant_dim
        quant_dim_name: Dimension name along which initial dicts in the initial dataset are provided.

    Returns:
        ds_out: Dictionary of DataArrays
    """
    dict_ds = {}
    # Get all dimension names except 'quant'
    other_dims = [d for d in ds_of_dicts.dims if
                  d != quant_dim_name]  # All non-quant dimensions in out_cont (e.g. dim_1, ..., dim_n)
    other_shape = tuple(ds_of_dicts.sizes[d] for d in other_dims)

    # Determine all dict keys from the first element of all other dims
    dict_first = ds_of_dicts.isel({d: 0 for d in other_dims}).item()
    keys = list(dict_first.keys())
    quant_dim_sz = dict_first[keys[0]].size
    if quant_dim_sz != len(quant_dim_vals):
        raise ValueError('Size mismatch in quant_dim_vals and ds_of_dicts')

    for key in keys:
        # Loop over all dimensions other than quant
        stacked = []
        for idx in np.ndindex(*other_shape):
            dict_ind_sel = {d: i for d, i in zip(other_dims, idx)}  # e.g. {'surf': 0, 'co2': 0, 'lat': 3}
            sel = ds_of_dicts.isel(dict_ind_sel)
            stacked.append(sel.item()[key])  # (quant,)

        data = np.stack(stacked, axis=0).reshape(*other_shape, quant_dim_sz)

        coords = {d: ds_of_dicts[d] for d in other_dims}
        coords[quant_dim_name] = quant_dim_vals

        da = xr.DataArray(
            data,  # (dim_1, ..., dim_n, quant)
            dims=(*other_dims, quant_dim_name),
            coords=coords,
        )
        dict_ds[key] = da
    return dict_ds


def apply_scale_factor_theory(ds_quant: xr.Dataset, ds_ref: xr.Dataset, p_ft: float, temp_surf_lcl_calc: float,
                              sCAPE_form: bool = False, co2_dim: str = 'co2', quant_dim: str = 'quant',
                              numerical: bool = False, lapse_coords: Literal['z', 'lnp'] = 'lnp') -> xr.Dataset:
    """
    Apply get_scale_factor_theory_numerical to an xarray.Dataset.
    Works in sCAPE or modParc form, depending on sCAPE_form parameter

    Args:
        ds_quant: Dataset with `quant_dim` dimension along which compute scaling factor
        ds_ref: Dataset with no `quant_dim` dimension but all other dimensions in `ds_quant`, that scaling factor
            is computed with respect to.
        p_ft: Free tropospheric pressure in Pa.
        temp_surf_lcl_calc: Temperature used to compute the LCL.
        sCAPE_form: If `True`, will include sCAPE mechanism instead of modified lapse rate mechanisms.
        co2_dim: Dimension of warming i.e. difference between climates is computed along this dimension.
        quant_dim: Dimension in scaling factor, i.e. compute anomalous temperature change along this dimension.
        numerical: Whether to compute scaling factor and contributions numerically
        lapse_coords: If `numerical=True`, have option to provide `lapse_D` and `lapse_M` in z coordinates,
            rather than log pressure coordinates. Specify here.

    Returns:
        Dataset with scaling factor, theoretical estimate and contribution from each mechanism.
    """
    input_core_dims = [
        [co2_dim],  # temp_surf_ref
        [co2_dim, quant_dim],
        [],  # rh_ref
        [co2_dim, quant_dim],
        [co2_dim, quant_dim],
        [],  # p_ft_ref
        [],  # p_surf_ref
        [co2_dim, quant_dim],
        [co2_dim, quant_dim] if not sCAPE_form else [],
        [co2_dim, quant_dim] if not sCAPE_form else [],
        [co2_dim, quant_dim] if sCAPE_form else []]

    func_kwargs = {'temp_surf_lcl_calc': temp_surf_lcl_calc}
    if numerical:
        # Numerical outputs sf_sum, sf_nl, and sf_linear, as well as dict of contributions
        output_core_dims = [[quant_dim], [quant_dim], [quant_dim], []]
        output_dtypes = [float, float, float, object]
        func_kwargs['lapse_coords'] = lapse_coords
    else:
        if lapse_coords == 'z':
            raise ValueError(f"lapse_coords='z' only possible for numerical=True")
        # Theoretical outputs sf_sum, as well as dict of gamma, variables, and contributions
        output_core_dims = [[quant_dim], [], [], []]
        output_dtypes = [float, object, object, object]

    out1, out2, out3, out_cont = xr.apply_ufunc(
        get_scale_factor_theory_numerical2 if numerical else get_scale_factor_theory,
        ds_ref["TREFHT"],  # (co2)
        ds_quant["TREFHT"],  # (co2, quant)
        ds_ref["rh_REFHT"].isel(**{co2_dim: 0}),
        ds_quant["rh_REFHT"],  # (co2, quant)
        ds_quant["T_ft_env"],  # (co2, quant)
        p_ft,  # (co2) or scalar
        ds_ref["PREFHT"].isel(**{co2_dim: 0}),  # scalar
        ds_quant["PREFHT"],  # (co2, quant) or None
        ds_quant[f"lapse_D{'z' if lapse_coords == 'z' else ''}"] if not sCAPE_form else None,  # (co2, quant)
        ds_quant[f"lapse_M{'z' if lapse_coords == 'z' else ''}"] if not sCAPE_form else None,  # (co2, quant)
        ds_quant['sCAPE'] if sCAPE_form else None,  # (co2, quant)
        kwargs=func_kwargs,
        input_core_dims=input_core_dims,
        output_core_dims=output_core_dims,
        vectorize=True,
        dask="parallelized",
        output_dtypes=output_dtypes,
    )

    # Expand the dictionary output entries into proper DataArrays
    dict_ds = {'scale_factor': ds_quant['TREFHT'].diff(dim=co2_dim).isel(co2=0, drop=True)
                               / ds_ref['TREFHT'].diff(dim=co2_dim).isel(co2=0, drop=True),
               **convert_ds_of_dicts(out_cont, ds_quant[quant_dim], quant_dim)}
    if numerical:
        ds_out = xr.Dataset({"scale_factor_sum_all_terms": out1, "scale_factor_nl": out2,
                             "scale_factor_linear": out3, **dict_ds})
    else:
        ds_out = xr.Dataset({"scale_factor_sum": out1, **dict_ds})
    # Drop coordinates with no dim
    ds_out = ds_out.drop_vars([v for v in ds_out.coords if v not in ds_out[v].dims])
    other_dims = [d for d in ds_out.dims if d != quant_dim]
    ds_out = ds_out.transpose(*other_dims, quant_dim)  # make quant the last dimension
    return ds_out
