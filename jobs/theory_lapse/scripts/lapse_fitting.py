import os
import numpy as np
import xarray as xr
import logging
import sys
from isca_tools.utils.base import print_log
from isca_tools.utils.xarray import print_ds_var_list, convert_ds_dtypes
from isca_tools.convection.base import lcl_metpy
from isca_tools.thesis.lapse_theory import get_var_at_plev
from geocat.comp.interpolation import interp_hybrid_to_pressure
from isca_tools.thesis.lapse_integral import fitting_2_layer_xr
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S',
                    stream=sys.stdout)

# File location Info
data_dir = '/Users/joshduffield/Documents/StAndrews/Isca/jobs/cesm/3_hour/hottest_quant'
quant_type = 'REFHT_quant99'
exp_name = ['pre_industrial', 'co2_2x']
processed_dir = [os.path.join(data_dir, exp_name[i], quant_type, 'lapse_fitting') for i in range(len(exp_name))]
processed_file_name = 'ds_lapse.nc'  # combined file from all samples


if __name__ == '__main__':
    logger = logging.getLogger()  # for printing to console time info


    from jobs.theory_lapse.scripts.lcl import ds, small_ds
    ds = ds.drop_vars('P_diff')

    # Compute LCL
    ds['p_lcl'], ds['T_lcl_parcel'] = lcl_metpy(ds.TREFHT, ds.QREFHT, ds.PREFHT)
    ds['T_lcl_env'] = get_var_at_plev(ds.T, ds.P, ds.p_lcl)
    print_log('LCL Computed', logger)

    # Interpolate data onto FT level
    p_ft = 400 * 100
    ds['T_ft_env'] = interp_hybrid_to_pressure(ds.T, ds.PS, ds.hyam, ds.hybm, ds.P0, np.atleast_1d(p_ft),
                                               lev_dim='lev')
    ds['T_ft_env'].load()
    ds['Z_ft_env'] = interp_hybrid_to_pressure(ds.Z3, ds.PS, ds.hyam, ds.hybm, ds.P0, np.atleast_1d(p_ft),
                                               lev_dim='lev')
    ds['Z_ft_env'].load()
    ds = ds.isel(plev=0)                    # remove plev as a dimension
    ds = ds.rename_vars({"plev": "p_ft"})   # change to p_ft
    print_log('FT Temp and Z Computed', logger)

    load_processed = [os.path.exists(os.path.join(processed_dir[i], processed_file_name)) for i in range(len(exp_name))]

    # Compute empirical estimate of LCL
    n_files = ds.co2.size * ds.sample.size      # One file for each co2 conc and sample due to speed
    print_log(f'Empirical lapse fitting for {n_files} Files | Start', logger)
    small = 1               # units of Pa just to ensure use pressure below LNB
    comp_level = 4
    n_lcl_mod = 7            # number of pressure values to use in the fine grid
    ds.attrs['p_lcl_log_mod'] = np.linspace(-0.06, 0.06, n_lcl_mod)
    # ds.attrs['p_lcl_log_mod'] = np.asarray([-10, -5, -1, 0, 1, 5, 10])   # check if work with rediculous lcl
    ds.attrs['n_lev_above_integral'] = 3
    ind_phys_lcl = int(np.where(ds.p_lcl_log_mod==0)[0][0])

    var_names = ['lapse', 'integral', 'error']
    for i in range(ds.co2.size):
        if load_processed[i]:
            print_log(f'Files already exist for {exp_name[i]}', logger)
            continue
        path_use = [os.path.join(processed_dir[i], f'sample{int(ds.sample[j])}.nc') for j in range(ds.sample.size)]
        for j in range(ds.sample.size):
            if os.path.exists(path_use[j]):
                print_log(f'File {i * ds.sample.size + j + 1}/{n_files} Exists Already', logger)
                continue
            print_log(f'File {i * ds.sample.size + j + 1}/{n_files} | Start', logger)
            ds_use = ds.isel(co2=i, sample=j)
            p_lcl_use = [(10 ** (np.log10(ds_use.p_lcl) + ds.p_lcl_log_mod[i])) for i in range(n_lcl_mod)]
            ds_use['p_lcl2'] = xr.concat(p_lcl_use, dim=xr.DataArray(ds.p_lcl_log_mod, name='lcl_mod', dims='lcl_mod'))
            ds_use['T_lcl_env2'] = get_var_at_plev(ds_use.T, ds_use.P, ds_use.p_lcl2)
            for key in ['const', 'mod_parcel']:
                var = fitting_2_layer_xr(ds_use.T, ds_use.P, ds_use.TREFHT, ds_use.PREFHT, ds_use.T_lcl_env2,
                                         ds_use.p_lcl2, ds_use.T_ft_env,
                                         float(ds.p_ft), n_lev_above_upper2_integral=ds.n_lev_above_integral,
                                         method_layer2=key)
                # Must include fillna as inf to deal with all nan slice.
                ind_best = var[2].sum(dim='layer', skipna=False).fillna(np.inf).argmin(dim='lcl_mod')
                for k, key2 in enumerate(var_names):
                    ds_use[f'{key}1_{key2}'] = var[k].isel(lcl_mod=ind_phys_lcl)
                    ds_use[f'{key}2_{key2}'] = var[k].isel(lcl_mod=ind_best)
                ds_use[f'{key}2_p_lcl'] = ds_use['p_lcl2'].isel(lcl_mod=ind_best)
                ds_use[f'{key}2_T_lcl_env'] = ds_use['T_lcl_env2'].isel(lcl_mod=ind_best)

            # Save data
            ds_use = ds_use.drop_vars(['p_lcl2', 'T_lcl_env2'])
            ds_use = ds_use.drop_dims('lcl_mod')
            ds_use['layer'] = xr.DataArray(['below lcl', 'above lcl'], name='layer', dims='layer')
            ds_use = convert_ds_dtypes(ds_use)
            if not os.path.exists(path_use[j]):
                ds_use.to_netcdf(path_use[j], format="NETCDF4",
                             encoding={var: {"zlib": True, "complevel": comp_level} for var in ds_use.data_vars})
            print_log(f'File {i*ds.sample.size+j+1}/{n_files} | Saved', logger)

        if not os.path.exists(os.path.join(processed_dir[i], processed_file_name)):
            # Combine all sample files into a single file for each experiment
            ds_lapse = xr.concat([xr.load_dataset(path_use[j]) for j in range(ds.sample.size)], dim=ds.sample)
            ds_lapse.to_netcdf(os.path.join(processed_dir[i], processed_file_name), format="NETCDF4",
                             encoding={var: {"zlib": True, "complevel": comp_level} for var in ds_lapse.data_vars})
            print_log(f'{exp_name[i]} | Combined samples into one {processed_file_name} File', logger)

    hi = 5