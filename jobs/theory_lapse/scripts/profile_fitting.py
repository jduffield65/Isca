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
import matplotlib
load_workspace('/Users/joshduffield/Desktop/workspace_shelf', globals())
show_plot = True
if show_plot:
    matplotlib.use("TkAgg")  # To show plots in debugging mode
import matplotlib.pyplot as plt

# Load in example dataset
data_dir = '/Users/joshduffield/Documents/StAndrews/Isca/jobs/cesm/3_hour/hottest_quant'
quant_type = 'REFHT_quant99'
exp_name = ['pre_industrial']
n_exp = len(exp_name)
co2_vals = np.arange(1, n_exp + 1)
lat_min = 40
lat_max = 50
lon_min = 0
lon_max = 20
ds = [xr.open_dataset(os.path.join(data_dir, exp_name[i], quant_type, 'output.nc')) for i in range(n_exp)]
ds = xr.concat(ds, dim=xr.DataArray(co2_vals, dims="co2", coords={"exp_name": ("co2", exp_name)}))
ds = ds.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))
ds = ds.load()

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

# Set FT info
p_ft = 500 * 100
ds['T_ft'] = get_var_at_plev(ds.T, ds.P, p_ft)
ds['Z_ft'] = get_var_at_plev(ds.Z3, ds.P, p_ft)
ds['mse_sat_ft'] = moist_static_energy(ds['T_ft'], sphum_sat(ds['T_ft'], p_ft), ds['Z_ft'])

# Compute MSE environmental profile (separate above and below LCL), then innterpolate
for key in ['below', 'above']:
    ds[f'mse_env_{key}'] = get_mse_env(ds.T, ds.P, ds.Z3, ds.T_at_lcl, ds.p_lcl, prof_type=f'{key}_lcl')

mse_env_pnorm = {'below': interp_var_to_pnorm(ds.mse_env_below, ds.P, ds.mse_env_below.isel(lev=-1), ds.P.isel(lev=-1),
                                              ds.mse_sat_at_lcl, ds.p_lcl, ds.lnb_ind),
                 'above': interp_var_to_pnorm(ds.mse_env_above, ds.P, ds.mse_sat_at_lcl, ds.p_lcl,
                                              ds.mse_sat_ft, p_ft, ds.lnb_ind)}

# Do PCA on mse on pnorm grid, to get starting point for scaled k-means
valid = {key: (lsm > 0) & (mse_env_pnorm[key][1] <= 2) & (np.abs(mse_env_pnorm[key][0]).max(dim='pnorm') <= 20) &
              (ds.lat > -60) & (ds.lat < 75) for key in mse_env_pnorm}
n_modes = 4
# slice so avoid pnorm=0 where var=0
pca_output = {key: list(pca_on_xarray(mse_env_pnorm[key][0].isel(pnorm=slice(1, 9999)),
                                      valid=valid[key], feature_dim_name='pnorm',
                                      standardize=True, n_modes=n_modes,
                                      reference_mean=False)) for key in mse_env_pnorm}

# Do scaled k-means using PC output as starting point
k_output = {}
pca_output_residual = {}
for key in pca_output:
    x_use = flatten_to_numpy(mse_env_pnorm[key][0].isel(pnorm=slice(1, 9999)), 'pnorm')
    k_output[key] = list(scaled_k_means(x_use, pca_output[key][0].to_numpy(), valid=flatten_to_numpy(valid[key])))
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
hi = 5

for i in range(n_modes):
    plt.plot(k_output[0][i], pca_output_residual[0].pnorm, color=f'C{i}')
    plt.plot(pca_output[key][0].isel(mode=i), pca_output_residual[0].pnorm, color=f'C{i}', linestyle='--')
    plt.plot(pca_output_residual[0].isel(mode=i), pca_output_residual[0].pnorm, color=f'C{i}', linestyle=':')