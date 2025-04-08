from isca_tools import cesm
from isca_tools.papers.byrne_2021 import get_quant_ind
from isca_tools.utils.moist_physics import moist_static_energy, sphum_sat
import sys
import os
import numpy as np
import f90nml

def get_exp_info_dict(input_file_path: str) -> dict:
    # From input nml file, this returns dict containing info required for getting npz file
    exp_info = f90nml.read(input_file_path)['script_info']
    if exp_info['out_dir'] is None:
        # Default directory to save output is the directory containing input nml file
        exp_info['out_dir'] = os.path.dirname(input_file_path)
    if exp_info['out_name'] is None:
        # Default name of saved data is output.npz
        exp_info['out_name'] = 'output.npz'
    elif exp_info['out_name'][-4:] != '.npz':
        # Ensure output file ends in npz
        exp_info['out_name'] = exp_info['out_name'] + '.npz'
    if exp_info['year_first'] is None:
        # Default first year of data is year 1
        exp_info['year_first'] = 1
    if exp_info['year_last'] is None:
        # Default last year of data is last year in cesm data
        exp_info['year_last'] = -1
    out_file = os.path.join(exp_info['out_dir'], exp_info['out_name'])
    if not exp_info['exist_ok'] and os.path.exists(out_file):
        # Raise error if output data already exists
        raise ValueError('Output file already exists at:\n{}'.format(out_file))
    return exp_info

def get_ds(exp_name, archive_dir, chunks_time, chunks_lat, chunks_lon, p_ft_approx_guess, p_surf_approx_guess,
           landfrac_thresh, year_first, year_last):
    # Load in datasets - one from atm and lnd components
    chunks={"time": chunks_time, "lat": chunks_lat, "lon": chunks_lon}
    ds = cesm.load_dataset(exp_name, archive_dir=archive_dir, hist_file=1,
                           year_first=year_first, year_last=year_last,chunks=chunks)
    ds_lnd = cesm.load_dataset(exp_name, archive_dir=archive_dir, hist_file=1, comp='lnd',
                               year_first=year_first, year_last=year_last,chunks=chunks)

    # Reduce size of datasets
    ind_ft = int(np.argmin(np.abs(ds.T.lev - p_ft_approx_guess).to_numpy()))
    ind_surf = int(np.argmin(np.abs(ds.T.lev - p_surf_approx_guess).to_numpy()))
    var_atm = ['T', 'Q', 'Z3', 'PS', 'P0', 'hyam', 'hybm']
    ds_atm = ds.isel(lev=[ind_surf, ind_ft])[var_atm]  # For atm dataset, only need a few variables and 2 pressure levels
    soil_liq = ds_lnd.SOILLIQ.sum(dim='levsoi')    # for lnd dataset, only need soil moisture

    is_land = ds_lnd.landfrac.isel(time=0, drop=True) > landfrac_thresh     # whether a coordinate is land or not
    return ds_atm, soil_liq, is_land


def main(input_file_path):
    exp_info = get_exp_info_dict(input_file_path)
    ds, soil_liq, is_land = get_ds(exp_info['exp_name'], exp_info['archive_dir'],
                                   exp_info['chunks_time'], exp_info['chunks_lat'], exp_info['chunks_lon'],
                                   exp_info['p_ft_approx_guess'], exp_info['p_surf_approx_guess'],
                                   exp_info['landfrac_thresh'], exp_info['year_first'], exp_info['year_last'])
    is_land = is_land.load()        # load as small dataset and quickens later steps

    # Get pressure info
    ind_surf = 0
    ind_ft = 1
    p_ref = float(ds.P0[0])
    hybrid_a_coef_ft = float(ds.hyam.isel(time=0, lev=ind_ft))
    hybrid_b_coef_ft = float(ds.hybm.isel(time=0, lev=ind_ft))
    ds = ds.drop_vars(["P0", "hyam", "hybm"])     # don't need ref pressure or hybrid coordinates anymore
    p_ft_approx = float(ds.T.lev[ind_ft]) * 100
    p_surf_approx = float(ds.T.lev[ind_surf]) * 100


    # Initialize dict to save output data - n_surf (land or ocean) x n_lat x n_quant
    n_lat = ds.lat.size
    quant_use = exp_info['quant']
    quant_range = exp_info['quant_range']   # find quantile by looking for all values in quantile range between quant_use-quant_range to quant_use+quant_range
    n_quant = len(quant_use)
    output_info = {var: np.zeros((2, n_lat, n_quant)) for var in
                   ['temp', 'temp_ft', 'sphum', 'z', 'z_ft', 'rh', 'mse', 'mse_sat_ft', 'mse_lapse',
                    'mse_sat_ft_p_approx', 'mse_lapse_p_approx', 'pressure_ft', 'soil_liq']}
    var_keys = [key for key in output_info.keys()]
    for var in var_keys:
        output_info[var + '_std'] = np.zeros((2, n_lat, n_quant))
    output_info['lon_most_common'] = np.zeros((2, n_lat, n_quant))
    output_info['lon_most_common_freq'] = np.zeros((2, n_lat, n_quant), dtype=int)
    output_info['n_grid_points'] = np.zeros((2, n_lat), dtype=int)  # number of grid points used at each location
    output_info['surface'] = ['land', 'ocean']
    # Record approx number of days used in quantile calculation. If quant_range=0.5 and 1 year used, this is just 0.01*365=3.65
    output_info['n_days_quant'] = get_quant_ind(np.arange(ds.time.size * n_lat), quant_use[0], quant_range,
                                                quant_range).size / n_lat


    # Loop through and get quantile info at each latitude and surface
    for i in range(n_lat):
        for k, surf in enumerate(output_info['surface']):
            if surf == 'land':
                is_surf = is_land.isel(lat=i)
            else:
                is_surf = (~is_land).isel(lat=i)

            if is_surf.sum() == 0:
                # If surface not at this latitude, record no data
                continue

            ds_use = ds.isel(lat=i).sel(lon=is_surf, drop=True)
            ds_use = ds_use.stack(lon_time=("lon", "time"), create_index=False).chunk(dict(lon_time=-1)).load()
            if surf == 'land':
                soil_liq_use = soil_liq.isel(lat=i).sel(lon=is_surf, drop=True)
                soil_liq_use = soil_liq_use.stack(lon_time=("lon", "time"),
                                                  create_index=False).chunk(dict(lon_time=-1)).load()
            output_info['n_grid_points'][k, i] = ds_use.lon.size
            for j in range(n_quant):
                # get indices corresponding to given near-surface temp quantile
                use_ind = get_quant_ind(ds_use.T.isel(lev=ind_surf), quant_use[j], quant_range, quant_range)
                ds_use_q = ds_use.isel(lon_time=use_ind, drop=True)
                var_use = {}
                var_use['temp'] = ds_use_q.T.isel(lev=ind_surf)
                var_use['temp_ft'] = ds_use_q.T.isel(lev=ind_ft)
                var_use['sphum'] = ds_use_q.Q.isel(lev=ind_surf)
                var_use['z'] = ds_use_q.Z3.isel(lev=ind_surf)
                var_use['z_ft'] = ds_use_q.Z3.isel(lev=ind_ft)
                var_use['rh'] = ds_use_q.Q.isel(lev=ind_surf) / sphum_sat(ds_use_q.T.isel(lev=ind_surf), p_surf_approx)
                var_use['mse'] = moist_static_energy(ds_use_q.T.isel(lev=ind_surf), ds_use_q.Q.isel(lev=ind_surf),
                                                     ds_use_q.Z3.isel(lev=ind_surf))
                var_use['mse_sat_ft_p_approx'] = moist_static_energy(ds_use_q.T.isel(lev=ind_ft),
                                                                     sphum_sat(ds_use_q.T.isel(lev=ind_ft), p_ft_approx),
                                                                     ds_use_q.Z3.isel(lev=ind_ft))
                var_use['mse_lapse_p_approx'] = var_use['mse'] - var_use['mse_sat_ft_p_approx']
                var_use['pressure_ft'] = cesm.get_pressure(ds_use_q.PS, p_ref, hybrid_a_coef_ft, hybrid_b_coef_ft)
                var_use['mse_sat_ft'] = moist_static_energy(ds_use_q.T.isel(lev=ind_ft),
                                                            sphum_sat(ds_use_q.T.isel(lev=ind_ft), var_use['pressure_ft']),
                                                            ds_use_q.Z3.isel(lev=ind_ft))
                var_use['mse_lapse'] = var_use['mse'] - var_use['mse_sat_ft']
                if surf == 'land':
                    var_use['soil_liq'] = soil_liq_use.isel(lon_time=use_ind)
                for key in var_use:
                    output_info[key][k, i, j] = var_use[key].mean()
                    output_info[key + '_std'][k, i, j] = var_use[key].std()
                lon_use = np.unique(ds_use.lon[use_ind], return_counts=True)

                # Record most common specific coordinate within grid to see if most of days are at a given location
                output_info['lon_most_common'][k, i, j] = lon_use[0][lon_use[1].argmax()]
                output_info['lon_most_common_freq'][k, i, j] = lon_use[1][lon_use[1].argmax()]
                print(i, k, j)
    # Add basic info of the dataset and averaging details used
    output_info['exp_name'] = exp_info['exp_name']
    output_info['date_start'] = ds.time.to_numpy()[0].strftime()
    output_info['date_end'] = ds.time.to_numpy()[-1].strftime()
    output_info['lat'] = ds.lat.to_numpy()
    output_info['lon'] = ds.lon.to_numpy()
    output_info['pressure_surf_approx'] = p_surf_approx
    output_info['pressure_ft_approx'] = p_ft_approx
    output_info['quant'] = quant_use
    output_info['quant_range'] = quant_range
    output_info['landfrac_thresh'] = exp_info['landfrac_thresh']

    # Save output to npz file
    np.savez_compressed(os.path.join(exp_info['out_dir'], exp_info['out_name']), **output_info)

if __name__ == "__main__":
    main(sys.argv[1])
