from isca_tools.run import run_experiment
import os
# import os
# from functools import wraps
#
# import sh
#
# mkdir = sh.mkdir.bake('-p')
# cd = sh.cd
# git = sh.git.bake('--no-pager')
#
# P = os.path.join
# codedir_git = git.bake('-C', os.environ['GFDL_BASE'])
# git_test = codedir_git.log('-1', '--format="%H"').stdout

# When calling the script with no arguments, it just runs the experiment.
jobs_dir = os.path.join(os.environ['HOME'], 'Isca', 'jobs')  # all jobs saved here - CHANGE FOR EACH EXPERIMENT
# exp_dir = os.path.join(jobs_dir, 'tau_sweep/land/meridional_band/depth=1/bucket_evap')
# exp_dir = os.path.join(jobs_dir, 'tau_sweep/aquaplanet/depth=1/k=1_5/')
# exp_dir = os.path.join(jobs_dir, 'play/bucket/depth=5/k=1_5')
# exp_dir = os.path.join(jobs_dir, 'rossbypalooza/depth=20/lat_10/rough')
# exp_dir = os.path.join(jobs_dir, 'rossbypalooza/depth=20/all_nh/lat_10/no_wishe/socrates/bucket')
# exp_dir= os.path.join(jobs_dir, 'tau_sweep/land/meridional_band/depth=1/bucket_file/evap=1/k=1_5')
# exp_dir= os.path.join(jobs_dir, 'sqe_profile/column/day150/ras_conv/')
# exp_dir = os.path.join(jobs_dir, 'aquaplanet/latent_heat_temp_only/base')
# exp_dir= os.path.join(jobs_dir, 'tau_sweep/land/meridional_band/bucket_file/evap=0_1/k=1_5')
# exp_dir = os.path.join(jobs_dir, 'rossbypalooza/sst_observation/test_vary_time')
# exp_dir = os.path.join(jobs_dir, 'aquaplanet/latent_heat_temp_only/k=0_6')
# exp_dir = os.path.join(jobs_dir, 'thesis_season/depth=5/k=1_const_drag_evap=0_1/')
# exp_dir = os.path.join(jobs_dir, 'thesis_season/depth=5/k=1_const_drag_evap=0_1/')
exp_dir = os.path.join(jobs_dir, 'nerc_srm/simulations/meridional_band/base/')


namelist_file = os.path.join(exp_dir, 'namelist.nml')
diag_table_file = os.path.join(exp_dir, 'diag_table')
run_experiment(namelist_file, diag_table_file, slurm=True, scheduler='pbs')
# diag_table_file = os.path.join(exp_dir, 'diag_table_6_hours')
# run_experiment(namelist_file, diag_table_file, slurm=True)



# Run multiple simulations
# for kappa in ['3']:
# for exp in os.listdir(os.path.join(jobs_dir, f'thesis_season/publish_exp/')):
#     if exp == 'advect':
#         continue
# exp = 'sw_abs_low'
# #for kappa in ['0_6', '0_8', '1', '1_5', '2', '2_5', '3', '3_5']:
# for kappa in ['0_6']:
#     exp_dir = os.path.join(jobs_dir, f'thesis_season/publish_exp/{exp}/k={kappa}/')
#     namelist_file = os.path.join(exp_dir, 'namelist.nml')
#     diag_table_file = os.path.join(exp_dir, 'diag_table')
#     # diag_table_file = os.path.join(exp_dir, 'diag_table_6_hours')
#     run_experiment(namelist_file, diag_table_file, slurm=True)

# for kappa in ['0_6', '0_8', '2', '2_5', '3', '3_5']:
# for lat in ['0', '23', '70', '90']:
# for lat in ['90']:
#     namelist_file = os.path.join(exp_dir, f'lat={lat}', 'namelist.nml')
#     diag_table_file = os.path.join(exp_dir, f'lat={lat}', 'diag_table_hourly')
#     run_experiment(namelist_file, diag_table_file, slurm=True)


# For creating co2 timeseries .nc file
# from isca_tools.time_series.base import create_time_series_file
# def co2_func(days, pressure, lat, lon):
#     co2_val = np.ones((days.shape[0], pressure.shape[0], lat.shape[0], lon.shape[0])) * 500    # last 50 years have 500 ppmv
#     co2_val[days < 360 * 5] = 360      # First 40 years have 300ppmv
#     return co2_val
# create_time_series_file(os.path.join(exp_dir, 'co2_timeseries_t42.nc'), namelist_file, 'co2', co2_func, 360)

# # For creating land .nc file
# from isca_tools.land.base import write_land
# from isca_tools.plot import show_land
# # write_land('land.nc', namelist_file, 'square', [-90, 90, 180 - 30, 180 + 30])
# write_land('land.nc', namelist_file, 'square', [-90, 90, 180 - 15, 180 + 15])
# show_land(os.path.join(exp_dir, 'land.nc'))
#
# from isca_tools.utils.land import get_ocean_coords
# get_ocean_coords(namelist_file)
