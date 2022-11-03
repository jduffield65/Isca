from run import run_experiment
import os
import numpy as np
jobs_dir = os.path.join(os.environ['HOME'], 'Isca', 'jobs')   # all jobs saved here
exp_dir = os.path.join(jobs_dir, 'aquaplanet/vary_co2')                      # specific experiment

namelist_file = os.path.join(exp_dir, 'd5_namelist.nml')
diag_table_file = os.path.join(exp_dir, 'd5_diag_table')
run_experiment(namelist_file, diag_table_file, slurm=True)




# from isca_tools.time_series.base import create_time_series_file
# def co2_func(days, pressure, lat, lon):
#     co2_val = np.zeros((days.shape[0], pressure.shape[0], lat.shape[0], lon.shape[0]))
#     co2_val[days < 360 * 5] = 300
#     co2_val[days >= 360 * 5] = 350
#     return co2_val
#
# create_time_series_file(os.path.join(exp_dir, 'co2_timeseries.nc'), namelist_file, 21, 'co2', co2_func,
#                         360)
