from run import run_experiment
import os
jobs_dir = os.path.join(os.environ['HOME'], 'Isca', 'jobs')   # all jobs saved here
exp_dir = os.path.join(jobs_dir, 'aquaplanet/spin_up')                      # specific experiment

namelist_file = os.path.join(exp_dir, 'd10_namelist.nml')
diag_table_file = os.path.join(exp_dir, 'd10_diag_table')
run_experiment(namelist_file, diag_table_file, slurm=True)
