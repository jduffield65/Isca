from run import run_experiment
import os
jobs_dir = os.path.join(os.environ['HOME'], 'Isca', 'jobs')   # all jobs saved here
# jobs_dir = os.path.join(os.environ['HOME'], 'Isca', 'isca_tools/exp')
exp_dir = os.path.join(jobs_dir, 'grid_files')                      # specific experiment

namelist_file = os.path.join(exp_dir, 'gridfile_namelist.nml')
diag_table_file = os.path.join(exp_dir, 'gridfile_diag_table')
run_experiment(namelist_file, diag_table_file, slurm=False)

