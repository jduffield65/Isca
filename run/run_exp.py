import os
from os import system
import numpy as np
import re
import sys
import f90nml

settings_file = "/home/jamd1/isca_jobs/run/settings_default.nml"
exp_details = f90nml.read(settings_file)['experiment_details']
diag_file = "/home/jamd1/isca_jobs/run/diag_table_default"

# Things to change
name = exp_details['name']  # python script to call is this in the benchmarking/experiment folder
n_months_total = exp_details['n_months']  # duration of total simulation
n_nodes = exp_details['n_nodes']  # how many nodes to run on
n_cores = exp_details['n_cores']  # cores per node to iterate over, all would be [8, 16, 32]
res = exp_details['res']  # horizontal resolution to iterate over, all would be ['T21', 'T42', 'T85']
partition = exp_details['partition']


# def extract_number(f: str) -> Tuple[int, str]:
#     # returns last number of string. If is no number, will return -1.
#     s = re.findall(r'\d+', f)
#     return int(s[-1]) if s else -1, f
#
#
# def get_max_folder_number(dir: str) -> int:
#     # Finds maximum number at end of folder within the directory dir.
#     # Will be -1 if no folders end in a number or if directory does not exist.
#     if os.path.exists(dir):
#         max_index = extract_number(max(os.listdir(dir), key=extract_number))[0]
#     else:
#         max_index = -1
#     return max_index
#

# get index of first iteration so does not overwrite previous data
# first index would be 1 if experiment not yet run
# shell_script = 'restart_run.sh'  # bash script to call experiment
# experiment = experiment.replace('.py', '')  # make sure experiment name does not have .py suffix.
# python_script = experiment+'.py'
# # error .txt files saved as output/experiment/error{ind}.txt and similar for the output file
# starting_ind = int(np.clip(get_max_folder_number('output') + 1, 1, np.inf))


# Iterate over all months and resolutions provided
for i in range(n_months_total):
    if i == 0:
        restart = False
    else:
        restart = True
    system(f'bash run_slurm.sh {name} {i} {restart} {partition} {n_nodes} {n_cores} {res} '
           f'{settings_file} {diag_file}')
