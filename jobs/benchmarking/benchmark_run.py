import os
from os import system
import numpy as np
import re
from typing import Tuple

# This script runs a single experiment with different resolution and number of cores. It then records the time
# taken for each.

# Things to change
experiment = 'frierson_test_case.py'  # python script to call is this in the benchmarking/experiment folder
n_months = 6  # duration of simulation
n_nodes = 2  # how many nodes to run on
n_cores_list = [32]  # cores per node to iterate over, all would be [8, 16, 32]
res_list = ['T85']  # horizontal resolution to iterate over, all would be ['T21', 'T42', 'T85']
csv_file = 'task_times.csv'  # where to save how long each task took.


def extract_number(f: str) -> Tuple[int, str]:
    # returns last number of string. If is no number, will return -1.
    s = re.findall(r'\d+', f)
    return int(s[-1]) if s else -1, f


def get_max_folder_number(dir: str) -> int:
    # Finds maximum number at end of folder within the directory dir.
    # Will be -1 if no folders end in a number or if directory does not exist.
    if os.path.exists(dir):
        max_index = extract_number(max(os.listdir(dir), key=extract_number))[0]
    else:
        max_index = -1
    return max_index


# get index of first iteration so does not overwrite previous data
# first index would be 1 if experiment not yet run
shell_script = 'benchmark.sh'  # bash script to call experiment
experiment = experiment.replace('.py', '')  # make sure experiment name does not have .py suffix.
python_script = os.path.join('experiments', experiment+'.py')
# error .txt files saved as output/experiment/error{ind}.txt and similar for the output file
starting_ind = int(np.clip(get_max_folder_number(f'output/{experiment}') + 1, 1, np.inf))

if n_nodes == 1 and n_months <= 3:
    # If 1 node use debug partition as quicker
    partition = 'debug'
else:
    # If more than 1 node, need to use other partition but will take much longer to run as need to queue.
    partition = 'parallel'

# Iterate over all n_cores and resolutions provided
ind = starting_ind
for n_cores in n_cores_list:
    for res in res_list:
        output_file = f'output/{experiment}/output_run{ind}.txt'  # name of file containing stuff printed to console
        error_file = f'output/{experiment}/error_run{ind}.txt'  # name of file containing errors printed to console
        system(f'bash {shell_script} {python_script} {ind} {partition} {n_nodes} {n_cores} {res} {n_months} {csv_file} '
               f'{output_file} {error_file}')
        ind += 1
