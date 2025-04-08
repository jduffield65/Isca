import os
import sys
import f90nml
import numpy as np
import datetime
from typing import List, Optional, Union, Literal

def add_list_to_str(var_str: str, var_list: Optional[Union[List, float, int, str]]) -> str:
    """
    Adds all entries in `var_list` to end of `var_str`.</br>
    If `var_str = 'hello'` and `var_list = ['Bob', 'Alice']` then will return `'hello Bob Alice'`.

    Args:
        var_str: Starting string, to which `var_list` will be added.
        var_list: List of entries to add to `var_str`.</br>
            If one of entries is itself a list, will convert that entry into comma separated string.

    Returns:
        var_str: Initial `var_str` with `var_list` entries added.
    """
    if isinstance(var_list, list):
        for arg in var_list:
            if isinstance(arg, list):
                # If one of the arguments is a list, convert into comma separated string so all info stil passed
                arg_comma_sep = ",".join(str(x) for x in arg)
                var_str += f" {arg_comma_sep}"
            else:
                var_str += f" {arg}"
    else:
        var_str += f" {var_list}"
    return var_str


def get_unique_dir_name(base_dir: str) -> str:
    """
    Return a unique directory name by appending a number if needed.
    E.g., 'results', 'results_1', 'results_2', ...

    Args:
        base_dir: Path to directory

    Returns:
        base_dir: Unique directory name
    """
    if not os.path.exists(base_dir):
        return base_dir

    i = 1
    while True:
        new_dir = f"{base_dir}_{i}"
        if not os.path.exists(new_dir):
            return new_dir
        i += 1


def run_script(script_path: str, script_args: Optional[Union[List, float, int, str]] = None,
               job_name: Optional[str] = None, time: str = '02:00:00', ntasks: int = 1,
               cpus_per_tasks: int = 1, mem: Union[float, int] = 16, partition: str = 'standard',
               qos: Optional[str] = None, account: str = 'global_ex', conda_env: str = 'myenv',
               exist_output_ok: Optional[bool] = None, slurm: bool = False) -> None:
    """
    Function to submit a python script located at `script_path` to JASMIN using *Slurm*.

    Args:
        script_path: Path of python script to run.
        script_args: Arguments to be passed to `script_path`. Can pass as many as you want, using a list.
        job_name: Name of job submitted to slurm. If not provided, will set to name of python script without `.py`
            extension, and without `$HOME/Isca/jobs` at the start.
        time: Maximum wall time for your job in format `hh:mm:ss`
        ntasks: Number of tasks to run (usually 1 for a single script).
        cpus_per_tasks: How many CPUs to allocate per task.
        mem: Memory to allocate for the job in GB.
        partition: Specifies the partition for the job.
            [Options](https://help.jasmin.ac.uk/docs/batch-computing/how-to-submit-a-job/#partitions-and-qos)
            are `standard`, `highres` and `debug`.
        qos: Quality of service
            [examples](https://help.jasmin.ac.uk/docs/software-on-jasmin/rocky9-migration-2024/#partitions-and-qos)
            include `debug`, `short` and `standard`. Each have a different max wall time and priority.</br>
            If `None`, will set to the same as `partition`.
        conda_env: Name of the conda environment on JASMIN to use.
        account: Account to use for submitting jobs. Should be able to find as a group workspace in the
            [myservices](https://accounts.jasmin.ac.uk/services/my_services/) section of your JASMIN account.
        exist_output_ok: Whether to run script if console_output for `job_name` already exists.</br>
            If None, will save output to directory with a number added e.g. if `output` exists, will save
            as `output_1`.
        slurm: If `True`, will submit job to LOTUS cluster using *Slurm* queue.
            Otherwise, it will just run the script interactively, with no submission to *Slurm*.

    """
    if slurm:
        if job_name is None:
            job_name = script_path.replace(os.path.join(os.environ['HOME'], 'Isca', 'jobs'), '')
            if job_name[0] == '/':
                job_name = job_name[1:]     # make sure does not start with '/'
            job_name = job_name.replace('.py', '')
        # Make directory where output saved
        job_output_dir = os.path.join(os.environ['HOME'], 'Isca/jobs/jasmin/console_output')
        dir_output = os.path.join(job_output_dir, job_name)
        if exist_output_ok is None:
            dir_output = get_unique_dir_name(dir_output)
            job_name = dir_output.replace(job_output_dir+'/', '')         # update job name so matches dir_output
            exist_output_ok = False
        os.makedirs(dir_output, exist_ok=exist_output_ok)
        slurm_script = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'run_slurm.sh')

        if qos is None:
            qos = partition

        # Note that sys.argv[0] is the path to the run_script.py script that was used to call this function.
        # We now call it again but with input arguments so that it runs the job on slurm.
        submit_string = f"bash {slurm_script} {job_name} {time} {ntasks} "\
                        f"{cpus_per_tasks} {mem} {partition} {qos} {account} {conda_env} {script_path}"
    else:
        submit_string = f"python {script_path}"
    os.system(add_list_to_str(submit_string, script_args))
