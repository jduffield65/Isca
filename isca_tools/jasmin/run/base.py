import os
import f90nml
from typing import List, Optional, Union, Callable
from operator import itemgetter
import importlib.util
import sys
import subprocess
import socket
# from ...run.base import get_unique_dir_name

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

def import_func_from_path(script_path: str, func_name: str="main",
                          module_name: str="my_dynamic_module") -> Optional[Callable]:
    """
    Loads in the function called `func_name` from `script_path`.

    Args:
        script_path: Path to python `.py` file which contains `func_name`.
        func_name: Function name to import.
        module_name: Temporary name for module being loaded dynamically. Could be anything.

    Returns:
        func: Desired function, or `None` if no function called `func_name` is found in `script_path`.
    """
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return getattr(module, func_name, None)  # Returns the main() function if it exists

def get_slurm_info_from_file(input_file_path: str) -> dict:
    """
    Reads in a `.nml` file with a `slurm_info` section, containing information about python script to run on slurm.

    Args:
        input_file_path: The `.nml` file path, with `slurm_info` section containing.

    Returns:
        info: Dictionary containing all entries in `slurm_info` section of `input_file_path`.</br>
            If `job_name` not provided, will set to directory containing `input_file_path`.</br>
            If `script_args` not provided, will set to `input_file_path`.
    """
    info = f90nml.read(input_file_path)['slurm_info']
    if info['script_args'] is None:
        # Default arg for script is address to this input file
        info['script_args'] = input_file_path
    if info['job_name'] is None:
        # Default job name is directory containing input file, without the jobs_dir
        info['job_name'] = os.path.dirname(input_file_path).replace(info['jobs_dir'], '')
    if info['job_name'][0] == '/':
        info['job_name'] = info['job_name'][1:]  # make sure does not start with '/'
    return info


def run_script(script_path: Optional[str] = None, script_args: Optional[Union[List, float, int, str]] = None,
               job_name: Optional[str] = None, time: str = '02:00:00', n_tasks: int = 1,
               cpus_per_task: int = 1, mem: Union[float, int] = 16, partition: str = 'standard',
               qos: Optional[str] = None, account: str = 'global_ex', conda_env: str = 'myenv',
               exist_output_ok: Optional[bool] = None, input_file_path: Optional[str] = None,
               slurm: bool = False, dependent_job_id: Optional[str] = None) -> Optional[str]:
    """
    Function to submit a python script located at `script_path` to JASMIN using *Slurm*.

    Args:
        script_path: Path of python script to run.
        script_args: Arguments to be passed to `script_path`. Can pass as many as you want, using a list.
        job_name: Name of job submitted to slurm. If not provided, will set to name of python script without `.py`
            extension, and without `$HOME/Isca/jobs` at the start.
        time: Maximum wall time for your job in format `hh:mm:ss`
        n_tasks: Number of tasks to run (usually 1 for a single script).
        cpus_per_task: How many CPUs to allocate per task.
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
        input_file_path: Give option to provide nml file containing all *Slurm* info within `slurm_info` section.
        slurm: If `True`, will submit job to LOTUS cluster using *Slurm* queue.
            Otherwise, it will just run the script interactively, with no submission to *Slurm*.
        dependent_job_id: Job should only run after job with this dependency has finished. Only makes a difference
            if `slurm=True`.

    """
    if input_file_path is not None:
        # If provide input nml file, get all slurm info from this file - need no other info
        if not os.path.exists(input_file_path):
            raise ValueError(f"Input file {input_file_path} does not exist")
        slurm_info = get_slurm_info_from_file(input_file_path)
        script_args, job_name, time, n_tasks, cpus_per_task, mem, partition, qos, conda_env, account, exist_output_ok = \
            itemgetter('script_args', 'job_name', 'time', 'n_tasks', 'cpus_per_task',
                       'mem', 'partition', 'qos', 'conda_env', 'account', 'exist_output_ok')(slurm_info)
        # make script path the full path - combine jobs_dir and script_path
        script_path = os.path.join(slurm_info['jobs_dir'], slurm_info['script_path'])
    if slurm:
        if job_name is None:
            job_name = script_path.replace(os.path.join(os.environ['HOME'], 'Isca', 'jobs'), '')
            if job_name[0] == '/':
                job_name = job_name[1:]     # make sure does not start with '/'
            job_name = job_name.replace('.py', '')
        # Make directory where output and error saved
        job_output_dir = os.path.join(os.environ['HOME'], 'Isca/jobs/jasmin/console_output')
        dir_output = os.path.join(job_output_dir, job_name)
        if exist_output_ok is None:
            dir_output = get_unique_dir_name(dir_output)
            job_name = dir_output.replace(job_output_dir+'/', '')     # update job name so matches dir_output
            exist_output_ok = False
        os.makedirs(dir_output, exist_ok=exist_output_ok)
        slurm_script = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'run_slurm.sh')

        if qos is None:
            qos = partition

        if dependent_job_id is None:
            dependent_job_id = ''

        # Note that sys.argv[0] is the path to the run_script.py script that was used to call this function.
        # We now call it again but with input arguments so that it runs the job on slurm.
        # TODO: have issues when script_args do not come from input_file_path but from script_args= e.g. load_temp_ft_climatology
        submit_string = f"bash {slurm_script if dependent_job_id == '' else slurm_script.replace('.sh','_depend.sh')} "\
                        f"{job_name} {time} {n_tasks} {cpus_per_task} {mem} {partition} {qos} {account} "\
                        f"{conda_env} {script_path} {dependent_job_id}"
        submit_string = add_list_to_str(submit_string.strip(), script_args)   # use strip to get rid of any empty spaces at start or end
        # output = subprocess.check_output(submit_string, shell=True).decode("utf-8").strip()  # get job just submitted info
        try:
            output = subprocess.check_output(
                submit_string,
                shell=True,
                stderr=subprocess.STDOUT,  # capture error text too
                text=True  # str instead of bytes
            ).strip()
        except subprocess.CalledProcessError as e:
            host = socket.gethostname()
            msg = (e.output or "").strip()
            hint = (
                "Job submission failed.\n"
                f"Host: {host}\n"
                f"Command: {e.cmd}\n"
                f"Exit code: {e.returncode}\n"
                "Likely cause: 'sbatch' not on this node (common on JASMIN login nodes).\n"
                "Fix: run on a scientific analysis server / LOTUS environment where Slurm client is available.\n"
            )
            raise RuntimeError(hint + ("\nOutput:\n" + msg if msg else "")) from e
        print(f"{output}{dependent_job_id if dependent_job_id == '' else ' (dependency: job ' + dependent_job_id + ')'}")
        return output.split()[-1]  # Save this job id (last word) for the next submission
    else:
        # Import main function from script - do this rather than submitting to console, as can then use debugging stuff
        main_func = import_func_from_path(script_path)
        if isinstance(script_args, list):
            main_func(*script_args)         # call function with all arguments
        elif script_args is None:
            main_func()                     # if no arguments, call function without providing any arguments
        else:
            main_func(script_args)          # if single argument, call with just that argument
        return None