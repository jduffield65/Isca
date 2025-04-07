import os
import sys
import f90nml
import numpy as np
import datetime
from typing import List, Optional, Union


def get_file_suffix(dir: str, suffix: str) -> List[str]:
    """
    Returns a list of all files in `dir` which end in `suffix`.

    This is the same function that is in `utils.load` but cannot do a relative import due to slurm job submission
    stuff.

    Args:
        dir: Directory of interest.
        suffix: Usually the file type of interest e.g. `.nml` or `.txt`.

    Returns:
        List of all files with the correct `suffix`.
    """
    file_name = []
    for file in os.listdir(dir):
        if file.endswith(suffix):
            file_name += [file]
    return file_name


def run_script(script_path: str, script_args: Optional[any] = None, job_name: Optional[str] = None,
               time: str = '02:00:00', ntasks: int = 1, cpus_per_tasks: int = 1, mem: Union[float, int] = 32,
               partition: str = 'test', slurm: bool = False):
    """

    Args:
        script_path: Path to python script to run.
        script_args: I give the option for arguments to be passed to `script_path`. These are provided here.
        job_name: Name of job submitted to slurm. If not provided, will set to name of python script without `.py`
            extension, and without `$HOME/Isca$ at the start.
        time: Maximum wall time for your job in format `hh:mm:ss`
        ntasks: Number of tasks to run (usually 1 for a single script).
        cpus_per_tasks: How many CPUs to allocate per task.
        mem: Memory to allocate for the job in GB.
        partition: Specifies the partition for the job e.g. `test`, `short-serial-4hr`, `short-serial`, `par-single`.
        slurm: If `True`, will submit job to LOTUS cluster using *Slurm* queue.
            Otherwise, it will just run the script interactively, with no submission to *Slurm*.

    Returns:

    """
    if slurm:
        if job_name is None:
            job_name = script_path.replace(os.path.join(os.environ['HOME'], 'Isca'), '')
            job_name = job_name.replace('.py', '')
        # Make directory where output saved
        job_output_dir = os.path.join(os.environ['HOME'], 'Isca/jobs/jasmin/console_output')
        os.makedirs(os.path.join(job_output_dir, job_name), exist_ok=False)
        slurm_script = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'run_slurm.sh')

        # Note that sys.argv[0] is the path to the run_script.py script that was used to call this function.
        # We now call it again but with input arguments so that it runs the job on slurm.
        os.system(f"bash {slurm_script} {exp_details['name']} {month_job[0]} {len(month_job)} "
                  f"{exp_details['partition']} {exp_details['n_nodes']} {exp_details['n_cores']} "
                  f"{namelist_file} {diag_table_file} {exp_details['max_walltime']} {run_job_script} "
                  f"{exp_details['nodelist']}")
    else:
        os.system(f"python {script_path} {script_args}")


if __name__ == "__main__":
    if len(sys.argv) == 5:
        # This is when calling the script from the isca_tools/run/run_slurm.sh shell script
        # Cannot include in isca_tools package due to relative import issues.
        start_time = datetime.datetime.utcnow()
        run_job(sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
        end_time = datetime.datetime.utcnow()
        # Print out how long it takes, so saves when running with slurm, to output txt file
        print(f"Simulation Start Month: {int(sys.argv[3])}")
        print(f"Simulation Length/Months: {int(sys.argv[4])}")
        print(f"Start Time: {start_time.strftime('%B %d %Y - %H:%M:%S')}")
        print(f"End Time: {end_time.strftime('%B %d %Y - %H:%M:%S')}")
        print(f"Duration/Seconds: {int(np.round((end_time - start_time).total_seconds()))}")

    elif sys.argv[1] in ["--help", "-h"]:
        print('To run_experiment, provide no arguments, just specify namelist and diag_table files in script.\n'
              'To run_job, need to provide 4 arguments:\n'
              'namelist_file - File that indicates physical parameters used in simulation.\n'
              'diag_table_file - File that specifies the outputs of the experiment.\n'
              'month_start - Index of month at which this job starts the simulation (starting with 1).\n'
              'month_duration - How many months to run simulation for in this job.')
    else:
        raise ValueError(f"{len(sys.argv) - 1} parameters provided but 4 expected.")
