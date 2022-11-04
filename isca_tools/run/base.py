import os
import sys
from isca import Experiment, IscaCodeBase, GFDL_BASE
from isca.diagtable import DiagTable
from ..utils.load import get_file_suffix
import f90nml
import numpy as np
from datetime import datetime


def run_experiment(namelist_file: str, diag_table_file: str, slurm: bool = False):
    """
    This splits the total simulation up into jobs based on info specified in `namelist_file` and
    then runs `run_job` for each job. If `slurm==True`, then each job will be submitted to a *Slurm*
    queue through the `run_slurm.sh` script.

    Args:
        namelist_file: File path to namelist `nml` file for the experiment.
            This specifies the physical parameters used for the simulation.
            Also contains `experiment_details` section which contains the following:

            - `name`: *string*. Name of experiment e.g. data saved in folder `$GFDL_DATA/{name}`
            - `input_dir`: *string*. Directory containing any input files e.g. `namelist.nml` or `co2.nc`.
            - `n_months_total`: *int*. Total duration of simulation in months.
            - `n_months_job`: *int*. Approximate duration of each job of the simulation in months.
            E.g. if `n_months_total=12` and `n_months_job=6`, the experiment would be split up into 2 jobs each
            of length 6 months.
            - `n_nodes`: *int*. Number of nodes to run job on (*Slurm* info).
            - `n_cores`: *int*. Number of cores for each node to run job on (*Slurm* info).
            - `resolution`: *string*. Horizontal resolution of experiment (options are `T21`, `T42` or `T85`).
            - `partition`: *string*. *Slurm* queue to submit job to.
            - `overwrite_data`: *bool*. If this is `True` and data already exists in `$GFDL_DATA/{name}`,
                then it will be overwritten. If it is `False` and the data exists, an error will occur.
            - `compile`: *bool*. If `True`, it will recompile the codebase before running the experiment.
            - `max_walltime`: *string*. Maximum time that job can run on *Slurm*. E.g. 1 hour would be '01:00:00'.
        diag_table_file: File path to the diagnostic table file for the experiment.
            This specifies the outputs of the experiment.
        slurm: If `True`, will split each job to a *Slurm* queue. Otherwise, it will just loop over each
            job locally.
    """
    exp_details = f90nml.read(namelist_file)['experiment_details']

    if exp_details['n_months_total'] % exp_details['n_months_job'] == 0:
        # If you give n_months_job which is exact multiple of n_months_total, do this many jobs
        n_jobs = int(np.ceil(exp_details['n_months_total'] / exp_details['n_months_job']))
    else:
        # Split simulation equally between jobs so no more than 1.5 multiplied by exp_details['n_months_job'] months
        # would be in each job.
        n_jobs = int(np.ceil(exp_details['n_months_total'] / (1.5 * exp_details['n_months_job'])))
    # month_jobs[i] are the months to simulate in job i.
    month_jobs = np.array_split(np.arange(1, exp_details['n_months_total'] + 1), n_jobs)

    slurm_script = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'run_slurm.sh')
    # Iterate over all jobs
    for month_job in month_jobs:
        if slurm:
            # Note that sys.argv[0] is the path to the run_script.py script that was used to call this function.
            # We now call it again but with input arguments so that it runs the job on slurm.
            os.system(f"bash {slurm_script} {exp_details['name']} {month_job[0]} {len(month_job)} "
                      f"{exp_details['partition']} {exp_details['n_nodes']} {exp_details['n_cores']} "
                      f"{namelist_file} {diag_table_file} {exp_details['max_walltime']} {sys.argv[0]}")
        else:
            run_job(namelist_file, diag_table_file, month_job[0], len(month_job))


def run_job(namelist_file: str, diag_table_file: str, month_start: int, month_duration: int):
    """
    Runs simulation for `month_duration` months starting with the month indicated by `month_start`.
    Will throw an error if no data found for months prior to `month_start`.

    Output will be saved in `$GFDL_DATA/{exp_name}` with a different folder for each month.

    Extended from [original script](https://github.com/ExeClim/Isca/blob/master/exp/run_isca/isca).
    Args:
        namelist_file: File path to namelist `nml` file for the experiment.
            This specifies the physical parameters used for the simulation.
            Also contains `experiment_details` section which contains the following:

            - `name`: *string*. Name of experiment e.g. data saved in folder `$GFDL_DATA/{name}`
            - `input_dir`: *string*. Directory containing any input files e.g. `namelist.nml` or `co2.nc`.
            - `n_months_total`: *int*. Total duration of simulation in months.
            - `n_months_job`: *int*. Approximate duration of each job of the simulation in months.
            E.g. if `n_months_total=12` and `n_months_job=6`, the experiment would be split up into 2 jobs each
            of length 6 months.
            - `n_nodes`: *int*. Number of nodes to run job on (*Slurm* info).
            - `n_cores`: *int*. Number of cores for each node to run job on (*Slurm* info).
            - `resolution`: *string*. Horizontal resolution of experiment (options are `T21`, `T42` or `T85`).
            - `partition`: *string*. *Slurm* queue to submit job to.
            - `overwrite_data`: *bool*. If this is `True` and data already exists in `$GFDL_DATA/{name}`,
                then it will be overwritten. If it is `False` and the data exists, an error will occur.
            - `compile`: *bool*. If `True`, it will recompile the codebase before running the experiment.
            - `max_walltime`: *string*. Maximum time that job can run on *Slurm*. E.g. 1 hour would be '01:00:00'.
        diag_table_file: File path to the diagnostic table file for the experiment.
            This specifies the outputs of the experiment.
        month_start: Index of month at which this job starts the simulation (starting with 1).
        month_duration: How many months to run simulation for in this job.
    """
    namelist = f90nml.read(namelist_file)
    diag_table = DiagTable.from_file(diag_table_file)
    exp_details = namelist['experiment_details']

    cb = IscaCodeBase.from_directory(GFDL_BASE)
    if exp_details['compile']:
        cb.compile()

    exp = Experiment(exp_details['name'], codebase=cb)
    # When passing to experiment, will have additional namelist called 'experiment_details' but does not
    # seem to be an issue.
    exp.namelist = namelist
    exp.diag_table = diag_table
    exp.inputfiles = [namelist_file, diag_table_file]

    # Get any additional input files e.g. co2 concentration or land - all of which should have a .nc suffix.
    nc_files = get_file_suffix(exp_details['input_dir'], '.nc')
    if len(nc_files) > 0:
        nc_files = [os.path.join(exp_details['input_dir'], val) for val in nc_files]
        exp.inputfiles += nc_files

    exp.set_resolution(exp_details['resolution'])  # set resolution
    if month_start == 1:
        # If first month, then there is no restart file to use
        use_restart = False
    else:
        use_restart = True
    exp.run(month_start, use_restart=use_restart, num_cores=exp_details['n_cores'],
            overwrite_data=exp_details['overwrite_data'])
    for i in range(month_start + 1, month_start + month_duration):
        # For all months but first, use latest restart file to set up simulation.
        exp.run(i, num_cores=exp_details['n_cores'], overwrite_data=exp_details['overwrite_data'])


# if __name__ == "__main__":
#     # When using slurm, it will call run_jobs by python base.py which uses this.
#     if sys.argv[1] in ["--help", "-h"]:
#         print('Need to provide 4 arguments to start run_job:\n'
#               'namelist_file - File that indicates physical parameters used in simulation.\n'
#               'diag_table_file - File that specifies the outputs of the experiment.\n'
#               'month_start - Index of month at which this job starts the simulation (starting with 1).\n'
#               'month_duration - How many months to run simulation for in this job.')
#     if len(sys.argv) == 5:
#         start_time = datetime.utcnow()
#         run_job(sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
#         end_time = datetime.utcnow()
#         # Print out how long it takes, so saves when running with slurm, to output txt file
#         print(f"Simulation Start Month: {int(sys.argv[3])}")
#         print(f"Simulation Length/Months: {int(sys.argv[4])}")
#         print(f"Start Time: {start_time.strftime('%B %d %Y - %H:%M:%S')}")
#         print(f"End Time: {end_time.strftime('%B %d %Y - %H:%M:%S')}")
#         print(f"Duration/Seconds: {int(np.round((end_time-start_time).total_seconds()))}")
#     else:
#         raise ValueError(f"Only {len(sys.argv)} parameters provided but 5 expected.")
