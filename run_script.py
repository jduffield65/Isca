from isca_tools import run_experiment, run_job
import sys
import datetime
import os
import numpy as np

if len(sys.argv) == 1:
    # When calling the script with no arguments, it just runs the experiment.
    jobs_dir = os.path.join(os.environ['HOME'], 'Isca', 'jobs')  # all jobs saved here - CHANGE FOR EACH EXPERIMENT
    exp_dir = os.path.join(jobs_dir, 'aquaplanet/vary_co2')  # specific experiment - CHANGE FOR EACH EXPERIMENT

    namelist_file = os.path.join(exp_dir, 'd5_namelist.nml')
    diag_table_file = os.path.join(exp_dir, 'd5_diag_table')
    run_experiment(namelist_file, diag_table_file, slurm=True)

elif len(sys.argv) == 5:
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
    raise ValueError(f"{len(sys.argv)-1} parameters provided but 0 or 4 expected.")
