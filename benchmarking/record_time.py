import sys as sys
import os
import pandas as pd
from datetime import datetime

# This records the time and environmental variables into a csv file for a given Isca experiment.

reformatted_GMT_timestamp = datetime.utcnow().strftime('[%Y,%m,%d,%H,%M,%S]')
file_name = sys.argv[1]    # where csv file with task info is saved
task_status = sys.argv[2]  # either start or end

if task_status.lower() == 'start':
    # Initialise dictionary with all things that we want to save
    task_dict = {'Job Name': [os.environ['SLURM_JOB_NAME']],
                 'Run Number': [int(os.environ['RUN_NO'])],
                 'Partition': [os.environ['SLURM_JOB_PARTITION']],
                 'Resolution': [os.environ['RES']],
                 'Number of Nodes': [int(os.environ['SLURM_NNODES'])],
                 'Tasks Per Node': [int(os.environ['SLURM_NTASKS_PER_NODE'])],
                 'Simulation Duration (Months)': [int(os.environ['NMONTHS'])],
                 'Start Time (Y,M,D,H,M,S)': [reformatted_GMT_timestamp],
                 'End Time (Y,M,D,H,M,S)': [None],  # Set when call with 'end'
                 'Time/Seconds': [None]}            # Set when call with 'end'
    if os.path.exists(file_name):
        # If file exists, append to existing dataframe.
        df = pd.read_csv(file_name)
        df = pd.concat([df, pd.DataFrame(data=task_dict)], ignore_index=True)
    else:
        # If no file, create new dataframe.
        df = pd.DataFrame(data=task_dict)
else:
    # If task finished, only record end time and duration of task
    df = pd.read_csv(file_name)
    df.loc[df.shape[0] - 1, 'End Time (Y,M,D,H,M,S)'] = reformatted_GMT_timestamp
    start_time = datetime.strptime(df.loc[df.shape[0] - 1, 'Start Time (Y,M,D,H,M,S)'], "[%Y,%m,%d,%H,%M,%S]")
    end_time = datetime.strptime(reformatted_GMT_timestamp, "[%Y,%m,%d,%H,%M,%S]")
    df.loc[df.shape[0] - 1, 'Time/Seconds'] = int((end_time-start_time).total_seconds())
df.to_csv(file_name, index=False)  # save dataframe to a .csv file
