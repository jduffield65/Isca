#!/bin/bash
sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=$(basename "$1" .py)  # make job name be the same as the python script without prefix and suffix.
#SBATCH --output=$9  # output to console saved as text file
#SBATCH --error=${10}    # errors to console saved as text file
#SBATCH --time=02:00:00 # maximum walltime for the job
#SBATCH --nodes=$4 # specify number of nodes
#SBATCH --ntasks-per-node=$5 # specify number of processors per node
#SBATCH --mail-type=END # send email at job completion
#SBATCH --mail-user=jamd1@st-andrews.ac.uk # email address
#SBATCH --partition=$3 # queue to run on

# Input parameters
# $1 - python script indicating experiment to run e.g. experiments/held_suarez_test_case.py
# $2 - index giving the run number for this experiment
# $3 - partition indicating which queue to run experiment on e.g. 'debug' or 'singlenode'
# $4 - n_nodes indicating number of nodes to run experiment on.
# $5 - n_cores indicating the number of cores to use per node.
# $6 - res indicating horizontal resolution to use for the experiment e.g. 'T42'
# $7 - n_months indicates duration of the simulation in months.
# $8 - csv file to save task times e.g. 'task_times.csv'
# $9 - name of txt file where stuff printed to console saved e.g. console_output.txt
# ${10} - name of txt file where errors printed to console saved e.g. console_error.txt

# Save some of input parameters for use in python scripts
export NMONTHS=$7
export RES=$6
export RUN_NO=$2

# Run python script for experiment and record how long it takes
cd /home/jamd1/isca_jobs/benchmarking/
python record_time.py $8 START
python $1
python record_time.py $8 END

exit 0
EOT
