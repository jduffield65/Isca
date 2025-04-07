#!/bin/bash
sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=$1  # make job name be the same as the python script without prefix and suffix.
#SBATCH --output=$HOME/Isca/jobs/jasmin/console_output/$1/time.txt  # output to console saved as text file in data directory for this experiment (May need to create directory first)
#SBATCH --error=$HOME/Isca/jobs/jasmin/console_output/$1/error.txt    # errors to console saved as text file in data directory for this experiment
#SBATCH --time=$9 # maximum walltime for the job
#SBATCH --nodes=$5 # specify number of nodes
#SBATCH --ntasks=$6 # specify number of processors per node
#SBATCH --cpus-per-task=$7
#SBATCH --mem=$8G
#SBATCH --partition=$4 # queue to run on

# Input parameters
# $1 - Name of job
# $2 - Starting month index, starting at 1
# $3 - Number of months to run per job
# $4 - partition indicating which queue to run experiment on e.g. 'debug' or 'singlenode'
# $5 - n_nodes indicating number of nodes to run experiment on.
# $6 - n_cores indicating the number of cores to use per node.
# $7 - namelist file
# $8 - diagnostic output file indicating what to save
# $9 - maximum allowed walltime to run job
# ${10} - python script path to run i.e. the path of run_job_script.py
# ${11} - Node to submit to: kennedy20, kennedy21 or kennedy22

# Run the base.py script for experiment and record how long it takes
source $HOME/miniforge3/bin/activate    # load miniforge3 module, where conda installed
conda activate myenv                                # Load conda environment which uses Isca on JASMIN to run script
python ${10} $7 $8 $2 $3   # run job

# Move txt documents containing errors and printed info to experiment data folder
#mkdir -p $HOME/jasmin_jobs/$1
#mv $GFDL_DATA/time$2.txt $GFDL_DATA/$1/console_output/time$2.txt
#mv $GFDL_DATA/error$2.txt $GFDL_DATA/$1/console_output/error$2.txt

exit 0
EOT
