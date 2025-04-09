sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=$1                   # name of job in slurm queue
#SBATCH --output=$HOME/Isca/jobs/jasmin/console_output/$1/out.txt    # output to console saved as text file in directory for this experiment
#SBATCH --error=$HOME/Isca/jobs/jasmin/console_output/$1/error.txt   # errors to console saved as text file in directory for this experiment
#SBATCH --time=$2                       # maximum walltime for the job
#SBATCH --ntasks=$3                     # specify number of processors per node
#SBATCH --cpus-per-task=$4
#SBATCH --mem=$5G
#SBATCH --partition=$6                  # queue to run on
#SBATCH --qos=$7                        # quality of service
#SBATCH --account=$8                    # Username on JASMIN

# Input parameters
# $1 - Name of job
# $2 - Maximum wall time for your job in format hh:mm:ss
# $3 - Number of tasks to run (usually 1 for a single script).
# $4 - How many CPUs to allocate per task.
# $5 - Memory to allocate for the job in GB.
# $6 - Specifies the partition for the job e.g. test, short-serial-4hr, short-serial, par-single.
# $7 - Quality of service, examples include debug, short and standard.
# $8 - Account to use for submitting jobs.
# $9 - Name of the conda environment on JASMIN to use.
# ${10} - Path of python script to run.
# ${11} - Arguments to be passed to python script (if multiple, will also be ${12}, ${13}, ...)

# Run the base.py script for experiment and record how long it takes
source $HOME/miniforge3/bin/activate    # load miniforge3 module, where conda installed
conda activate $9                       # Load conda environment on JASMIN to run script
python ${10} ${@:11}                    # run python script with all but first 10 arguments as input to script

exit 0
EOT
