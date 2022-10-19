#!/bin/bash
sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=$1  # make job name be the same as the python script without prefix and suffix.
#SBATCH --output=$GFDL_DATA/time$2.txt  # output to console saved as text file in data directory for this experiment
#SBATCH --error=$GFDL_DATA/error$2.txt    # errors to console saved as text file in data directory for this experiment
#SBATCH --time=$9 # maximum walltime for the job
#SBATCH --nodes=$5 # specify number of nodes
#SBATCH --ntasks-per-node=$6 # specify number of processors per node
#SBATCH --mail-type=END # send email at job completion
#SBATCH --mail-user=$USER@st-andrews.ac.uk # email address
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

# Run python script for experiment and record how long it takes
python $(dirname "$BASH_SOURCE")/base.py $7 $8 $2 $3   # run job

# Move txt documents containing erros and printed info to experiment data folder
mkdir -p $GFDL_DATA/$1/console_output
mv $GFDL_DATA/time$2.txt $GFDL_DATA/$1/console_output/time$2.txt
mv $GFDL_DATA/error$2.txt $GFDL_DATA/$1/console_output/error$2.txt

exit 0
EOT
