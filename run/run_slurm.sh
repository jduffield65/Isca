#!/bin/bash
sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=$1  # make job name be the same as the python script without prefix and suffix.
#SBATCH --output=console_output.txt  # output to console saved as text file
#SBATCH --error=console_error.txt    # errors to console saved as text file
#SBATCH --time=02:00:00 # maximum walltime for the job
#SBATCH --nodes=$5 # specify number of nodes
#SBATCH --ntasks-per-node=$6 # specify number of processors per node
#SBATCH --mail-type=END # send email at job completion
#SBATCH --mail-user=jamd1@st-andrews.ac.uk # email address
#SBATCH --partition=$4 # queue to run on

# Input parameters
# $1 - Name of job
# $2 - Run number for this job, starting at 0.
# $3 - True or False, whether to restart experiment or not
# $4 - partition indicating which queue to run experiment on e.g. 'debug' or 'singlenode'
# $5 - n_nodes indicating number of nodes to run experiment on.
# $6 - n_cores indicating the number of cores to use per node.
# $7 - res indicating horizontal resolution to use for the experiment e.g. 'T42'
# $8 - namelist file
# $9 - diagnostic output file indicating what to save

# Run python script for experiment and record how long it takes
cd /home/jamd1/isca_jobs/run/
./run_isca --restart $3 --num_cores $6 --res $7 --namelist $8 --diag $9

# Rename output folders so no overwriting issues later
mv $GFDL_DATA/$1/run0001 $GFDL_DATA/$1/run$2

# Save txt file outputs to same output folder as rest of data
mv console_output.txt $GFDL_DATA/$1/run$2/console_output.txt
mv console_error.txt $GFDL_DATA/$1/run$2/console_error.txt

exit 0
EOT
