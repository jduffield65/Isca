#!/bin/bash
qsub <<EOT
#!/bin/bash
#PBS -N $1
#PBS -o ${11}/time$2.txt
#PBS -e ${11}/error$2.txt
#PBS -l walltime=$9
#PBS -l select=$5:ncpus=$6:mpiprocs=$6${14}
#PBS -V

# Input parameters
# $1 - Name of job
# $2 - Starting month index, starting at 1
# $3 - Number of months to run per job
# $4 - Unused queue placeholder (Imperial routes jobs automatically)
# $5 - Number of nodes
# $6 - Number of cores per node
# $7 - Namelist file
# $8 - Diagnostic output file
# $9 - Maximum allowed walltime
# ${10} - Python script path to run
# ${11} - Directory for saving console output
# ${12} - Unused node-list placeholder (PBS selects nodes)
# ${13} - Unused dependency placeholder
# ${14} - Optional PBS resource suffix, e.g. :mem=16gb

python "${10}" "$7" "$8" "$2" "$3"
EOT
