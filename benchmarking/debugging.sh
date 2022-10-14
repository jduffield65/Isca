# Just exports some variables to terminal which are needed to run the
# benchmarking/experiments/held_suarez_test_case.py and
# benchmarking/record_time.py scripts
# This is to do not in SLURM but just live in the interface when first log in to kennedy.

export SLURM_JOB_NAME=socrates_ape_aquaplanet
export RES=T21
export RUN_NO=1
export NMONTHS=1
export SLURM_NTASKS_PER_NODE=8  # Only can use 8 cores in this interface
export SLURM_JOB_PARTITION=None
export SLURM_NNODES=1
