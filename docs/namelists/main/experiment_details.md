# Experiment details

The `experiment_details` namelist contains information on how to run the simulation, most of which is 
relevant for submitting jobs to *Slurm*. It is an addition I made to the namelists in the *Isca* source code, and 
all options must be specified to run the simulation using `isca_tools`.

## Options
### `name`
*string*</br>
Name of experiment e.g. data saved in folder `$GFDL_DATA/{name}`.</br>
You can use `name='exp/run1'` to set create a new `exp` directory and then save the `run1` data within it. 
This may be useful for running similar experiments with different parameters.</br>
**Default:** N/A

### `input_dir`
*string*</br>
Directory containing any input files required for the experiment such as `namelist.nml`, `diag_table` or 
[`co2.nc`](../radiation/two_stream_gray.md#co2_file).</br>
**Default:** N/A

### `n_months_total`
*int*</br>
Total duration of simulation in months.</br>
**Default:** N/A

### `n_months_job`
*int*</br>
Approximate duration of each job of the simulation in months.</br>
E.g. if `n_months_total=12` and `n_months_job=6`, the experiment would be split up into 2 jobs each
of length 6 months.</br>
**Default:** N/A

### `n_nodes`
*int*</br>
Number of nodes to run job on (Slurm info).</br>
**Default:** N/A

### `n_cores`
*int*</br>
Number of cores for each node to run job on (Slurm info).</br>
Must be a power of $2$ so typically $8$, $16$ or $32$.</br>
**Default:** N/A

### `resolution`
*string*</br>
Horizontal resolution of experiment.</br>
Options are `T21`, `T42` or `T85`.</br>
**Default:** N/A

### `partition`
*string*</br>
*Slurm* queue to submit job to.</br>
**Default:** N/A

### `overwrite_data`
*bool*</br>
If this is `True` and data already exists in `$GFDL_DATA/{name}`, then it will be overwritten. </br>
If it is `False` and the data exists, an error will occur.</br>
If a previous run failed part way through, running the same experiment again with `overwrite_data=.false.` will
start using the last restart file created in that previous run.
Typically set to `False`. </br>
**Default:** N/A

### `compile`
*bool*</br>
If `True`, it will recompile the codebase before running the experiment.</br>
Typically set to `False`. </br>
Set this to `True` if you have made any changes to the underlying source code of Isca.</br>
**Default:** N/A

### `max_walltime`
*string*</br>
Maximum time that job can run on *Slurm*.</br> 
$1$ hour would be `'01:00:00'` and $30$ minutes would be `'30:00'`.</br>
**Default:** N/A

### `delete_restart_files`
*bool*</br>
If `True`, only the restart file for the final month will be kept.
Otherwise, a restart file will be generated for every month.</br>
**Default:** N/A

### `nodelist`
*string*</br>
Specify node to submit to.</br>
Options on `debug` partition on St Andrews kennedy are `kennedy20`, `kennedy21` or `kennedy22`.</br>
Number listed needs to match `n_nodes`.</br>
`kennedy[20-22]` would request the 3 nodes `20, 21, 22`.</br>
If not given, will just submit to default nodes. If multiple jobs, this will likely result in
jobs being submit in the wrong order, and an error occuring.</br>
**Default:** N/A