# Slurm

To run a job on [kennedy](kennedy.md), it must be submitted to a Slurm queue. To do this, you basically
modify a [*.sh* script](shell_scripting.md) with [some headers](#sbatch) indicating which node to submit the job to.

## Useful commands [🔗](https://slurm.schedmd.com/pdfs/summary.pdf)  
### Monitoring jobs  
**These can be used to monitor the current state of the queues and your jobs.**

- `squeue -u jamd1` - shows all jobs submitted by the username `jamd1`  
- `scancel 158329` - cancels job with id `158329`. Job ID is shown with the above command or `squeue`  
- `scancel -u jamd1` - cancels all jobs by the username `jamd1`  
- `squeue -t RUNNING` - shows all jobs that are currently running  
- `squeue -p debug` - shows all jobs queued and running for the partition called `debug`  
### SBATCH
**These must be added to top of *.sh* script, just below the *shebang* line.** [🔗](https://slurm.schedmd.com/sbatch.html)

- `#SBATCH --job-name=` - name of job e.g. `test`  
- `#SBATCH --output=` - file where things printed to console are saved e.g. `output.txt`  
- `#SBATCH --error=` - file where any errors are saved e.g. `error.txt`  
- `#SBATCH --time=` - maximum walltime for the job e.g. `01:00:00` for 1 hour or `01:00` for 1 minute.  
- `#SBATCH --nodes=` - number of nodes used e.g. `1`  
- `#SBATCH --ntasks-per-node=` - Number of processors per node e.g. `16`.
- `#SBATCH --partition=` - the queue to submit a job to. </br>
??? note "Options for [kennedy](kennedy.md)"
      - **singlenode** </br> Jobs requiring one node, with a maximum of 32 cores. Maximum run-time 30 days.
      - **parallel** </br> Jobs running parallel across multiple nodes. 
      There is no upper limit on the number of nodes (other than how many there are), but remember that you are using a shared 
      resource. Maximum run-time 30 days.
      - **gpu** </br> Jobs requesting one or both of kennedy's GPU nodes.
      Maximum run-time 30 days. 
      Note that you still need to request the GPUs with the `--gres` flag. 
      The following line would request both GPUs on a node:
      `#SBATCH --gres=gpu:2`
      - **debug** </br> Small and short jobs, usually meant for tests or debugging. 
      This partition is limited to one node and a maximum of two hours run-time. </br>
      *I would use this one for starting off with and running short scripts*.
- `#SBATCH --mail-type=` - indicates when you would like to receive email notifications, `ALL` or `END`.  
- `#SBATCH --mail-user=` - email address e.g. `jamd1@st-andrews.ac.uk`  

## Submitting a job
To submit a script which prints 'hello world' to the `debug` queue with
16 tasks per node with the output saved to the file `example_output.txt`, the `example.sh` script would look like this:

```bash
#!/bin/bash
#SBATCH --job-name=example
#SBATCH --output="example_output.txt"
#SBATCH --error="example_error.txt"
#SBATCH --time=01:00 # maximum walltime for the job
#SBATCH --nodes=1 # specify number of nodes
#SBATCH --ntasks-per-node=16 # specify number of processors per node
#SBATCH --mail-type=END # send email at job completion
#SBATCH --mail-user=$USER@st-andrews.ac.uk # email address
#SBATCH --partition=debug # queue to run on

echo hello world
```

To submit this job, you then need to [login](kennedy.md#login) to kennedy and 
[transfer `example.sh`](kennedy.md#file-transfer) from your local computer to kennedy.

Then make sure your current directory contains the `example.sh` file and run `sbatch example.sh`. 
This should produce the files `example_output.txt` and `example_error.txt` in your current directory
when it is finished running (use `squeue -p debug` to see its place in the queue).

### Multiple jobs with environmental variables
The submission of multiple instances of the same job but using different numbers of cores can be achieved 
through an overarching python script which then runs the shell (`.sh`) script with different input 
parameters.

For example, if the three files below are all in the current directory then running `python example_run.py`
will send two jobs to the `debug` queue, both with job name `example` but one on 8 cores per node
and one on 16.

???+ note "Using *CONDA*"
    Because this uses python, you probably want to run it with a relatively modern version of python.
    To do this, [activate a *CONDA* environment](kennedy.md#conda) before running `python example_run.py`.

=== "example_run.py"

    ``` python
    from os import system
    job_name = 'example'
    shell_script = 'example.sh'
    n_nodes = 1
    output_text = 'hello world'
    for n_cores in [8, 16]:
	    system(f'bash {shell_script} {job_name} {n_nodes} {n_cores} {output_text')
    ```

=== "example.sh"

    ``` bash
    #!/bin/bash
    sbatch <<EOT
    #!/bin/bash
    #SBATCH --job-name=$1
    #SBATCH --output="outFile"$3".txt"
    #SBATCH --error="errFile"$3".txt"
    #SBATCH --time=01:00 # maximum walltime for the job
    #SBATCH --nodes=$2 # specify number of nodes
    #SBATCH --ntasks-per-node=$3 # specify number of processors per node
    #SBATCH --mail-type=END # send email at job completion
    #SBATCH --mail-user=$USER@st-andrews.ac.uk # email address
    #SBATCH --partition=debug # queue to run on

    export OUTPUT_TEXT=$4   # export so can be used by python script
    python example_print.py
    exit 0
    EOT
    ```

=== "example_print.py"

    ```python
    import os
    print(f'Job Name: {os.environ['SLURM_JOB_NAME']}')
    print(f'Number of nodes: {int(os.environ['SLURM_NNODES'])}')
    print(f'Number of tasks per node: {int(os.environ['SLURM_NTASKS_PER_NODE'])}')
    print(f'Output text: {os.environ['OUTPUT_TEXT']}')
    ```

??? note "Wrapper in `.sh` script"
    The `example.sh` script is slightly different when it is called from a python script.
    It needs a [wrapper](https://stackoverflow.com/questions/36279200/sbatch-pass-job-name-as-input-argument)
    which is what the `EOT` stuff is.

When both jobs have been completed, this will then produce 4 files in the current directory, 
`outFile8.txt`, `outFile16.txt`, `errFile8.txt` and `errFile16.txt`.

=== "outFile8.txt"

    ```
    Job Name: example
    Number of nodes: 1
    Number of tasks per node: 8
    Output text: hello world
    ```
=== "outFile16.txt"

    ```
    Job Name: example
    Number of nodes: 1
    Number of tasks per node: 16
    Output text: hello world
    ```

This example shows that the `#SBATCH` commands in the `example.sh` script produces 
[environmental variables](https://slurm.schedmd.com/sbatch.html#SECTION_OUTPUT-ENVIRONMENT-VARIABLES)
with a `SLURM` prefix which can then be accessed. These variables cannot be accessed from the `example.sh` script
itself though e.g. `echo $SLURM_SLURM_JOB_NAME` would not print anything if it was included in
`example.sh`.

## Debugging
It can be annoying if you are running a small job to submit it to a `SLURM` queue and then wait for 
it to get to the front.

As an alternative, you can just [login](kennedy.md#login) to kennedy and run the `example_print.py` script.
To do this, you first need to set the environmental variables that the script uses by running
`source example_params.sh` where the `example_params.sh` script is given below:

```bash
export SLURM_JOB_NAME=example
export SLURM_NTASKS_PER_NODE=8
export OUTPUT_TEXT=hello world
```

Doing this, you are limited to one node and 8 tasks per node, so you may get problems 
with Isca if you try to run a simulation with more than 8 tasks per node.