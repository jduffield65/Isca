# First Run
Here I go through a step by step example of running a slab ocean experiment, with present day initialization.
This is the same experiment as for the [Archer2 tutorial](../first_run.md).

## Step 1 - Login
First [login](https://icl-rcs-user-guide.readthedocs.io/en/latest/hpc/getting-started/using-ssh/) to the Imperial HPC.

## Step 2 - Load the CESM environment

Initialise Conda and activate the [`cesm` environment](./installation.md#create-the-conda-environment):

```bash
eval "$("$HOME/miniforge3/bin/conda" shell.bash hook)"
conda activate cesm
```

Source the [CESM environment-setup script](./installation.md#update-the-environment-setup-script) to load the required modules and set the installation paths:

```bash
CESM215_ENV="$HOME/cesm/config_files/cesm2.1.5_cx3p2.bashrc"
source "$CESM215_ENV"
```

## Step 3 - Create a case

Create the parent directory for your cases:

```bash
mkdir -p "$CESM215_MAIN/runs"
```

Set the case directory and choose whether to use Express access:

```bash
case_name="$CESM215_MAIN/runs/e.e20.ETEST.f19_g17.test"
run_express=false
```

Create the case using the `ETEST` component set and `f19_g17` resolution. If using Express access, replace `exp-?????` with your Express project ID before running:

```bash
if [[ "$run_express" == "true" ]]; then
    machine_name=cx3p2exp
    project_id="exp-?????"

    "$CIME/scripts/create_newcase" \
        --case "$case_name" \
        --compset ETEST \
        --res f19_g17 \
        --mach "$machine_name" \
        --project "$project_id" \
        --run-unsupported
else
    machine_name=cx3p2

    "$CIME/scripts/create_newcase" \
        --case "$case_name" \
        --compset ETEST \
        --res f19_g17 \
        --mach "$machine_name" \
        --run-unsupported
fi
```

### Express machine

To use Express access, set `run_express=true` and replace `exp-?????` with your own Express project ID before running the commands. This selects the supplied `cx3p2exp` machine configuration and passes the project ID through `--project`.

For standard access, leave `run_express=false`. This selects `cx3p2`; you do not need to provide a project ID or explicitly specify a queue in the case-creation command.

Express access is a paid service and requires access to an Express account. See the [Imperial RCS Express Access documentation](https://icl-rcs-user-guide.readthedocs.io/en/latest/hpc/legacy-systems/cx3-legacy/#express-access) for account and funding details.

## Step 4 - Set the processor layout

Move into the case directory and configure the case to use 32 MPI tasks, with all components starting at MPI rank 0:

```bash
cd "$case_name"
./xmlchange NTASKS=32
./xmlchange ROOTPE=0
```

The 32-task layout above works for this test case, but other processor counts and configurations may also be suitable. See Imperial's [job-sizing guidance](https://icl-rcs-user-guide.readthedocs.io/en/latest/hpc/queues/job-sizing-guidance/) for supported resource requests and node limits. You do not need to specify a queue explicitly: PBS routes the job according to the resources requested.

??? danger "Must be done before setup"
    Set the processor layout before running `./case.setup` in the [next step](#step-5-setup).

    If you change these settings after running setup, reset the case setup to apply the updated layout:

    ```bash
    ./case.setup --reset
    ```

## Step 5 - Setup

Run the case setup, following the same procedure as on [ARCHER2](../first_run.md#step-4-setup):

```bash
cd "$case_name"
./case.setup
```

## Step 6 - Customize namelists
As for the [ARCHER2 example](../first_run.md#step-5-customize-namelists), for this example, we keep the default experiment parameters so don't need to 
[change the namelist files](../basics.md#step-5-customize-namelists).

## Step 7 - Build
From the case directory, submit the [build script](./installation.md#update-the-build-script) to the batch scheduler:

```bash
qsub "$CESM215_BUILD_SCRIPT"
```

Monitor the case status:

```bash
tail -f CaseStatus
```

The `tail -f` command displays the last few lines of `CaseStatus` and continues to display new entries as they are added. Press `Ctrl+C` to stop monitoring (when it shows `case.build success`); this does not cancel the submitted build job.

??? note "PBS submission"
    Unlike the [ARCHER2 instructions](../first_run.md#step-6-build), submit the build using `qsub` rather than running `./case.build` directly on the login node. This runs the build on a compute node and avoids consuming excessive resources on the shared login node.

## Step 8 - Download input data

From the case directory, download the input data required by the case:

```bash
./check_input_data --download
```

??? "Sanity check"
    As the command runs, it may report missing input files before attempting to download them. Successful downloads are followed by `SUCCESS`, for example:

    ```text
    Model clm missing file fsnowoptics = '/rds/general/user/jduffiel/home/cesm/data/inputdata/lnd/clm2/snicardata/snicar_optics_5bnd_c090915.nc'
    Trying to download file: 'lnd/clm2/snicardata/snicar_optics_5bnd_c090915.nc' to path '/rds/general/user/jduffiel/home/cesm/data/inputdata/lnd/clm2/snicardata/snicar_optics_5bnd_c090915.nc' using SVN protocol.
    SUCCESS
    ```

    The paths will reflect your own home directory.

    Once the download command has finished, check again without requesting downloads:

    ```bash
    ./check_input_data
    ```

    There should be no messages reporting missing input files. For example:

    ```text
    Setting resource.RLIMIT_STACK to -1 from (8388608, -1)
    Loading input file list: 'Buildconf/clm.input_data_list'
    Loading input file list: 'Buildconf/cpl.input_data_list'
    Loading input file list: 'Buildconf/cice.input_data_list'
    Loading input file list: 'Buildconf/docn.input_data_list'
    Loading input file list: 'Buildconf/mosart.input_data_list'
    Loading input file list: 'Buildconf/cam.input_data_list'
    ```

## Step 9 - Run model

Set the run length to two model months and request a wall-clock limit of three hours, in the same way as on [ARCHER2](../first_run.md#step-8-run-model):

```bash
cd "$case_name"

# Remove explicit queue settings when using standard access
if [[ "$run_express" == "false" ]]; then
    sed -i '/JOB_QUEUE/d' env_batch.xml
fi

./xmlchange JOB_WALLCLOCK_TIME=03:00:00
./xmlchange STOP_OPTION=nmonths
./xmlchange STOP_N=2
```

??? note "Express machine"
    As described in the [Express machine section](#express-machine), `run_express=false` selects standard access. In this configuration, PBS selects the queue according to the requested resources, so remove all lines containing `JOB_QUEUE` from `env_batch.xml` rather than specifying a queue explicitly.

    When `run_express=true`, the command leaves the Express queue settings unchanged.

Now we can run the job in the same way as on [ARCHER2](../first_run.md#step-8-run-model):

```bash
./case.submit
```

??? "Sanity check: nodes and cores"
    Before submitting, check that the generated run script requests a single node with 32 cores:

    ```bash
    grep -n '^#PBS' .case.run
    ```

    Look for the following resource request; the line number may differ:

    ```text
    8:#PBS  -l select=1:ncpus=32:mpiprocs=32:ompthreads=1:mem=400GB:cpu_type=rome
    ```

    Then preview the run configuration:

    ```bash
    ./preview_run
    ```

    The `CASE INFO` section should show one node, 32 total tasks and one thread per task:

    ```text
    CASE INFO:
      nodes: 1
      total tasks: 32
      tasks per node: 32
      thread count: 1
    ```

    Check that these values match before submitting the case.

## Model Output
The [model output](./basics.md#model-output) should be located in 
`$HOME/cesm/output/archive/` if successful with files 
corresponding to each of the [model components](./basics.md#code-components), as well as restart and
log files.

The PBS output file should be stored within the `$case_name` directory with a name something like `run.e.e20.ETEST.f19_g17.test.o4635723`.

Timing files are created and are `$case_name/timing/cesm_timing.$CASE.$date` as for [ARCHER2](../first_run.md#timing).

Log files are created and are `$HOME/cesm/output/archive/$CASE/logs` as on [ARCHER2](../first_run.md#log).

A first plot can be made by transferring data to JASMIN as on [ARCHER2](../first_run.md#first-plot).
