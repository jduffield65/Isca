# Basics

This page just gives some useful information regarding the Community Earth System Model.

## Resources
* [Discussion Forum](https://bb.cgd.ucar.edu/cesm/)
* [Tutorial](https://ncar.github.io/CESM-Tutorial/README.html) (particularly useful)
* [Practical](https://www2.cesm.ucar.edu/events/tutorials/2022/files/)
* [Analysis Example](https://brian-rose.github.io/ClimateLaboratoryBook/courseware/introducing-cesm.html#)
* [My code](../code/cesm/load.md) to help with analysis 

## Paths on ARCHER2
Paths pointing to different parts of the CESM model are [given below](https://ncar.github.io/CESM-Tutorial/notebooks/basics/cesm_workspaces.html).

The paths that are not case specific should be available to use in ARCHER2 after [loading the CESM module](#step-2-load-modules) through
`module load CESM2/2.1.3`.

![](https://ncar.github.io/CESM-Tutorial/_images/CESM2_Workspaces_Detail.png)


* `$CESM_ROOT = /work/$GROUP/$GROUP/$USER/cesm/CESM2.1.3` <br/>
This is the overall directory containing all CESM stuff (for me, `$GROUP=n02` and `$USER=jamd`, giving: `$CESM_ROOT = /work/n02/n02/jamd/cesm/CESM2.1.3`)
* `$CESMDATA = $CESM_ROOT/cesm_inputdata`<br/>Path to input data.
* `$SRCROOT = $CESM_ROOT/my_cesm_sandbox`<br/>Path to CESM source code.
* `$CIMEROOT = $CESM_ROOT/my_cesm_sandbox/cime`<br/>Path to the 
[Common Infrastructure for Modeling Earth](https://esmci.github.io/cime/versions/master/html/what_cime/index.html) part of the source code.
* `$CASEROOT = $CESM_ROOT/runs/$CASE/`<br/>Path to a particular experiment, as indicated by `$CASE`.
* `$EXEROOT = $CESM_ROOT/runs/$CASE/bld/`<br/>Path to the build directories of a particular experiment.
* `$RUNDIR = $CESM_ROOT/runs/$CASE/run/`<br/>Path to the run directories of a particular experiment.
* `$DOUT_S_ROOT = $CESM_ROOT/archive/$CASE/`<br/>Path to the archive3d model output.

## Code Components
CESM consists of several sub models [listed below](https://ncar.github.io/CESM-Tutorial/notebooks/basics/code/cesm_code_explore.html#step-3-examine-the-cesm-components-area). 
Output data is saved in a different location for each.

![CESM directories](https://ncar.github.io/CESM-Tutorial/_images/CESM2_Code_Components_List.png)

## Workflow
Here, I go through the [general steps](https://ncar.github.io/CESM-Tutorial/notebooks/basics/basics_overview.html) 
for running an CESM experiment on ARCHER2.

At any point, the [file](https://ncar.github.io/CESM-Tutorial/notebooks/basics/cesm_workflow/checking_jobs_and_status.html#casestatus-file) 
`$CESM_ROOT/runs/$CASE/CaseStatus` records commands run and whether each step has been successful.

#### Step 1 - Login
First, you need to login to ARCHER2 [using ssh](../hpc_basics/kennedy.md#login).

#### Step 2 - Load modules
Each time you login to ARCHER2, you need to load the python and CESM modules:

```bash
module load cray-python
module load CESM2/2.1.3
```

#### Step 3 - Create a case
Create a new case using:
```bash
create_newcase --case $CESM_ROOT/runs/CASE --compset COMPSET --res RES --project PROJECT
```

where for me, `PROJECT=n02-GLOBALEX`.

??? note "Casename convection"
    Experiment casenames have a [convention](https://www.cesm.ucar.edu/models/cesm2/naming-conventions#casenames):
    
    ```
    `<compset char>.<code base>.<compset shortname>.<res shortname>[.opt_desc_string].<nnn>[opt_char]`
    ```

    An example `$CASE = e.e20.ETEST.f19_g17.test`.

??? note "Compsets"
    Compsets are listed [here](https://docs.cesm.ucar.edu/models/cesm2/config/compsets.html) and the names are 
    [explained here](https://ncar.github.io/CESM-Tutorial/notebooks/basics/cesm_workflow/create_newcase.html#compset):

    ![](https://ncar.github.io/CESM-Tutorial/_images/CESM2_Create_Newcase_Compset.png)

??? note "Resolution"
    Resolutions are listed [here](https://docs.cesm.ucar.edu/models/cesm2/config/grids.html) and the names are 
    [explained here](https://ncar.github.io/CESM-Tutorial/notebooks/basics/cesm_workflow/create_newcase.html#resolution):

    ![](https://ncar.github.io/CESM-Tutorial/_images/CESM2_Create_Newcase_Resolution.png)

#### Step 4 - Setup
Navigate to `$CASEROOT` and then invoke using `case.setup`, e.g. for `$CASE = e.e20.ETEST.f19_g17.test`:
```bash
cd $CESM_ROOT/runs/e.e20.ETEST.f19_g17.test
./case.setup
```

#### Step 5 - Customize namelists
At this stage, you need to specify the details of the experiment by [modifying the namelists](#namelists) and/or 
customizing the output.

#### Step 6 - Build
Next, the executable should be built through `case.build`:
```bash
./case.build
```
Again, this should be run from `$CASEROOT`.

#### Step 7 - Download input data
Next, the required input data, from which to start the model, should be downloaded:
```bash
./check_input_data --download
```
Again, this should be run from `$CASEROOT`.

#### Step 8 - Run model
Finally, you can run the model with `case.submit`:
```bash
./case.submit
```
Again, this should be run from `$CASEROOT`.


## Model Output
If the model run is successful, the [CESM netcdf output](https://ncar.github.io/CESM-Tutorial/notebooks/basics/cesm_workflow/model_output.html) 
history files are automatically moved to the short term archive, located at `$DOUT_S_ROOT`. Otherwise, they are in `$RUNDIR`.

Output files should be moved somewhere else for more long term storage. This is likely to be JASMIN, and the 
files can be transferred with globus.

[Timing information](https://ncar.github.io/CESM-Tutorial/notebooks/modifications/xml/run_length/timing_files.html) 
is saved as `$CASEROOT/timing/cesm_timing.$CASE.$date`. The model throughput is the estimated number 
of model years that you can run in a wallclock day.

The `cpl.log` file at `$CESM_ROOT/archive/$CASE/logs` indicates whether successful. 
[It should](https://docs.archer2.ac.uk/research-software/cesm213_run/#run-the-case) end with 
`SUCCESSFUL TERMINATION OF CPL7-cesm`.

## XML Modifications


## Namelists
[Namelists](https://ncar.github.io/CESM-Tutorial/notebooks/namelist/overview.html) 
can be modified through the `user_nl_xxx` files in `$CASEROOT`:

![](https://ncar.github.io/CESM-Tutorial/_images/CESM_directories_and_namelists.png))

This should be done after [setup](#step-4-setup) but before [build](#step-6-build). Note that the 
`_in` files only appear in `$CASEROOT` after `./case.build` and these should not be edited.

Optionally, can run `./preview_namelists` from `$CASEROOT` after editing namelists, but this is done anyway in 
`./case.build`