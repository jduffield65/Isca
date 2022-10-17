# Getting Started
Isca needs to be run on a powerful computer, so to use it, you first need to `ssh`
into a high performance computer e.g. [kennedy](../hpc_basics/kennedy.md#login).

## Installation
### Copy Source Code ([Step 2](https://github.com/ExeClim/Isca#getting-started))
- Once logged into the high performance computer, run </br> 
`git clone https://github.com/ExeClim/Isca`
- Then change the current directory </br>
`cd Isca`

### Create *CONDA* environment ([Step 3](https://github.com/ExeClim/Isca#getting-started))
- In terminal, run `conda env create -f ci/environment-py3.9.yml`. This takes around 5 minutes.
- Then, [run](../hpc_basics/kennedy.md#create-environment) `chmod u+x /gpfs1/apps/conda/$USER/conda/envs/*/bin/*` </br>
to make sure it is using the correct python version.

- Now you should be able to activate the *CONDA* environment: `conda activate isca_env`. I would probably
double check it is using the correct python version by running `which python`. </br>
It should return `/gpfs1/apps/conda/jamd1/conda/envs/isca_env/bin/python`.

### Install in development mode ([Step 4](https://github.com/ExeClim/Isca#getting-started))
- Change directory to where the *setup.py* file is: </br> `cd /gpfs1/home/jamd1/Isca/src/extra/python` </br>
- Install the `isca` python module in development mode: </br> `pip install -e .`

### Set environment and where Isca saves data
- Using [FileZilla](../hpc_basics/kennedy.md#file-transfer), create the directories 
*/gpfs1/scratch/other/\$USER/isca_work* and */gpfs1/scratch/other/\$USER/isca_output*.
- Add the following to the */gpfs1/home/jamd1/.bashrc* file, to indicate the Isca environment and where to 
save data. </br>
```bash
# directory of the Isca source code
export GFDL_BASE=/gpfs1/home/$USER/Isca
# "environment" configuration for emps-gv4
export GFDL_ENV=ubuntu_conda
# temporary working directory used in running the model
export GFDL_WORK=/gpfs1/scratch/other/$USER/isca_work
# directory for storing model output
export GFDL_DATA=/gpfs1/scratch/other/$USER/isca_output
```
- The full *.bashrc* file now looks like this: </br>
![image.png](../images/Isca/bashrc.png){width="500"}
- Exit the `ssh` and log back in for the new *.bashrc* script to take effect.
- This is close to the 
[Compiling for the first time section](https://github.com/ExeClim/Isca#compiling-for-the-first-time).

### Add Fortran compiler flags
- Check version of `gfortran` installed using `conda list -n isca_env`: </br>
```bash
gfortran                  10.4.0              h0c96582_10    conda-forge
gfortran_impl_linux-64    10.4.0              h44b2e72_16    conda-forge
gfortran_linux-64         10.4.0              h69d5af5_10    conda-forge
```
- If version is 10 or greater, the file </br> 
*/gpfs1/home/$USER/Isca/src/extra/python/isca/templates/mkmf.template.ubuntu_conda* </br>
needs to be changed, with the following: </br>
`-w -fallow-argument-mismatch -fallow-invalid-boz` </br>
added to the existing FFlags.
- The final file should look like this: </br>
![image.png](../images/Isca/ubuntu_conda.png){width="500"}

##Held Suarez
A simple experiment to run to check that the installation has worked is the 
[*Held Suarez*](https://github.com/ExeClim/Isca/blob/master/exp/test_cases/held_suarez/held_suarez_test_case.py) 
experiment.

To run this, you can do the following:

- [Login](kennedy.md#login) to kennedy.
- Run `conda activate isca_env` to activate the Isca *CONDA* environment.
- Create the following script, titled `held_suarez_run.sh`, and transfer it to a suitable location
on kennedy.
```bash
#!/bin/bash
#SBATCH --job-name=held_suarez_test
#SBATCH --output="held_suarez_test_output.txt"
#SBATCH --error="held_suarez_test_error.txt"
#SBATCH --time=02:00:00 # maximum walltime for the job
#SBATCH --nodes=1 # specify number of nodes
#SBATCH --ntasks-per-node=16 # specify number of processors per node
#SBATCH --mail-type=END # send email at job completion
#SBATCH --mail-user=$USER@st-andrews.ac.uk # email address
#SBATCH --partition=debug # queue to run on

python $GFDL_BASE/exp/test_cases/held_suarez/held_suarez_test_case.py
```
    - This will save the things printed to the console in `held_suarez_test_error.txt`, which
    will just be added to the same directory that the script is in. 
    You may want to change this to put it in another location.
    - This will use the debug queue as it should only take around 20 minutes to run.
- Submit the script: `sbatch held_suarez_run.sh` (make sure you are in the same directory as the file first).

As well as the `held_suarez_test_output.txt` and `held_suarez_test_error.txt` files, this should also generate
some output data in the folder </br> `/gpfs1/scratch/jamd1/isca_output/held_suarez_default/run0001`:

![image.png](../images/Isca/held_suarez_output.png){width="500"}
