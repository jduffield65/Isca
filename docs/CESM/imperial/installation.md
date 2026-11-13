# Installation on Imperial HPC
Here, I go through the procedure for installing CESM version 2.1.5 on the Imperial HPC.

!!! note "This is a different version to 2.1.3 used for [ARCHER2](../installation.md)."

## Step 1 - Upload configuration files

You should have access to a set of configuration files for installing and running CESM2.1.5 on Imperial HPC. These include machine-specific porting settings, environment-setup files and model namelists.

Create a directory to store these files:

```bash
mkdir -p "$HOME/cesm/config_files"
```

Upload the supplied files and folders into this directory, preserving their names and folder structure. The resulting layout should be:

```text
$HOME/cesm/config_files/
├── cesm_environment.yml
├── cesm2_cx3p2.sh
├── cesm2.1.5_cx3p2.bashrc
├── cime_configuration/
└── namelists/
```

The files and folders are intended for the following parts of the setup:

| File or folder | Purpose |
| --- | --- |
| `cesm_environment.yml` | Software-environment definition for the CESM workflow. |
| `cesm2_cx3p2.sh` | Shell script associated with setting up or running CESM on the `cx3p2` system. |
| `cesm2.1.5_cx3p2.bashrc` | Shell-environment settings for CESM2.1.5 on `cx3p2`. |
| `cime_configuration/` | Machine-specific CIME configuration files used to adapt CESM to the HPC system. |
| `namelists/` | Model namelist files containing settings for the simulations. |

Ensure that `cime_configuration/` and `namelists/` contain their supplied files, rather than uploading only the empty folders.

At this stage, `$HOME/cesm/config_files` is a staging directory for the supplied configuration files, not the CESM source-code directory.

## Step 2 - Copy CIME configuration files

Create the `$HOME/.cime` directory, then copy the contents of the supplied `cime_configuration` folder into it:

```bash
mkdir -p "$HOME/.cime"
cp -r "$HOME/cesm/config_files/cime_configuration/." "$HOME/.cime/"
```

The configuration files should sit directly inside `$HOME/.cime/`, rather than inside a nested `cime_configuration` folder.

Check that the files have been copied:

```bash
ls -la "$HOME/.cime"
```

## Step 3 - Install Perl dependencies

Install `XML::LibXML` into a local Perl library directory at `$HOME/perl5`.

First, load the required modules:

```bash
module purge
module load tools/prod
module load ESMF/8.4.2-foss-2022a
module load Perl/5.34.1-GCCcore-11.3.0
```

Then install `XML::LibXML`:

```bash
perl "$(command -v cpanm)" --local-lib-contained "$HOME/perl5" XML::LibXML
```

This installs the Perl dependency locally; it does not build a new Perl interpreter. The `$HOME/perl5` directory is created during installation if it does not already exist.

??? note "Sanity check"
    As a sanity check, add the local Perl library directory to `PERL5LIB` and check that `XML::LibXML` can be loaded:

    ```bash
    export PERL5LIB="$HOME/perl5/lib/perl5${PERL5LIB:+:$PERL5LIB}"
    perl -MXML::LibXML -e 'print "XML::LibXML is available\n";'
    ```

    If the installation was successful, this should print:

    ```text
    XML::LibXML is available
    ```

## Step 4 - Update machine configuration paths

Edit `$HOME/.cime/config_machines.xml` to update the Perl library and CESM output paths. The file was created in [Step 2](#step-2-copy-cime-configuration-files).

### Update the Perl library path

Replace all four instances of:

```xml
<PERL5LIB>/rds/general/user/seastham/home/perl5/lib/perl5</PERL5LIB>
```

with:

```xml
<PERL5LIB>$HOME/perl5/lib/perl5</PERL5LIB>
```

This replaces the original user's directory with your own.

### Update the CESM output path

Replace all four instances of:

```xml
<CIME_OUTPUT_ROOT>/rds/general/user/$USER/home/cesmoutput</CIME_OUTPUT_ROOT>
```

with:

```xml
<CIME_OUTPUT_ROOT>$HOME/cesm/output</CIME_OUTPUT_ROOT>
```

Save your changes, then create the output directory:

```bash
mkdir -p "$HOME/cesm/output"
```

This directory will store the output produced by CESM.

## Step 5 - Create the Conda environment and download CESM2.1.5

### Create the Conda environment

Initialise Conda, assuming Miniforge is installed at `$HOME/miniforge3`:

```bash
eval "$("$HOME/miniforge3/bin/conda" shell.bash hook)"
```

Create the environment definition in `$HOME/cesm/config_files`. This overwrites the supplied `cesm_environment.yml` with the following contents:

```bash
cd "$HOME/cesm/config_files"

cat > cesm_environment.yml <<'EOF'
name: cesm
channels:
  - conda-forge
dependencies:
  - python=3.11
  - subversion
EOF
```

Create the environment:

```bash
conda env create --file cesm_environment.yml
```

!!! note "Python 3.11 is used because the checkout tooling requires the `imp` module, which was removed in Python 3.12."

### Download CESM2.1.5

Activate the environment and clone the CESM2.1.5 release into `$HOME/cesm/CESM2.1.5`:

```bash
conda activate cesm
cd "$HOME/cesm"

git clone -b release-cesm2.1.5 [https://github.com/ESCOMP/CESM.git](https://github.com/ESCOMP/CESM.git) CESM2.1.5
```

Move into the source directory and download the external components:

```bash
cd "$HOME/cesm/CESM2.1.5"
./manage_externals/checkout_externals
```


??? note "Sanity check"
    While the `cesm` environment is still active, check the status of the external components:

    ```bash
    ./manage_externals/checkout_externals -S
    ```

    The output should include the following component directories. The absolute paths in the preceding messages will reflect your own home directory.

    ```text
        ./cime
        ./components/cam
        ./components/cam/chem_proc
        ./components/cam/src/physics/carma/base
        ./components/cam/src/physics/clubb
        ./components/cam/src/physics/cosp2/src
        ./components/cice
        ./components/cism
        ./components/cism/source_cism
        ./components/clm
        ./components/clm/src/fates
        ./components/mosart
        ./components/pop
        ./components/pop/externals/CVMix
        ./components/pop/externals/MARBL
        ./components/rtm
        ./components/ww3
    ```

Finally, deactivate the environment:

```bash
conda deactivate
```

The CESM source code and its external components should now be located in `$HOME/cesm/CESM2.1.5`.

## Step 6 - Update the environment and build scripts

Update the two scripts in `$HOME/cesm/config_files` so that they use your installation paths and activate the `cesm` Conda environment.

### Update the environment-setup script

Replace the contents of `$HOME/cesm/config_files/cesm2.1.5_cx3p2.bashrc` with:

```bash
#!/bin/bash

module purge

# Convenient variables used by the setup scripts
CESM215_MAIN="$HOME/cesm/CESM2.1.5"
CESM215_CONFIG="$HOME/cesm/config_files"
CESM215_BUILD_SCRIPT="${CESM215_CONFIG}/cesm2_cx3p2.sh"
CESM215_NAMELISTS="${CESM215_CONFIG}/namelists"
CESM215_OUTPUT="$HOME/cesm/output"

# Variables used directly by CESM
export CESMDATAROOT="$HOME/cesm/data"
export CIME_MODEL=cesm
export CIME="${CESM215_MAIN}/cime"

module load tools/prod
module load ESMF/8.4.2-foss-2022a
module load Perl/5.34.1-GCCcore-11.3.0
module load CMake/3.23.1-GCCcore-11.3.0

export PERL5LIB="$HOME/perl5/lib/perl5"
```

### Update the build script

Replace the contents of `$HOME/cesm/config_files/cesm2_cx3p2.sh` with:

```bash
#!/bin/bash
#PBS -N CESM2_build
#PBS -l select=1:ncpus=1:mem=8gb:cpu_type=rome
#PBS -l walltime=1:00:00
#PBS -j oe

if [[ -n "$PBS_O_WORKDIR" ]]; then
    cd "$PBS_O_WORKDIR" || exit 1
fi

model_version=$(grep MODEL_VERSION env_case.xml | cut -d'"' -f4)
if [[ -z "$model_version" ]]; then
    echo "Could not determine model version."
    exit 102
fi

# Initialise the shell environment and activate the CESM Conda environment
source "$HOME/.bashrc"
eval "$("$HOME/miniforge3/bin/conda" shell.bash hook)"
conda activate cesm || exit 1

module purge
if [[ "$model_version" == "release-cesm2.2.2" ]]; then
    echo "Loading bashrc for CESM 2.2.2"
    source "$HOME/bash_startup/cesm2_cx3p2.bashrc" || exit 1
elif [[ "$model_version" == "release-cesm2.1.5" ]]; then
    echo "Loading bashrc for CESM 2.1.5"
    source "$HOME/cesm/config_files/cesm2.1.5_cx3p2.bashrc" || exit 1
else
    echo "Model version $model_version not recognized."
    exit 101
fi

# Uncomment the following line if a clean rebuild is required
# ./case.build --clean
./case.build
```
