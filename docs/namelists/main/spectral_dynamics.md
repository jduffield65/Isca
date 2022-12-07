# Spectral Dynamics
The [`spectral_dynamics_nml`](https://github.com/ExeClim/Isca/blob/master/src/atmos_spectral/model/spectral_dynamics.F90) 
contains options relating to Isca's spectral dynamical core. 
The [*Frierson* example script](https://github.com/ExeClim/Isca/blob/master/exp/test_cases/frierson/frierson_test_case.py)
uses this namelist and some of the [diagnostics](#diagnostics).</br>
Some of the most common options are described below:

## Options
### `num_levels`
*integer*</br>
The number of pressure coordinates i.e. number of vertical coordinates. </br>
**Default:** `18`

### `vert_coord_option`
*string*</br>
How to specify the vertical coordinates, there are two options: </br>

* `even_sigma`: Levels are equally separated in $\sigma$ such that there are [`num_levels`](#num_levels) levels. </br>
* `uneven_sigma`: Not really sure what this does but it is an option.
* `input`: Each coordinate is explicitly specified using the 
[`vert_coordinate_nml`](https://github.com/ExeClim/Isca/blob/master/src/atmos_spectral/init/vert_coordinate.F90) 
namelist. </br>

    ??? note "`vert_coordinate_nml`"
         In this case, the namelist 
         [`vert_coordinate_nml`](https://github.com/ExeClim/Isca/blob/master/src/atmos_spectral/init/vert_coordinate.F90)
         needs to be specified through two options:

         * `bk` - The $\sigma$ coordinates. There should be [`num_levels+1`](#num_levels) of these including $0$ and $1$.
         * `pk` - The corresponding pressure coordinates. Again, there should be [`num_levels+1`](#num_levels) of these.
         If they are all set to $0$, they will be computed automatically.
        
         For an example using this , see the the 
         [`frierson_test_case`](https://github.com/ExeClim/Isca/blob/master/exp/test_cases/frierson/frierson_test_case.py).


**Default:** `even_sigma`

## Diagnostics
The diagnostics for 
[this module](https://github.com/ExeClim/Isca/blob/master/src/atmos_spectral/model/spectral_dynamics.F90) 
can be specified using the `module_name` of `dynamics` in the 
diagnostic table file. Some available diagnostics are listed  on 
[Isca's website](https://execlim.github.io/Isca/modules/diag_manager_mod.html#output-fields). Some 
of the more common ones are also given below:

### `ps`
Surface pressure.</br>
*Dimensions: time, lat, lon*</br>
*Units: $Pa$*

### `bk`
Vertical coordinate $\sigma$ values.</br>
*Dimensions: pressure*</br>
*Units: $Pa$*

### `pk`
Vertical coordinate pressure values.</br>
*Dimensions: pressure*</br>
*Units: $Pa$*

### `sphum`
Specific humidity.</br>
*Dimensions: time, lat, lon, pressure*</br>
*Units: $kg/kg$*

### `rh`
Relative humidity.</br>
*Dimensions: time, lat, lon, pressure*</br>
*Units: $%$*

### `temp`
Temperature.</br>
*Dimensions: time, lat, lon, pressure*</br>
*Units: $K$*

### `ucomp`
Zonal component of the horizontal winds.</br>
*Dimensions: time, lat, lon, pressure*</br>
*Units: $ms^{-1}$*

### `vcomp`
Meridional component of the horizontal winds.</br>
*Dimensions: time, lat, lon, pressure*</br>
*Units: $ms^{-1}$*

### `omega`
Vertical velocity.</br>
*Dimensions: time, lat, lon, pressure*</br>
*Units: $Pas^{-1}$*

### `vor`
Vorticity.</br>
*Dimensions: time, lat, lon, pressure*</br>
*Units: $s^{-1}$*

### `div`
Divergence.</br>
*Dimensions: time, lat, lon, pressure*</br>
*Units: $s^{-1}$*

### `height`
Geopotential height at full model levels.</br>
*Dimensions: time, lat, lon, pressure*</br>
*Units: $m$*
