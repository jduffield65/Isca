# Damping Driver
The [`damping_driver_nml`](https://github.com/ExeClim/Isca/blob/master/src/atmos_param/damping_driver/damping_driver.f90) 
only ever needs to be specified if 
[`do_damping=.true.`](../main/idealized_moist_physics.md#do_damping) in 
`idealized_moist_phys_nml`.
This namelist accounts for subgrid-scale processes which decelerate fast winds at upper levels. 
It is described on [Isca's website](https://execlim.github.io/Isca/modules/damping_driver.html) and is used
in the 
[*Frierson* example script](https://github.com/ExeClim/Isca/blob/master/exp/test_cases/frierson/frierson_test_case.py). 
Some of the most common options are also described below:

## Options
### `do_rayleigh`
*bool*</br> 
On/off switch for Rayleigh friction. </br>
**Default:** `False`

## Diagnostics
The diagnostics for 
[this module](https://github.com/ExeClim/Isca/blob/master/src/atmos_param/damping_driver/damping_driver.f90) 
can be specified using the `module_name` of `damping` in the 
diagnostic table file. The list of available diagnostics is available on 
[Isca's website](https://execlim.github.io/Isca/modules/damping_driver.html#diagnostics). 
Some are also given below:

### `udt_rdamp`
Zonal wind tendency for Rayleigh damping.</br>
Can only be returned if [`do_rayleigh=True`](#do_rayleigh).</br>
*Dimensions: time, lat, lon, pressure*</br>
*Units: $ms^{-2}$*
