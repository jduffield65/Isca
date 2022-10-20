# Damping Driver
The [`damping_driver_nml`](https://github.com/ExeClim/Isca/blob/master/src/atmos_param/damping_driver/damping_driver.f90) 
only ever needs to be specified if 
[`do_damping=.true.`](../main/idealized_moist_physics.md#do_damping) in 
`idealized_moist_phys_nml`.
This namelist accounts for subgrid-scale processes which decelerate fast winds at upper levels. 
It is described on [Isca's website](https://execlim.github.io/Isca/modules/damping_driver.html) and 
some of the most common options are also described below:

## Options
### `do_rayleigh`
*bool*</br> 
On/off switch for Rayleigh friction. </br>
**Default:** `False`
