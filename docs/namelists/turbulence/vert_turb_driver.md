# Vertical Turbulence Driver
The [`vert_turb_nml`](https://github.com/ExeClim/Isca/blob/master/src/atmos_param/vert_turb_driver/vert_turb_driver.F90) 
namelist only ever needs to be specified if 
[`turb = .true.`](../main/idealized_moist_physics.md#turb) in 
`idealized_moist_phys_nml`. The module computes the vertical diffusion coefficients.
Some of the most common options for configuring this are described below:

## Options
### `do_diffusivity`
*bool*</br> If `True`, the `diffusivity` routine in the `diffusivity` module is run and the 
[`diffusivity_nml`](diffusivity.md) namelist needs to be specified.</br>
**Default:** `False`

### `do_molecular_diffusion`
*bool*</br> If `True`, the `molecular_diffusion` routine in the `diffusivity` module is run. 
[`do_diffusivity`](#do_diffusivity) must be `True` for this variable to make any difference. </br>
**Default:** `False`

