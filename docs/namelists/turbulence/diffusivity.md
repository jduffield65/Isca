# Vertical Turbulence Driver
The [`diffusivity_nml`](https://github.com/ExeClim/Isca/blob/master/src/atmos_param/diffusivity/diffusivity.F90) 
namelist only ever needs to be specified if 
[`do_diffusivity = .true.`](vert_turb_driver.md#do_diffusivity) in 
`vert_turb_nml`. The module computes the atmospheric diffusivities in the planetary boundary layer and in 
the free atmosphere.
Some of the most common options for configuring this are described below:

## Options
### `do_simple`
*bool*</br> If `True`, a simplified calculation is used.</br>
**Default:** `False`
