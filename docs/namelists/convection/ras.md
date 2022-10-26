# Relaxed Arakawa Schubert Convection
The [`ras_nml`](https://github.com/ExeClim/Isca/blob/master/src/atmos_param/ras/ras.f90) 
only ever needs to be specified if 
[`convection_scheme = RAS_CONV`](../main/idealized_moist_physics.md#convection_scheme) in 
`idealized_moist_phys_nml`.
Some of the most common options for configuring this convection scheme are described below:

## Options
### `fracs`
*float*</br> Fraction of planetary boundary layer mass allowed to be used by a cloud-type in time $DT$.</br>
**Default:** `0.25`

## Diagnostics
The diagnostics for 
[this module](https://github.com/ExeClim/Isca/blob/master/src/atmos_param/ras/ras.f90) 
can be specified using the `module_name` of `ras` in the 
diagnostic table file.
Some available diagnostics are given below:

### `tdt_conv`
Temperature tendency from *RAS*.</br>
*Dimensions: time, lat, lon, pressure*</br>
*Units: $Ks^{-1}$*

### `qdt_conv`
Specific humidity tendency from *RAS*.</br>
*Dimensions: time, lat, lon, pressure*</br>
*Units: $kgkg^{-1}s^{-1}$*

### `prec_conv`
Precipitation rate from *RAS*.</br>
*Dimensions: time, lat, lon, pressure*</br>
*Units: $kgm^{-2}s^{-1}$*
