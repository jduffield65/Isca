# RRTM Radiation
The [`rrtm_radiation_nml`](https://github.com/ExeClim/Isca/blob/master/src/atmos_param/rrtm_radiation/rrtm_radiation.f90) 
only ever needs to be specified if 
[`do_rrtm_radiation = .true.`](../main/idealized_moist_physics.md#do_rrtm_radiation) in 
`idealized_moist_phys_nml`. If this is the case, then the Rapid Radiative Transfer Model will be
the radiation scheme that is used.
Some of the most common options for configuring this radiation scheme are described below:

## Options
### `co2ppmv`
*float*</br> Concentration of $CO_2$ (*ppmv*).</br>
**Default:** `300`

## Diagnostics
The diagnostics for 
[this module](https://github.com/ExeClim/Isca/blob/master/src/atmos_param/rrtm_radiation/rrtm_radiation.f90) 
can be specified using the `module_name` of `rrtm_radiation` in the 
diagnostic table file.
Some available diagnostics are given below:

### `olr`
Top of atmosphere longwave flux (up is positive).</br>
*Dimensions: time, lat, lon*</br>
*Units: $Wm^{-2}$*

### `toa_sw`
Top of atmosphere net shortwave flux (down is positive).</br>
*Dimensions: time, lat, lon*</br>
*Units: $Wm^{-2}$*

### `flux_lw`
Longwave flux at the surface (down only).</br>
*Dimensions: time, lat, lon*</br>
*Units: $Wm^{-2}$*

### `flux_sw`
Net shortwave flux at the surface (down is positive).</br>
*Dimensions: time, lat, lon*</br>
*Units: $Wm^{-2}$*

