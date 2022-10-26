# SOCRATES
The [`socrates_rad_nml`](https://github.com/ExeClim/Isca/blob/master/src/atmos_param/socrates/interface/socrates_config_mod.f90) 
only ever needs to be specified if 
[`do_socrates_radiation = .true.`](../main/idealized_moist_physics.md#do_socrates_radiation) in 
`idealized_moist_phys_nml`.
If this is the case, then SOCRATES will be the radiation scheme that is used. 
This is described on [Isca's website](https://execlim.github.io/Isca/modules/socrates.html). 
Isca also includes an 
[example script](https://github.com/ExeClim/Isca/blob/master/exp/test_cases/socrates_test/socrates_aquaplanet.py).</br>
Some of the most common options are described below:

## Options
### `inc_co2`
*bool*</br> Includes radiative effects of $CO_2$ if `True`.</br>
**Default:** `True`

## Diagnostics
The diagnostics for 
[this module](https://github.com/ExeClim/Isca/blob/master/src/atmos_param/socrates/interface/socrates_interface.F90) 
can be specified using the `module_name` of `socrates` in the 
diagnostic table file. The list of available diagnostics is available on 
[Isca's website](https://execlim.github.io/Isca/modules/socrates.html#diagnostics). 
Some are also given below:

### `soc_olr`
Top of atmosphere longwave flux (up is positive).</br>
*Dimensions: time, lat, lon*</br>
*Units: $Wm^{-2}$*

### `soc_toa_sw`
Top of atmosphere net shortwave flux (down is positive).</br>
*Dimensions: time, lat, lon*</br>
*Units: $Wm^{-2}$*

### `soc_surf_flux_lw`
Net longwave flux at the surface (up is positive).</br>
*Dimensions: time, lat, lon*</br>
*Units: $Wm^{-2}$*

### `soc_surf_flux_sw`
Net shortwave flux at the surface (down is positive).</br>
*Dimensions: time, lat, lon*</br>
*Units: $Wm^{-2}$*
