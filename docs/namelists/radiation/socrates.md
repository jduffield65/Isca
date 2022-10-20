# SOCRATES
The [`socrates_rad_nml`](https://github.com/ExeClim/Isca/blob/master/src/atmos_param/socrates/interface/socrates_config_mod.f90) 
only ever needs to be specified if 
[`do_socrates_radiation = .true.`](../main/idealized_moist_physics.md#do_socrates_radiation) in 
`idealized_moist_phys_nml`.
If this is the case, then SOCRATES will be the radiation scheme that is used. 
This is described on [Isca's website](https://execlim.github.io/Isca/modules/socrates.html). </br>
Some of the most common options are described below:

## Options
### `inc_co2`
*bool*</br> Includes radiative effects of $CO_2$ if `True`.</br>
**Default:** `True`
