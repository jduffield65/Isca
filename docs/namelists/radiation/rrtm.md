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
