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
