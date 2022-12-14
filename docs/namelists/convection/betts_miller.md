# Betts-Miller Convection
The [`betts_miller_nml`](https://github.com/ExeClim/Isca/blob/master/src/atmos_param/betts_miller/betts_miller.f90) 
only ever needs to be specified if 
[`convection_scheme = FULL_BETTS_MILLER_CONV`](../main/idealized_moist_physics.md#convection_scheme) in 
`idealized_moist_phys_nml`.
*Isca* give an 
[example script](https://github.com/ExeClim/Isca/blob/master/exp/test_cases/realistic_continents/namelist_basefile.nml)
using this namelist.</br>
Some of the most common options for configuring this convection scheme are described below:

## Options
### `tau_bm`
*float*</br> Relaxation timescale, $\tau$ (seconds).</br>
**Default:** `7200`
