# Dry Convection
The [`dry_convection_nml`](https://github.com/ExeClim/Isca/blob/master/src/atmos_param/qe_moist_convection/qe_moist_convection.F90) 
only ever needs to be specified if 
[`convection_scheme = DRY`](../main/idealized_moist_physics.md#convection_scheme) in 
`idealized_moist_phys_nml`.
The key options are also described below:

## Options
### `tau`
*float*</br> Relaxation timescale, $\tau$ (seconds). 
The [equivalent parameter for moist convection](./qe_moist_convection.md#tau_bm)
is set to `7200`.</br>
**Default:** `N/A`

### `gamma`
*float*</br> Prescribed lapse rate. Set to `1` for dry lapse rate.</br>
**Default:** `N/A`
