# Atmosphere

The [`atmosphere_nml`](https://github.com/ExeClim/Isca/blob/master/src/atmos_spectral/driver/solo/atmosphere.F90) 
namelist contains the following options:

## Options
### `idealized_moist_model`
*bool*</br> If `False`, the [`hs_forcing_nml`](held_suarez.md) namelist needs to be specified
to configure the Newtonian cooling thermal relaxation profile (simple and dry alternative to
moist physics).</br> If `True`, the [`idealized_moist_phys_nml`](idealized_moist_physics.md) namelist needs to 
be specified.</br>
**Default:** `False`