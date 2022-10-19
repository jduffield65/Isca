# Quasi Equilibrium Moist Convection
The [`qe_moist_convection_nml`](https://github.com/ExeClim/Isca/blob/master/src/atmos_param/qe_moist_convection/qe_moist_convection.F90) 
only ever needs to be specified if 
[`convection_scheme = SIMPLE_BETTS_MILLER_CONV`](idealized_moist_physics.md#convection_scheme) in 
`idealized_moist_phys_nml`.
This convection scheme is described on [Isca's website](https://execlim.github.io/Isca/modules/convection_simple_betts_miller.html)
Some of the most common options are also described below:

## Options
### `tau_bm`
*float*</br> Relaxation timescale, $\tau$ (seconds).</br>
**Default:** `7200`
