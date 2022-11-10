# Held Suarez

The [`hs_forcing_nml`](https://github.com/ExeClim/Isca/blob/master/src/atmos_param/hs_forcing/hs_forcing.F90) 
only ever needs to be specified if 
[`idealized_moist_model = False` in `atmosphere_nml`](atmosphere.md#idealized_moist_model).
It contains options which specify the Newtonian cooling thermal relaxation profile. Some of the most common
ones are described below:

*Isca* gives an 
[example script](https://github.com/ExeClim/Isca/blob/master/exp/test_cases/held_suarez/held_suarez_test_case.py) 
indicating typical usage of the `hs_forcing_nml`.

## Options
### `t_zero`
*float*</br>
Temperature at reference pressure at equator (Kelvin).</br>
**Default:** `315`

### `t_strat`
*float*</br>
Stratosphere temperature (Kelvin).</br>
**Default:** `200`

### `delh`
*float*</br>
Equator-pole temperature gradient (Kelvin).</br>
**Default:** `60`

### `delv`
*float*</br>
Lapse rate (Kelvin).</br>
**Default:** `60`

### `eps`
*float*</br>
Stratospheric latitudinal variation.</br>
**Default:** `0`

### `sigma_b`
*float*</br>
Boundary layer friction height ($\sigma =  p/p_s$).</br>
**Default:** `0.7`

### `ka`
*float*</br>
Constant Newtonian cooling timescale (negative sign is a flag indicating that the units are days).</br>
**Default:** `-40`

### `ks`
*float*</br>
Boundary layer dependent cooling timescale (negative sign is a flag indicating that the units are days).</br>
**Default:** `-4`

### `kf`
*float*</br>
BL momentum frictional timescalee (negative sign is a flag indicating that the units are days).</br>
**Default:** `-1`

### `do_conserve_energy`
*bool*</br>
Convert dissipated momentum into heat if `True`. </br>
**Default:** `True`

## Diagnostics
The diagnostics for 
[this module](https://github.com/ExeClim/Isca/blob/master/src/atmos_param/hs_forcing/hs_forcing.F90) 
can be specified using the `module_name` of `hs_forcing` in the 
diagnostic table file.
Some available diagnostics are given below:

### `h_trop`
Height of tropopause.</br>
*Dimensions: time, lat, lon*</br>
*Units: $km$*

