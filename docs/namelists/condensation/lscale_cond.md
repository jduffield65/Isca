# Large Scale Condensation and Precipitation
The [`lscale_cond_nml`](https://github.com/ExeClim/Isca/blob/master/src/atmos_param/lscale_cond/lscale_cond.F90) 
only ever needs to be specified if 
[`convection_scheme`](../main/idealized_moist_physics.md#convection_scheme) in 
`idealized_moist_phys_nml` is either `SIMPLE_BETTS_CONV`, `FULL_BETTS_MILLER_CONV` or `RAS_CONV`.
This namelist is described on [Isca's website](https://execlim.github.io/Isca/modules/lscale_cond.html) 
and is used in the 
[*Frierson* example script](https://github.com/ExeClim/Isca/blob/master/exp/test_cases/frierson/frierson_test_case.py).</br>
All the options are also described below:

## Options
### `hc`
*float*</br> The relative humidity at which large scale condensation occurs. </br>
$0.0 \leq hc \leq 1.0$.</br>
**Default:** `1.0`

### `do_evap`
*bool*</br> 
The flag for the re-evaporation of moisture in sub-saturated layers below, 
if `True` then re-evaporation is performed. </br>
**Default:** `False`

### `do_simple`
*bool*</br> 
If `True` then all precipitation is rain/liquid precipitation, there is no snow/frozen precipitation. </br>
**Default:** `False`
