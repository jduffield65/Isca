# Surface Flux
The [`surface_flux_nml`](https://github.com/ExeClim/Isca/blob/master/src/coupler/surface_flux.F90) 
only ever needs to be specified if 
[`mixed_layer_bc = .true.`](../main/idealized_moist_physics.md#mixed_layer_bc) in 
`idealized_moist_phys_nml`.
It contains options which deal with the exchange of heat, momentum at the surface, including the difference 
between ocean and land. It is described on 
[Isca's website](https://execlim.github.io/Isca/modules/surface_flux.html).</br>
Some of the most common options are described below:

## Options
### **Land**
There are [4 ways that land is implemented in Isca](../main/idealized_moist_physics.md#land-and-hydrology).

#### `land_humidity_factor`
*float*</br> 
Factor that multiplies the surface specific humidity over land.</br>
This is included to make land *dry*. If it is equal to 1, land behaves like ocean. </br>
If it is between 0 and 1, this will decrease the evaporative heat flux in areas of land. </br>
The evaporative flux formula is given on [Isca's website](https://execlim.github.io/Isca/modules/surface_flux.html#land).
</br> Only ever required if [`land_option`](mixed_layer.md#land_option) is not `none`.</br>
??? warning "Instability"
    Note that this can lead to sign changes in the evaporative flux, 
    and we find this becomes unstable over very shallow mixed layer depths.
**Default:** `1.0`

#### `land_evap_prefactor`
*float*</br> 
Factor that multiplies the evaporative flux over land.</br>
This is included to make land *dry*. If it is equal to 1, land behaves like ocean. </br>
If it is between 0 and 1, this will decrease the evaporative heat flux in areas of land. </br>
The evaporative flux formula is given on [Isca's website](https://execlim.github.io/Isca/modules/surface_flux.html#land).
</br> Only ever required if [`land_option`](mixed_layer.md#land_option) is not `none`.</br>
??? note "Stability"
    This formulation avoids sign changes in the evaporative flux and remains stable over very 
    shallow mixed layer depths.
**Default:** `1.0`