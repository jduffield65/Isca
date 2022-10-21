# Mixed Layer
The [`mixed_layer_nml`](https://github.com/ExeClim/Isca/blob/master/src/atmos_spectral/driver/solo/mixed_layer.F90) 
only ever needs to be specified if 
[`mixed_layer_bc = .true.`](../main/idealized_moist_physics.md#mixed_layer_bc) in 
`idealized_moist_phys_nml`.
It contains options which deal with the mixed layer boundary condition and is described on 
[Isca's website](https://execlim.github.io/Isca/modules/mixedlayer.html).</br>
Some of the most common options are described below:

## Options
### `evaporation`
*bool*</br> Switch for surface evaporation.</br>
**Default:** `True`

### `depth`
*float*</br> Depth of mixed layer ($m$).</br>
**Default:** `40.0`
</br>
</br>
### **Q-flux**

#### `do_qflux`
*bool*</br> Switch to calculate time-independent Q-flux. </br>
**Default:** `False`

#### `qflux_amp`
*float*</br> Amplitude of time-independent Q-flux. </br> 
Only ever required if [`do_qflux = .true.`](#do_qflux).</br>
**Default:** `0.0`

#### `qflux_width`
*float*</br> Width of time-independent Q-flux. </br> 
Only ever required if [`do_qflux = .true.`](#do_qflux).</br>
**Default:** `16.0`

#### `load_qflux`
*bool*</br> Switch to use input file to load in a time-independent or time-dependent Q-flux. </br>
**Default:** `False`

#### `qflux_file_name`
*string*</br> 
Name of file among input files, from which to get Q-flux. </br> 
Only ever required if [`load_qflux = .true.`](#load_qflux).</br>
**Default:** `16.0`

#### `time_varying_qflux`
*bool*</br> 
Flag that determines whether input Q-flux file is time dependent. </br> 
Only ever required if [`load_qflux = .true.`](#load_qflux).</br>
**Default:** `False`

#### `qflux_field_name`
*string*</br> 
Name of field name in Q-flux file name, from which to get Q-flux. </br> 
Only ever required if [`time_varying_qflux = .false.`](#time_varying_qflux), otherwise assumes 
field_name=[file_name](#qflux_file_name).</br>
**Default:** `16.0`
</br>
</br>

### **Surface Temperature**
#### `prescribe_initial_dist`
*bool*</br>
If `True`, an 
[initial surface temperature distribution](https://execlim.github.io/Isca/modules/mixedlayer.html#slab-ocean) is set 
up which is then allowed to evolve based on the surface fluxes.</br>
**Default:** `False`

#### `tconst`
*float*</br>
The parameter $T_{surf}$ for the initial temperature distribution which follows:</br>
$T_s = T_{surf} -\frac{1}{3} dT \left(3\sin(\lambda)^2-1\right)$</br>
Only ever required if [`prescribe_initial_dist = .true.`](#prescribe_initial_dist).</br>
**Default:** `305.0`

#### `delta_T`
*float*</br>
The parameter $dT$ for the initial temperature distribution which follows:</br>
$T_s = T_{surf} -\frac{1}{3} dT \left(3\sin(\lambda)^2-1\right)$</br>
Only ever required if [`prescribe_initial_dist = .true.`](#prescribe_initial_dist).</br>
**Default:** `40.0`

#### `do_read_sst`
*bool*</br>
If `True`, surface temperatures will be prescribed from an [input file](#sst_file) and will be fixed. </br>
**Default:** `False`

#### `do_sc_sst`
*bool*</br>
As far as I can tell, this is exactly the same as [`do_read_sst`](#do_read_sst).</br>
**Default:** `False`

#### `sst_file`
*string*</br>
Name of *NetCDF* file containing fixed surface temperatures. </br>
The [file](https://execlim.github.io/Isca/modules/mixedlayer.html#input-sst-file) can be time independent or
vary with time. </br>
Only ever required if [`do_read_sst = .true.`](#do_read_sst).</br>
**Default:** N/A

#### `specify_sst_over_ocean_only`
*bool*</br>
Flag to specify surface temperature only over ocean. </br>
Only ever required if [`do_read_sst = .true.`](#do_read_sst).</br>
**Default:** N/A

#### `do_ape_sst`
*bool*</br>
If `True`, surface temperatures will be fixed and prescribed from the 
[APE aquaplanet](https://execlim.github.io/Isca/modules/mixedlayer.html#ape-aquaplanet-analytic-sst) equation: </br>
$T_s = 27 \left( 1 - \sin^2\left( \frac{3}{2} \lambda \right) \right)$ </br>
**Default:** `False`

#### `add_latent_heat_flux_anom`
*bool*</br>
Flag to add an anomalous latent heat flux. </br>
**Default:** `False`

#### `do_warmpool`
*bool*</br>
Flag to call the `warmpool` routine from the 
[`qflux`](https://github.com/ExeClim/Isca/blob/master/src/atmos_param/qflux/qflux.f90) module, 
which returns `ocean_qflux`. </br>
**Default:** `False`
</br>
</br>
### **Surface Albedo**

#### `albedo_choice`
*integer*</br>
There are 5 choices of surface albedo, $\alpha$, which can be specified by the indices below:

1. *Constant*</br>$\alpha$ set to the constant indicated by [`albedo_value`](#albedo_value).
2. *Glacier (One hemisphere)*</br>
    * [`lat_glacier>0`](#lat_glacier)
   
         * $\lambda$ > `lat_glacier`: $\alpha$ = [`higher_albedo`](#higher_albedo)
         * $\lambda$ < `lat_glacier`: $\alpha$ = [`albedo_value`](#albedo_value)
      
    * [`lat_glacier<0`](#lat_glacier)
   
         * $\lambda$ < `lat_glacier`: $\alpha$ = [`higher_albedo`](#higher_albedo)
         * $\lambda$ > `lat_glacier`: $\alpha$ = [`albedo_value`](#albedo_value)
      
3. *Glacier (Both hemispheres)*</br>
    * $\lambda$ > $|$[`lat_glacier`](#lat_glacier)$|$: $\alpha$ = [`higher_albedo`](#higher_albedo)
    * $\lambda$ < $|$[`lat_glacier`](#lat_glacier)$|$: $\alpha$ = [`albedo_value`](#albedo_value)

4. *Exponent*</br>
    $\alpha$ = [`albedo_value`](#albedo_value) + ([`higher_albedo`](#higher_albedo) - [`albedo_value`](#albedo_value))
    $\times (\frac{\lambda}{90})$^[`albedo_exp`](#albedo_exp)
5. *Tanh*</br>
    This is an increase in $\alpha$ at latitude indicated by [`albedo_cntr`](#albedo_cntr) with width 
[`albedo_wdth`](#albedo_wdth). </br>
    ```
   α(λ) = albedo_value + (higher_albedo-albedo_value)* 0.5 *(1+tanh((λ-albedo_cntr)/albedo_wdth))
   ```

**Default:** `1`

#### `albedo_value`
*float*</br>
Parameter to determine the surface albedo.</br>
Required for every value of  [`albedo_choice`](#do_read_sst).</br>
**Default:** `0.06`

#### `higher_albedo`
*float*</br>
Parameter to determine the surface albedo.</br>
Only ever required if [`albedo_choice`](#do_read_sst) is `2`, `3`, `4` or `5`.</br>
**Default:** `0.10`

#### `lat_glacier`
*float*</br>
Parameter that sets the glacier ice latitude for determining the surface albedo.</br>
Only ever required if [`albedo_choice`](#do_read_sst) is `2` or `3`.</br>
**Default:** `60.0`

#### `albedo_exp`
*float*</br>
Parameter that sets the latitude dependence for determining the surface albedo.</br>
Only ever required if [`albedo_choice`](#do_read_sst) is `4`.</br>
**Default:** `2.`

#### `albedo_cntr`
*float*</br>
Parameter that sets the central latitude for determining the surface albedo.</br>
Only ever required if [`albedo_choice`](#do_read_sst) is `5`.</br>
**Default:** `45.`

#### `albedo_wdth`
*float*</br>
Parameter that sets the latitude width for determining the surface albedo.</br>
Only ever required if [`albedo_choice`](#do_read_sst) is `5`.</br>
**Default:** `10.`
</br>
</br>
### **Land**