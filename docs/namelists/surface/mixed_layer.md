# Mixed Layer
The [`mixed_layer_nml`](https://github.com/ExeClim/Isca/blob/master/src/atmos_spectral/driver/solo/mixed_layer.F90) 
only ever needs to be specified if 
[`mixed_layer_bc = .true.`](../main/idealized_moist_physics.md#mixed_layer_bc) in 
`idealized_moist_phys_nml`.
It contains options which deal with the mixed layer boundary condition, including the difference 
between ocean and land. 
It is described on [Isca's website](https://execlim.github.io/Isca/modules/mixedlayer.html) and is 
used in numerous 
[example scripts](https://github.com/ExeClim/Isca/blob/master/exp/test_cases/realistic_continents/namelist_basefile.nml).
</br>Some of the most common options are described below:

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

### **Ice**

#### `update_albedo_from_ice`
*bool*</br>
Flag to set the surface albedo to [`ice_albedo_value`](#ice_albedo_value) 
where there is ice as specified by [`ice_file_name`](#ice_file_name).</br>
**Default:** `False`

#### `ice_albedo_value`
*float*</br>
Value for the ice albedo.</br>
Expect this to be much larger than [`albedo_value`](#albedo_value) because ice is more reflective than ocean.
Only ever required if [`update_albedo_from_ice=.true.`](#land_option).</br>
**Default:** `0.7`

#### `ice_file_name`
*string*</br>
Name of file containing sea ice concentration.</br>
Only ever required if [`update_albedo_from_ice=.true.`](#land_option).</br>
**Default:** `siconc_clim_amip`

#### `ice_concentration_threshold`
*float*</br>
Value of sea ice concentration above which albedo should be set to [`ice_albedo_value`](#ice_albedo_value).</br>
**Default:** `0.5`
</br>
</br>

### **Land**
There are [4 ways that land is implemented in Isca](../main/idealized_moist_physics.md#land-and-hydrology).

#### `land_option`
*string*</br>
There are 4 choices of the land mask in *Isca* given below. This parameter should be set to the same value
as [`land_option`](../main/idealized_moist_physics.md#land_option) in the 
[`idealized_moist_phys_nml`](../main/idealized_moist_physics.md) namelist. **If it is not specified here but is 
elsewhere, then none of the land parameters set in this module will be used.**

??? note "Heat capacity calculation"
    The heat capacity calculation is different over land for the different options of `land_option`.
    If it is `input`, the heat capacity at a particular location is 
    [set](https://github.com/ExeClim/Isca/blob/9560521e1ba5ce27a13786ffdcb16578d0bd00da/src/atmos_spectral/driver/solo/mixed_layer.F90#L542) 
    to [`land_h_capacity_prefactor`](#land_h_capacity_prefactor) multiplied
    by the ocean heat capacity at that location.</br>
    If it is either `zsurf` or `lonlat`, it is 
    [computed](https://github.com/ExeClim/Isca/blob/9560521e1ba5ce27a13786ffdcb16578d0bd00da/src/atmos_spectral/driver/solo/mixed_layer.F90#L521-L540) 
    as for ocean but with a [different depth](#land_depth).

* `input` - Read land mask from input file indicated by the 
[`land_file_name`](../main/idealized_moist_physics.md#land_field_name) parameter in the 
[`idealized_moist_phys_nml`](../main/idealized_moist_physics.md) namelist.
* `zsurf` - Define land where surface geopotential height at model initialisation exceeds a threshold of 10.
* `lonlat` - Define land to be in the longitude/latitude box set by [`slandlon[k]`](#slandlon), [`elandlon[k]`](#elandlon),
[`slandlat[k]`](#slandlat), [`elandlat[k]`](#elandlat) for all $k$.
* `none` - Do not apply land mask.

**Default:** `none`

#### `land_h_capacity_prefactor`
*float*</br>
Factor by which to multiply ocean heat capacity to get land heat capacity. </br>
Would expect this to be less than `1` as land has a smaller heat capacity than ocean. </br>
Only ever required if [`land_option='input'`](#land_option).</br>
**Default:** `1.0`

#### `land_albedo_prefactor`
*float*</br>
Factor by which to multiply ocean albedo to get land albedo. </br>
Would expect this to be more than `1` as land is more reflective than ocean. </br>
Only ever required if [`land_option='input'`](#land_option).</br>
**Default:** `1.0`

#### `land_depth`
*float*</br>
Depth of land mixed layer ($m$).</br>
Only ever required if [`land_option`](#land_option) is `'zsurf'` or `'lonlat'`.</br>
If it is 
[negative](https://github.com/ExeClim/Isca/blob/9560521e1ba5ce27a13786ffdcb16578d0bd00da/src/atmos_spectral/driver/solo/mixed_layer.F90#L305),
it just uses the ocean [`depth`](#depth).</br>
**Default:** `-1`

#### `slandlon`
*list - float*</br>
`slandlon[k]` is the start longitude of land box $k$.</br>
Only ever required if [`land_option`](#land_option) is `'lonlat'`.</br>
**Default:** `0`

#### `slandlat`
*list - float*</br>
`slandlat[k]` is the start latitude of land box $k$.</br>
Only ever required if [`land_option`](#land_option) is `'lonlat'`.</br>
**Default:** `0`

#### `elandlon`
*list - float*</br>
`elandlon[k]` is the end longitude of land box $k$.</br>
Only ever required if [`land_option`](#land_option) is `'lonlat'`.</br>
**Default:** `-1`

#### `elandlat`
*list - float*</br>
`elandlat[k]` is the end latitude of land box $k$.</br>
Only ever required if [`land_option`](#land_option) is `'lonlat'`.</br>
**Default:** `-1`

## Diagnostics
The diagnostics for 
[this module](https://github.com/ExeClim/Isca/blob/master/src/atmos_spectral/driver/solo/mixed_layer.F90) 
can be specified using the `module_name` of `mixed_layer` in the 
diagnostic table file. The list of available diagnostics is available on 
[Isca's website](https://execlim.github.io/Isca/modules/mixedlayer.html#diagnostics). 
They are also given below:

### `t_surf`
Surface temperature.</br>
*Dimensions: time, lat, lon*</br>
*Units: $K$*

### `delta_t_surf`
Surface temperature change.</br>
*Dimensions: time, lat, lon*</br>
*Units: $K$*

### `flux_t`
Surface sensible heat flux (up is positive).</br>
*Dimensions: time, lat, lon*</br>
*Units: $Wm^{-2}$*

### `flux_lhe`
Surface latent heat flux (up is positive).</br>
*Dimensions: time, lat, lon*</br>
*Units: $Wm^{-2}$*

### `flux_oceanq`
Oceanic Q-flux (will be $0$ if [`do_qflux=False`](#do_qflux)).</br>
*Dimensions: time, lat, lon*</br>
*Units: $Wm^{-2}$*

### `ml_heat_cap`
Mixed layer heat capacity.</br>
On the [website](https://execlim.github.io/Isca/modules/mixedlayer.html#diagnostics), it calls this 
`land_sea_heat_capacity` but in the 
[code](https://github.com/ExeClim/Isca/blob/9560521e1ba5ce27a13786ffdcb16578d0bd00da/src/atmos_spectral/driver/solo/mixed_layer.F90#L355-L356), 
I think it is `ml_heat_cap` so I am not sure which is correct.</br>
*Dimensions: time, lat, lon*</br>
*Units: $Jm^{-2}K^{-1}$*

### `albedo`
Surface albedo.</br>
I think this will remain constant in time unless [`update_albedo_from_ice=True`](#update_albedo_from_ice).</br>
*Dimensions: time, lat, lon*</br>
*Units: N/A*

### `ice_conc`
Sea ice concentration.</br>
Can only be returned if [`update_albedo_from_ice=True`](#update_albedo_from_ice).</br>
*Dimensions: time, lat, lon*</br>
*Units: N/A*
