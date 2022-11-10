# Topography
The [`spectral_init_cond_nml`](https://github.com/ExeClim/Isca/blob/master/src/atmos_spectral/init/spectral_init_cond.F90) 
only ever needs to be specified if 
[`land_option`](../main/idealized_moist_physics.md#land_option) is `input` in 
`idealized_moist_phys_nml`.
It contains options which specify the topography of the land. It is described on 
[Isca's website](https://execlim.github.io/Isca/modules/topography.html#spectral-init-cond) and there is an
[example script](https://github.com/ExeClim/Isca/blob/master/exp/test_cases/realistic_continents/namelist_basefile.nml)
using topography.

The options are described below:

## Options
### `topography_option`
*string*</br> 
This indicates how the topography is specified. There are 4 options:</br>

* `input` - Get topography from input file.
* `flat` - Surface geopotential is 0 (not widely used).
* `interpolated` - Not currently used.
* `gaussian` - Simple Gaussian-shaped mountains are generated from specified parameters (not widely used).

**Default:** `flat`

### `topog_file_name`
*string*</br> 
File that contains the topography information. </br> This should be the same as 
[`land_file_name`](../main/idealized_moist_physics.md#land_file_name) but without `INPUT` i.e.
if the file is called `land.nc` and is in the [`input_dir`](../main/experiment_details.md#input_dir) then 
`topog_file_name` should be `land.nc`.</br>
**Default:** `topography.data.nc`

### `topog_field_name`
*string*</br>
The height field name in the [input file](#topog_file_name).</br>
**Default:** `zsurf`

### `land_field_name`
*string*</br>
The land field name in the [input file](#topog_file_name).</br>
This should be the same as 
[`land_field_name`](../main/idealized_moist_physics.md#land_field_name) in the `idealized_moist_phys_nml` namelist.</br>
**Default:** `land_mask`