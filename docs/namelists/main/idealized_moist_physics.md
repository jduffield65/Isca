# Idealized Moist Physics
The [`idealized_moist_phys_nml`](https://github.com/ExeClim/Isca/blob/master/src/atmos_spectral/driver/solo/idealized_moist_phys.F90) 
only ever needs to be specified if 
[`idealized_moist_model = True` in `atmosphere_nml`](atmosphere.md#idealized_moist_model).
It contains options which specify the various modules associated with Isca’s moist physics configurations and is
described on [Isca's website](https://execlim.github.io/Isca/modules/idealised_moist_phys.html).
Some of the most common options are described below:

## Options
### `convection_scheme`
*string*</br> There are 4 choices of convection schemes, as well as no convection, in *Isca*:

* `SIMPLE_BETTS_CONV` - Use Frierson Quasi-Equilibrium convection scheme 
[*[Frierson2007]*](https://execlim.github.io/Isca/references.html#frierson2007).</br>
If this is selected, the [`lscale_cond_nml`](../condensation/lscale_cond.md) and [`qe_moist_convection_nml`](../convection/qe_moist_convection.md) 
namelists needs to be specified.
* `FULL_BETTS_MILLER_CONV` - Use the Betts-Miller convection scheme 
[*[Betts1986]*](https://execlim.github.io/Isca/references.html#betts1986),
[*[BettsMiller1986]*](https://execlim.github.io/Isca/references.html#bettsmiller1986). </br>
If this is selected, the [`lscale_cond_nml`](../condensation/lscale_cond.md) and [`betts_miller_nml`](../convection/betts_miller.md) namelists needs to be specified.
* `RAS_CONV` - Use the relaxed Arakawa Schubert convection scheme 
[*[Moorthi1992]*](https://execlim.github.io/Isca/references.html#moorthi1992). </br>
If this is selected, the [`lscale_cond_nml`](../condensation/lscale_cond.md) and [`ras_nml`](../convection/ras.md) namelists needs to be specified.
* `DRY_CONV` - Use the dry convection scheme 
[*[Schneider2006]*](https://execlim.github.io/Isca/references.html#schneider2006). </br>
This [module](https://github.com/ExeClim/Isca/blob/master/src/atmos_param/dry_convection/dry_convection.f90) does not 
need any additional namelist specified.
* `NO_CONV` - Use no convection scheme.
* `UNSET` - Will raise an error.

**Default:** `UNSET`

### `do_simple`
*bool*</br>
If `True`, use a simplification when calculating relative humidity. </br>
**Default:** `False`

### `do_damping`
*bool*</br>
If `True`, the 
[`damping_driver_nml`](../damping/index.md) namelist needs
to be specified. </br>
**Default:** `False`
</br>
</br>
### **Radiation**
#### `two_stream_gray`
*bool*</br>
This is one of three choices for radiation, the others being [`do_rrtm_radiation`](#do_rrtm_radiation) and 
[`do_socrates_radiation`](#do_socrates_radiation). If `True`, the 
[`two_stream_gray_nml`](../radiation/two_stream_gray.md) namelist needs
to be specified. </br>
**Default:** `True`

#### `do_rrtm_radiation`
*bool*</br>
This is one of three choices for radiation, the others being [`two_stream_gray`](#two_stream_gray) and 
[`do_socrates_radiation`](#do_socrates_radiation). If `True`, the 
[`rrtm_radiation_nml`](../radiation/rrtm.md) namelist needs
to be specified. </br>
**Default:** `False`

#### `do_socrates_radiation`
*bool*</br>
This is one of three choices for radiation, the others being [`two_stream_gray`](#two_stream_gray) and 
[`do_rrtm_radiation`](#do_rrtm_radiation). If `True`, the 
[`socrates_rad_nml`](../radiation/socrates.md) namelist needs
to be specified. </br>
**Default:** `False`
</br>
</br>
### **Turbulence**
These options are all related to how Isca computes surface exchange of heat, momentum and humidity.

#### `turb`
*bool*</br>
If `True`, vertical diffusion is enabled and the [`vert_turb_driver_nml`](../turbulence/vert_turb_driver.md) 
namelist needs to be specified. </br>
**Default:** `False`

#### `do_virtual`
*bool*</br>
If `True`, the virtual temperature is used in the 
[vertical diffusion module](https://github.com/ExeClim/Isca/blob/master/src/atmos_param/vert_diff/vert_diff.F90). </br>
**Default:** `False`

#### `roughness_moist`
*float*</br>
Roughness length for use in surface moisture exchange.</br>
**Default:** `0.05`

#### `roughness_mom`
*float*</br>
Roughness length for use in surface momentum exchange.</br>
**Default:** `0.05`

#### `roughness_heat`
*float*</br>
Roughness length for use in surface heat exchange.</br>
**Default:** `0.05`
</br>
</br>

### **Land and Hydrology**
Land and hydrology processes are predominantly dealt with in the [`surface_flux_nml`](../surface/surface_flux.md) and 
the [`mixed_layer_nml`](../surface/mixed_layer.md) namelists, but land and bucket hydrology options are initialised 
with the following namelist parameters.

Land is [implemented in Isca](https://execlim.github.io/Isca/modules/surface_flux.html#land) 
via adjustment of the [roughness length](#land_roughness_prefactor) (larger over land), 
[evaporative flux](../surface/surface_flux.md#land) (smaller over land), 
[albedo](../surface/mixed_layer.md#land_albedo_prefactor) (larger over land) and 
[mixed layer depth](../surface/mixed_layer.md#land_depth)/
[heat capacity](../surface/mixed_layer.md#land_h_capacity_prefactor) (smaller over land).


#### `mixed_layer_bc`
*bool*</br>
If `True`, the `mixed_layer` module is called and the [`mixed_layer_nml`](../surface/mixed_layer.md)
namelist needs to be specified. </br>
**Default:** `False`

#### `land_option`
*string*</br>
There are 3 choices of the land mask in *Isca*:

* `input` - Read land mask from input file.
* `zsurf` - Define land where surface geopotential height at model initialisation exceeds a threshold of 10.
* `none` - Do not apply land mask.

**Default:** `none`

#### `land_file_name`
*string*</br>
Filename for the input land-mask.</br>Only ever required if [`land_option = 'input'`](#land_option).</br>
**Default:** `'INPUT/land.nc'`

#### `land_field_name`
*string*</br>
Field name in the input land-mask *netcdf*.</br>Only ever required if [`land_option = 'input'`](#land_option).</br>
**Default:** `land_mask`

#### `land_roughness_prefactor`
*float*</br>
Multiplier on the [roughness lengths](#turbulence) to allow land-ocean contrast.</br>
Expect this to be greater than `1` because land is rougher than ocean. </br>
Only ever required if [`land_option`](#land_option) is not `none`.</br>
**Default:** `1.0`

#### `bucket`
*bool*</br>
If `True`, use bucket hydrology. </br>
**Default:** `False`

#### `init_bucket_depth`
*float*</br>
Value at which to initialise bucket water depth over ocean (in $m$). Should be large. </br>
Only ever required if [`bucket = .true.`](#bucket).</br>
**Default:** `1000`

#### `init_bucket_depth_land`
*float*</br>
Value at which to initialise bucket water depth over land (in $m$). </br>
Only ever required if [`bucket = .true.`](#bucket).</br>
**Default:** `20`

#### `max_bucket_depth_land`
*float*</br>
Maximum depth of water in bucket over land following initialisation.. </br>
Only ever required if [`bucket = .true.`](#bucket).</br>
**Default:** `0.15`

#### `robert_bucket`
*float*</br>
Robert coefficient for [Roberts-Asselin-Williams filter](https://execlim.github.io/Isca/references.html#williams2011) 
on bucket leapfrog timestepping. </br>
Only ever required if [`bucket = .true.`](#bucket).</br>
**Default:** `0.04`

#### `raw_bucket`
*float*</br>
RAW coefficient for [Roberts-Asselin-Williams filter](https://execlim.github.io/Isca/references.html#williams2011) 
on bucket leapfrog timestepping. </br>
Only ever required if [`bucket = .true.`](#bucket).</br>
**Default:** `0.53`

## Diagnostics
The diagnostics for 
[this module](https://execlim.github.io/Isca/modules/idealised_moist_phys.html#diagnostics) 
can be specified using the `module_name` of `atmosphere` in the 
diagnostic table file. The list of available diagnostics is available on 
[Isca's website](https://execlim.github.io/Isca/modules/mixedlayer.html#diagnostics). 
Some are also given below:

### `precipitation`
Rain and Snow from resolved and parameterised condensation/convection.</br>
*Dimensions: time, lat, lon*</br>
*Units: $kgm^{-2}s^{-1}$*

### **Condensation**
#### `condensation_rain`
Rain from condensation.</br>
*Dimensions: time, lat, lon*</br>
*Units: $kgm^{-2}s^{-1}$*

#### `dt_qg_condensation`
Moisture tendency from condensation.</br>
*Dimensions: time, lat, lon, pressure*</br>
*Units: $kgkg^{-1}s^{-1}$*

#### `dt_qg_condensation`
Temperature tendency from condensation.</br>
*Dimensions: time, lat, lon, pressure*</br>
*Units: $Ks^{-1}$*


### **Convection**
#### `convection_rain`
Convective precipitation.</br>
*Dimensions: time, lat, lon*</br>
*Units: $kgm^{-2}s^{-1}$*

#### `dt_qg_convection`
Moisture tendency from convection.</br>
*Dimensions: time, lat, lon, pressure*</br>
*Units: $kgkg^{-1}s^{-1}$*

#### `dt_qg_convection`
Temperature tendency from convection.</br>
*Dimensions: time, lat, lon, pressure*</br>
*Units: $Ks^{-1}$*

#### `cape`
Convective Available Potential Energy.</br>
*Dimensions: time, lat, lon*</br>
*Units: $JK^{-1}$*

#### `cin`
Convective Inhibition.</br>
*Dimensions: time, lat, lon*</br>
*Units: $JK^{-1}$*

### **Near Surface Variables**
#### `temp_2m`
Air temperature $2m$ above surface.</br>
*Dimensions: time, lat, lon*</br>
*Units: $K$*

#### `sphum_2m`
Specific humidity $2m$ above surface.</br>
*Dimensions: time, lat, lon*</br>
*Units: $kg/kg$*

#### `rh_2m`
Relative humidity $2m$ above surface.</br>
*Dimensions: time, lat, lon*</br>
*Units: $%$*

#### `u_10m`
Zonal wind $10m$ above surface.</br>
*Dimensions: time, lat, lon*</br>
*Units: $ms^{-1}$*

#### `v_10m`
Meridional wind $10m$ above surface.</br>
*Dimensions: time, lat, lon*</br>
*Units: $ms^{-1}$*
