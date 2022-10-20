# Idealized Moist Physics
The [`idealized_moist_phys_nml`](https://github.com/ExeClim/Isca/blob/master/src/atmos_spectral/driver/solo/idealized_moist_phys.F90) 
only ever needs to be specified if 
[`idealized_moist_model = True` in `atmosphere_nml`](atmosphere.md#idealized_moist_model).
It contains options which specify the various modules associated with Isca’s moist physics configurations and is
described on [Isca's website](https://execlim.github.io/Isca/modules/idealised_moist_phys.html).
Some of the most common ones are described below:

## Options
### `convection_scheme`
*string*</br> There are 4 choices of convection schemes, as well as no convection, in *Isca*:

* `SIMPLE_BETTS_CONV` - Use Frierson Quasi-Equilibrium convection scheme 
[*[Frierson2007]*](https://execlim.github.io/Isca/references.html#frierson2007).</br>
If this is selected, the [`lscale_cond_nml`](../lscale_cond/index.md) and [`qe_moist_convection_nml`](../convection/qe_moist_convection.md) 
namelists needs to be specified.
* `FULL_BETTS_MILLER_CONV` - Use the Betts-Miller convection scheme 
[*[Betts1986]*](https://execlim.github.io/Isca/references.html#betts1986),
[*[BettsMiller1986]*](https://execlim.github.io/Isca/references.html#bettsmiller1986). </br>
If this is selected, the [`lscale_cond_nml`](../lscale_cond/index.md) and [`betts_miller_nml`](../convection/betts_miller.md) namelists needs to be specified.
* `RAS_CONV` - Use the relaxed Arakawa Schubert convection scheme 
[*[Moorthi1992]*](https://execlim.github.io/Isca/references.html#moorthi1992). </br>
If this is selected, the [`lscale_cond_nml`](../lscale_cond/index.md) and [`ras_nml`](../convection/ras.md) namelists needs to be specified.
* `DRY_CONV` - Use the dry convection scheme 
[*[Schneider2006]*](https://execlim.github.io/Isca/references.html#schneider2006). </br>
This [module](https://github.com/ExeClim/Isca/blob/master/src/atmos_param/dry_convection/dry_convection.f90) does not 
need any additional namelist specified.
* `NO_CONV` - Use no convection scheme.
* `UNSET` - Will raise an error.

**Default:** `UNSET`

### `two_stream_gray`
*bool*</br>
This is one of three choices for radiation, the others being [`do_rrtm_radiation`](#do_rrtm_radiation) and 
[`do_socrates_radiation`](#do_socrates_radiation). If `True`, the 
[`two_stream_gray_nml`](../radiation/two_stream_gray.md) namelist needs
to be specified. </br>
**Default:** `True`

### `do_rrtm_radiation`
*bool*</br>
This is one of three choices for radiation, the others being [`two_stream_gray`](#two_stream_gray) and 
[`do_socrates_radiation`](#do_socrates_radiation). If `True`, the 
[`rrtm_radiation_nml`](../radiation/rrtm.md) namelist needs
to be specified. </br>
**Default:** `False`

### `do_socrates_radiation`
*bool*</br>
This is one of three choices for radiation, the others being [`two_stream_gray`](#two_stream_gray) and 
[`do_rrtm_radiation`](#do_rrtm_radiation). If `True`, the 
[`socrates_rad_nml`](../radiation/socrates.md) namelist needs
to be specified. </br>
**Default:** `False`

### `do_damping`
*bool*</br>
If `True`, the 
[`damping_driver_nml`](../damping/index.md) namelist needs
to be specified. </br>
**Default:** `False`