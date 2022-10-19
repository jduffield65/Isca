# Idealized Moist Physics
The [`idealized_moist_phys_nml`](https://github.com/ExeClim/Isca/blob/master/src/atmos_spectral/driver/solo/idealized_moist_phys.F90) 
only ever needs to be specified if 
[`idealized_moist_model = True` in `atmosphere_nml`](atmosphere.md#idealized_moist_model).
It contains options which specify the various modules associated with Isca’s moist physics configurations. 
Some of the most common ones are described below:

## Options
### `convection_scheme`
*string*</br> There are 4 choices of convection schemes, as well as no convection, in *Isca*:

* `SIMPLE_BETTS_CONV` - Use Frierson Quasi-Equilibrium convection scheme 
[*[Frierson2007]*](https://execlim.github.io/Isca/references.html#frierson2007).</br>
If this is selected, the [`lscale_cond_nml`](lscale_cond.md) and [`qe_moist_convection_nml`](qe_moist_convection.md) 
namelists needs to be specified.
* `FULL_BETTS_MILLER_CONV` - Use the Betts-Miller convection scheme 
[*[Betts1986]*](https://execlim.github.io/Isca/references.html#betts1986),
[*[BettsMiller1986]*](https://execlim.github.io/Isca/references.html#bettsmiller1986). </br>
If this is selected, the [`lscale_cond_nml`](lscale_cond.md) and [`betts_miller_nml`](betts_miller.md) namelists needs to be specified.
* `RAS_CONV` - Use the relaxed Arakawa Schubert convection scheme 
[*[Moorthi1992]*](https://execlim.github.io/Isca/references.html#moorthi1992). </br>
If this is selected, the [`lscale_cond_nml`](lscale_cond.md) and [`ras_nml`](ras.md) namelists needs to be specified.
* `DRY_CONV` - Use the dry convection scheme 
[*[Schneider2006]*](https://execlim.github.io/Isca/references.html#schneider2006). </br>
This [module](https://github.com/ExeClim/Isca/blob/master/src/atmos_param/dry_convection/dry_convection.f90) does not 
need any additional namelist specified.
* `NO_CONV` - Use no convection scheme.
* `UNSET` - Will raise an error.

**Default:** `UNSET`
