# Namelists
To run a simulation, a [namelist *.nml* file is required](../Isca/getting_started.md#required-files), 
specifying the value of configuration options in a variety of categories or namelists. A diagnostic table 
file is also required to specify what diagnostics to save for the experiment.

Each page in this section is devoted to a particular *namelist*, indicating the source code where it is specified 
and any relevant pages on [Isca's website](https://execlim.github.io/Isca/index.html). There are then two sections, 
[*Options*](#options) and [*Diagnostics*](#diagnostics):

## Options
This section is related to the namelist *.nml* file and indicates the available options that can be configured
in that particular namelist, as well as their default values.

These options can be found, by looking at the [module source code](#mod_name) (e.g. 
[idealized_moist_phys.F90](https://github.com/ExeClim/Isca/blob/master/src/atmos_spectral/driver/solo/idealized_moist_phys.F90) for
[`idealized_moist_phys_nml`](main/idealized_moist_physics.md)) and searching for the word *namelist*.
You should then find some code like that below, indicating all the options in the namelist.
```fortran
namelist / idealized_moist_phys_nml / turb, lwet_convection, do_bm, do_ras, &
                                      roughness_heat, do_cloud_simple,      &
                                      two_stream_gray, do_rrtm_radiation,   &
                                      do_damping
```
Then just above this code, the options will be initialized with their default values.

## Diagnostics
This section is related to the diagnostic table file and indicates the diagnostics relevant to the namelist that 
can be saved to the data directory.

These diagnostics can be found, by looking at the [module source code](#mod_name) (e.g. 
[idealized_moist_phys.F90](https://github.com/ExeClim/Isca/blob/master/src/atmos_spectral/driver/solo/idealized_moist_phys.F90) for
[`idealized_moist_phys_nml`](main/idealized_moist_physics.md)) and searching for the word *mod_name*.
You should then find some code like that below, indicating the available diagnostics for the namelist.
```fortran
id_cond_dt_qg = register_diag_field(mod_name, 'dt_qg_condensation',        &
     axes(1:3), Time, 'Moisture tendency from condensation','kg/kg/s')
id_cond_dt_tg = register_diag_field(mod_name, 'dt_tg_condensation',        &
     axes(1:3), Time, 'Temperature tendency from condensation','K/s')
```

### `mod_name`
The `mod_name` used in the [diagnostic table file](../Isca/getting_started.md#required-files) is not always the same
as the corresponding *namelist* associated with it. The table below gives the `mod_name` associated with each 
*namelist*. There is a [flowchart on Isca's website](https://execlim.github.io/Isca/isca_structure.html#isca-structure) 
which is quite useful for understanding how the modules are related to each other.

| Module                                                                                                                               | *namelist*                                                             | `mod_name`                                                  |
|--------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------|-------------------------------------------------------------|
|                                                                                                                                      | **Main**                                                               |                                                             |
| [`atmos_model.F90`](https://github.com/ExeClim/Isca/blob/master/src/atmos_solo/atmos_model.F90)                                      | [`main_nml`](main/index.md#options)                                    | N/A                                                         |
| [`atmosphere.F90`](https://github.com/ExeClim/Isca/blob/master/src/atmos_spectral/driver/solo/atmosphere.F90)                        | [`atmosphere_nml`](main/atmosphere.md#options)                         | N/A                                                         |
| [`hs_forcing.F90`](https://github.com/ExeClim/Isca/blob/master/src/atmos_param/hs_forcing/hs_forcing.F90)                            | [`hs_forcing_nml`](main/held_suarez.md#options)                        | [`hs_forcing`](main/held_suarez.md#diagnostics)             |
| [`idealized_moist_phys.F90`](https://github.com/ExeClim/Isca/blob/master/src/atmos_spectral/driver/solo/idealized_moist_phys.F90)    | [`idealized_moist_phys_nml`](main/idealized_moist_physics.md#options)  | [`atmosphere`](main/idealized_moist_physics.md#diagnostics) |
| [`spectral_dynamics.F90`](https://github.com/ExeClim/Isca/blob/master/src/atmos_spectral/model/spectral_dynamics.F90)                | [`spectral_dynamics_nml`](main/spectral_dynamics.md#options)           | [`dynamics`](main/spectral_dynamics.md#diagnostics)         |
|                                                                                                                                      | **Convection**                                                         |                                                             |
| [`qe_moist_convection.F90`](https://github.com/ExeClim/Isca/blob/master/src/atmos_param/qe_moist_convection/qe_moist_convection.F90) | [`qe_moist_convection_nml`](convection/qe_moist_convection.md#options) | [N/A](convection/index.md)                                  |
| [`betts_miller.f90`](https://github.com/ExeClim/Isca/blob/master/src/atmos_param/betts_miller/betts_miller.f90)                      | [`betts_miller_nml`](convection/betts_miller.md#options)               | [N/A](convection/index.md)                                  |
| [`ras.F90`](https://github.com/ExeClim/Isca/blob/master/src/atmos_param/ras/ras.f90)                                                 | [`ras_nml`](convection/ras.md#options)                                 | [`ras`](convection/ras.md#diagnostics)                      |
|                                                                                                                                      | **Condensation**                                                       |                                                             |
| [`lscale_cond.F90`](https://github.com/ExeClim/Isca/blob/master/src/atmos_param/lscale_cond/lscale_cond.F90)                         | [`lscale_cond_nml`](condensation/lscale_cond.md#options)               | [N/A](condensation/index.md)                                |
| [`sat_vapor_pres.F90`](https://github.com/ExeClim/Isca/blob/master/src/shared/sat_vapor_pres/sat_vapor_pres.F90)                     | [`sat_vapor_pres_nml`](condensation/lscale_cond.md#options)            | [N/A](condensation/index.md)                                |
|                                                                                                                                      | **Radiation**                                                          |                                                             |
| [`two_stream_gray_rad.F90`](https://github.com/ExeClim/Isca/blob/master/src/atmos_param/two_stream_gray_rad/two_stream_gray_rad.F90) | [`two_stream_gray_rad_nml`](radiation/two_stream_gray.md#options)      | [`two_stream`](radiation/two_stream_gray.md#diagnostics)    |
| [`rrtm_radiation.F90`](https://github.com/ExeClim/Isca/blob/master/src/atmos_param/rrtm_radiation/rrtm_radiation.f90)                | [`rrtm_radiation_nml`](radiation/rrtm.md#options)                      | [`rrtm_radiation`](radiation/rrtm.md#diagnostics)           |
| [`socrates_config_mod.F90`](https://github.com/ExeClim/Isca/blob/master/src/atmos_param/socrates/interface/socrates_config_mod.f90)  | [`socrates_rad_nml`](radiation/socrates.md#options)                    | [`socrates`](radiation/socrates.md#diagnostics)             |
|                                                                                                                                      | **Surface**                                                            |                                                             |
| [`mixed_layer.F90`](https://github.com/ExeClim/Isca/blob/master/src/atmos_spectral/driver/solo/mixed_layer.F90)                      | [`mixed_layer_nml`](surface/mixed_layer.md#options)                    | [`mixed_layer`](surface/mixed_layer.md#diagnostics)         |
| [`surface_flux.F90`](https://github.com/ExeClim/Isca/blob/master/src/coupler/surface_flux.F90)                                       | [`surface_flux_nml`](surface/surface_flux.md#options)                  | N/A                                                         |
| [`spectral_init_cond.F90`](https://github.com/ExeClim/Isca/blob/master/src/atmos_spectral/init/spectral_init_cond.F90)               | [`spectral_init_cond_nml`](surface/topography.md#options)              | N/A                                                         |
|                                                                                                                                      | **Damping**                                                            |                                                             |
| [`damping_driver.F90`](https://github.com/ExeClim/Isca/blob/master/src/atmos_param/damping_driver/damping_driver.f90)                | [`damping_driver_nml`](damping/index.md#options)                       | [`damping`](damping/index.md#diagnostics)                   |
|                                                                                                                                      | **Turbulence**                                                         |                                                             |
| [`vert_turb_driver.F90`](https://github.com/ExeClim/Isca/blob/master/src/atmos_param/vert_turb_driver/vert_turb_driver.F90)          | [`vert_turb_driver_nml`](turbulence/vert_turb_driver.md#options)       | [`vert_turb`](turbulence/vert_turb_driver.md#diagnostics)   |
| [`diffusivity.F90`](https://github.com/ExeClim/Isca/blob/master/src/atmos_param/diffusivity/diffusivity.F90)                         | [`diffusivitiy_nml`](turbulence/diffusivity.md#options)                | N/A                                                         |

