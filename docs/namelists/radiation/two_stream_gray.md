# Gray Radiation
The [`two_stream_gray_rad_nml`](https://github.com/ExeClim/Isca/blob/master/src/atmos_param/two_stream_gray_rad/two_stream_gray_rad.F90) 
namelist only ever needs to be specified if 
[`two_stream_gray = .true.`](../main/idealized_moist_physics.md#two_stream_gray) in 
`idealized_moist_phys_nml`.
It contains options which specify the configuration to use to solve the two stream radiative transfer equations, as well
as configuring the incoming solar radiation.
It is described on [Isca's website](https://execlim.github.io/Isca/modules/two_stream_gray_rad.html). </br>
Some of the most common options are described below:

## Options
### `rad_scheme`
*string*</br> There are 4 choices of configuration for solving the two stream radiative transfer equations in *Isca*:

* [`FRIERSON`](https://execlim.github.io/Isca/modules/two_stream_gray_rad.html#frierson-byrne-schemes) - Semi-gray 
scheme with prescribed longwave and shortwave optical depths.</br> 
Changing the [$CO_2$](#do_read_co2) concentration does not affect this scheme.
* [`BYRNE`](https://execlim.github.io/Isca/modules/two_stream_gray_rad.html#frierson-byrne-schemes) - Semi-gray scheme 
with longwave optical depth dependent on water vapour content and $CO_2$ concentration. 
Shortwave optical depth is prescribed.
* [`GEEN`](https://execlim.github.io/Isca/modules/two_stream_gray_rad.html#geen-scheme) - Multi-band scheme with 
two longwave bands and one shortwave band. One longwave band corresponds to an infrared window region ($8-14\mu m$) 
and the second corresponds to all other infrared wavelengths ($>4\mu m$). 
Longwave and shortwave optical depths depend on water vapour content and  concentration.
* [`SCHNEIDER`](https://execlim.github.io/Isca/modules/two_stream_gray_rad.html#schneider-giant-planet-scheme) - Semi-gray 
scheme for use in **giant planet simulations**. Longwave and shortwave optical depths are prescribed. 
Does not require a surface temperature as input, and allows specification of an interior heat flux. </br>
Changing the [$CO_2$](#do_read_co2) concentration does not affect this scheme.

??? note "Reference Pressure, $P_0$"
    A reference pressure, $P_0$, is used in the `FRIERSON`/`BYRNE`/`SCHNEIDER` shortwave optical depth, as well as in 
    the `FRIERSON`/`SCHNEIDER` longwave optical depth. The value of this is set to `pstd_mks` in the 
    [`constants_nml`](https://github.com/ExeClim/Isca/blob/master/src/shared/constants/constants.F90) namelist.
    This has a default value of $10^5 Pa$ i.e. surface pressure on Earth.

**Default:** `FRIERSON`
</br>
</br>

### **Incoming Solar Radiation**
There is a specific section on 
[Isca's website](https://execlim.github.io/Isca/modules/two_stream_gray_rad.html#incoming-solar-radiation)
that explains this.

#### `do_seasonal`
*bool*</br>

* `False`: A diurnally and seasonally averaged insolation is selected. Incoming solar radiation takes the form: </br>
$$
S = \frac{S_{0}}{4}[1+\Delta_{S}P_{2}(\theta)+\Delta_{\text{sw}}\sin\theta]
$$
    * $P_{2} = (1 - 3\sin^{2}\theta)/4$ is the second legendre polynomial.
    * $S_0$ is the [`solar_constant`](#solar_constant).
    * $\Delta_s$ is [`del_sol`](#del_sol).
    * $\Delta_{sw}$ is [`del_sw`](#del_sw).
  
    ??? note "Schneider Insolation Profile"
        If [`rad_scheme`](#rad_scheme) is `SCHNEIDER`, then the insolation with `do_seasonal = False` is:
        $$
        S = \frac{S_{0}}{\pi}\cos\theta 
        $$

* `True`: The time dependent insolation has the form:
$$
S = S_{0}\cos\zeta\left(\frac{a}{r}\right)^{2}
$$
    * $\zeta$ is the zenith angle.
    * $a$ is the semi-major axis of the orbital ellipse.
    * $r$ is the time-varying planet-star distance.

**Default:** `False`

#### `solar_constant`
*float*</br>
The solar constant, $S_0$, in the [insolation equation](#do_seasonal) ($Wm^{-2}$).</br>
**Default:** `1360.0`

#### `del_sol`
*float*</br>
Parameter, $\Delta_s$, in the [insolation equation](#do_seasonal).</br>
It sets the amplitude of the $P_2$ insolation profile between the equator and the pole.</br>
Only ever required if [`do_seasonal = .false.`](#do_seasonal) and [`rad_scheme`](#rad_scheme) is not `SCHNEIDER`.</br>
**Default:** `1.4`

#### `del_sw`
*float*</br>
Parameter, $\Delta_{sw}$, in the [insolation equation](#do_seasonal).</br>
It defines the magnitude of $\sin \theta$ modification to the $P_2$ insolation profile.</br>
Only ever required if [`do_seasonal = .false.`](#do_seasonal) and [`rad_scheme`](#rad_scheme) is not `SCHNEIDER`.</br>
**Default:** `0.0`

#### `use_time_average_coszen`
*bool*</br>
If `True`, average $\cos\zeta$ over the period [`dt_rad_avg`](#dt_rad_avg). </br>
For example, for the Earth's diurnal period, `use_time_average_coszen=True` and `dt_rad_avg=86400.` 
would achieve diurnally averaged insolation. </br>
Only ever required if [`do_seasonal = .true.`](#do_seasonal).</br>
**Default:** `False`

#### `dt_rad_avg`
*float*</br>
Averaging period (seconds) for time-dependent insolation $\Delta t_{\text{avg}}$. 
If equal to `-1`, it sets averaging period to model timestep. </br>
Only ever required if [`do_seasonal = .true.`](#do_seasonal).</br>
**Default:** `-1`

#### `solday`
*integer*</br>
Day of year to run time-dependent insolation perpetually. </br>
If negative, the option to run perpetually on a specific day is not used. </br>
Only ever required if [`do_seasonal = .true.`](#do_seasonal).</br>
**Default:** `-10`

#### `equinox_day`
*float*</br>
Fraction of year (between $0$ and $1$) where Northern Hemisphere autumn equinox occurs.</br>
A value of `0.75` would mean the end of September for 360 day year.</br>
Only ever required if [`do_seasonal = .true.`](#do_seasonal).</br>
**Default:** `0.75`
</br>
</br>

### $CO_2$

#### `do_read_co2`
*bool*</br>
If `True`, reads time-varying $CO_2$ concentration from an [input file](#co2_file). </br> 
The input file needs to be 4D (3 spatial dimensions and time), but no spatial variation should be defined 
(the code only reads in maximum value at a given time). </br>
???+ warning "Compatible [`rad_schemes`](#rad_scheme)"
    Varying $CO_2$ concentration can only be using if [`rad_scheme`](#rad_scheme) is `byrne` or `geen`.
**Default:** `False`

#### `co2_file`
*string*</br>
Name of $CO_2$ file to read. </br>
The file should be in the [`input_dir`](../main/experiment_details.md#input_dir) and
have a *.nc* appendix but that should be left out here. </br>
File is produced using the `create_time_series_file` function in `isca_tools` which is extended from a 
[python script](https://github.com/ExeClim/Isca/blob/master/src/extra/python/scripts/create_co2_timeseries.py) 
provided by *Isca*.</br>
Only ever required if [`do_read_co2 = .true.`](#do_read_co2).</br>
**Default:** `co2`

#### `co2_variable_name`
*string*</br>
Name of $CO_2$ variable in $CO_2$ file.</br>
Only ever required if [`do_read_co2 = .true.`](#do_read_co2).</br>
**Default:** `co2`

#### `carbon_conc`
*float*</br>
Prescribed concentration (in $ppmv$) of $CO_2$ which remains constant throughout the simulation.</br>
Only ever required if [`do_read_co2 = .false.`](#do_read_co2) and [`rad_scheme`](#rad_scheme) is either 
`byrne` or `geen`.</br> For other [`rad_schemes`](#rad_scheme), optical depth is prescribed so $CO_2$ concentration
has no effect.</br>
**Default:** `360.0`

## Diagnostics
The diagnostics for 
[this module](https://github.com/ExeClim/Isca/blob/9560521e1ba5ce27a13786ffdcb16578d0bd00da/src/atmos_param/two_stream_gray_rad/two_stream_gray_rad.F90#L153-L160) 
can be specified using the `module_name` of `two_stream` in the 
diagnostic table file. The list of available diagnostics is available on 
[Isca's website](https://execlim.github.io/Isca/modules/two_stream_gray_rad.html#diagnostics). Some 
of the more common ones are also given below.

### `co2`
Carbon dioxide concentration.</br>
*Dimensions: time*</br>
*Units: $ppmv$*

### **Radiation**
#### `olr`
Outgoing Longwave radiation. May be useful, along with [`swdn_toa`](#swdn_toa) to investigate how long experiment takes 
to spin up.</br>
*Dimensions: time, lat, lon*</br>
*Units: $Wm^{-2}$*

#### `swdn_toa`
Shortwave flux down at top of atmosphere. May be useful, along with [`olr`](#olr) to investigate how long 
experiment takes to spin up.</br>
*Dimensions: time, lat, lon*</br>
*Units: $Wm^{-2}$*

#### `swdn_sfc`
Absorbed shortwave flux at the surface.</br>
*Dimensions: time, lat, lon*</br>
*Units: $Wm^{-2}$*

#### `lwdn_sfc`
Downward longwave flux at the surface.</br>
*Dimensions: time, lat, lon*</br>
*Units: $Wm^{-2}$*

#### `lwup_sfc`
Upward longwave flux at the surface.</br>
*Dimensions: time, lat, lon*</br>
*Units: $Wm^{-2}$*
