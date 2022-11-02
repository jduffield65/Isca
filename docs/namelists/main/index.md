# Main

The [`main_nml`](https://github.com/ExeClim/Isca/blob/master/src/atmos_solo/atmos_model.F90) 
namelist contains options relating to the 
[frequency at which data is saved](https://execlim.github.io/Isca/modules/diag_manager_mod.html#output-files).
Some of the most common options are described below.

## Options
### `days`
*integer*</br>
Data is saved after this many days.</br> E.g. if `days=10`, the first output folder `run0001` will contain data
for the first 10 days of the simulation.</br>
Can also use the options `hours`, `minutes` and `seconds` if you want to increase the precision in this duration.</br>
**Must be specified, typical would be `30`.**</br>
**Default:** `0`

### `calendar`
*string*</br>
Specifies the calendar used for the experiment. </br>
Options are `thirty_day`, `julian`, `noleap` or `no_calendar`.</br>
Typical is `thirty_day` which means months of $30$ days each so a year would be $360$ days.
**Default:** N/A

### `dt_atmosphere`
*integer*</br>
Duration of each time step in the simulation (seconds).</br>
**Must be specified, typical would be `720`.**</br>
**Default:** `0`

### `current_date`
*list - integer*</br>
The start date of the simulation.</br> Not exactly sure, but I think the 6 values are 
`[Year, Month, Day, Hour, Minute, Second]`.</br>
In the example experiments, they use 
[`[1,1,1,0,0,0]`](https://github.com/ExeClim/Isca/blob/master/exp/test_cases/frierson/frierson_test_case.py) and 
[`[2000,1,1,0,0,0]`](https://github.com/ExeClim/Isca/blob/master/exp/test_cases/held_suarez/held_suarez_test_case.py).
</br>
**Default:** `[0,0,0,0,0,0]`
