# Saturation Vapor Pressure
The [`sat_vapor_pres_nml`](https://github.com/ExeClim/Isca/blob/master/src/shared/sat_vapor_pres/sat_vapor_pres.F90) 
namelist specifies options for computing the saturation vapor pressure, specific humidity and vapor mixing ratio
at a specified relative humidity. 
Some of the most common options are described below:

## Options
### `do_simple`
*bool*</br> 
If `True`, the calculation that is performed is simplified. </br>
This option seems to be set to `True` for most example experiments e.g. 
[*bucket_hydrology*](https://github.com/ExeClim/Isca/blob/master/exp/test_cases/bucket_hydrology/bucket_model_test_case.py)
and [*frierson*](https://github.com/ExeClim/Isca/blob/master/exp/test_cases/frierson/frierson_test_case.py). </br>
**Default:** `False`
