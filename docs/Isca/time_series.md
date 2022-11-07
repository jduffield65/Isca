# Time Series
To run an experiment with a variable e.g. $CO_2$ concentration that varies during the simulation, 
a timeseries *.nc* file needs to be created indicating the value of the variable at a given time, pressure, 
latitude and longitude.

This can be done using the 
[`create_time_series_file`](../code/time_series/base.md#isca_tools.time_series.base.create_time_series_file) function.

## Carbon dioxide Example
To produce a file called `co2.nc` for an experiment that had $CO_2$ concentration of $300 ppmv$ for the first 
$5$ years and $600 ppmv$ thereafter, I would run the following:

=== "Code"
    ```python
    from isca_tools import create_time_series_file
    import numpy as np
    
    def co2_func(days, pressure, lat, lon):
        co2_val = np.ones((days.shape[0], pressure.shape[0],
                           lat.shape[0], lon.shape[0])) * 600
        co2_val[days < 360 * 5] = 300
        return co2_val
    
    create_time_series_file('/gpfs1/home/jamd1/Isca/jobs/experiment/co2.nc', 
                            '/gpfs1/home/jamd1/Isca/jobs/experiment/namelist.nml',
                            'co2',co2_func, 360)
    ```
=== "`namelist.nml`"
    ```nml
    &experiment_details
    name = 'experiment'             
    input_dir = '/gpfs1/home/jamd1/Isca/jobs/experiment/'   
    n_months_total = 120          ! 10 year simulation
    n_months_job = 12             ! Each job is 1 year
    n_nodes = 1                 
    n_cores = 16                 
    resolution = 'T21'           
    partition = 'debug'         
    overwrite_data = .false.    
    compile = .false.           
    max_walltime = '01:00:00'   
    /

    &main_nml
        calendar = 'thirty_day'
        current_date = 1, 1, 1, 0, 0, 0
        days = 30
    /

    &atmosphere_nml
        idealized_moist_model = .true.
    /

    &idealized_moist_phys_nml
        two_stream_gray = .true.
    /

    &two_stream_gray_rad_nml
        rad_scheme = 'byrne'            !Must be 'byrne' or 'geen' otherwise optical depth independent of CO2
        do_read_co2 = .true.            !Read in CO2 timeseries from input file
        co2_file = 'co2'                !Tell model name of co2 input file
    /
    ```

The `co2_func` function given as an input to the `create_time_series_file` must take as arguments, time (in days),
pressure, latitude and longitude. It then must output an 
$n_{time} \times n_{pressure} \times n_{latitude} \times n_{longitude}$ numpy array giving the $CO_2$ concentration
at each location for each time.

??? "Pressure values"

    The `create_time_series_file` function loads in pressure coordinates from the `t_{res}_grid.nc` files 
    in `isca_tools/time_series/grid_files`. These files only contain 2 pressure values so if more resolution
    is required in the vertical, new `t_{res}_grid.nc` files need to be created.

    To do this, modify the `gridfile_namelist.nml` file so that 
    [`num_levels`](../namelists/main/spectral_dynamics.md#num_levels) indicates the number of desired pressure values.

    Then run the experiment using this updated file. Copy the `atmos_monthly.nc` file that this creates and place
    it in the `isca_tools/time_series/grid_files` folder with the name `t_{res}_exp.nc`. Now you can run 
    [`create_grid_file`](../code/time_series/base.md#isca_tools.time_series.base.create_grid_file) 
    to get the `t_{res}_grid.nc` file.

    Note that if the variable is `co2` for the 
    [`two_stream_gray_rad_nml`](../namelists/radiation/two_stream_gray.md#co2_file) namelist, 
    no spatial variation is allowed and the max value at each time step is read in.

The `namelist.nml` file indicates the bare minimum variables that need to be specified to run this time varying
$CO_2$ experiment. Because the [calendar](../namelists/main/index.md#calendar) is `thirty_day`, each year is $360$ days
hence why $360$ appears in `co2_func`.

The [`co2_file`](../namelists/radiation/two_stream_gray.md#co2_file) option in the namelist indicates the name of 
the file within the [`input_dir`](../namelists/main/experiment_details.md#input_dir). It must be specified without 
the `.nc` suffix.

To allow for varying $CO_2$ concentration to have any effect, 
[`rad_scheme`](../namelists/radiation/two_stream_gray.md#rad_scheme) must be either `byrne` or `geen`. Otherwise,
the optical depth is prescribed and independent of $CO_2$ concentration.

The same `co2.nc` file can be used with the *Rapid Radiative Transfer Model* too by including 
[`rrtm_radiation_nml`](../namelists/radiation/rrtm.md) in the namelist file:

```nml
&rrtm_radiation_nml
    do_read_co2 = .true.            !Read in CO2 timeseries from input file
    co2_file = 'co2'                !Tell model name of co2 input file
/
```

