import os
import numpy as np
import xarray as xr
from netCDF4 import Dataset
from ..utils.load import load_namelist
from typing import Optional, Tuple, Callable, Union
from .cmip_time import day_number_to_date, FakeDT
import numpy_indexed
import warnings


# def write_var(file_name: str, exp_dir: str, var_array: np.ndarray, var_name: Optional[str] = None,
#               namelist_file: Optional[str] = 'namelist.nml') -> None:
#     """
#     This function generates a *.nc* file containing the variable `sst`, indicating the sst values at each coordinate.
#
#     Args:
#         file_name: *.nc* file containing the variable will be saved with this name
#             in the folder given by `input_dir` in the `experiment_details` namelist of the `namelist_file`.
#         namelist_file: File path to namelist `nml` file for the experiment.
#             This specifies the physical parameters e.g. resolution used for the simulation.
#         var_array: `float [n_lat x n_lon]`
#             Variable values at each coordinate.
#         var_name: `str`
#             Name of variable to be saved within `file_name`. If not given, weill set to `file_name`.
#
#     """
#     namelist = load_namelist(namelist_file=os.path.join(exp_dir, namelist_file))
#     # Load in grid file containing longitude/latitude info for the resolution used for this experiment
#     res = int(namelist['experiment_details']['resolution'][1:])  # resolution for experiment read in
#     grid_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'grid_files')
#     grid_file = os.path.join(grid_dir, f"t{res}_grid.nc")
#
#     resolution_file = Dataset(grid_file, 'r', format='NETCDF3_CLASSIC')
#     lons = resolution_file.variables['lon'][:]
#     lats = resolution_file.variables['lat'][:]
#     nlon = lons.shape[0]
#     nlat = lats.shape[0]
#
#     lonb = resolution_file.variables['lonb'][:]
#     latb = resolution_file.variables['latb'][:]
#     nlonb = lonb.shape[0]
#     nlatb = latb.shape[0]
#
#
#     # Write land and topography arrays to file
#     file_name = file_name.replace('.nc', '')
#     file_name = file_name + '.nc'
#     file_name = os.path.join(exp_dir, file_name)
#     if os.path.exists(file_name):
#         raise ValueError(f"The file {file_name} already exists. Delete or re-name this to continue.")
#     sst_file = Dataset(file_name, 'w', format='NETCDF3_CLASSIC')
#     sst_file.createDimension('lat', nlat)
#     sst_file.createDimension('lon', nlon)
#     latitudes = sst_file.createVariable('lat','float64',('lat',))
#     longitudes = sst_file.createVariable('lon','float64',('lon',))
#     # Set units otherwise get error when reading in with Isca
#     latitudes.units = resolution_file.variables['lat'].units
#     longitudes.units = resolution_file.variables['lon'].units
#     latitudes.cartesian_axes = resolution_file.variables['lat'].cartesian_axis
#     longitudes.cartesian_axes = resolution_file.variables['lon'].cartesian_axis
#     latitudes.edges = resolution_file.variables['lat'].edges
#     longitudes.edges = resolution_file.variables['lon'].edges
#     latitudes.long_name = resolution_file.variables['lat'].long_name
#     longitudes.long_name = resolution_file.variables['lon'].long_name
#
#     sst_file.createDimension('latb', nlatb)
#     sst_file.createDimension('lonb', nlonb)
#     latitudesb = sst_file.createVariable('latb','float64',('latb',))
#     longitudesb = sst_file.createVariable('lonb','float64',('lonb',))
#     latitudesb.units = resolution_file.variables['latb'].units
#     longitudesb.units = resolution_file.variables['lonb'].units
#
#     sst_array_netcdf = sst_file.createVariable('sst','f4',('lat','lon',))
#     latitudes[:] = lats
#     longitudes[:] = lons
#     latitudesb[:] = latb
#     longitudesb[:] = lonb
#     sst_array_netcdf[:] = var_array
#     sst_file.close()
#     print('Output written to: ' + file_name)


def create_grid_file(res: int = 21):
    """
    Function to create a grid file e.g. `t21_grid.nc` for a specified resolution.
    This grid file is required to create timeseries files and includes the variables `lon`, `lonb`, `lat`, `latb`,
    `pfull` and `phalf`.

    ??? note "Pressure values"
        The `'_exp.nc'` files in the `grid_files` folder for each resolution contain only 2 pressure values.
        To change this, the `'_exp.nc'` files will be need to be created again using different pressure levels
        as specified in the `gridfile_namelist.nml` file. I.e. a quick experiment needs to be run
        using a modified version of `gridfile_namelist.nml`.

    Function is extended from an
    [Isca script](https://github.com/ExeClim/Isca/blob/master/src/extra/python/scripts/gfdl_grid_files/grid_file_generator.py)
    but also includes pressure level info.

    Args:
        res: Experiment resolution. Must be either `21`, `42` or `85`.

    """
    # Read in an example data output file for the specified resolution that has suffix '_exp'
    # This data was produced using the gridfile_namelist.nml and gridfile_diag_table input files.
    grid_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'grid_files')
    input_file_name = os.path.join(grid_dir, f"t{res}_exp.nc")
    if not os.path.exists(input_file_name):
        raise ValueError(f"The file {input_file_name} does not exist.\n"
                         f"res provided was {res} but it must be either 21, 42 or 85.")
    resolution_file = Dataset(input_file_name, 'r', format='NETCDF3_CLASSIC')

    # Load longitude/latitude information from the file
    lons = resolution_file.variables['lon'][:]
    lats = resolution_file.variables['lat'][:]
    lonsb = resolution_file.variables['lonb'][:]  # longitude edges
    latsb = resolution_file.variables['latb'][:]  # latitude edges
    # lonb always has one more value than lon, because lonb refers to longitude edges. Same with lat and latb.

    # Load in pressure information from the file
    pfull = resolution_file.variables['pfull'][:]
    phalf = resolution_file.variables['phalf'][:]  # always one more value in phalf than in pfull

    # Save grid data for this resolution to the same folder as input
    output_file_name = os.path.join(grid_dir, f"t{res}_grid.nc")
    if os.path.exists(output_file_name):
        raise ValueError(f"The file {output_file_name} already exists.")
    else:
        print('Grid file written to: ' + output_file_name)
    output_file = Dataset(output_file_name, 'w', format='NETCDF3_CLASSIC')

    output_file.createDimension('lat', lats.shape[0])
    output_file.createDimension('lon', lons.shape[0])
    output_file.createDimension('latb', latsb.shape[0])
    output_file.createDimension('lonb', lonsb.shape[0])
    output_file.createDimension('pfull', pfull.shape[0])
    output_file.createDimension('phalf', phalf.shape[0])

    # Create variable for each dimension and give units and axis.
    latitudes = output_file.createVariable('lat', 'f4', ('lat',))
    latitudes.units = 'degrees_N'.encode('utf-8')
    latitudes.cartesian_axis = 'Y'
    latitudes.long_name = 'latitude'

    longitudes = output_file.createVariable('lon', 'f4', ('lon',))
    longitudes.units = 'degrees_E'.encode('utf-8')
    longitudes.cartesian_axis = 'X'
    longitudes.long_name = 'longitude'

    latitudesb = output_file.createVariable('latb', 'f4', ('latb',))
    latitudes.edges = 'latb'
    latitudesb.units = 'degrees_N'.encode('utf-8')
    latitudesb.cartesian_axis = 'Y'
    latitudesb.long_name = 'latitude edges'

    longitudesb = output_file.createVariable('lonb', 'f4', ('lonb',))
    longitudes.edges = 'lonb'
    longitudesb.units = 'degrees_E'.encode('utf-8')
    longitudesb.cartesian_axis = 'X'
    longitudesb.long_name = 'longitude edges'

    pfulls = output_file.createVariable('pfull', 'f4', ('pfull',))
    pfulls.units = 'hPa'
    pfulls.cartesian_axis = 'Z'
    pfulls.positive = 'down'
    pfulls.long_name = 'full pressure level'

    phalfs = output_file.createVariable('phalf', 'f4', ('phalf',))
    phalfs.units = 'hPa'
    phalfs.cartesian_axis = 'Z'
    phalfs.positive = 'down'
    phalfs.long_name = 'half pressure level'

    # Assign values to each dimension variable.
    latitudes[:] = lats
    longitudes[:] = lons
    latitudesb[:] = latsb
    longitudesb[:] = lonsb
    pfulls[:] = pfull
    phalfs[:] = phalf

    output_file.close()


def create_time_arr(duration_days: int, time_spacing: int, start_year: int = 0, start_month: int = 1,
                    start_day: int = 1, start_time: str = '00:00:00',
                    calendar: str = '360_day') -> Tuple[FakeDT, np.ndarray, str]:
    """
    Creates a `time_arr` which indicates the times in the simulation when a specified variable e.g. $CO_2$ concentration
    may be varied.

    Function is extended from an
    [`create_time_arr`](https://github.com/ExeClim/Isca/blob/master/src/extra/python/scripts/create_timeseries.py)
    given by *Isca*.

    Args:
        duration_days: Length of time in days that a variable is varying in the simulation.
        time_spacing: Spacing between change in variable value in days.
            E.g. if `time_spacing=360` then you would change the value of the variable every year if
            `calendar='360_day'`.
        start_year: Start year of simulation, must be either `0` or a 4 digit integer e.g. `2000`.
        start_month: Start month of simulation, January is `1`.
        start_day: Start day of simulation, first day of month is `1`.
        start_time: Start time of simulation in the form `hh:mm:ss`.
        calendar: Calendar used for simulation.
            Valid options are  `standard`, `gregorian`, `proleptic_gregorian`, `noleap`, `365_day`, `360_day`,
            `julian`, `all_leap`, `366_day`.

    Returns:
        `time_arr`: `cftime.datetime [duration_days/time_spacing]`.
            A `FakeDT` object containing a `cftime.datetime` date for each date in `day_number`.
        `day_number`: `int [duration_days/time_spacing]`.
            Index of days of simulation on which variable changes.
        `time_units`: time units in the form `'days since <ref_date> <ref_time>'`.</br>
            `<ref_date>` is in the form `yyyy-mm-dd`.</br>
            `<ref_time>` is in the form `hh:mm:ss`.</br>
    """
    day_number = np.arange(0, duration_days, time_spacing)
    time_units = f"days since {start_year:04d}-{start_month:02d}-{start_day:02d} {start_time}"
    time_arr = day_number_to_date(day_number, calendar, time_units)
    return time_arr, day_number, time_units


def write_var(file_name: str, exp_dir: str,
              var_array: np.ndarray,
              lat_var: np.ndarray, lon_var: np.ndarray,
              time_var: Optional[np.ndarray] = None, pressure_var: Optional[np.ndarray] = None,
              lat_interpolate: bool = False, lon_interpolate: bool = False,
              time_interpolate: Union[bool, str] = False, pressure_interpolate: bool = False,
              var_name: Optional[str] = None, namelist_file: str='namelist.nml'):
    """
    Creates a *.nc* file containing the value of `var_name` as a function of time, pressure, latitude and longitude to
    be used during a simulation using pressure, latitude and longitude information from the `t{res}_grid.nc` file.

    Output file will contain array of dimension `[n_time_out, n_pressure_out, n_lat_out, n_lon_out]`. With the
    time and pressure dimension only included if `time_var` and `pressure_var` are specified.

    ??? note "Var saved in Isca output data"
        Note that Isca does interpolation on these files, e.g. if provide `var` at day 0 and day 1, the value of `var`
        output by Isca will be given for time=0.5 days, and will be an average between day 0 and day 1, because
        the day=0.5 value is an average of all time steps between day 0 and day 1.

    ??? note "Pressure values"
        The `'_exp.nc'` files in the `grid_files` folder for each resolution contain only 2 pressure values.
        To change this, the `'_exp.nc'` files will be need to be created again using different pressure levels
        as specified in the `gridfile_namelist.nml` file. I.e. a quick experiment needs to be run
        using a modified version of `gridfile_namelist.nml`.

        Then `create_grid_file` needs to be run to create the new `'_grid.nc'` file for the given resolution.

    Args:
        file_name: *.nc* file containing the value of `var_name` as a function of time, pressure,
            latitude and longitude will be saved with this name in the folder given by `exp_dir`.
        exp_dir: Path to directory of the experiment.
        var_array: `float [n_time_in x n_pressure_in x n_lat_in x n_lon_in]`.</br>
            `var_array[i, j, k, n]` is the value of `var_name` at `time_var[i]`, `pressure_var[j]`, `lat_var[k]`,
            `lon_var[n]`.</br>Do not have to give `time_var` or `pressure_var` though,
            in which case a 2 or 3-dimensional array is expected i.e. if neither provided then `var_array[k, n]`
            is value of `var_name` at `lat_var[k]`, `lon_var[n]`.
        lat_var: `float [n_lat_in]`.</br>
            Latitudes in degrees, that provided variable info for in `var_array`.
        lon_var: `float [n_lon_in]`.</br>
            Longitudes in degrees (from 0 to 360), that provided variable info for in `var_array`.
        time_var: `int [n_time_in]`.</br>
            Time in days, that provided variable info for in `var_array`. First day would be 0, second day 1...
        pressure_var: `float [n_pressure_in]`.</br>
            Pressure in hPa, that provided variable info for in `var_array`.
        lat_interpolate: Output file will have `var` defined at `lat_out`, with `n_lat_out` latitudes, specified by
            resolution indicated in `experiment_details` section of `namelist_file` within `exp_dir`.</br>
            If `False`, will require that `lat_var` contains all these `lat_out`. Otherwise, will set value of `var`
            at `lat_out` in output file to nearest latitude in `lat_var`.
        lon_interpolate: Output file will have `var` defined at `lon_out`, with `n_lon_out` longitudes, specified by
            resolution indicated in `experiment_details` section of `namelist_file` within `exp_dir`.</br>
            If `False`, will require that `lon_var` contains all these `lon_out`. Otherwise, will set value of `var`
            at `lon_out` in output file to nearest longitude in `lon_var`.
        time_interpolate: Output file will have `var` defined at `time_out`, with `n_time_out` days, specified by
            `n_months_total` indicated in `experiment_details` section of `namelist_file` within `exp_dir`.</br>
            If `False`, will require that `time_var` contains all these `time_out`. If `True`, will set value of `var`
            at `time_out` in output file to nearest time in `time_var`.</br>
            If `wrap`, similar to `True`, but will assume `var` is periodic with period of `time_var[-1]+1`.
            E.g. if provide `time_var=np.arange(360)`, so give each value for a year.
            Then the output value of `var` for `time_out=360` will be set to `var_array` at `time_var=360%(359+1)=0`.
        pressure_interpolate: Output file will have `var` defined at `pressure_out`, with `n_pressure_out` pressures,
            specified by the `'_exp.nc'` files in the `grid_files` folder (typically only 2 values).</br>
            If `False`, will require that `pressure_var` contains all these `pressure_out`.
            Otherwise, will set value of `var` at `pressure_out` in output file to nearest pressure in `pressure_var`.</br>
            May need to create new `grid_file` if need additional pressure values in output file.
        var_name: Name of variable that file is being created for e.g. `'co2'`.</br>
            If not provided, will set to `file_name` (without the `.nc` extension).
        namelist_file: Name of namelist `nml` file for the experiment, within the `exp_dir`.
            This specifies the physical parameters used for the simulation.

    """
    if var_name is None:
        var_name = file_name.replace('.nc', '')
    namelist = load_namelist(namelist_file=os.path.join(exp_dir, namelist_file))
    res = int(namelist['experiment_details']['resolution'][1:])     # resolution for experiment read in

    # Create copy of base file for this resolution and save to file_name
    grid_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'grid_files')
    base_file_name = os.path.join(grid_dir, f"t{res}_grid.nc")
    if not os.path.exists(base_file_name):
        # Create base nc file with all the dimensions if it does not exist.
        create_grid_file(res)
    dataset_base = xr.open_dataset(base_file_name)

    var_dims = ('lat','lon',)

    # Find which latitudes in var_val_array to output onto grid file with specific resolution
    input_lat_inds_for_out = np.argmin(np.abs(dataset_base.lat.to_numpy() - lat_var[:, np.newaxis]), axis=0)
    interpolated_lat_ind = np.where([np.round(dataset_base.lat.to_numpy()[i], 0)
                                     not in np.round(lat_var, 0) for i in range(dataset_base.lat.size)])[0]
    if len(interpolated_lat_ind)>0:
        if not lat_interpolate:
            raise ValueError(f"Simulation in {exp_dir}\nis for resolution=T{res}, but lat_var does not include"
                             f" {np.round(dataset_base.lat.to_numpy()[interpolated_lat_ind], 0)}.\n"
                             f"Closest is {np.round(lat_var[input_lat_inds_for_out[interpolated_lat_ind]], 0)}.\n"
                             f"May need to create new grid file with different resolution using create_grid_file function,"
                             f"or use lat_interpolate=True.")
        if lat_interpolate:
            warnings.warn(f'\nValues in var_array for lat={np.round(lat_var[input_lat_inds_for_out[interpolated_lat_ind]], 0)}\n'
                f'output onto lat={np.round(dataset_base.lat.to_numpy()[interpolated_lat_ind], 0)} in output file which has resolution=T{res}.')

    # Find which longitudes in var_val_array to output onto grid file with specific resolution
    input_lon_inds_for_out = np.argmin(np.abs(dataset_base.lon.to_numpy() - lon_var[:, np.newaxis]), axis=0)
    interpolated_lon_ind = np.where([np.round(dataset_base.lon.to_numpy()[i], 0)
                                     not in np.round(lon_var, 0) for i in range(dataset_base.lon.size)])[0]
    if len(interpolated_lon_ind)>0:
        if not lon_interpolate:
            raise ValueError(f"Simulation in {exp_dir}\nis for resolution=T{res}, but lon_var does not include"
                             f" {np.round(dataset_base.lon.to_numpy()[interpolated_lon_ind], 0)}.\n"
                             f"Closest is {np.round(lon_var[input_lon_inds_for_out[interpolated_lon_ind]], 0)}.\n"
                             f"May need to create new grid file with different resolution using create_grid_file function,"
                             f"or use lon_interpolate=True.")
        if lat_interpolate:
            warnings.warn(f'\nValues in var_array for lon={np.round(lon_var[input_lon_inds_for_out[interpolated_lon_ind]], 0)}\n'
                f'output onto lon={np.round(dataset_base.lon.to_numpy()[interpolated_lon_ind], 0)} in output file which has resolution=T{res}.')

    # Find which pressure values in var_val_array to output onto grid file with specific resolution
    if pressure_var is None:
        dataset_base = dataset_base.drop_dims(['pfull', 'phalf'])       # get rid of pressure dimension
    else:
        var_dims = ('pfull', ) + var_dims
        input_pressure_inds_for_out = np.argmin(np.abs(dataset_base.pfull.to_numpy() -
                                                       pressure_var[:, np.newaxis]), axis=0)
        interpolated_pressure_ind = np.where([np.round(dataset_base.pfull.to_numpy()[i], 0)
                                         not in np.round(pressure_var, 0) for i in range(dataset_base.pfull.size)])[0]
        if len(interpolated_pressure_ind)>0:
            if not pressure_interpolate:
                raise ValueError(f"Using resolution file={base_file_name}\nwhich contains pfull="
                                 f"{np.round(dataset_base.pfull.to_numpy()[interpolated_pressure_ind], 0)}hPa."
                                 f"\nThese are not provided in pressure_var. Closest is "
                                 f"{np.round(pressure_var[input_pressure_inds_for_out[interpolated_pressure_ind]], 0)}hPa\n"
                                 f"May need to create new grid file with different pressure levels using create_grid_file function,"
                                 f"or use pressure_interpolate=True.")
            else:
                warnings.warn(f'\nValues in var_array for pressure='
                              f'{np.round(pressure_var[input_pressure_inds_for_out[interpolated_pressure_ind]], 0)}hPa\n'
                              f'output onto pfull={np.round(dataset_base.pfull.to_numpy()[interpolated_pressure_ind], 0)}hPa'
                              f' in output file.')
    # make sure output file has .nc suffix
    file_name = file_name.replace('.nc', '')
    file_name = file_name + '.nc'
    file_name = os.path.join(exp_dir, file_name)
    if os.path.exists(file_name):
        raise ValueError(f"The file {file_name} already exists. Delete or re-name this to continue.")
    dataset_base.to_netcdf(file_name)

    # Load copy of base file in append mode so can add
    out_file = Dataset(file_name, 'a', format='NETCDF3_CLASSIC')

    if time_var is not None:
        if time_var.size != var_array.shape[0]:
            raise ValueError(f'time_var has {time_var.size} dimensions, but first dimension of var_array has '
                             f'{var_array.shape[0]} elements')
        if time_var.dtype != int:
            raise ValueError(f"time_var={time_var} must be an integer type. 0 refers to first day.")
        var_dims = ('time',) + var_dims
        # Load in namelist file to get details about the calendar used for the experiment
        calendar = namelist['main_nml']['calendar']
        if calendar.lower() == 'thirty_day':
            calendar = '360_day'
        if calendar.lower() == 'no_calendar':
            raise ValueError(f"Calendar for this experiment is {calendar}.\n"
                             f"Not sure what calendar to pass to create_time_arr function.")
        if 'current_date' not in namelist['main_nml']:
            # default start date is 0 year, first month, first day I THINK - NOT SURE.
            current_date = [0, 1, 1, 0, 0, 0]
        else:
            current_date = namelist['main_nml']['current_date']

        # Add time as a dimension and variable
        out_file.createDimension('time', 0)  # Key point is to have the length of the time axis 0, or 'unlimited'.
                                             # This seems necessary to get the code to run properly.
        # Time calendar details
        start_time = f'{current_date[3]:02d}:{current_date[4]:02d}:{current_date[5]:02d}'
        duration_days = namelist['experiment_details']['n_months_total'] * namelist['main_nml']['days']
        # last value in day_number must be more than last day in simulation so can interpolate variable value on all days.
        day_number = np.arange(0, duration_days+1)
        time_units = f"days since {current_date[0]:04d}-{current_date[1]:02d}-{current_date[2]:02d} {start_time}"

        times = out_file.createVariable('time', 'd', ('time',))
        times.units = time_units
        if calendar == '360_day':
            calendar = 'thirty_day_months'
        times.calendar = calendar.upper()
        times.calendar_type = calendar.upper()
        times.cartesian_axis = 'T'
        times[:] = day_number

        # What indices of input variable do we use for output variable - i.e. for each time in output, interpolate
        # to find corresponding nearest time in input
        if time_interpolate == 'wrap':
            time_period = time_var[-1]+1
            input_time_inds_for_out = np.argmin(np.abs(day_number % time_period - time_var[:, np.newaxis]), axis=0)
            interpolated_time_ind = np.where([day_number[i]%time_period not in time_var for i in range(day_number.size)])[0]
        else:
            input_time_inds_for_out = np.argmin(np.abs(day_number - time_var[:, np.newaxis]), axis=0)
            interpolated_time_ind = np.where([day_number[i] not in time_var for i in range(day_number.size)])[0]
        if len(interpolated_time_ind) > 0:
            if not time_interpolate:
                raise ValueError(f"Simulation in {exp_dir}\nis for {duration_days}, but time_var does not include"
                                 f"days {day_number[interpolated_time_ind]}.\n"
                                 f"Closest is {time_var[input_time_inds_for_out[interpolated_time_ind]]}).\n"
                                 f"Need to provide value in var_val_array for each day of simulation, "
                                 f"or use time_interpolate=True.")
            else:
                warnings.warn(f'\nNot all times provided in time_var, had to do some interpolation\nValues in '
                              f'var_array for days={time_var[input_time_inds_for_out[interpolated_time_ind]]}\n'
                              f'output onto days={day_number[interpolated_time_ind]} in output file.')


    # Add variable info to file - allow to vary in 4 dimensions.
    var_out = out_file.createVariable(var_name, 'f4', var_dims)
    # Output variable
    if ('time' in var_dims) and ('pfull' in var_dims):
        var_out[:] = var_array[np.ix_(input_time_inds_for_out, input_pressure_inds_for_out,
                                      input_lat_inds_for_out, input_lon_inds_for_out)]
    elif 'time' in var_dims:
        var_out[:] = var_array[np.ix_(input_time_inds_for_out, input_lat_inds_for_out,
                                      input_lon_inds_for_out)]
    elif 'pfull' in var_dims:
        var_out[:] = var_array[np.ix_(input_pressure_inds_for_out, input_lat_inds_for_out,
                                      input_lon_inds_for_out)]
    else:
        var_out[:] = var_array[np.ix_(input_lat_inds_for_out, input_lon_inds_for_out)]
    out_file.close()
    print('Output written to: ' + file_name)
