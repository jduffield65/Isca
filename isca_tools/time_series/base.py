from netCDF4 import Dataset, date2num
import os
import numpy as np
from .cmip_time import day_number_to_date, FakeDT
from ..utils.load import load_namelist
from typing import Tuple, Callable
import xarray as xr


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


def create_time_series_file(file_name: str, namelist_file: str, var_name: str,
                            var_val_func: Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray],
                            time_spacing: int):
    """
    Creates a *.nc* file containing the value of `var_name` as a function of time, pressure, latitude and longitude to
    be used during a simulation using pressure, latitude and longitude information from the `t{res}_grid.nc` file.

    ??? note "Pressure values"
        The `'_exp.nc'` files in the `grid_files` folder for each resolution contain only 2 pressure values.
        To change this, the `'_exp.nc'` files will be need to be created again using different pressure levels
        as specified in the `gridfile_namelist.nml` file. I.e. a quick experiment needs to be run
        using a modified version of `gridfile_namelist.nml`.

        Then `create_grid_file` needs to be run to create the new `'_grid.nc'` file for the given resolution.

    Args:
        file_name: *.nc* file containing the value of `var_name` as a function of time, pressure,
            latitude and longitude will be saved with this name in the folder given by `input_dir` in the
            `experiment_details` namelist of the `namelist_file`.
        namelist_file: File path to namelist `nml` file for the experiment.
            This specifies the physical parameters used for the simulation.
        var_name: Name of variable that a time series is being created for e.g. `'co2'`.
        var_val_func: Function which takes as arguments `time`, `pressure`, `latitude` and `longitude` and ouputs
            the value of `var_name` in a [`n_time` x `n_pressure` x `n_lat` x `n_lon`] numpy array.
        time_spacing: Time interval in days at which the value of `var_name` can change.

    """
    # TODO: Maybe change this function so it runs a simple experiment, copies long,lat,pressure info then deletes
    #   all files. I.e. make it so it does not rely on the files in the grid_files folder except for the namelist files.

    namelist = load_namelist(namelist_file=namelist_file)
    res = int(namelist['experiment_details']['resolution'][1:])     # resolution for experiment read in

    # Create copy of base file for this resolution and save to file_name
    grid_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'grid_files')
    base_file_name = os.path.join(grid_dir, f"t{res}_grid.nc")
    if not os.path.exists(base_file_name):
        # Create base nc file with all the dimensions if it does not exist.
        create_grid_file(res)
    dataset_base = xr.open_dataset(base_file_name)

    # make sure output file has .nc suffix
    file_name = file_name.replace('.nc', '')
    file_name = file_name + '.nc'
    file_name = os.path.join(namelist['experiment_details']['input_dir'], file_name)
    if os.path.exists(file_name):
        raise ValueError(f"The file {file_name} already exists. Delete or re-name this to continue.")
    dataset_base.to_netcdf(file_name)

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

    # Load copy of base file in append mode so can add
    out_file = Dataset(file_name, 'a', format='NETCDF3_CLASSIC')
    # Add time as a dimension and variable
    out_file.createDimension('time', 0)  # Key point is to have the length of the time axis 0, or 'unlimited'.
                                         # This seems necessary to get the code to run properly.
    # Time calendar details
    start_time = f'{current_date[3]:02d}:{current_date[4]:02d}:{current_date[5]:02d}'
    duration_days = namelist['experiment_details']['n_months_total'] * namelist['main_nml']['days']
    # last value in day_number must be more than last day in simulation so can interpolate variable value on all days.
    day_number = np.arange(0, duration_days + time_spacing, time_spacing)
    time_units = f"days since {current_date[0]:04d}-{current_date[1]:02d}-{current_date[2]:02d} {start_time}"

    times = out_file.createVariable('time', 'd', ('time',))
    times.units = time_units
    if calendar == '360_day':
        calendar = 'thirty_day_months'
    times.calendar = calendar.upper()
    times.calendar_type = calendar.upper()
    times.cartesian_axis = 'T'
    times[:] = day_number

    # Add variable info to file - allow to vary in 4 dimensions.
    var_array = out_file.createVariable(var_name, 'f4', ('time', 'pfull', 'lat', 'lon',))
    var_array[:] = var_val_func(np.asarray(out_file.variables['time']), np.asarray(out_file.variables['pfull']),
                                np.asarray(out_file.variables['lat']), np.asarray(out_file.variables['lon']))
    out_file.close()
