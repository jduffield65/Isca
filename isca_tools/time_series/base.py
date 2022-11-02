from netCDF4 import Dataset
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


def create_time_series_file(file_name: str, namelist_file: str, res: int, var_name: str,
                            var_val_func: Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray],
                            duration_days: int, time_spacing: int):
    # Maybe give namelist file as input and work out automatically from it e.g. start time and calendar.
    grid_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'grid_files')
    base_file_name = os.path.join(grid_dir, f"t{res}_grid.nc")
    if not os.path.exists(base_file_name):
        # Create base nc file with all the dimensions if it does not exist.
        create_grid_file(res)
    # Create copy of base file for this resolution and save to file_name
    dataset_base = xr.open_dataset(base_file_name)
    dataset_base.to_netcdf(file_name)
    exp_details = load_namelist(namelist_file=namelist_file)['experiment_details']
    time_arr, day_number, time_units = create_time_arr(duration_days, time_spacing, start_year, start_month, start_day,
                                                       start_time, calendar)

    out_file = Dataset(file_name, 'a', format='NETCDF3_CLASSIC')   # Load copy of base file in append mode so can add
    var_array = out_file.createVariable(var_name, 'f4', ('time', 'pfull', 'lat', 'lon',))
    var_array[:] = var_val_func(day_number, out_file.variables['pfull'], out_file.variables['lat'],
                                out_file.variables['lon'])
    out_file.close()
