from netCDF4 import Dataset
import os


def create_grid_file(res: int = 21):
    """
    Function to create a grid file e.g. `t21_grid.nc` for a specified resolution.
    This grid file is required to create timeseries files and includes the variables `lon`, `lonb`, `lat`, `latb`,
    `p_full` and `p_half`.

    Function is extended from an
    [Isca script](https://github.com/ExeClim/Isca/blob/master/src/extra/python/scripts/gfdl_grid_files/grid_file_generator.py).
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
    lonsb = resolution_file.variables['lonb'][:]        # longitude edges
    latsb = resolution_file.variables['latb'][:]        # latitude edges
    # lonb always has one more value than lon, because lonb refers to longitude edges. Same with lat and latb.

    # Load in pressure information from the file
    p_full = resolution_file.variables['pfull'][:]
    p_half = resolution_file.variables['phalf'][:]    # always one more value in p_half than in p_full


    # Save grid data for this resolution to the same folder as input
    output_file_name = os.path.join(grid_dir, f"t{res}_grid.nc")
    if os.path.exists(output_file_name):
        raise ValueError(f"The file {output_file_name} already exists.")
    output_file = Dataset(output_file_name, 'w', format='NETCDF3_CLASSIC')

    output_file.createDimension('lat', lats.shape[0])
    output_file.createDimension('lon', lons.shape[0])
    output_file.createDimension('latb', latsb.shape[0])
    output_file.createDimension('lonb', lonsb.shape[0])
    output_file.createDimension('p_full', p_full.shape[0])
    output_file.createDimension('p_half', p_half.shape[0])
    latitudes = output_file.createVariable('lat', 'f4', ('lat',))
    longitudes = output_file.createVariable('lon', 'f4', ('lon',))
    latitudesb = output_file.createVariable('latb', 'f4', ('latb',))
    longitudesb = output_file.createVariable('lonb', 'f4', ('lonb',))
    pressure_full = output_file.createVariable('p_full', 'f4', ('p_full',))
    pressure_half = output_file.createVariable('p_half', 'f4', ('p_half',))

    latitudes[:] = lats
    longitudes[:] = lons
    latitudesb[:] = latsb
    longitudesb[:] = lonsb
    pressure_full[:] = p_full
    pressure_half[:] = p_half

    output_file.close()
