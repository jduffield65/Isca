from netCDF4 import Dataset
import os


def create_grid_file(res: int = 21):
    """
    Function to create a grid file e.g. `t21_grid.nc` for a specified resolution.
    This grid file is required to create timeseries files.
    Function is extended from an
    [Isca script](https://github.com/ExeClim/Isca/blob/master/src/extra/python/scripts/gfdl_grid_files/grid_file_generator.py).

    Args:
        res: Experiment resolution. Must be either `21`, `42` or `85`.

    """
    # Read in an example data output file for the specified resolution that has suffix '_exp'
    grid_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'grid_files')
    input_file_name = os.path.join(grid_dir, f"t{res}_exp.nc")
    if not os.path.exists(input_file_name):
        raise ValueError(f"The file {input_file_name} does not exist.\n"
                         f"res provided was {res} but it must be either 21, 42 or 85.")
    resolution_file = Dataset(input_file_name, 'r', format='NETCDF3_CLASSIC')

    # Load longitude/latitude information from the file
    lons = resolution_file.variables['lon'][:]
    lats = resolution_file.variables['lat'][:]
    lonsb = resolution_file.variables['lonb'][:]
    latsb = resolution_file.variables['latb'][:]
    nlon = lons.shape[0]
    nlat = lats.shape[0]
    # nlonb is always one more than nlon, so I assume nlonb refers to full levels and nlon refers to the half levels.
    # Same with nlat and nlatb.
    nlonb = lonsb.shape[0]
    nlatb = latsb.shape[0]

    # Save grid data for this resolution to the same folder as input
    output_file_name = os.path.join(grid_dir, f"t{res}_grid.nc")
    if os.path.exists(output_file_name):
        raise ValueError(f"The file {output_file_name} already exists.")
    output_file = Dataset(output_file_name, 'w', format='NETCDF3_CLASSIC')

    lat = output_file.createDimension('lat', nlat)
    lon = output_file.createDimension('lon', nlon)
    latb = output_file.createDimension('latb', nlatb)
    lonb = output_file.createDimension('lonb', nlonb)
    latitudes = output_file.createVariable('lat', 'f4', ('lat',))
    longitudes = output_file.createVariable('lon', 'f4', ('lon',))
    latitudesb = output_file.createVariable('latb', 'f4', ('latb',))
    longitudesb = output_file.createVariable('lonb', 'f4', ('lonb',))
    latitudes[:] = lats
    longitudes[:] = lons
    latitudesb[:] = latsb
    longitudesb[:] = lonsb

    output_file.close()
