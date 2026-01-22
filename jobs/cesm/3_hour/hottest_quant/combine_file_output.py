# The save_info script outputs 1 file per each input file. Here, we concatenate across those files to give just one output file.
# Script version of the notebook

import xarray as xr
import os
import fnmatch
import re
import logging
import sys
# Set up logging configuration to output to console and don't output milliseconds, and stdout so saved to out file
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S',
                    stream=sys.stdout)


if __name__ == "__main__":
    try:
        exp_name = sys.argv[1]
    except IndexError:
        exp_name = 'pre_industrial'
    complevel = 4
    print(os.getcwd())
    logger = logging.getLogger()
    output_dir = f'/Users/joshduffield/Desktop/{exp_name}/'
    output_dir = os.path.join(output_dir, 'T_Q_PS_all_lat')
    out_files_all = os.listdir(output_dir)
    # only keep files of correct format
    out_files_all = [file for file in out_files_all if
                      re.fullmatch(r"output\d+\.nc", file)]     # must match output{any_number}.nc
    n_out_files = len(out_files_all)
    logger.info(f"Loading {n_out_files} files from {output_dir}")
    ds = []
    for i in range(n_out_files):
        ds.append(xr.load_dataset(os.path.join(output_dir, out_files_all[i])))
        if i > 0:
            # Concatenate as we go to save memory
            ds = xr.concat(ds, dim="file")
            ds = [ds.max(dim='file')]
        logger.info(f"Finished loading {i+1}/{n_out_files} file")

    # Save concatenated file
    out_file = os.path.join(output_dir, 'output.nc')
    if os.path.exists(out_file):
        raise ValueError('Output file already exists at {}'.format(out_file))
    ds[0].to_netcdf(out_file, format="NETCDF4",
                    encoding={var: {"zlib": True, "complevel": complevel} for var in ds[0].data_vars})
    logger.info(f"Saved file to {out_file}")