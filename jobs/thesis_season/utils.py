import xarray as xr
from typing import List, Optional, Union

from jobs.thesis_season.column.utils import get_fit_coef_complex_xr, lat_min, lat_max

var_keep = ['temp', 'ps', 'sphum', 'olr', 'swdn_toa', 'swdn_sfc', 'lwdn_sfc', 'lwup_sfc', 'flux_t',
            'flux_lhe', 't_surf']       # just the fluxes, no variables

#
# def load_ds(exp_name: str, exp_dir: str, var_keep: List = var_keep,
#             lat_min: float = lat_min, lat_max: float = lat_max,
#             first_month_file: Optional[int] = None) -> xr.Dataset:
