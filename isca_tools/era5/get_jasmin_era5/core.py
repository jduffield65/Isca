from datetime import datetime
import pathlib
import numpy as np
import pandas as pd
import xarray as xr
import warnings
from typing import Literal

from .utils import get_pl, get_gz, sel_era5, convert_lnsp_to_sp


class Find_era5:
    """

    """
    def __init__(self, archive: Literal[None, 1, 't'] = None):
        """
        Initialise object to load ERA5 data from JASMIN
        Args:
            archive: There are three types of ERA5 archives:

                * `None` to use default ERA5 archive at `/badc/ecmwf-era5`

                * `1` to use ERA5.1 at `/badc/ecmwf-era51`,
                    which is suggested for model level data in years 2000-2006 inclusive.

                * `t` to use Preliminary at `/badc/ecmwf-era5t`, near real-time data
        """
        self._init_vars(archive)
        self.pl = Pressure_levels_era5(archive)
        self.gz = Geopotential_levels_era5(archive)
        self.enda = Ensemble_era5(archive)

    def _init_vars(self, archive: Literal[None, 1, 't'] = None):
        self.archive = '' if archive is None else str(archive)
        self.path = pathlib.Path(f"/badc/ecmwf-era5{self.archive}/data/")
        self._INVARIANTS = [
            "anor",
            "cl",
            "cvh",
            "cvl",
            "dl",
            "isor",
            "lsm",
            "sdfor",
            "sdor",
            "slor",
            "slt",
            "tvh",
            "tvl",
            "z",
        ]
        self._INVARIANT_DATE = datetime(2000, 1, 1)

        self._ML_VARS = ['sp', 'lnsp', 'o3', 'q', 't', 'u', 'v', 'vo', 'z']       # variables on model levels
        self._SURF_VARS = ['10u', '10v', '2d', '2t', 'asn', 'cape', 'ci',   # variables on surface level
                           'msl', 'sd', 'skt', 'sst', 'tcc', 'tcwv']
        self._ML_WARNING_YEARS = np.arange(2000, 2007).tolist()  # in these years model level data suffer from statospheric cold biases - should use ERA5.1

    def __getitem__(self, args):
        var = args[0]
        date = args[1]
        sel = {}
        if len(args) > 2 and args[2] is not None:
            sel["level"] = args[2]
        if len(args) > 3 and args[3] is not None:
            sel["longitude"] = args[3]
        if len(args) > 4 and args[4] is not None:
            sel["latitude"] = args[4]

        if len(args) > 5:
            model = args[5]
        else:
            model = "oper"

        if isinstance(date, slice):
            if date.step is None:
                freq = "1h"
            else:
                freq = date.step
            dates = (
                pd.date_range(
                    pd.to_datetime(date.start),
                    pd.to_datetime(date.stop),
                    inclusive="left",
                    freq=freq,
                )
                .to_pydatetime()
                .tolist()
            )
        else:
            dates = [pd.to_datetime(date).to_pydatetime()]

        if isinstance(var, str):
            var = [var]

        # If requested surface pressure, must get log of surface pressure first and convert later
        # Record info here
        sp_info = {'in_var': 'sp' in var}
        if sp_info['in_var']:
            var.remove('sp')
            if 'lnsp' not in var:
                # Requested just sp
                var.append('lnsp')
                sp_info['delete_lnsp'] = True
            else:
                # Requested lnsp and sp
                sp_info['delete_lnsp'] = False

        self.warn_missing_years(var, dates)

        files = sum(
            [
                self.find_files(v, dates, model=model)
                for v in var
                if v not in self._INVARIANTS
            ],
            [],
        )
        ds = None
        if len(files) > 0:
            ds = xr.open_mfdataset(files, combine="by_coords")
            ds = sel_era5(ds, sel)
        invar_files = sum(
            [
                self.find_files(v, dates, model=model)
                for v in var
                if v in self._INVARIANTS
            ],
            [],
        )
        if len(invar_files) > 0:
            invar_ds = xr.open_mfdataset(invar_files, combine="by_coords").squeeze(
                drop=True
            )
            invar_ds = sel_era5(invar_ds, sel)
            if len(files) > 0:
                for invar in invar_ds.data_vars:
                    ds[invar] = invar_ds[invar]
            else:
                ds = invar_ds
        if ds is None:
            # If no data found
            raise ValueError(f'No data found for ecmwf-era5{self.archive}, var={var} and date={date}.\n'
                             f'Model level variables = {self._ML_VARS}\n'
                             f'Surface variables = {self._SURF_VARS}\n'
                             f'Invariant variables = {self._INVARIANTS}')

        if sp_info['in_var']:
            ds = convert_lnsp_to_sp(ds, delete_lnsp=sp_info['delete_lnsp'])

        return ds

    def find_files(
        self, var: str, dates: list[datetime], model: str = "oper"
    ) -> list[str]:
        if var in self._INVARIANTS:
            return self.find_invariant(var)
        else:
            return sum(
                [self.find_single_file(var, date, model=model) for date in dates], []
            )

    def find_invariant(self, var: str) -> list[str]:
        if self.archive != '':
            warnings.warn(f'Using base archive (ecmwf-era5), for invariant var={var} '
                          f'despite requested archive of ecmwf-era5{self.archive}.')
        date = self._INVARIANT_DATE
        files = sorted(
            list(
                self.path.glob(
                    f"invariants/ecmwf-era5_oper_an_sfc_{date.year:04d}{date.month:02d}{date.day:02d}0000.{var}.inv.nc"
                )
            )
        )

        return files

    def find_single_file(
        self, var: str, date: datetime, model: str = "oper"
    ) -> list[str]:
        if model == "enda":
            level_type = "em_sfc"
        else:
            level_type = "*"
        files = sorted(
            list(
                self.path.glob(
                    f"{model}/{level_type}/{date.year:04d}/{date.month:02d}/{date.day:02d}/"
                    f"ecmwf-era5{self.archive}_{model}_*_{date.year:04d}{date.month:02d}{date.day:02d}{date.hour:02d}*.{var}.nc"
                )
            )
        )
        return files

    def warn_missing_years(self, var: list[str], dates: list[datetime], model: str = "oper") -> None:
        """
        Warn about some years that might be missing model level data for ERA5 - print README in relevant directory
        Args:
            var: List of variables requested
            dates: List of dates requested
            model: Model requested

        Returns:

        """
        var_to_warn = [v for v in var if v in self._ML_VARS]
        if self.archive=='' and (model == "oper") and (len(var_to_warn) > 0):
            years_to_warn = list({d.year for d in dates if d.year in self._ML_WARNING_YEARS})
            if len(years_to_warn) > 0:
                dir_use = list(self.path.glob(f'oper/an_ml/{years_to_warn[0]}'))[0]
                with open(f"{dir_use}/00README", "r") as f:
                    contents = f.read()
                warnings.warn(f"README for year {years_to_warn[0]} (Can use ERA5.1 by setting `archive=1`):\n{contents}")


class Pressure_levels_era5(Find_era5):
    def __init__(self, archive: Literal[None, 1, 't'] = None):
        self._init_vars(archive)

    def __getitem__(self, args):
        ds_args = (
            "lnsp",
            args[0],
            None,
        )
        if len(args) > 2:
            ds_args = ds_args + args[2:]
        ps = super().__getitem__(ds_args)
        pl = get_pl(np.exp(ps.lnsp.to_numpy()), 137).swapaxes(0, 1)
        sel = {}
        if len(args) > 1 and args[1] is not None:
            sel["level"] = args[1]
            if isinstance(sel["level"], slice):
                if sel["level"].start is None:
                    sel["level"] = slice(1, sel["level"].stop, sel["level"].step)
                if sel["level"].stop is None:
                    sel["level"] = slice(sel["level"].start, 137, sel["level"].step)
        return xr.DataArray(
            pl,
            dims=("time", "level", "latitude", "longitude"),
            coords={
                "level": range(1, 138),
                "time": ps.time,
                "longitude": ps.longitude,
                "latitude": ps.latitude,
            },
        ).sel(sel)


class Geopotential_levels_era5(Find_era5):
    def __init__(self, archive: Literal[None, 1, 't'] = None):
        self._init_vars(archive)

    def __getitem__(self, args):
        ds_args = (
            ["lnsp", "z", "t", "q"],
            args[0],
            None,
        )
        if len(args) > 2:
            ds_args = ds_args + args[2:]
        ds = super().__getitem__(ds_args)
        gz = get_gz(
            np.exp(ds.lnsp.to_numpy()),
            ds.z.to_numpy() * 9.81,
            ds.t.to_numpy().swapaxes(0, 1),
            ds.q.to_numpy().swapaxes(0, 1),
            137,
        ).swapaxes(0, 1)
        sel = {}
        if len(args) > 1 and args[1] is not None:
            sel["level"] = args[1]
            if isinstance(sel["level"], slice):
                if sel["level"].start is None:
                    sel["level"] = slice(1, sel["level"].stop, sel["level"].step)
                if sel["level"].stop is None:
                    sel["level"] = slice(sel["level"].start, 137, sel["level"].step)
        return xr.DataArray(
            gz,
            dims=("time", "level", "latitude", "longitude"),
            coords={
                "level": range(1, 138),
                "time": ds.time,
                "longitude": ds.longitude,
                "latitude": ds.latitude,
            },
        ).sel(sel)


class Ensemble_era5(Find_era5):
    def __init__(self, archive: Literal[None, 1, 't'] = None):
        self._init_vars(archive)

    def __getitem__(self, args):
        var = args[0]
        date = args[1]
        sel = {}
        if len(args) > 2 and args[2] is not None:
            sel["level"] = args[2]
        if len(args) > 3 and args[3] is not None:
            sel["longitude"] = args[3]
        if len(args) > 4 and args[4] is not None:
            sel["latitude"] = args[4]
        if isinstance(date, slice):
            if date.step is None:
                freq = "1H"
            else:
                freq = date.step
            dates = (
                pd.date_range(
                    pd.to_datetime(date.start),
                    pd.to_datetime(date.stop),
                    inclusive="left",
                    freq=freq,
                )
                .to_pydatetime()
                .tolist()
            )
        else:
            dates = [pd.to_datetime(date).to_pydatetime()]

        if isinstance(var, str):
            var = [var]
        files = [
            self.find_files(v, dates, model="enda")
            for v in var
            if v not in self._INVARIANTS
        ]
        if len(files) > 0:
            ds_list = [
                [
                    xr.open_mfdataset(
                        f, combine="nested", concat_dim=("ensemble_member")
                    ).assign_coords({"ensemble_member": range(1, len(f) + 1)})
                    for f in fl
                    if len(fl)
                ]
                for fl in files
            ]
            ds = xr.merge([xr.concat(dsl, dim="time") for dsl in ds_list if len(dsl)])
            ds = sel_era5(ds, sel)
        invar_files = sum(
            [
                self.find_files(v, dates, model="oper")
                for v in var
                if v in self._INVARIANTS
            ],
            [],
        )
        if len(invar_files) > 0:
            invar_ds = xr.open_mfdataset(invar_files, combine="by_coords").squeeze(
                drop=True
            )
            invar_ds = sel_era5(invar_ds, sel)
            if len(files) > 0:
                for invar in invar_ds.data_vars:
                    ds[invar] = invar_ds[invar]
            else:
                ds = invar_ds
        return ds

    def find_files(
        self, var: str, dates: list[datetime], model: str = "enda"
    ) -> list[str]:
        if var in self._INVARIANTS:
            return self.find_invariant(var)
        else:
            return [self.find_single_file(var, date, model=model) for date in dates]

    def find_invariant(self, var: str) -> list[str]:
        date = self._INVARIANT_DATE
        files = sorted(
            list(
                self.path.glob(
                    f"invariants/ecmwf-era5_oper_an_sfc_{date.year:04d}{date.month:02d}{date.day:02d}0000.{var}.inv.nc"
                )
            )
        )

        return files

    def find_single_file(
        self, var: str, date: datetime, model: str = "enda"
    ) -> list[str]:
        if model == "enda":
            level_type = "an_sfc"
        else:
            level_type = "*"
        files = sorted(
            list(
                self.path.glob(
                    f"{model}/{level_type}/{date.year:04d}/{date.month:02d}/{date.day:02d}/ecmwf-era5_{model}_*_{date.year:04d}{date.month:02d}{date.day:02d}{date.hour:02d}*.{var}.nc"
                )
            )
        )
        return files
