from datetime import datetime
import pathlib
import numpy as np
import pandas as pd
import xarray as xr

from .utils import get_pl, get_gz, sel_era5


class Find_era5:
    def __init__(self):
        self._init_vars()
        self.pl = Pressure_levels_era5()
        self.gz = Geopotential_levels_era5()
        self.enda = Ensemble_era5()

    def _init_vars(self):
        self.path = pathlib.Path("/badc/ecmwf-era5/data/")
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
        files = sum(
            [
                self.find_files(v, dates, model=model)
                for v in var
                if v not in self._INVARIANTS
            ],
            [],
        )
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
                    f"{model}/{level_type}/{date.year:04d}/{date.month:02d}/{date.day:02d}/ecmwf-era5_{model}_*_{date.year:04d}{date.month:02d}{date.day:02d}{date.hour:02d}*.{var}.nc"
                )
            )
        )
        return files


class Pressure_levels_era5(Find_era5):
    def __init__(self):
        self._init_vars()

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
    def __init__(self):
        self._init_vars()

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
    def __init__(self):
        self._init_vars()

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
