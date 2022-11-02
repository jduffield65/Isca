import numpy as np
import cftime
# Module for handling of CMIP5 date/time, wraps complicated stuff with fake and/or
# real datetime/DatetimeIndex objects


class FakeDT(object):
    def __init__(self, dates: np.ndarray, units: str ='hours since 1800-01-01 00:00:00',
                 calendar='standard'):
        """
        An object created to mimic the behavior of a *pandas* `DatetimeIndex` object, but
        one that allows for dates from non-standard calendars (e.g. 360 day or no leap).

        Copied from [Isca code](https://github.com/ExeClim/Isca/blob/master/src/extra/python/scripts/cmip_time.py).

        Args:
            dates: Array or list of `cftime.Datetime` values to be converted into a `FakeDT` object
            units: Time units (e.g. `'hours since...'`)
            calendar: Calendar to which `dates` belong.
        """
        self.dates = np.array(dates)
        self.units = units
        self.calendar = calendar
        self.ndates = len(self.dates)
        self.dtype = type(dates[0])
        if self.ndates == 1:
            self.year = dates.year
            self.month = dates.month
            self.day = dates.day
            self.hour = dates.hour
            self.minute = dates.minute
            try:
                self.dayofyear = dates.timetuple().tm_yday
            except AttributeError:
                self.dayofyear = dates.timetuple()[7]
        else:
            self.year = np.array([dk.year for dk in dates])
            self.month = np.array([dk.month for dk in dates])
            self.day = np.array([dk.day for dk in dates])
            self.hour = np.array([dk.hour for dk in dates])
            self.minute = np.array([dk.minute for dk in dates])
            try:
                self.dayofyear = np.array([dk.timetuple().tm_yday for dk in dates])
            except AttributeError:
                self.dayofyear = np.array([dk.timetuple()[7] for dk in dates])

    def __getitem__(self, idx):
        # If <idx> is array_like, return a new FakeDT object restricted to those
        # indicies, if not, just return the member at a particular location
        if isinstance(idx, (list, np.ma.MaskedArray, np.ndarray)):
            return FakeDT(self.dates[idx], self.units, self.calendar)
        else:
            return self.dates[idx]

    def __str__(self):
        if self.ndates == 1:
            return "[ {}, dtype={} ]".format(self.dates, type(self.dates))
        else:
            out_s = "[ "
            for k in range(self.ndates - 1):
                if k % 5 == 0:
                    out_s += "{},\n".format(self.dates[k])
                else:
                    out_s += "{}, ".format(self.dates[k])
            out_s += "{}, dtype={} ]".format(self.dates[-1], type(self.dates))
        return out_s

    def __reduce__(self):
        # Special method for pickle to output in binary format
        return (self.__class__, (self.dates, self.units, self.calendar))

    def __len__(self):
        return self.ndates

    def get_loc(self, date: cftime.datetime) -> int:
        """
        FakeDT class method for returning the index of a particular date
        raises KeyError if the date is not found. Uses bisection method

        Args:
            date: `netcdftime.datetime` or `datetime.datetime` date for which to search

        Returns:
            Index of `date` in `self.dates`
        """
        a, b = 0, len(self.dates) - 1
        niter = 0
        # Compare dates using the .timetuple() method, since this works if <date> is
        # a datetime or netcdftime .datetime, otherwise only == works, not > or <
        while True and niter < len(self.dates):
            if self.dates[a] == date:
                return a
            elif self.dates[b] == date:
                return b
            elif self.dates[a].timetuple() < date.timetuple() \
                    and self.dates[b].timetuple() > date.timetuple():
                c = a + (b - a) / 2
                if self.dates[c] == date:
                    return c
                elif self.dates[c].timetuple() > date.timetuple():
                    b = c
                elif self.dates[c].timetuple() < date.timetuple():
                    a = c
            else:
                # First error string only raised if 'c' has been assigned
                if 'c' in locals():
                    raise KeyError('Date not found {}, a({}): {}, b({}): {}, '
                                   'c({}):{}'.format(date, a, self.dates[a],
                                                     b, self.dates[b],
                                                     c, self.dates[c]))
                else:
                    raise KeyError('Date not found {}, a({}): {},'
                                   ' b({}): {}'.format(date, a, self.dates[a],
                                                       b, self.dates[b]))
            niter += 1
        return c


def day_number_to_date(time_in: np.ndarray, calendar: str = '360_day',
                       time_units: str = 'days since 0001-01-01 00:00:00') -> FakeDT:
    """
    Aim is to make the time array have attributes like .month, or .year etc. This doesn't work with
    normal datetime objects, so FakeDT does this for you. First step is to turn input times
    into an array of datetime objects, and then FakeDT makes the array have the attributes of the
    elements themselves.

    Based on [Isca code](https://github.com/ExeClim/Isca/blob/master/src/extra/python/scripts/calendar_calc.py).

    Args:
        time_in: Array of dates with units of `<time units>` in `units_in`.
        calendar: Calendar that the dates in `time_in` correspond to.
            Valid options are  `standard`, `gregorian`, `proleptic_gregorian`, `noleap`, `365_day`, `360_day`,
            `julian`, `all_leap`, `366_day`.
        time_units: time units in the form `'<time_units> since <ref_date> <ref_time>'`.</br>
            `<time units>` can be `days`, `hours`, `minutes`, `seconds`.</br>
            `<ref_date>` is in the form `yyyy-mm-dd`.</br>
            `<ref_time>` is in the form `hh:mm:ss`.</br>

    Returns:
        Dates given in `time_in` converted to the `FakeDT` format.
    """
    dates = cftime.num2date(time_in, time_units, calendar)
    cdftime = FakeDT(dates, units=time_units, calendar=calendar)
    return cdftime
