import cdsapi

def main():
    c = cdsapi.Client()

    c.retrieve(
        'reanalysis-era5-pressure-levels',
        {
            'product_type': 'reanalysis',
            'variable': 'temperature',
            'pressure_level': '500',
            'year': [str(year) for year in [2022]],
            'month': [f'{month:02d}' for month in [1,2]],
            'day': [f'{day:02d}' for day in [1,2,3]],
            'time': [f'{hour:02d}:00' for hour in range(0, 24)],
            'format': 'netcdf',
        },
        '/gws/nopw/j04/global_ex/jamd1/era5/test.nc')

if __name__ == '__main__':
    main()