# get_jasmin_era5
A small python package to find and load ECMWF ERA5 data from the BADC archive on jasmin

**Requirements:** `numpy`, `pandas` and `xarray`

**Installation:**
```
git clone https://github.com/w-k-jones/get_jasmin_era5.git
pip install get_jasmin_era5/
```

**Example usage:**

Import and initialise era5 object:

```
from get_jasmin_era5 import Find_era5
era5 = Find_era5()
```
Load temperature (t) for one time step:

`era5["t", "2020-06-01-12:00:00"]`

```
OUT:
<xarray.Dataset>
Dimensions:    (longitude: 1440, latitude: 721, level: 137, time: 1)
Coordinates:
  * longitude  (longitude) float32 0.0 0.25 0.5 0.75 ... 359.0 359.2 359.5 359.8
  * latitude   (latitude) float32 90.0 89.75 89.5 89.25 ... -89.5 -89.75 -90.0
  * level      (level) int32 1 2 3 4 5 6 7 8 ... 130 131 132 133 134 135 136 137
  * time       (time) datetime64[ns] 2020-06-01T12:00:00
Data variables:
    t          (time, level, latitude, longitude) float32 dask.array<chunksize=(1, 137, 721, 1440), meta=np.ndarray>
Attributes:
    Conventions:  CF-1.6
    history:      2020-12-18 00:24:54 GMT by grib_to_netcdf-2.19.1: grib_to_n...
```

Load temperature (t) for all time steps on 2020/6/1:

`era5["t", "2020-06-01":"2020-06-02"]`

```
OUT: 
<xarray.Dataset>
Dimensions:    (longitude: 1440, latitude: 721, level: 137, time: 24)
Coordinates:
  * longitude  (longitude) float32 0.0 0.25 0.5 0.75 ... 359.0 359.2 359.5 359.8
  * latitude   (latitude) float32 90.0 89.75 89.5 89.25 ... -89.5 -89.75 -90.0
  * level      (level) int32 1 2 3 4 5 6 7 8 ... 130 131 132 133 134 135 136 137
  * time       (time) datetime64[ns] 2020-06-01 ... 2020-06-01T23:00:00
Data variables:
    t          (time, level, latitude, longitude) float32 dask.array<chunksize=(1, 137, 721, 1440), meta=np.ndarray>
Attributes:
    Conventions:  CF-1.6
    history:      2020-12-18 00:18:51 GMT by grib_to_netcdf-2.19.1: grib_to_n...
```

Slicing across the prime meridian and using ascending latitude values is also enabled:

`era5["t", "2020-06-01":"2020-06-02", None, -90:90, -60:60]`

```
OUT:
<xarray.Dataset>
Dimensions:    (longitude: 721, latitude: 481, level: 137, time: 24)
Coordinates:
  * longitude  (longitude) float32 270.0 270.2 270.5 270.8 ... 89.5 89.75 90.0
  * latitude   (latitude) float32 60.0 59.75 59.5 59.25 ... -59.5 -59.75 -60.0
  * level      (level) int32 1 2 3 4 5 6 7 8 ... 130 131 132 133 134 135 136 137
  * time       (time) datetime64[ns] 2020-06-01 ... 2020-06-01T23:00:00
Data variables:
    t          (time, level, latitude, longitude) float32 dask.array<chunksize=(1, 137, 481, 721), meta=np.ndarray>
Attributes:
    Conventions:  CF-1.6
    history:      2020-12-18 00:18:51 GMT by grib_to_netcdf-2.19.1: grib_to_n...
```

Load temperature (t), specific humidity (q), 2m temperature (2t) and surface height (z) for every three hours on 2020/6/1, for the 100th level downward, and between 90-270 degrees longtiude and -60-60 degrees latitude:

`era5[("t", "q", "2t", "z"), "2020-06-01":"2020-06-02":"3H", 100:, 90:270, -60:60]`

```
OUT:
<xarray.Dataset>
Dimensions:    (longitude: 721, latitude: 481, level: 38, time: 8)
Coordinates:
  * longitude  (longitude) float32 90.0 90.25 90.5 90.75 ... 269.5 269.8 270.0
  * latitude   (latitude) float32 60.0 59.75 59.5 59.25 ... -59.5 -59.75 -60.0
  * level      (level) int32 100 101 102 103 104 105 ... 132 133 134 135 136 137
  * time       (time) datetime64[ns] 2020-06-01 ... 2020-06-01T21:00:00
Data variables:
    q          (time, level, latitude, longitude) float32 dask.array<chunksize=(1, 38, 481, 721), meta=np.ndarray>
    t          (time, level, latitude, longitude) float32 dask.array<chunksize=(1, 38, 481, 721), meta=np.ndarray>
    t2m        (time, latitude, longitude) float32 dask.array<chunksize=(1, 481, 721), meta=np.ndarray>
    z          (latitude, longitude) float32 dask.array<chunksize=(481, 721), meta=np.ndarray>
Attributes:
    Conventions:  CF-1.6
    history:      2020-12-18 00:18:45 GMT by grib_to_netcdf-2.19.1: grib_to_n...
```

Calculate pressure levels for the previous example:

`era5.pl["2020-06-01":"2020-06-02":"3H", 100:, 90:270, -60:60]`

```
OUT:
<xarray.DataArray (time: 8, level: 38, latitude: 481, longitude: 721)>
array([[[[ 57872.48474194,  57851.04773663,  57737.57761903, ...,
           59308.89351633,  59309.34814012,  59312.7428857 ],
         [ 57819.39331036,  57858.96549122,  57897.17048227, ...,
           59314.18639922,  59313.68863594,  59313.68863594],
         [ 57719.0044118 ,  57721.79188617,  57792.87911958, ...,
           59317.08669994,  59320.93275089,  59319.48260053],
         ...,
...
         [ 97330.26117977,  97328.03725574,  97325.90697063, ...,
          100171.16415324, 100144.3366066 , 100115.30854566],
         [ 97374.73185699,  97376.86994535,  97377.99361223, ...,
          100127.62206184, 100100.7945152 , 100078.54747172],
         [ 97401.85592684,  97413.84170684,  97422.48769918, ...,
          100085.22704703, 100067.38103215, 100052.87480492]]]])
Coordinates:
  * level      (level) int64 100 101 102 103 104 105 ... 132 133 134 135 136 137
  * time       (time) datetime64[ns] 2020-06-01 ... 2020-06-01T21:00:00
  * longitude  (longitude) float32 90.0 90.25 90.5 90.75 ... 269.5 269.8 270.0
  * latitude   (latitude) float32 60.0 59.75 59.5 59.25 ... -59.5 -59.75 -60.0
```

Calculate geopotential heights for the same example:

`era5.gz["2020-06-01":"2020-06-02":"3H", 100:, 90:270, -60:60]`

```
OUT:
<xarray.DataArray (time: 8, level: 38, latitude: 481, longitude: 721)>
array([[[[5.11737957e+04, 5.20996878e+04, 5.44298830e+04, ...,
          4.13218602e+04, 4.12940971e+04, 4.12325377e+04],
         [5.19596141e+04, 5.17204796e+04, 5.12408262e+04, ...,
          4.13251723e+04, 4.13314302e+04, 4.13199145e+04],
         [5.43623733e+04, 5.43935385e+04, 5.32772366e+04, ...,
          4.13783058e+04, 4.12982295e+04, 4.12765517e+04],
         ...,
...
         [1.01424387e+02, 1.01402244e+02, 9.25518078e+01, ...,
          7.53837186e+01, 7.54326293e+01, 8.43107208e+01],
         [9.25974757e+01, 9.25697266e+01, 9.25488523e+01, ...,
          8.42433633e+01, 1.19629061e+02, 1.19669439e+02],
         [1.01434764e+02, 1.01407126e+02, 9.25514372e+01, ...,
          1.19581624e+02, 9.31415615e+01, 6.66779947e+01]]]])
Coordinates:
  * level      (level) int64 100 101 102 103 104 105 ... 132 133 134 135 136 137
  * time       (time) datetime64[ns] 2020-06-01 ... 2020-06-01T21:00:00
  * longitude  (longitude) float32 90.0 90.25 90.5 90.75 ... 269.5 269.8 270.0
  * latitude   (latitude) float32 60.0 59.75 59.5 59.25 ... -59.5 -59.75 -60.0
```

It will also load ensemble datasets. By default, if you pass the "enda" argument it will load the ensemble mean:

`era5[("2t","2d"), "2020-06-01", None, None, None, "enda"]`

```
OUT:
<xarray.Dataset>
Dimensions:    (longitude: 1440, latitude: 721, time: 1)
Coordinates:
  * longitude  (longitude) float32 0.0 0.25 0.5 0.75 ... 359.0 359.2 359.5 359.8
  * latitude   (latitude) float32 90.0 89.75 89.5 89.25 ... -89.5 -89.75 -90.0
  * time       (time) datetime64[ns] 2020-06-01
Data variables:
    d2m        (time, latitude, longitude) float32 dask.array<chunksize=(1, 721, 1440), meta=np.ndarray>
    t2m        (time, latitude, longitude) float32 dask.array<chunksize=(1, 721, 1440), meta=np.ndarray>
Attributes:
    Conventions:  CF-1.6
    history:      2020-12-22 18:23:11 GMT by grib_to_netcdf-2.19.1: grib_to_n...
```

However you can also get the ensemble members using `.enda`:

`era5.enda[("2t", "2d"), "2020-06-01":"2020-06-02":"3H", 100:, 90:270, -60:60]`

```
OUT:
<xarray.Dataset>
Dimensions:          (longitude: 721, latitude: 481, time: 8,
                      ensemble_member: 10)
Coordinates:
  * longitude        (longitude) float32 90.0 90.25 90.5 ... 269.5 269.8 270.0
  * latitude         (latitude) float32 60.0 59.75 59.5 ... -59.5 -59.75 -60.0
  * time             (time) datetime64[ns] 2020-06-01 ... 2020-06-01T21:00:00
  * ensemble_member  (ensemble_member) int64 1 2 3 4 5 6 7 8 9 10
Data variables:
    t2m              (ensemble_member, time, latitude, longitude) float32 dask.array<chunksize=(1, 1, 481, 721), meta=np.ndarray>
    d2m              (ensemble_member, time, latitude, longitude) float32 dask.array<chunksize=(1, 1, 481, 721), meta=np.ndarray>
Attributes:
    Conventions:  CF-1.6
    history:      2021-01-15 04:09:35 GMT by grib_to_netcdf-2.19.1: grib_to_n...
```
