# Aircraft wind snapshot

`era5_wind.npz` is a numerical extraction of the committed
`_static/era5_mtl_20230601_12.grib` file. The snapshot is valid at
2023-06-01 12:00 UTC, with wind components at eight pressure levels over
40–50 degrees north and 80–70 degrees west. It contains no future forecast
ensemble. The original request is recorded in `_static/openap_fetch_era5.py`.

The extraction keeps the raw eastward and northward wind values and sorts the
pressure, latitude, and longitude axes in ascending order. Array fields are:

| Field | Shape / units |
|---|---|
| `pressure_hpa` | 8 pressure levels, hPa |
| `lat_deg` | 41 latitudes, degrees north |
| `lon_deg` | 41 longitudes, degrees east, negative westward |
| `u_mps`, `v_mps` | `(pressure, latitude, longitude)`, m/s |
| `valid_time_unix_s` | UTC Unix timestamp, seconds |

`era5_wind.json` records source and artifact checksums, coverage, reader
versions, and processing. Regenerate from the existing GRIB file with:

```bash
uv run --no-project --with eccodes==2.48.0 --with numpy==2.5.3 python scripts/prepare_aircraft_wind.py
```

This one-time conversion needs ecCodes. Ordinary book builds and numerical
replays need only the checked-in NPZ. No credentials or weather downloads are
required. The aircraft model interpolates these levels numerically; below
925 hPa or above 200 hPa it uses the nearest available pressure boundary.

The aircraft experiment superimposes a declared correlated gust process on
this fixed mean field. The snapshot does not calibrate the gust variance or
correlation time, and the synthetic future gusts are not ERA5 observations.

Dataset citation: [ERA5 hourly data on pressure levels](https://doi.org/10.24381/cds.bd0915c6), Copernicus Climate Change Service (2018).
