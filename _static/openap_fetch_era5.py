"""
Download a tiny ERA5 pressure-level wind GRIB around Montreal/Toronto for OpenAP.top.

Requirements:
  pip install cdsapi
  Create ~/.cdsapirc with your CDS API key:
    url: https://cds.climate.copernicus.eu/api/v2
    key: <UID>:<API_KEY>

Produces a small GRIB file suitable for top.tools.read_grids().
"""
from __future__ import annotations

import os
from pathlib import Path

try:
    import cdsapi
except Exception as e:
    raise SystemExit("cdsapi is required. Install with: python -m pip install cdsapi") from e


def download_era5_grib(out_path: str | os.PathLike = "_static/era5_mtl_20230601_12.grib") -> str:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    c = cdsapi.Client()
    # ERA5 pressure levels: U/V wind at several standard levels
    # Area bbox: [North, West, South, East] (deg); Montreal/Toronto region
    req = {
        "product_type": "reanalysis",
        "format": "grib",
        "variable": [
            "u_component_of_wind",
            "v_component_of_wind",
        ],
        "pressure_level": ["200", "250", "300", "400", "500", "700", "850", "925"],
        "year": "2023",
        "month": "06",
        "day": "01",
        "time": ["12:00"],
        "area": [50, -80, 40, -70],
    }

    c.retrieve("reanalysis-era5-pressure-levels", req, str(out))
    return str(out)


if __name__ == "__main__":
    path = download_era5_grib()
    print(f"Saved: {path}")


