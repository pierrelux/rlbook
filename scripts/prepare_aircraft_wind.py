"""Extract the committed ERA5 snapshot without a live weather service.

Regenerate with a temporary reader environment, leaving the book environment
unchanged:

    uv run --no-project --with eccodes==2.48.0 --with numpy==2.5.3 \
        python scripts/prepare_aircraft_wind.py

The numerical artifact is read by the aircraft demonstration. ecCodes is only
needed here, not while building or viewing the book.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def extract(source: Path, output: Path) -> dict:
    import eccodes as ec

    records = {}
    timestamps = set()
    with source.open("rb") as stream:
        while (handle := ec.codes_grib_new_from_file(stream)) is not None:
            try:
                name = ec.codes_get(handle, "shortName")
                if name not in {"u", "v"}:
                    continue
                if ec.codes_get(handle, "typeOfLevel") != "isobaricInhPa":
                    raise ValueError("expected pressure-level wind messages")
                level = float(ec.codes_get(handle, "level"))
                date = ec.codes_get(handle, "validityDate")
                time = int(ec.codes_get(handle, "validityTime"))
                stamp = datetime.strptime(f"{date}{time:04d}", "%Y%m%d%H%M")
                timestamps.add(stamp.replace(tzinfo=timezone.utc).timestamp())
                lat = ec.codes_get_array(handle, "latitudes")
                lon = (ec.codes_get_array(handle, "longitudes") + 180) % 360 - 180
                values = ec.codes_get_values(handle)
                if not np.isfinite(values).all():
                    raise ValueError("missing or nonfinite wind data")
                records[name, level] = (lat, lon, values)
            finally:
                ec.codes_release(handle)

    if not records or len(timestamps) != 1:
        raise ValueError("expected one nonempty wind snapshot")
    pressure = np.array(sorted({key[1] for key in records}))
    lat0, lon0, _ = next(iter(records.values()))
    latitude, longitude = np.unique(lat0), np.unique(lon0)
    arrays = {}
    for component in ("u", "v"):
        grid = np.full((len(pressure), len(latitude), len(longitude)), np.nan)
        for i, level in enumerate(pressure):
            lat, lon, values = records[component, level]
            j, k = np.searchsorted(latitude, lat), np.searchsorted(longitude, lon)
            if len(values) != len(latitude) * len(longitude):
                raise ValueError("incomplete rectangular grid")
            grid[i, j, k] = values
        if not np.isfinite(grid).all():
            raise ValueError("incomplete wind component")
        arrays[f"{component}_mps"] = grid

    output.mkdir(parents=True, exist_ok=True)
    numerical = output / "era5_wind.npz"
    np.savez_compressed(
        numerical,
        lat_deg=latitude,
        lon_deg=longitude,
        pressure_hpa=pressure,
        valid_time_unix_s=np.array(next(iter(timestamps))),
        **arrays,
    )
    metadata = {
        "source": str(source.relative_to(ROOT)),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "artifact_sha256": hashlib.sha256(numerical.read_bytes()).hexdigest(),
        "valid_time_utc": datetime.fromtimestamp(next(iter(timestamps)), timezone.utc).isoformat(),
        "reader": {name: importlib.metadata.version(name) for name in ("eccodes", "numpy")},
        "pressure_hpa": pressure.tolist(),
        "latitude_range_deg": [float(latitude[0]), float(latitude[-1])],
        "longitude_range_deg": [float(longitude[0]), float(longitude[-1])],
        "shape": list(arrays["u_mps"].shape),
        "units": {"u_mps": "eastward m/s", "v_mps": "northward m/s"},
        "processing": "Extracted raw pressure-level values, sorted all axes ascending, wrapped longitudes to [-180,180). No fitting, temporal interpolation, or additional weather observations.",
    }
    (output / "era5_wind.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "_static/era5_mtl_20230601_12.grib")
    parser.add_argument("--output", type=Path, default=ROOT / "data/aircraft")
    args = parser.parse_args()
    print(json.dumps(extract(args.source, args.output), indent=2))
