#!/usr/bin/env python3
"""Derive the small textbook BIXI dataset from the official 2024 archive.

The 2.6 GB trip file is never copied into the repository. This maintenance
command verifies the two official inputs, streams the archive once, and writes
only the 3-station, 3-hour aggregates and showcase events used by the book.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import date, datetime, timedelta, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import zipfile
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data" / "bixi"
MONTREAL = ZoneInfo("America/Toronto")

TRIP_ARCHIVE_SHA256 = "b186787487149d111b5d1a414a25f5e8430d7424fd4c982d36e33ce282eb7d58"
STATION_INFORMATION_SHA256 = "75c4434b4a1e6c32824ecbc9133ab868c4aa098d34e3c426932acdbeca0c522b"
TRIP_ARCHIVE_URL = (
    "https://cdn.bixi.com/wp-content/uploads/2025/01/"
    "DonneesOuvertes2024_010203040506070809101112.zip"
)
STATION_INFORMATION_URL = "https://gbfs.velobixi.com/gbfs/en/station_information.json"
LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/"
DATA_PAGE_URL = "https://bixi.com/en/open-data/"

STATION_NAMES = (
    "Berri / Cherrier",
    "Prince-Arthur / St-Urbain",
    "de Maisonneuve / Aylmer (ouest)",
)
SHOWCASE_DATE = date(2024, 7, 4)
TRAINING_START = date(2024, 5, 1)
TRAINING_END = date(2024, 6, 28)
HORIZON_START_MINUTE = 7 * 60
HORIZON_END_MINUTE = 10 * 60
BIN_MINUTES = 15
EXPECTED_SHOWCASE_COUNTS = {
    "Berri / Cherrier": (79, 36),
    "Prince-Arthur / St-Urbain": (48, 20),
    "de Maisonneuve / Aylmer (ouest)": (12, 84),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify(path: Path, expected: str, label: str) -> None:
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f"{label} checksum mismatch: expected {expected}, got {actual}")


def _training_days() -> tuple[date, ...]:
    days: list[date] = []
    current = TRAINING_START
    while current <= TRAINING_END:
        if current.weekday() < 5:
            days.append(current)
        current += timedelta(days=1)
    return tuple(days)


def _local_datetime(milliseconds: str) -> datetime:
    return datetime.fromtimestamp(int(milliseconds) / 1000.0, tz=timezone.utc).astimezone(
        MONTREAL
    )


def _bin_index(value: datetime) -> int | None:
    minute = value.hour * 60 + value.minute
    if not HORIZON_START_MINUTE <= minute < HORIZON_END_MINUTE:
        return None
    return (minute - HORIZON_START_MINUTE) // BIN_MINUTES


def _haversine_km(lat_a: float, lon_a: float, lat_b: float, lon_b: float) -> float:
    radius_km = 6371.0088
    phi_a, phi_b = math.radians(lat_a), math.radians(lat_b)
    delta_phi = math.radians(lat_b - lat_a)
    delta_lambda = math.radians(lon_b - lon_a)
    value = (
        math.sin(delta_phi / 2.0) ** 2
        + math.cos(phi_a) * math.cos(phi_b) * math.sin(delta_lambda / 2.0) ** 2
    )
    return 2.0 * radius_km * math.asin(math.sqrt(value))


def _station_records(path: Path) -> list[dict[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    stations = {row["name"]: row for row in payload["data"]["stations"]}
    missing = [name for name in STATION_NAMES if name not in stations]
    if missing:
        raise ValueError(f"station-information snapshot is missing {missing}")
    return [stations[name] for name in STATION_NAMES]


def derive(trip_archive: Path, station_information: Path, output: Path) -> None:
    _verify(trip_archive, TRIP_ARCHIVE_SHA256, "BIXI 2024 trip archive")
    _verify(
        station_information,
        STATION_INFORMATION_SHA256,
        "BIXI station-information snapshot",
    )
    training_days = _training_days()
    training_set = set(training_days)
    counts: Counter[tuple[date, int, str, str]] = Counter()
    showcase_events: list[dict[str, object]] = []
    scanned_rows = 0

    with zipfile.ZipFile(trip_archive) as archive:
        members = [item for item in archive.infolist() if item.filename.endswith(".csv")]
        if len(members) != 1:
            raise ValueError("the verified BIXI archive should contain exactly one CSV")
        with archive.open(members[0]) as raw:
            text = io.TextIOWrapper(raw, encoding="utf-8", newline="")
            reader = csv.DictReader(text)
            expected_columns = {
                "STARTSTATIONNAME",
                "ENDSTATIONNAME",
                "STARTTIMEMS",
                "ENDTIMEMS",
            }
            if not expected_columns.issubset(reader.fieldnames or []):
                raise ValueError("the BIXI trip schema has changed")
            for row in reader:
                scanned_rows += 1
                for name_key, time_key, kind in (
                    ("STARTSTATIONNAME", "STARTTIMEMS", "rental"),
                    ("ENDSTATIONNAME", "ENDTIMEMS", "return"),
                ):
                    station_name = row[name_key]
                    if station_name not in STATION_NAMES:
                        continue
                    local = _local_datetime(row[time_key])
                    bin_index = _bin_index(local)
                    if bin_index is None:
                        continue
                    if local.date() in training_set:
                        counts[(local.date(), bin_index, station_name, kind)] += 1
                    if local.date() == SHOWCASE_DATE:
                        horizon_start = local.replace(hour=7, minute=0, second=0, microsecond=0)
                        showcase_events.append(
                            {
                                "time_minutes": (local - horizon_start).total_seconds() / 60.0,
                                "timestamp_ms": row[time_key],
                                "timestamp_local": local.isoformat(timespec="milliseconds"),
                                "station_name": station_name,
                                "kind": kind,
                            }
                        )

    output.mkdir(parents=True, exist_ok=True)
    station_rows = _station_records(station_information)
    distances = []
    for source in station_rows:
        distances.append(
            [
                _haversine_km(
                    float(source["lat"]),
                    float(source["lon"]),
                    float(destination["lat"]),
                    float(destination["lon"]),
                )
                for destination in station_rows
            ]
        )
    stations_payload = {
        "schema_version": 1,
        "stations": [
            {
                "station_id": str(row["station_id"]),
                "external_id": str(row["external_id"]),
                "short_name": str(row["short_name"]),
                "name": str(row["name"]),
                "lat": float(row["lat"]),
                "lon": float(row["lon"]),
                "capacity": int(row["capacity"]),
            }
            for row in station_rows
        ],
        "scenario": {
            "horizon_start_local": "2024-07-04T07:00:00-04:00",
            "horizon_minutes": 180,
            "control_period_minutes": 15,
            "initial_inventory": [30, 18, 8],
            "truck_capacity": 16,
            "transfer_limit": 8,
            "initial_truck_inventory": 0,
            "initial_truck_station": 2,
            "travel_steps": [[0, 1, 1], [1, 0, 1], [1, 1, 0]],
            "distance_km": distances,
            "objective": {
                "service_failure": 1.0,
                "bike_moved": 0.05,
                "truck_distance_km": 0.2,
                "terminal_imbalance": 0.1,
            },
        },
    }
    stations_path = output / "stations.json"
    stations_path.write_text(
        json.dumps(stations_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    profile_path = output / "completed_event_profile.csv"
    with profile_path.open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "bin_index",
            "bin_start_minutes",
            "local_time",
            "station_name",
            "mean_completed_starts",
            "mean_completed_ends",
            "training_day_count",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for bin_index in range(12):
            minute = HORIZON_START_MINUTE + bin_index * BIN_MINUTES
            for station_name in STATION_NAMES:
                starts = sum(
                    counts[(day, bin_index, station_name, "rental")]
                    for day in training_days
                ) / len(training_days)
                ends = sum(
                    counts[(day, bin_index, station_name, "return")]
                    for day in training_days
                ) / len(training_days)
                writer.writerow(
                    {
                        "bin_index": bin_index,
                        "bin_start_minutes": bin_index * BIN_MINUTES,
                        "local_time": f"{minute // 60:02d}:{minute % 60:02d}",
                        "station_name": station_name,
                        "mean_completed_starts": f"{starts:.12f}",
                        "mean_completed_ends": f"{ends:.12f}",
                        "training_day_count": len(training_days),
                    }
                )

    showcase_events.sort(
        key=lambda row: (
            float(row["time_minutes"]),
            str(row["station_name"]),
            str(row["kind"]),
        )
    )
    observed_counts: dict[str, tuple[int, int]] = {}
    for station_name in STATION_NAMES:
        starts = sum(
            row["station_name"] == station_name and row["kind"] == "rental"
            for row in showcase_events
        )
        ends = sum(
            row["station_name"] == station_name and row["kind"] == "return"
            for row in showcase_events
        )
        observed_counts[station_name] = (starts, ends)
    if observed_counts != EXPECTED_SHOWCASE_COUNTS:
        raise ValueError(
            f"unexpected showcase counts: {observed_counts}; expected {EXPECTED_SHOWCASE_COUNTS}"
        )
    events_path = output / "showcase_events.csv"
    with events_path.open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "time_minutes",
            "timestamp_ms",
            "timestamp_local",
            "station_name",
            "kind",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in showcase_events:
            row = dict(row)
            row["time_minutes"] = f"{float(row['time_minutes']):.6f}"
            writer.writerow(row)

    station_snapshot = json.loads(station_information.read_text(encoding="utf-8"))
    last_updated = datetime.fromtimestamp(
        int(station_snapshot["last_updated"]), tz=timezone.utc
    ).isoformat()
    manifest = {
        "schema_version": 1,
        "attribution": "BIXI Montréal",
        "license": "Creative Commons Attribution 4.0 International (CC BY 4.0)",
        "license_url": LICENSE_URL,
        "source_page_url": DATA_PAGE_URL,
        "sources": {
            "trip_archive": {
                "url": TRIP_ARCHIVE_URL,
                "sha256": TRIP_ARCHIVE_SHA256,
                "archive_member": members[0].filename,
                "rows_scanned": scanned_rows,
            },
            "station_information": {
                "url": STATION_INFORMATION_URL,
                "sha256": STATION_INFORMATION_SHA256,
                "gbfs_version": station_snapshot["version"],
                "snapshot_last_updated_utc": last_updated,
            },
        },
        "derivation": {
            "timezone": "America/Toronto",
            "stations": list(STATION_NAMES),
            "training_dates_inclusive": [
                TRAINING_START.isoformat(),
                TRAINING_END.isoformat(),
            ],
            "training_day_filter": "Monday-Friday, including statutory holidays",
            "training_day_count": len(training_days),
            "local_time_window": "07:00:00 inclusive to 10:00:00 exclusive",
            "bin_minutes": BIN_MINUTES,
            "showcase_date": SHOWCASE_DATE.isoformat(),
            "showcase_selection": "Chosen after inspection for a legible teaching contrast; not an unbiased evaluation sample.",
            "event_semantics": "Starts and ends of completed trips, not attempted demand; operator relocations are absent.",
            "archive_exclusions": "The source excludes trips shorter than one minute or longer than two hours.",
        },
        "derived_files": {
            "stations.json": sha256(stations_path),
            "completed_event_profile.csv": sha256(profile_path),
            "showcase_events.csv": sha256(events_path),
        },
        "showcase_completed_counts": {
            name: {"rentals": values[0], "returns": values[1]}
            for name, values in observed_counts.items()
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trip-archive", type=Path, required=True)
    parser.add_argument("--station-information", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    derive(args.trip_archive, args.station_information, args.output)


if __name__ == "__main__":
    main()
