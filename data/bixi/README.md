# BIXI teaching subset

This directory contains only the derived data used by the modeling chapter:
three station records, 43-weekday mean completed-event rates in 15-minute
bins, and completed-trip events for one three-hour showcase morning. The raw
2024 archive is 2.6 GB and is deliberately not vendored.

The source is BIXI Montréal's official 2024 open-data archive and official
GBFS station-information feed. BIXI publishes the trip data under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). See
`manifest.json` for exact URLs, SHA-256 checksums, filters, snapshot time, and
derived-file checksums.

The archive records completed trips, not attempted rentals or returns, and it
does not identify operator relocation. BIXI also removes trips shorter than
one minute and longer than two hours. Consequently, the derived rates are an
input to a transparent teaching simulator—not an estimate of the real
system's unmet demand.

Station capacities and coordinates come from the checksum-pinned 2 September
2026 GBFS snapshot because the trip archive does not contain capacities. They
may not exactly match the station hardware present in July 2024.

Reproduce the subset from verified downloads at `/tmp/bixi2024.zip` and
`/tmp/bixi_station_information.json`:

```bash
uv run python scripts/prepare_bixi_data.py \
  --trip-archive /tmp/bixi2024.zip \
  --station-information /tmp/bixi_station_information.json
```
