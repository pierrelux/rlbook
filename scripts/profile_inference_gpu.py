#!/usr/bin/env python3
"""Measure Qwen2.5-7B inference and thermal dynamics on an NVIDIA L4.

This is a maintainer-only measurement program.  It requires a real NVIDIA L4,
root access to ``nvidia-smi`` clock controls, Docker, and network access for the
specified vLLM image and model.  It has no synthetic or CPU fallback.  The
normal textbook build reads its committed CSV and JSON outputs and never calls
this program.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
from itertools import combinations
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import threading
import time
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET


MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
MODEL_REVISION = "acbd96531cda22292a3ceaa67e984955d3965282"
VLLM_IMAGE = "vllm/vllm-openai:v0.28.0"
SERVED_MODEL_NAME = "qwen2.5-7b-instruct"
PROFILE_SCHEMA_VERSION = 2
TELEMETRY_PERIOD_S = 0.1
VLLM_PREFIX_CACHING_ENABLED = False
SWEEP_COOLDOWN_MARGIN_C = 16.0
SWEEP_COOLDOWN_STABLE_SAMPLES = 10
SWEEP_COOLDOWN_TIMEOUT_S = 15.0 * 60.0
SWEEP_COOLDOWN_POLL_S = 0.25
THERMAL_SAFE_DOWN_C = 77.0
THERMAL_ABORT_C = 79.0
THERMAL_STALE_TELEMETRY_S = 1.0
THERMAL_COOLDOWN_TARGET_C = 52.0
THERMAL_COOLDOWN_STABILITY_C = 1.0
THERMAL_COOLDOWN_STABILITY_S = 60.0
THERMAL_COOLDOWN_TIMEOUT_S = 10.0 * 60.0
THERMAL_MINIMUM_POWER_W = 40.0
THERMAL_REQUESTED_CLOCK_MHZ = 2040

RAW_PROFILE_COLUMNS = [
    "phase",
    "prompt_tokens",
    "output_tokens",
    "concurrency",
    "requested_clock_mhz",
    "realized_clock_mhz",
    "repeat",
    "ttft_s",
    "tpot_s",
    "total_s",
    "energy_j",
    "mean_power_w",
    "peak_power_w",
    "peak_temp_c",
    "telemetry_sample_count",
    "telemetry_fallback_used",
    "request_index",
    "completion_tokens",
    "prompt_tokens_observed",
]

PROFILE_COLUMNS = [
    "clock_mhz",
    "prefill_tokens_per_s",
    "decode_tokens_per_s",
    "idle_power_w",
    "prefill_power_w",
    "decode_power_w",
]

TELEMETRY_COLUMNS = [
    "elapsed_s",
    "utc",
    "phase",
    "graphics_clock_mhz",
    "memory_clock_mhz",
    "power_w",
    "temperature_c",
    "utilization_percent",
    "memory_used_mib",
]

THERMAL_TELEMETRY_COLUMNS = TELEMETRY_COLUMNS + [
    "split",
    "sequence",
    "block_id",
    "block_role",
    "requested_power_limit_w",
    "requested_clock_mhz",
    "workload_phase",
]

THERMAL_REQUEST_COLUMNS = [
    "split",
    "sequence",
    "block_id",
    "workload_phase",
    "prompt_tokens",
    "output_tokens",
    "concurrency",
    "requested_power_limit_w",
    "requested_clock_mhz",
    "batch_index",
    "request_index",
    "prompt_tokens_observed",
    "completion_tokens",
    "ttft_s",
    "tpot_s",
    "total_s",
    "start_elapsed_s",
    "end_elapsed_s",
]


class ProfilingError(RuntimeError):
    """Raised when required hardware behavior or measurement data is absent."""


CommandRunner = Callable[..., subprocess.CompletedProcess[str]]


@dataclass(frozen=True)
class GpuMetadata:
    name: str
    uuid: str
    driver_version: str
    cuda_version: str | None
    default_power_limit_w: float
    minimum_power_limit_w: float
    maximum_power_limit_w: float
    slowdown_temperature_c: float | None
    shutdown_temperature_c: float | None


@dataclass(frozen=True)
class BenchmarkCondition:
    phase: str
    prompt_tokens: int
    output_tokens: int
    concurrency: int


@dataclass(frozen=True)
class RequestTiming:
    request_index: int
    prompt_tokens_observed: int
    completion_tokens: int
    ttft_s: float
    tpot_s: float
    total_s: float
    start_elapsed_s: float
    end_elapsed_s: float


@dataclass(frozen=True)
class ThermalIdentificationBlock:
    """One prespecified input block in the thermal identification protocol."""

    block_id: str
    split: str
    sequence: str
    role: str
    requested_power_limit_w: float
    duration_s: float
    condition: BenchmarkCondition


def run_command(
    arguments: Sequence[str],
    *,
    check: bool = True,
    timeout: float | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run one local command without a shell."""

    return subprocess.run(
        list(arguments),
        check=check,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _number(text: str) -> float:
    match = re.search(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)", text)
    if not match:
        raise ProfilingError(f"Could not parse a number from nvidia-smi value {text!r}.")
    return float(match.group(0))


def parse_supported_clocks(output: str) -> dict[int, list[int]]:
    """Parse ``nvidia-smi --query-supported-clocks`` CSV output."""

    clocks: dict[int, set[int]] = {}
    for row in csv.reader(output.splitlines()):
        if len(row) < 2:
            continue
        try:
            memory = int(round(_number(row[0])))
            graphics = int(round(_number(row[1])))
        except ProfilingError:
            continue
        clocks.setdefault(memory, set()).add(graphics)
    if not clocks:
        raise ProfilingError("nvidia-smi did not report any supported memory/graphics clocks.")
    return {memory: sorted(values) for memory, values in clocks.items()}


def select_clock_levels(
    supported: Mapping[int, Sequence[int]],
    *,
    requested_levels: int = 5,
    minimum_levels: int = 4,
) -> tuple[int, list[int]]:
    """Choose evenly spaced graphics clocks at the highest memory clock."""

    if requested_levels < minimum_levels:
        raise ValueError("requested_levels cannot be smaller than minimum_levels.")
    if not supported:
        raise ProfilingError("No supported clock table was supplied.")
    memory_clock = max(int(value) for value in supported)
    available = sorted({int(value) for value in supported[memory_clock]})
    if len(available) < minimum_levels:
        raise ProfilingError(
            f"Only {len(available)} graphics clocks are available at {memory_clock} MHz; "
            f"at least {minimum_levels} are required."
        )
    count = min(requested_levels, len(available))
    if count == 1:
        return memory_clock, [available[0]]
    targets = [
        available[0] + index * (available[-1] - available[0]) / (count - 1)
        for index in range(count)
    ]
    chosen: set[int] = set()
    for target in targets:
        chosen.add(min(available, key=lambda value: (abs(value - target), value)))
    while len(chosen) < count:
        remaining = [value for value in available if value not in chosen]
        next_value = max(
            remaining,
            key=lambda value: min(abs(value - selected) for selected in chosen),
        )
        chosen.add(next_value)
    return memory_clock, sorted(chosen)


def parse_temperature_thresholds(
    xml_output: str,
) -> tuple[float | None, float | None]:
    """Read slowdown and shutdown thresholds from ``nvidia-smi -q -x``."""

    try:
        root = ET.fromstring(xml_output)
    except ET.ParseError as error:
        raise ProfilingError(f"Could not parse nvidia-smi XML: {error}") from error

    def find(names: Sequence[str]) -> float | None:
        for name in names:
            element = root.find(f".//{name}")
            if element is not None and element.text and element.text.strip() not in {"N/A", "[N/A]"}:
                return _number(element.text)
        return None

    slowdown = find(("gpu_temp_slow_threshold", "gpu_slowdown_temp"))
    shutdown = find(("gpu_temp_max_threshold", "gpu_shutdown_temp"))
    return slowdown, shutdown


def select_thermal_limit(
    slowdown_temperature_c: float | None,
) -> tuple[float, str]:
    """Apply the protocol limit without fabricating missing GPU metadata."""

    if slowdown_temperature_c is None:
        return 80.0, "protocol_absolute_80_c_no_reported_slowdown_threshold"
    return (
        min(80.0, slowdown_temperature_c - 5.0),
        "min_80_c_and_5_c_below_reported_slowdown",
    )


def query_gpu_metadata(
    gpu_index: int = 0, *, runner: CommandRunner = run_command
) -> GpuMetadata:
    fields = (
        "name,uuid,driver_version,power.default_limit,power.min_limit,power.max_limit"
    )
    result = runner(
        [
            "nvidia-smi",
            "-i",
            str(gpu_index),
            f"--query-gpu={fields}",
            "--format=csv,noheader,nounits",
        ],
        timeout=20,
    )
    row = next(csv.reader([result.stdout.strip()]), [])
    if len(row) < 6:
        raise ProfilingError(f"Unexpected nvidia-smi GPU metadata: {result.stdout!r}")
    name = row[0].strip()
    if "L4" not in name.upper():
        raise ProfilingError(
            f"This protocol requires an NVIDIA L4; nvidia-smi reported {name!r}."
        )
    xml_result = runner(
        ["nvidia-smi", "-i", str(gpu_index), "-q", "-x"], timeout=20
    )
    slowdown, shutdown = parse_temperature_thresholds(xml_result.stdout)
    plain_result = runner(["nvidia-smi", "-i", str(gpu_index)], timeout=20)
    cuda_match = re.search(r"CUDA Version:\s*([0-9.]+)", plain_result.stdout)
    return GpuMetadata(
        name=name,
        uuid=row[1].strip(),
        driver_version=row[2].strip(),
        cuda_version=cuda_match.group(1) if cuda_match else None,
        default_power_limit_w=_number(row[3]),
        minimum_power_limit_w=_number(row[4]),
        maximum_power_limit_w=_number(row[5]),
        slowdown_temperature_c=slowdown,
        shutdown_temperature_c=shutdown,
    )


def query_supported_clocks(
    gpu_index: int = 0, *, runner: CommandRunner = run_command
) -> dict[int, list[int]]:
    result = runner(
        [
            "nvidia-smi",
            "-i",
            str(gpu_index),
            "--query-supported-clocks=memory,graphics",
            "--format=csv,noheader,nounits",
        ],
        timeout=20,
    )
    return parse_supported_clocks(result.stdout)


class GpuControls:
    """Clock and power controls with unconditional reset in the outer context."""

    def __init__(
        self,
        gpu_index: int,
        runner: CommandRunner,
        *,
        minimum_power_limit_w: float | None = None,
        maximum_power_limit_w: float | None = None,
    ) -> None:
        self.gpu_index = gpu_index
        self.runner = runner
        self.minimum_power_limit_w = minimum_power_limit_w
        self.maximum_power_limit_w = maximum_power_limit_w
        self._control_lock = threading.Lock()

    def lock_graphics(self, frequency_mhz: int) -> None:
        try:
            with self._control_lock:
                self.runner(
                    [
                        "nvidia-smi",
                        "-i",
                        str(self.gpu_index),
                        "-lgc",
                        f"{frequency_mhz},{frequency_mhz}",
                    ],
                    timeout=20,
                )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
            raise ProfilingError(
                f"Could not lock graphics clock at {frequency_mhz} MHz. "
                "The protocol will not substitute an unlocked measurement."
            ) from error

    def set_power_limit(self, power_limit_w: float, *, verify: bool = True) -> float:
        """Set a power cap and return its hardware readback."""

        if not math.isfinite(power_limit_w):
            raise ProfilingError("The requested power limit must be finite.")
        if (
            self.minimum_power_limit_w is not None
            and power_limit_w < self.minimum_power_limit_w - 1e-9
        ):
            raise ProfilingError(
                f"Requested {power_limit_w:.1f} W below the GPU minimum power limit."
            )
        if (
            self.maximum_power_limit_w is not None
            and power_limit_w > self.maximum_power_limit_w + 1e-9
        ):
            raise ProfilingError(
                f"Requested {power_limit_w:.1f} W above the GPU maximum power limit."
            )
        try:
            with self._control_lock:
                self.runner(
                    [
                        "nvidia-smi",
                        "-i",
                        str(self.gpu_index),
                        "-pl",
                        f"{power_limit_w:.1f}",
                    ],
                    timeout=20,
                )
                if not verify:
                    return power_limit_w
                result = self.runner(
                    [
                        "nvidia-smi",
                        "-i",
                        str(self.gpu_index),
                        "--query-gpu=power.limit",
                        "--format=csv,noheader,nounits",
                    ],
                    timeout=20,
                )
            realized = _number(result.stdout)
        except (
            ProfilingError,
            subprocess.CalledProcessError,
            subprocess.TimeoutExpired,
        ) as error:
            raise ProfilingError(
                f"Could not set and read back the {power_limit_w:.1f} W power limit."
            ) from error
        if not math.isclose(realized, power_limit_w, rel_tol=0.0, abs_tol=0.51):
            raise ProfilingError(
                f"Power-limit readback {realized:.2f} W does not match "
                f"the requested {power_limit_w:.2f} W."
            )
        return realized

    def emergency_safe_down(
        self, *, power_limit_w: float, graphics_clock_mhz: int
    ) -> tuple[str, ...]:
        """Attempt both conservative controls without masking either failure."""

        errors: list[str] = []
        with self._control_lock:
            for arguments in (
                [
                    "nvidia-smi",
                    "-i",
                    str(self.gpu_index),
                    "-pl",
                    f"{power_limit_w:.1f}",
                ],
                [
                    "nvidia-smi",
                    "-i",
                    str(self.gpu_index),
                    "-lgc",
                    f"{graphics_clock_mhz},{graphics_clock_mhz}",
                ],
            ):
                try:
                    self.runner(arguments, timeout=20)
                except Exception as error:
                    errors.append(f"{' '.join(arguments)}: {error}")
        return tuple(errors)


@contextmanager
def managed_gpu_controls(
    metadata: GpuMetadata,
    memory_clock_mhz: int,
    *,
    gpu_index: int = 0,
    runner: CommandRunner = run_command,
) -> Iterator[tuple[GpuControls, dict[str, Any]]]:
    """Apply the fixed power/memory settings and always restore defaults."""

    if os.geteuid() != 0:
        raise ProfilingError(
            "Clock profiling must run as root so nvidia-smi lock and reset commands are reliable."
        )
    target_power = max(
        metadata.minimum_power_limit_w,
        min(metadata.maximum_power_limit_w, 0.9 * metadata.default_power_limit_w),
    )
    memory_locked = False
    memory_error: str | None = None
    controls = GpuControls(
        gpu_index,
        runner,
        minimum_power_limit_w=metadata.minimum_power_limit_w,
        maximum_power_limit_w=metadata.maximum_power_limit_w,
    )
    try:
        runner(["nvidia-smi", "-i", str(gpu_index), "-pm", "1"], timeout=20)
        runner(
            ["nvidia-smi", "-i", str(gpu_index), "-pl", f"{target_power:.1f}"],
            timeout=20,
        )
        try:
            runner(
                [
                    "nvidia-smi",
                    "-i",
                    str(gpu_index),
                    "-lmc",
                    f"{memory_clock_mhz},{memory_clock_mhz}",
                ],
                timeout=20,
            )
            memory_locked = True
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
            memory_error = str(error)
        yield controls, {
            "power_limit_w": target_power,
            "memory_clock_requested_mhz": memory_clock_mhz,
            "memory_clock_locked": memory_locked,
            "memory_clock_error": memory_error,
        }
    finally:
        reset_errors: list[str] = []
        for arguments in (
            ["nvidia-smi", "-i", str(gpu_index), "-rgc"],
            ["nvidia-smi", "-i", str(gpu_index), "-rmc"],
            [
                "nvidia-smi",
                "-i",
                str(gpu_index),
                "-pl",
                f"{metadata.default_power_limit_w:.1f}",
            ],
        ):
            try:
                runner(arguments, timeout=20)
            except Exception as error:  # Reset every remaining setting before reporting.
                reset_errors.append(f"{' '.join(arguments)}: {error}")
        if reset_errors and sys.exc_info()[0] is None:
            raise ProfilingError("GPU reset failed: " + "; ".join(reset_errors))


def _telemetry_row(output: str, *, elapsed_s: float, phase: str) -> dict[str, Any]:
    row = next(csv.reader([output.strip()]), [])
    if len(row) < 6:
        raise ProfilingError(f"Unexpected telemetry row from nvidia-smi: {output!r}")
    return {
        "elapsed_s": elapsed_s,
        "utc": datetime.now(timezone.utc).isoformat(),
        "phase": phase,
        "graphics_clock_mhz": _number(row[0]),
        "memory_clock_mhz": _number(row[1]),
        "power_w": _number(row[2]),
        "temperature_c": _number(row[3]),
        "utilization_percent": _number(row[4]),
        "memory_used_mib": _number(row[5]),
    }


class TelemetrySampler:
    """Collect nvidia-smi telemetry at an approximately fixed wall-time period."""

    def __init__(
        self,
        start_time: float,
        *,
        gpu_index: int = 0,
        period_s: float = TELEMETRY_PERIOD_S,
        runner: CommandRunner = run_command,
    ) -> None:
        self.start_time = start_time
        self.gpu_index = gpu_index
        self.period_s = period_s
        self.runner = runner
        self.rows: list[dict[str, Any]] = []
        self.error: Exception | None = None
        self._phase = "initialization"
        self._context: dict[str, Any] = {}
        self._phase_lock = threading.Lock()
        self._rows_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)

    def __enter__(self) -> "TelemetrySampler":
        self._thread.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self._stop.set()
        self._thread.join(timeout=max(2.0, 5.0 * self.period_s))
        if self._thread.is_alive() and exc_type is None:
            raise ProfilingError("Telemetry thread did not stop.")
        if self.error is not None and exc_type is None:
            raise ProfilingError(f"Telemetry collection failed: {self.error}") from self.error

    def set_phase(self, phase: str) -> None:
        with self._phase_lock:
            self._phase = phase

    def set_context(self, phase: str, **context: Any) -> None:
        """Atomically label subsequent samples with an experiment block."""

        with self._phase_lock:
            self._phase = phase
            self._context = dict(context)

    def snapshot(self) -> list[dict[str, Any]]:
        with self._rows_lock:
            return [dict(row) for row in self.rows]

    def _sample_loop(self) -> None:
        query = [
            "nvidia-smi",
            "-i",
            str(self.gpu_index),
            "--query-gpu=clocks.current.graphics,clocks.current.memory,power.draw,"
            "temperature.gpu,utilization.gpu,memory.used",
            "--format=csv,noheader,nounits",
        ]
        while not self._stop.is_set():
            iteration_start = time.monotonic()
            try:
                result = self.runner(query, timeout=max(2.0, 5.0 * self.period_s))
                with self._phase_lock:
                    phase = self._phase
                    context = dict(self._context)
                row = _telemetry_row(
                    result.stdout,
                    elapsed_s=time.monotonic() - self.start_time,
                    phase=phase,
                )
                row.update(context)
                with self._rows_lock:
                    self.rows.append(row)
            except Exception as error:
                self.error = error
                self._stop.set()
                return
            remaining = self.period_s - (time.monotonic() - iteration_start)
            if remaining > 0:
                self._stop.wait(remaining)

    def ensure_healthy(self) -> None:
        if self.error is not None:
            raise ProfilingError(f"Telemetry collection failed: {self.error}") from self.error

    def between(self, start_elapsed_s: float, end_elapsed_s: float) -> list[dict[str, Any]]:
        return [
            row
            for row in self.snapshot()
            if start_elapsed_s <= float(row["elapsed_s"]) <= end_elapsed_s
        ]


class ThermalSafetyWatchdog:
    """Monitor telemetry independently of blocking inference requests."""

    def __init__(
        self,
        sampler: TelemetrySampler,
        controls: GpuControls,
        *,
        profile_start: float,
        safe_clock_mhz: int,
        safe_power_limit_w: float = THERMAL_MINIMUM_POWER_W,
        safe_down_temperature_c: float = THERMAL_SAFE_DOWN_C,
        abort_temperature_c: float = THERMAL_ABORT_C,
        stale_after_s: float = THERMAL_STALE_TELEMETRY_S,
        poll_s: float = 0.05,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        if abort_temperature_c <= safe_down_temperature_c:
            raise ValueError("The abort threshold must exceed the safe-down threshold.")
        self.sampler = sampler
        self.controls = controls
        self.profile_start = profile_start
        self.safe_clock_mhz = safe_clock_mhz
        self.safe_power_limit_w = safe_power_limit_w
        self.safe_down_temperature_c = safe_down_temperature_c
        self.abort_temperature_c = abort_temperature_c
        self.stale_after_s = stale_after_s
        self.poll_s = poll_s
        self.monotonic = monotonic
        self.safe_down = threading.Event()
        self.abort = threading.Event()
        self.stop_requested = threading.Event()
        self.reason: str | None = None
        self.safe_down_errors: tuple[str, ...] = ()
        self.started_monotonic: float | None = None
        self._state_lock = threading.Lock()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def __enter__(self) -> "ThermalSafetyWatchdog":
        self.started_monotonic = self.monotonic()
        self._thread.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.stop_requested.set()
        self._thread.join(timeout=max(2.0, 10.0 * self.poll_s))
        if self._thread.is_alive() and exc_type is None:
            raise ProfilingError("The thermal safety watchdog did not stop.")

    def _trip(self, reason: str, *, hard_abort: bool) -> None:
        with self._state_lock:
            first_trip = not self.safe_down.is_set()
            if first_trip:
                self.reason = reason
                self.safe_down_errors = self.controls.emergency_safe_down(
                    power_limit_w=self.safe_power_limit_w,
                    graphics_clock_mhz=self.safe_clock_mhz,
                )
                self.safe_down.set()
            if hard_abort or self.safe_down_errors:
                self.abort.set()
                if hard_abort:
                    self.reason = reason

    def abort_for_control_failure(self, reason: str) -> None:
        self._trip(reason, hard_abort=True)

    def inspect_once(self) -> None:
        """Inspect current state once; exposed for deterministic safety tests."""

        if self.sampler.error is not None:
            self._trip(
                f"telemetry failure: {type(self.sampler.error).__name__}: {self.sampler.error}",
                hard_abort=True,
            )
            return
        rows = self.sampler.snapshot()
        now = self.monotonic()
        if not rows:
            started = self.started_monotonic if self.started_monotonic is not None else now
            if now - started > self.stale_after_s:
                self._trip("telemetry did not start within the safety deadline", hard_abort=True)
            return
        latest = rows[-1]
        sample_time = self.profile_start + float(latest["elapsed_s"])
        if now - sample_time > self.stale_after_s:
            self._trip(
                f"telemetry is stale by {now - sample_time:.3f} s",
                hard_abort=True,
            )
            return
        temperature_c = float(latest["temperature_c"])
        if not math.isfinite(temperature_c):
            self._trip("telemetry temperature is non-finite", hard_abort=True)
        elif temperature_c >= self.abort_temperature_c:
            self._trip(
                f"temperature reached the {self.abort_temperature_c:.1f} C abort threshold",
                hard_abort=True,
            )
        elif temperature_c >= self.safe_down_temperature_c:
            self._trip(
                f"temperature reached the {self.safe_down_temperature_c:.1f} C safe-down threshold",
                hard_abort=False,
            )

    def raise_if_stopped(self) -> None:
        if self.abort.is_set():
            detail = "; ".join(self.safe_down_errors)
            suffix = f" Safe-down errors: {detail}" if detail else ""
            raise ProfilingError(f"Thermal watchdog aborted: {self.reason}.{suffix}")
        if self.safe_down.is_set():
            raise ProfilingError(f"Thermal watchdog stopped excitation: {self.reason}.")

    def _run(self) -> None:
        while not self.stop_requested.is_set():
            self.inspect_once()
            self.stop_requested.wait(self.poll_s)


def apply_verified_thermal_power_limit(
    controls: GpuControls,
    watchdog: ThermalSafetyWatchdog,
    power_limit_w: float,
) -> float:
    """Apply one scheduled cap or force the card into its safe state."""

    try:
        return controls.set_power_limit(power_limit_w, verify=True)
    except Exception as error:
        watchdog.abort_for_control_failure(
            f"power-limit command or readback failed at {power_limit_w:.1f} W: {error}"
        )
        raise ProfilingError(
            f"Thermal power-limit application failed at {power_limit_w:.1f} W."
        ) from error


def wait_for_thermal_identification_cooldown(
    sampler: TelemetrySampler,
    controls: GpuControls,
    watchdog: ThermalSafetyWatchdog,
    *,
    block: ThermalIdentificationBlock,
    profile_start: float,
    cooldown_clock_mhz: int,
    events: list[dict[str, Any]],
    target_temperature_c: float = THERMAL_COOLDOWN_TARGET_C,
    stability_band_c: float = THERMAL_COOLDOWN_STABILITY_C,
    stability_window_s: float = THERMAL_COOLDOWN_STABILITY_S,
    timeout_s: float = THERMAL_COOLDOWN_TIMEOUT_S,
    poll_s: float = SWEEP_COOLDOWN_POLL_S,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Reach the same cool, slowly varying state before one isolated pulse."""

    if stability_window_s <= 0.0 or timeout_s <= stability_window_s:
        raise ValueError("Cooldown timeout must exceed its positive stability window.")
    controls.lock_graphics(cooldown_clock_mhz)
    apply_verified_thermal_power_limit(
        controls, watchdog, THERMAL_MINIMUM_POWER_W
    )
    phase = f"thermal_identification_cooldown_before_{block.block_id}"
    sampler.set_context(
        phase,
        split="conditioning",
        sequence=block.sequence,
        block_id=f"cooldown_before_{block.block_id}",
        block_role="cooldown",
        requested_power_limit_w=THERMAL_MINIMUM_POWER_W,
        requested_clock_mhz=cooldown_clock_mhz,
        workload_phase="idle",
    )
    started = monotonic()
    event: dict[str, Any] = {
        "sequence": block.sequence,
        "before_block_id": block.block_id,
        "phase": phase,
        "started_elapsed_s": started - profile_start,
        "target_temperature_c": target_temperature_c,
        "stability_band_c": stability_band_c,
        "stability_window_s": stability_window_s,
        "timeout_s": timeout_s,
        "status": "running",
    }
    events.append(event)
    while True:
        sampler.ensure_healthy()
        watchdog.raise_if_stopped()
        phase_rows = [
            row for row in sampler.snapshot() if str(row.get("phase")) == phase
        ]
        if phase_rows:
            latest_elapsed = float(phase_rows[-1]["elapsed_s"])
            window = [
                row
                for row in phase_rows
                if float(row["elapsed_s"]) >= latest_elapsed - stability_window_s
            ]
            temperatures = [float(row["temperature_c"]) for row in window]
            spans_window = (
                float(window[-1]["elapsed_s"]) - float(window[0]["elapsed_s"])
                >= stability_window_s - max(0.2, 2.0 * sampler.period_s)
            )
            if (
                spans_window
                and max(temperatures) <= target_temperature_c
                and max(temperatures) - min(temperatures) <= stability_band_c
            ):
                event.update(
                    {
                        "status": "complete",
                        "completed_elapsed_s": latest_elapsed,
                        "duration_s": monotonic() - started,
                        "final_temperature_c": temperatures[-1],
                        "window_min_temperature_c": min(temperatures),
                        "window_max_temperature_c": max(temperatures),
                    }
                )
                return event
        elapsed = monotonic() - started
        if elapsed >= timeout_s:
            event.update({"status": "failed_timeout", "duration_s": elapsed})
            watchdog.abort_for_control_failure(
                f"standardized cooldown before {block.block_id} timed out"
            )
            raise ProfilingError(
                f"Thermal cooldown before {block.block_id} did not stabilize "
                f"within {timeout_s:.0f} s."
            )
        sleep(min(poll_s, timeout_s - elapsed))


def _server_json(url: str, timeout_s: float) -> Any:
    try:
        with urlopen(url, timeout=timeout_s) as response:
            return json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as error:
        raise ProfilingError(f"Could not read vLLM endpoint {url}: {error}") from error


class VllmServer:
    """Launch the pinned vLLM container or validate an explicitly external one."""

    def __init__(
        self,
        *,
        server_url: str,
        image: str,
        model: str,
        revision: str,
        served_name: str,
        log_path: Path,
        launch: bool,
        runner: CommandRunner = run_command,
        readiness_timeout_s: float = 900.0,
    ) -> None:
        self.server_url = server_url.rstrip("/")
        self.image = image
        self.model = model
        self.revision = revision
        self.served_name = served_name
        self.log_path = log_path
        self.launch = launch
        self.runner = runner
        self.readiness_timeout_s = readiness_timeout_s
        self.process: subprocess.Popen[str] | None = None
        self.log_stream: Any = None
        self.image_digest: str | None = None
        self.server_version: str | None = None

    def __enter__(self) -> "VllmServer":
        if self.launch:
            if shutil.which("docker") is None:
                raise ProfilingError("Docker is required to launch the pinned vLLM server.")
            self.runner(["docker", "pull", self.image], timeout=1800)
            digest = self.runner(
                [
                    "docker",
                    "image",
                    "inspect",
                    "--format={{if .RepoDigests}}{{index .RepoDigests 0}}{{else}}{{.Id}}{{end}}",
                    self.image,
                ],
                timeout=30,
            ).stdout.strip()
            if not digest:
                raise ProfilingError("Docker did not report an image digest or image ID.")
            self.image_digest = digest
            self.runner(
                [
                    "docker",
                    "run",
                    "--rm",
                    "--gpus",
                    "device=0",
                    "--entrypoint",
                    "nvidia-smi",
                    self.image,
                    "-L",
                ],
                timeout=60,
            )
            port_match = re.search(r":(\d+)(?:/|$)", self.server_url)
            if not port_match:
                raise ProfilingError("server_url must include an explicit local TCP port.")
            host_port = port_match.group(1)
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self.log_stream = self.log_path.open("w", encoding="utf-8")
            command = [
                "docker",
                "run",
                "--rm",
                "--gpus",
                "device=0",
                "--ipc=host",
                "-p",
                f"{host_port}:8000",
                self.image,
                "--model",
                self.model,
                "--revision",
                self.revision,
                "--served-model-name",
                self.served_name,
                "--dtype",
                "auto",
                "--max-model-len",
                "8192",
                "--gpu-memory-utilization",
                "0.90",
                # Repeated fixed-length prompts are part of the measurement
                # matrix.  vLLM 0.28 enables prefix caching by default, which
                # would turn later prefill trials into cache-hit benchmarks.
                "--no-enable-prefix-caching",
            ]
            self.process = subprocess.Popen(
                command,
                stdout=self.log_stream,
                stderr=subprocess.STDOUT,
                text=True,
            )
        else:
            self.image_digest = "external-server-not-inspected"

        deadline = time.monotonic() + self.readiness_timeout_s
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            if self.process is not None and self.process.poll() is not None:
                raise ProfilingError(
                    f"vLLM exited with code {self.process.returncode}; inspect {self.log_path}."
                )
            try:
                with urlopen(f"{self.server_url}/health", timeout=3):
                    models = _server_json(f"{self.server_url}/v1/models", 10)
                    identifiers = {
                        str(item.get("id"))
                        for item in models.get("data", [])
                        if isinstance(item, Mapping)
                    }
                    if self.served_name not in identifiers:
                        raise ProfilingError(
                            f"vLLM is healthy but does not expose {self.served_name!r}: {sorted(identifiers)}"
                        )
                    try:
                        version_payload = _server_json(f"{self.server_url}/version", 10)
                        if isinstance(version_payload, Mapping):
                            self.server_version = str(version_payload.get("version") or "") or None
                    except ProfilingError:
                        self.server_version = None
                    return self
            except Exception as error:
                last_error = error
                time.sleep(2.0)
        raise ProfilingError(
            f"vLLM was not ready after {self.readiness_timeout_s:.0f} s: {last_error}"
        )

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=10)
        if self.log_stream is not None:
            self.log_stream.close()


def _completion_request(
    server_url: str,
    served_model: str,
    *,
    prompt_tokens: int,
    output_tokens: int,
    request_index: int,
    profile_start: float,
    barrier: threading.Barrier | None,
    timeout_s: float,
) -> RequestTiming:
    if barrier is not None:
        barrier.wait(timeout=30)
    token_id = 1000 + (request_index % 17)
    payload = {
        "model": served_model,
        "prompt": [token_id] * prompt_tokens,
        "max_tokens": output_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "ignore_eos": True,
    }
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    request = Request(
        f"{server_url.rstrip('/')}/v1/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    start = time.monotonic()
    first_token_time: float | None = None
    completion_tokens: int | None = None
    observed_prompt_tokens: int | None = None
    try:
        with urlopen(request, timeout=timeout_s) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8").strip()
                if not line.startswith("data:"):
                    continue
                item = line[5:].strip()
                if item == "[DONE]":
                    break
                message = json.loads(item)
                choices = message.get("choices") or []
                if choices and choices[0].get("text") and first_token_time is None:
                    first_token_time = time.monotonic()
                usage = message.get("usage")
                if isinstance(usage, Mapping):
                    completion_tokens = int(usage.get("completion_tokens", 0))
                    observed_prompt_tokens = int(usage.get("prompt_tokens", 0))
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as error:
        raise ProfilingError(f"Streaming vLLM request failed: {error}") from error
    end = time.monotonic()
    if first_token_time is None or completion_tokens is None or observed_prompt_tokens is None:
        raise ProfilingError("vLLM stream omitted token timing or final usage metadata.")
    if observed_prompt_tokens != prompt_tokens:
        raise ProfilingError(
            f"Requested {prompt_tokens} prompt token IDs but vLLM reported {observed_prompt_tokens}."
        )
    if completion_tokens != output_tokens:
        raise ProfilingError(
            f"Requested {output_tokens} generated tokens but vLLM reported {completion_tokens}."
        )
    ttft = first_token_time - start
    tpot = (
        (end - first_token_time) / (completion_tokens - 1)
        if completion_tokens > 1
        else 0.0
    )
    return RequestTiming(
        request_index=request_index,
        prompt_tokens_observed=observed_prompt_tokens,
        completion_tokens=completion_tokens,
        ttft_s=ttft,
        tpot_s=tpot,
        total_s=end - start,
        start_elapsed_s=start - profile_start,
        end_elapsed_s=end - profile_start,
    )


def run_concurrent_batch(
    condition: BenchmarkCondition,
    *,
    server_url: str,
    served_model: str,
    profile_start: float,
    timeout_s: float,
) -> list[RequestTiming]:
    barrier = threading.Barrier(condition.concurrency)
    with ThreadPoolExecutor(max_workers=condition.concurrency) as executor:
        futures = [
            executor.submit(
                _completion_request,
                server_url,
                served_model,
                prompt_tokens=condition.prompt_tokens,
                output_tokens=condition.output_tokens,
                request_index=request_index,
                profile_start=profile_start,
                barrier=barrier,
                timeout_s=timeout_s,
            )
            for request_index in range(condition.concurrency)
        ]
        return [future.result() for future in futures]


def profile_conditions() -> list[BenchmarkCondition]:
    conditions: list[BenchmarkCondition] = []
    for prompt_tokens in (128, 512, 2048, 4096):
        for concurrency in (1, 4, 8):
            conditions.append(
                BenchmarkCondition("prefill", prompt_tokens, 1, concurrency)
            )
    for prompt_tokens in (128, 1024):
        for concurrency in (1, 4, 8):
            conditions.append(
                BenchmarkCondition("decode", prompt_tokens, 128, concurrency)
            )
    return conditions


_THERMAL_DECODE_CONDITION = BenchmarkCondition("decode", 128, 32, 8)
_THERMAL_PREFILL_CONDITION = BenchmarkCondition("prefill", 4096, 1, 8)
_THERMAL_TRAINING_DURATION_CEILINGS_S: dict[int, float] = {
    40: 120.0,
    46: 105.0,
    52: 90.0,
    58: 75.0,
    64: 60.0,
}
_THERMAL_TRAINING_PULSES: dict[str, tuple[tuple[int, float], ...]] = {
    # Both repeats contain every training cap exactly once. The different order
    # and duration at each cap separate repeatability from pulse-length effects.
    "training_a": (
        (40, 120.0),
        (52, 90.0),
        (64, 60.0),
        (46, 105.0),
        (58, 75.0),
    ),
    "training_b": (
        (58, 60.0),
        (46, 90.0),
        (40, 105.0),
        (64, 45.0),
        (52, 75.0),
    ),
}
_THERMAL_VALIDATION_SPEC: tuple[
    tuple[int, float, BenchmarkCondition, str], ...
] = (
    (43, 90.0, _THERMAL_DECODE_CONDITION, "intermediate_cap"),
    (55, 60.0, _THERMAL_DECODE_CONDITION, "intermediate_cap"),
    (49, 75.0, _THERMAL_DECODE_CONDITION, "intermediate_cap"),
    (61, 45.0, _THERMAL_DECODE_CONDITION, "intermediate_cap"),
    (55, 60.0, _THERMAL_PREFILL_CONDITION, "workload_transfer"),
    (55, 60.0, _THERMAL_DECODE_CONDITION, "workload_transfer"),
)
_THERMAL_PHASE_TRAINING_SPEC: tuple[
    tuple[int, float, BenchmarkCondition], ...
] = (
    (46, 75.0, _THERMAL_DECODE_CONDITION),
    (61, 45.0, _THERMAL_PREFILL_CONDITION),
    (46, 75.0, _THERMAL_PREFILL_CONDITION),
    (61, 45.0, _THERMAL_DECODE_CONDITION),
)
_THERMAL_PHASE_VALIDATION_SPEC: tuple[
    tuple[int, float, BenchmarkCondition], ...
] = (
    (55, 60.0, _THERMAL_DECODE_CONDITION),
    (55, 60.0, _THERMAL_PREFILL_CONDITION),
)


def _thermal_training_blocks(
    sequence: str, pulses: Sequence[tuple[int, float]]
) -> tuple[ThermalIdentificationBlock, ...]:
    return tuple(
        ThermalIdentificationBlock(
            block_id=f"{sequence}_pulse_{index:02d}",
            split="training",
            sequence=sequence,
            role="training_pulse",
            requested_power_limit_w=float(power),
            duration_s=duration_s,
            condition=_THERMAL_DECODE_CONDITION,
        )
        for index, (power, duration_s) in enumerate(pulses)
    )


def thermal_identification_sequences(
) -> tuple[tuple[str, tuple[ThermalIdentificationBlock, ...]], ...]:
    """Return the immutable, cold-start pulse train/validation schedule."""

    training_a = _thermal_training_blocks(
        "training_a", _THERMAL_TRAINING_PULSES["training_a"]
    )
    training_b = _thermal_training_blocks(
        "training_b", _THERMAL_TRAINING_PULSES["training_b"]
    )
    validation = tuple(
        ThermalIdentificationBlock(
            block_id=f"validation_pulse_{index:02d}",
            split="validation",
            sequence="validation",
            role=role,
            requested_power_limit_w=float(power),
            duration_s=duration_s,
            condition=condition,
        )
        for index, (power, duration_s, condition, role) in enumerate(
            _THERMAL_VALIDATION_SPEC
        )
    )
    sequences = (
        ("training_a", training_a),
        ("training_b", training_b),
        ("validation", validation),
    )
    validate_thermal_identification_schedule(sequences)
    return sequences


def validate_thermal_identification_schedule(
    sequences: Sequence[tuple[str, Sequence[ThermalIdentificationBlock]]],
) -> None:
    """Fail closed if the prespecified pulse split or safety ceiling changes."""

    if tuple(name for name, _ in sequences) != (
        "training_a",
        "training_b",
        "validation",
    ):
        raise ProfilingError("Thermal identification sequences are not in the fixed order.")
    blocks = [block for _, sequence_blocks in sequences for block in sequence_blocks]
    identifiers = [block.block_id for block in blocks]
    if len(identifiers) != len(set(identifiers)):
        raise ProfilingError("Thermal identification block identifiers must be unique.")
    if any(block.duration_s <= 0.0 for block in blocks):
        raise ProfilingError("Thermal identification pulse durations must be positive.")
    if any(block.role == "plateau" for block in blocks):
        raise ProfilingError("The cold-start protocol must not contain plateaus.")

    training_orders: list[tuple[int, ...]] = []
    for sequence_name in ("training_a", "training_b"):
        sequence_blocks = next(
            items for name, items in sequences if name == sequence_name
        )
        if len(sequence_blocks) != 5 or any(
            block.split != "training"
            or block.role != "training_pulse"
            or block.condition != _THERMAL_DECODE_CONDITION
            for block in sequence_blocks
        ):
            raise ProfilingError(
                f"{sequence_name} must contain five fixed decode training pulses."
            )
        cap_to_duration = {
            int(block.requested_power_limit_w): block.duration_s
            for block in sequence_blocks
        }
        if set(cap_to_duration) != set(_THERMAL_TRAINING_DURATION_CEILINGS_S):
            raise ProfilingError(
                f"{sequence_name} must contain every training cap exactly once."
            )
        if any(
            cap_to_duration[cap] > ceiling
            for cap, ceiling in _THERMAL_TRAINING_DURATION_CEILINGS_S.items()
        ):
            raise ProfilingError(f"{sequence_name} exceeds a pulse-duration ceiling.")
        training_orders.append(
            tuple(int(block.requested_power_limit_w) for block in sequence_blocks)
        )
    if training_orders[0] == training_orders[1]:
        raise ProfilingError("The two deterministic training orders must differ.")
    if any(
        first.duration_s == second.duration_s
        for first in sequences[0][1]
        for second in sequences[1][1]
        if first.requested_power_limit_w == second.requested_power_limit_w
    ):
        raise ProfilingError("Training pulse durations must vary between repeats.")

    validation_blocks = next(items for name, items in sequences if name == "validation")
    if any(block.split != "validation" for block in validation_blocks):
        raise ProfilingError("Thermal validation blocks must remain held out.")
    intermediate = [
        block for block in validation_blocks if block.role == "intermediate_cap"
    ]
    if len(intermediate) != 4 or {
        int(block.requested_power_limit_w) for block in intermediate
    } != {43, 49, 55, 61}:
        raise ProfilingError(
            "Thermal validation must contain one cold pulse at every intermediate cap."
        )
    if any(block.duration_s > 90.0 for block in validation_blocks):
        raise ProfilingError("Thermal validation pulses must not exceed 90 seconds.")
    matched = [
        block for block in validation_blocks if block.role == "workload_transfer"
    ]
    if len(matched) != 2 or {
        (block.requested_power_limit_w, block.duration_s) for block in matched
    } != {(55.0, 60.0)} or {
        block.condition.phase for block in matched
    } != {"prefill", "decode"}:
        raise ProfilingError(
            "Thermal validation requires a matched 55 W prefill/decode pulse pair."
        )


def thermal_phase_identification_sequences(
) -> tuple[tuple[str, tuple[ThermalIdentificationBlock, ...]], ...]:
    """Return the immutable train/validation schedule for the phase follow-up."""

    training = tuple(
        ThermalIdentificationBlock(
            block_id=f"phase_training_pulse_{index:02d}",
            split="training",
            sequence="phase_training",
            role="phase_training",
            requested_power_limit_w=float(power),
            duration_s=duration_s,
            condition=condition,
        )
        for index, (power, duration_s, condition) in enumerate(
            _THERMAL_PHASE_TRAINING_SPEC
        )
    )
    validation = tuple(
        ThermalIdentificationBlock(
            block_id=f"phase_validation_pulse_{index:02d}",
            split="validation",
            sequence="phase_validation",
            role="phase_validation",
            requested_power_limit_w=float(power),
            duration_s=duration_s,
            condition=condition,
        )
        for index, (power, duration_s, condition) in enumerate(
            _THERMAL_PHASE_VALIDATION_SPEC
        )
    )
    sequences = (("phase_training", training), ("phase_validation", validation))
    validate_thermal_phase_identification_schedule(sequences)
    return sequences


def validate_thermal_phase_identification_schedule(
    sequences: Sequence[tuple[str, Sequence[ThermalIdentificationBlock]]],
) -> None:
    """Fail closed if the confirmatory phase schedule or split changes."""

    if tuple(name for name, _ in sequences) != (
        "phase_training",
        "phase_validation",
    ):
        raise ProfilingError("Thermal phase sequences are not in the fixed order.")
    blocks = [block for _, sequence_blocks in sequences for block in sequence_blocks]
    if len(blocks) != 6 or len({block.block_id for block in blocks}) != 6:
        raise ProfilingError("Thermal phase identification requires six unique pulses.")

    def signature(block: ThermalIdentificationBlock) -> tuple[Any, ...]:
        return (
            block.block_id,
            block.split,
            block.sequence,
            block.role,
            int(block.requested_power_limit_w),
            block.duration_s,
            block.condition.phase,
            block.condition.prompt_tokens,
            block.condition.output_tokens,
            block.condition.concurrency,
        )

    expected = (
        (
            "phase_training_pulse_00",
            "training",
            "phase_training",
            "phase_training",
            46,
            75.0,
            "decode",
            128,
            32,
            8,
        ),
        (
            "phase_training_pulse_01",
            "training",
            "phase_training",
            "phase_training",
            61,
            45.0,
            "prefill",
            4096,
            1,
            8,
        ),
        (
            "phase_training_pulse_02",
            "training",
            "phase_training",
            "phase_training",
            46,
            75.0,
            "prefill",
            4096,
            1,
            8,
        ),
        (
            "phase_training_pulse_03",
            "training",
            "phase_training",
            "phase_training",
            61,
            45.0,
            "decode",
            128,
            32,
            8,
        ),
        (
            "phase_validation_pulse_00",
            "validation",
            "phase_validation",
            "phase_validation",
            55,
            60.0,
            "decode",
            128,
            32,
            8,
        ),
        (
            "phase_validation_pulse_01",
            "validation",
            "phase_validation",
            "phase_validation",
            55,
            60.0,
            "prefill",
            4096,
            1,
            8,
        ),
    )
    observed = tuple(signature(block) for block in blocks)
    if observed != expected:
        raise ProfilingError("Thermal phase pulse order, split, or condition changed.")


class ContinuousThermalLoad:
    """Keep one fixed inference workload active for a bounded thermal pulse."""

    def __init__(
        self,
        condition: BenchmarkCondition,
        initial_block: ThermalIdentificationBlock,
        *,
        server_url: str,
        served_model: str,
        profile_start: float,
        timeout_s: float,
        request_rows: list[dict[str, Any]],
        request_rows_lock: threading.Lock,
    ) -> None:
        self.condition = condition
        self.server_url = server_url
        self.served_model = served_model
        self.profile_start = profile_start
        self.timeout_s = timeout_s
        self.request_rows = request_rows
        self.request_rows_lock = request_rows_lock
        self.block = initial_block
        self._stop = threading.Event()
        self._error: Exception | None = None
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def ensure_healthy(self) -> None:
        if self._error is not None:
            raise ProfilingError(
                f"Continuous thermal workload failed: {type(self._error).__name__}: {self._error}"
            ) from self._error
        if not self._thread.is_alive() and not self._stop.is_set():
            raise ProfilingError("Continuous thermal workload stopped unexpectedly.")

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=min(120.0, max(30.0, self.timeout_s + 5.0)))
        if self._thread.is_alive():
            raise ProfilingError("Continuous thermal workload did not stop after its request.")
        self.ensure_healthy()

    def _run(self) -> None:
        batch_index = 0
        try:
            while not self._stop.is_set():
                timings = run_concurrent_batch(
                    self.condition,
                    server_url=self.server_url,
                    served_model=self.served_model,
                    profile_start=self.profile_start,
                    timeout_s=self.timeout_s,
                )
                recorded = [
                    {
                        "split": self.block.split,
                        "sequence": self.block.sequence,
                        "block_id": self.block.block_id,
                        "workload_phase": self.block.condition.phase,
                        "prompt_tokens": self.block.condition.prompt_tokens,
                        "output_tokens": self.block.condition.output_tokens,
                        "concurrency": self.block.condition.concurrency,
                        "requested_power_limit_w": self.block.requested_power_limit_w,
                        "requested_clock_mhz": THERMAL_REQUESTED_CLOCK_MHZ,
                        "batch_index": batch_index,
                        "request_index": timing.request_index,
                        "prompt_tokens_observed": timing.prompt_tokens_observed,
                        "completion_tokens": timing.completion_tokens,
                        "ttft_s": timing.ttft_s,
                        "tpot_s": timing.tpot_s,
                        "total_s": timing.total_s,
                        "start_elapsed_s": timing.start_elapsed_s,
                        "end_elapsed_s": timing.end_elapsed_s,
                    }
                    for timing in timings
                ]
                with self.request_rows_lock:
                    self.request_rows.extend(recorded)
                batch_index += 1
        except Exception as error:
            self._error = error


def checkpoint_thermal_block(
    *,
    output_directory: Path,
    telemetry_path: Path,
    requests_path: Path,
    manifest_path: Path,
    sampler: TelemetrySampler,
    request_rows: Sequence[Mapping[str, Any]],
    manifest: dict[str, Any],
    block: ThermalIdentificationBlock,
    started_elapsed_s: float,
    ended_elapsed_s: float,
    status: str,
    marker_prefix: str = "thermal-block",
) -> dict[str, Any]:
    """Atomically preserve all acquisition progress after one scheduled block."""

    telemetry_rows = sampler.snapshot()
    _write_csv(telemetry_path, telemetry_rows, THERMAL_TELEMETRY_COLUMNS)
    _write_csv(requests_path, request_rows, THERMAL_REQUEST_COLUMNS)
    block_rows = [
        row
        for row in telemetry_rows
        if started_elapsed_s <= float(row["elapsed_s"]) <= ended_elapsed_s
    ]
    checkpoint: dict[str, Any] = {
        "block": asdict(block),
        "status": status,
        "started_elapsed_s": started_elapsed_s,
        "ended_elapsed_s": ended_elapsed_s,
        "actual_duration_s": max(0.0, ended_elapsed_s - started_elapsed_s),
        "telemetry_rows": len(telemetry_rows),
        "request_rows": len(request_rows),
    }
    if block_rows:
        checkpoint.update(
            {
                "block_telemetry_rows": len(block_rows),
                "mean_power_w": sum(float(row["power_w"]) for row in block_rows)
                / len(block_rows),
                "minimum_temperature_c": min(
                    float(row["temperature_c"]) for row in block_rows
                ),
                "maximum_temperature_c": max(
                    float(row["temperature_c"]) for row in block_rows
                ),
                "median_realized_clock_mhz": _median(
                    [float(row["graphics_clock_mhz"]) for row in block_rows]
                ),
            }
        )
    manifest.setdefault("block_checkpoints", []).append(checkpoint)
    manifest["latest_block_checkpoint"] = checkpoint
    if status == "complete":
        manifest.setdefault("completed_block_ids", []).append(block.block_id)
    _write_json(manifest_path, manifest)
    marker_suffix = "done" if status == "complete" else "failed"
    _write_json(
        output_directory / f"{marker_prefix}-{block.block_id}.{marker_suffix}",
        checkpoint,
    )
    return checkpoint


def run_thermal_identification_sequence(
    blocks: Sequence[ThermalIdentificationBlock],
    *,
    controls: GpuControls,
    sampler: TelemetrySampler,
    watchdog: ThermalSafetyWatchdog,
    server_url: str,
    served_model: str,
    profile_start: float,
    timeout_s: float,
    output_directory: Path,
    telemetry_path: Path,
    requests_path: Path,
    manifest_path: Path,
    request_rows: list[dict[str, Any]],
    request_rows_lock: threading.Lock,
    manifest: dict[str, Any],
    cooldown_clock_mhz: int,
    cooldown_events: list[dict[str, Any]],
    marker_prefix: str = "thermal-block",
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    """Run cold-start pulses, stopping each workload before the next cooldown."""

    for block in blocks:
        wait_for_thermal_identification_cooldown(
            sampler,
            controls,
            watchdog,
            block=block,
            profile_start=profile_start,
            cooldown_clock_mhz=cooldown_clock_mhz,
            events=cooldown_events,
            monotonic=monotonic,
            sleep=sleep,
        )
        _write_csv(telemetry_path, sampler.snapshot(), THERMAL_TELEMETRY_COLUMNS)
        _write_json(manifest_path, manifest)
        try:
            controls.lock_graphics(THERMAL_REQUESTED_CLOCK_MHZ)
        except Exception as error:
            watchdog.abort_for_control_failure(
                f"graphics-clock command failed before {block.block_id}: {error}"
            )
            raise
        sampler.set_context(
            f"thermal_identification_{block.block_id}",
            split=block.split,
            sequence=block.sequence,
            block_id=block.block_id,
            block_role=block.role,
            requested_power_limit_w=block.requested_power_limit_w,
            requested_clock_mhz=THERMAL_REQUESTED_CLOCK_MHZ,
            workload_phase=block.condition.phase,
        )
        started_elapsed_s = monotonic() - profile_start
        status = "failed"
        pulse_error: Exception | None = None
        active_load: ContinuousThermalLoad | None = None
        try:
            applied_power = apply_verified_thermal_power_limit(
                controls, watchdog, block.requested_power_limit_w
            )
            if not math.isclose(
                applied_power,
                block.requested_power_limit_w,
                rel_tol=0.0,
                abs_tol=0.51,
            ):
                watchdog.abort_for_control_failure(
                    "verified power limit changed unexpectedly"
                )
                watchdog.raise_if_stopped()
            active_load = ContinuousThermalLoad(
                block.condition,
                block,
                server_url=server_url,
                served_model=served_model,
                profile_start=profile_start,
                timeout_s=timeout_s,
                request_rows=request_rows,
                request_rows_lock=request_rows_lock,
            )
            active_load.start()
            deadline = monotonic() + block.duration_s
            while monotonic() < deadline:
                sampler.ensure_healthy()
                watchdog.raise_if_stopped()
                active_load.ensure_healthy()
                sleep(min(0.1, max(0.0, deadline - monotonic())))
            status = "complete"
        except Exception as error:
            pulse_error = error
        finally:
            ended_elapsed_s = monotonic() - profile_start
            sampler.set_context(
                f"thermal_identification_transition_after_{block.block_id}",
                split="conditioning",
                sequence=block.sequence,
                block_id=f"transition_after_{block.block_id}",
                block_role="workload_stop",
                requested_power_limit_w=THERMAL_MINIMUM_POWER_W,
                requested_clock_mhz=cooldown_clock_mhz,
                workload_phase="transition_idle",
            )
            try:
                apply_verified_thermal_power_limit(
                    controls, watchdog, THERMAL_MINIMUM_POWER_W
                )
                controls.lock_graphics(cooldown_clock_mhz)
            except Exception as error:
                watchdog.abort_for_control_failure(
                    f"safe controls failed after {block.block_id}: {error}"
                )
                if pulse_error is None:
                    pulse_error = error
                status = "failed"
            if active_load is not None:
                try:
                    active_load.stop()
                except Exception as error:
                    if pulse_error is None:
                        pulse_error = error
                    status = "failed"
            with request_rows_lock:
                saved_requests = [dict(row) for row in request_rows]
            checkpoint_thermal_block(
                output_directory=output_directory,
                telemetry_path=telemetry_path,
                requests_path=requests_path,
                manifest_path=manifest_path,
                sampler=sampler,
                request_rows=saved_requests,
                manifest=manifest,
                block=block,
                started_elapsed_s=started_elapsed_s,
                ended_elapsed_s=ended_elapsed_s,
                status=status,
                marker_prefix=marker_prefix,
            )
        if pulse_error is not None:
            raise pulse_error


def _telemetry_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    start_elapsed_s: float,
    end_elapsed_s: float,
    concurrency: int,
) -> dict[str, float]:
    if not rows:
        raise ProfilingError("No telemetry samples overlap a measured request batch.")
    times = [float(row["elapsed_s"]) for row in rows]
    powers = [float(row["power_w"]) for row in rows]
    if len(rows) == 1:
        energy = powers[0] * max(end_elapsed_s - start_elapsed_s, 0.0)
    else:
        energy = 0.0
        for left, right, p_left, p_right in zip(
            times[:-1], times[1:], powers[:-1], powers[1:]
        ):
            energy += 0.5 * (p_left + p_right) * max(0.0, right - left)
        uncovered = max(0.0, (end_elapsed_s - start_elapsed_s) - (times[-1] - times[0]))
        energy += sum(powers) / len(powers) * uncovered
    return {
        "realized_clock_mhz": sum(float(row["graphics_clock_mhz"]) for row in rows)
        / len(rows),
        "energy_j": energy / max(concurrency, 1),
        "mean_power_w": sum(powers) / len(powers),
        "peak_power_w": max(powers),
        "peak_temp_c": max(float(row["temperature_c"]) for row in rows),
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def aggregate_profile(
    rows: Sequence[Mapping[str, Any]],
    telemetry_rows: Sequence[Mapping[str, Any]],
    requested_clocks_mhz: Sequence[int],
) -> list[dict[str, float]]:
    """Reduce request-level measurements to the plant table used by the book.

    Each concurrent batch is first reduced to one throughput and one power
    observation.  Medians are then taken across the prespecified prompt sizes,
    concurrency levels, and five repeats.  This prevents an eight-request batch
    from receiving eight times the weight of a one-request batch.
    """

    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    for row in rows:
        key = (
            str(row["phase"]),
            int(round(float(row["requested_clock_mhz"]))),
            int(row["prompt_tokens"]),
            int(row["output_tokens"]),
            int(row["concurrency"]),
            int(row["repeat"]),
        )
        grouped.setdefault(key, []).append(row)

    batch_rows: list[dict[str, float | str]] = []
    for key, requests in grouped.items():
        phase, clock, _, _, _, _ = key
        if phase == "prefill":
            work = sum(float(row["prompt_tokens_observed"]) for row in requests)
            duration = max(float(row["total_s"]) for row in requests)
        elif phase == "decode":
            work = sum(float(row["completion_tokens"]) for row in requests)
            duration = max(
                float(row["total_s"])
                - float(row["ttft_s"])
                + float(row["tpot_s"])
                for row in requests
            )
        else:
            continue
        if not math.isfinite(work) or not math.isfinite(duration) or work <= 0 or duration <= 0:
            raise ProfilingError(f"Invalid measured {phase} batch at {clock} MHz.")
        batch_rows.append(
            {
                "phase": phase,
                "clock_mhz": float(clock),
                "tokens_per_s": work / duration,
                "power_w": sum(float(row["mean_power_w"]) for row in requests)
                / len(requests),
            }
        )

    aggregate: list[dict[str, float]] = []
    for clock in sorted({int(value) for value in requested_clocks_mhz}):
        idle_power = [
            float(row["power_w"])
            for row in telemetry_rows
            if str(row["phase"]) == f"idle_f{clock}"
        ]
        prefill = [
            row
            for row in batch_rows
            if row["phase"] == "prefill" and int(row["clock_mhz"]) == clock
        ]
        decode = [
            row
            for row in batch_rows
            if row["phase"] == "decode" and int(row["clock_mhz"]) == clock
        ]
        if not idle_power or not prefill or not decode:
            raise ProfilingError(
                f"Cannot aggregate {clock} MHz: idle, prefill, or decode measurements are missing."
            )
        aggregate.append(
            {
                "clock_mhz": float(clock),
                "prefill_tokens_per_s": _median(
                    [float(row["tokens_per_s"]) for row in prefill]
                ),
                "decode_tokens_per_s": _median(
                    [float(row["tokens_per_s"]) for row in decode]
                ),
                "idle_power_w": _median(idle_power),
                "prefill_power_w": _median(
                    [float(row["power_w"]) for row in prefill]
                ),
                "decode_power_w": _median(
                    [float(row["power_w"]) for row in decode]
                ),
            }
        )
    return aggregate


def _median(values: Sequence[float]) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ProfilingError("Cannot take the median of an empty measurement set.")
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return 0.5 * (ordered[middle - 1] + ordered[middle])


def batch_balanced_realized_clock_medians(
    rows: Sequence[Mapping[str, Any]],
    requested_clocks_mhz: Sequence[int],
) -> dict[int, float]:
    """Give every condition/repeat batch one vote in its clock median."""

    requested = [int(value) for value in requested_clocks_mhz]
    batches: dict[tuple[int, str, int, int, int, int], list[float]] = {}
    for row in rows:
        clock = int(round(float(row["requested_clock_mhz"])))
        if clock not in requested:
            raise ProfilingError(f"Unexpected requested clock in raw profile: {clock} MHz.")
        key = (
            clock,
            str(row["phase"]),
            int(row["prompt_tokens"]),
            int(row["output_tokens"]),
            int(row["concurrency"]),
            int(row["repeat"]),
        )
        batches.setdefault(key, []).append(float(row["realized_clock_mhz"]))

    by_clock: dict[int, list[float]] = {clock: [] for clock in requested}
    for key, values in batches.items():
        if not values or any(not math.isfinite(value) or value <= 0.0 for value in values):
            raise ProfilingError(f"Invalid realized clock values for batch {key}.")
        if max(values) - min(values) > 1e-9:
            raise ProfilingError(
                f"Concurrent requests disagree on the batch realized clock for {key}."
            )
        by_clock[key[0]].append(_median(values))
    if any(not values for values in by_clock.values()):
        raise ProfilingError("At least one requested clock has no realized-clock batches.")
    return {clock: _median(values) for clock, values in by_clock.items()}


def select_usable_clock_levels(
    rows: Sequence[Mapping[str, Any]],
    aggregate_rows: Sequence[Mapping[str, Any]],
    requested_clocks_mhz: Sequence[int],
    *,
    minimum_levels: int = 4,
) -> tuple[list[int], dict[str, Any]]:
    """Select a prespecified monotone measured subset without altering values.

    We retain the maximum-cardinality subsequence that is strictly increasing
    in batch-balanced realized clock and in both measured service rates.  Ties
    use the lexicographically earliest requested-clock indices.  This rule is
    fixed before profile consumption and prevents a power-capped nominal clock
    inversion from being smoothed or relabeled as a different measurement.
    """

    requested = [int(value) for value in requested_clocks_mhz]
    if len(requested) < minimum_levels or any(
        right <= left for left, right in zip(requested, requested[1:])
    ):
        raise ProfilingError("Requested clock levels must be strictly increasing.")
    aggregate_by_clock = {
        int(round(float(row["clock_mhz"]))): row for row in aggregate_rows
    }
    if set(aggregate_by_clock) != set(requested):
        raise ProfilingError("The full aggregate profile does not cover every requested clock.")
    realized = batch_balanced_realized_clock_medians(rows, requested)

    def strictly_usable(indices: Sequence[int]) -> bool:
        for left_index, right_index in zip(indices, indices[1:]):
            left_clock = requested[left_index]
            right_clock = requested[right_index]
            left = aggregate_by_clock[left_clock]
            right = aggregate_by_clock[right_clock]
            if not (
                realized[right_clock] > realized[left_clock]
                and float(right["prefill_tokens_per_s"])
                > float(left["prefill_tokens_per_s"])
                and float(right["decode_tokens_per_s"])
                > float(left["decode_tokens_per_s"])
            ):
                return False
        return True

    chosen_indices: tuple[int, ...] | None = None
    for size in range(len(requested), minimum_levels - 1, -1):
        for candidate in combinations(range(len(requested)), size):
            if strictly_usable(candidate):
                chosen_indices = candidate
                break
        if chosen_indices is not None:
            break
    if chosen_indices is None:
        raise ProfilingError(
            "Fewer than four requested clocks form a strictly increasing measured "
            "realized-clock and service-rate profile."
        )

    modeled = [requested[index] for index in chosen_indices]
    excluded = [clock for clock in requested if clock not in modeled]
    return modeled, {
        "rule": (
            "maximum-cardinality subsequence strictly increasing in batch-balanced "
            "median realized clock, prefill rate, and decode rate; ties use the "
            "lexicographically earliest requested-clock indices"
        ),
        "minimum_usable_levels": minimum_levels,
        "realized_clock_median_method": (
            "one realized-clock observation per phase/prompt/output/concurrency/repeat batch"
        ),
        "realized_clock_median_mhz_by_requested": {
            str(clock): realized[clock] for clock in requested
        },
        "excluded_requested_clocks_mhz": excluded,
    }


def baseline_latency(
    rows: Sequence[Mapping[str, Any]], maximum_clock_mhz: int
) -> tuple[float, float]:
    """Return the prespecified single-request, 1,024-prompt decode baseline."""

    selected = [
        row
        for row in rows
        if str(row["phase"]) == "decode"
        and int(round(float(row["requested_clock_mhz"]))) == maximum_clock_mhz
        and int(row["prompt_tokens"]) == 1024
        and int(row["concurrency"]) == 1
    ]
    if len(selected) != 5:
        raise ProfilingError(
            "The latency baseline requires exactly five max-clock, concurrency-1, "
            f"1,024-prompt decode rows; found {len(selected)}."
        )
    ttft = _median([float(row["ttft_s"]) for row in selected])
    tpot = _median([float(row["tpot_s"]) for row in selected])
    if ttft <= 0.0 or tpot <= 0.0:
        raise ProfilingError("The measured TTFT/TPOT baseline must be positive.")
    return ttft, tpot


def _solve_three_by_three(matrix: list[list[float]], vector: list[float]) -> list[float]:
    augmented = [row[:] + [value] for row, value in zip(matrix, vector)]
    for column in range(3):
        pivot = max(range(column, 3), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) < 1e-12:
            raise ProfilingError("Thermal regression is rank deficient.")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        scale = augmented[column][column]
        augmented[column] = [value / scale for value in augmented[column]]
        for row in range(3):
            if row == column:
                continue
            factor = augmented[row][column]
            augmented[row] = [
                value - factor * pivot_value
                for value, pivot_value in zip(augmented[row], augmented[column])
            ]
    return [augmented[index][3] for index in range(3)]


def fit_thermal_rc(telemetry_rows: Sequence[Mapping[str, Any]]) -> dict[str, float | str]:
    """Fit ``dT/dt = c_T T + c_P P + c_0`` from recorded load/cool cycles."""

    thermal = sorted(
        (
            row
            for row in telemetry_rows
            if str(row.get("phase", "")).startswith("thermal_")
        ),
        key=lambda row: float(row["elapsed_s"]),
    )
    observations: list[tuple[list[float], float]] = []
    for index, left in enumerate(thermal):
        left_phase = str(left["phase"])
        left_time = float(left["elapsed_s"])
        right_index: int | None = None
        for candidate in range(index + 1, len(thermal)):
            right = thermal[candidate]
            if str(right["phase"]) != left_phase:
                break
            elapsed = float(right["elapsed_s"]) - left_time
            if elapsed >= 0.8:
                right_index = candidate
                break
        if right_index is None:
            continue
        right = thermal[right_index]
        elapsed = float(right["elapsed_s"]) - left_time
        if elapsed > 1.6:
            continue
        window = thermal[index : right_index + 1]
        mean_power = sum(float(row["power_w"]) for row in window) / len(window)
        temperature = 0.5 * (
            float(left["temperature_c"]) + float(right["temperature_c"])
        )
        derivative = (
            float(right["temperature_c"]) - float(left["temperature_c"])
        ) / elapsed
        observations.append(([temperature, mean_power, 1.0], derivative))
    if len(observations) < 12:
        raise ProfilingError(
            f"Thermal RC fitting needs at least 12 one-second load/cool observations; "
            f"found {len(observations)}."
        )
    normal = [[0.0] * 3 for _ in range(3)]
    right_hand_side = [0.0] * 3
    for features, target in observations:
        for row in range(3):
            right_hand_side[row] += features[row] * target
            for column in range(3):
                normal[row][column] += features[row] * features[column]
    coefficient_temperature, coefficient_power, intercept = _solve_three_by_three(
        normal, right_hand_side
    )
    if coefficient_temperature >= -1e-9 or coefficient_power <= 0.0:
        raise ProfilingError(
            "Measured thermal regression does not have stable cooling and positive power gain."
        )
    time_constant = -1.0 / coefficient_temperature
    resistance = coefficient_power * time_constant
    ambient = intercept * time_constant
    if not (1.0 <= time_constant <= 2_000.0 and 0.001 <= resistance <= 20.0):
        raise ProfilingError(
            f"Measured thermal fit is outside physical bounds: tau={time_constant:.3g} s, "
            f"R={resistance:.3g} °C/W."
        )
    predictions = [
        coefficient_temperature * features[0]
        + coefficient_power * features[1]
        + intercept
        for features, _ in observations
    ]
    targets = [target for _, target in observations]
    mean_target = sum(targets) / len(targets)
    residual_sum = sum((target - prediction) ** 2 for target, prediction in zip(targets, predictions))
    total_sum = sum((target - mean_target) ** 2 for target in targets)
    return {
        "method": "one-second finite-difference least squares on two load/cool cycles",
        "thermal_time_constant_s": time_constant,
        "thermal_resistance_c_per_w": resistance,
        "fitted_ambient_temperature_c": ambient,
        "thermal_fit_r_squared": 1.0 - residual_sum / total_sum if total_sum > 0 else 0.0,
        "thermal_fit_observations": float(len(observations)),
    }


def completed_profile_metadata(
    rows: Sequence[Mapping[str, Any]],
    telemetry_rows: Sequence[Mapping[str, Any]],
    *,
    maximum_clock_mhz: int,
    metadata: GpuMetadata,
    experiment_power_limit_w: float,
    measured_on: str,
) -> dict[str, Any]:
    """Build the fields consumed by ``inference_serving.load_profile``."""

    baseline_ttft_s, baseline_tpot_s = baseline_latency(rows, maximum_clock_mhz)
    thermal = fit_thermal_rc(telemetry_rows)
    return {
        "profile_status": "measured_l4",
        "source_label": (
            "Measured NVIDIA L4 profile for Qwen/Qwen2.5-7B-Instruct "
            "served by pinned vLLM"
        ),
        "baseline_ttft_s": baseline_ttft_s,
        "baseline_tpot_s": baseline_tpot_s,
        "default_power_limit_w": metadata.default_power_limit_w,
        "experiment_power_limit_w": experiment_power_limit_w,
        "measured_on": measured_on,
        **thermal,
    }


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_revision() -> str | None:
    try:
        return run_command(["git", "rev-parse", "HEAD"], timeout=10).stdout.strip()
    except Exception:
        return None


def _run_thermal_cycles(
    sampler: TelemetrySampler,
    *,
    thermal_limit_c: float,
    cycles: int,
    load_seconds: float,
    cool_seconds: float,
    server_url: str,
    served_model: str,
    profile_start: float,
    timeout_s: float,
) -> None:
    condition = BenchmarkCondition("thermal_load", 4096, 128, 8)
    for cycle in range(cycles):
        sampler.set_phase(f"thermal_load_{cycle + 1}")
        deadline = time.monotonic() + load_seconds
        while time.monotonic() < deadline:
            run_concurrent_batch(
                condition,
                server_url=server_url,
                served_model=served_model,
                profile_start=profile_start,
                timeout_s=timeout_s,
            )
            sampler.ensure_healthy()
            if sampler.rows and float(sampler.rows[-1]["temperature_c"]) > thermal_limit_c:
                raise ProfilingError(
                    f"Thermal cycle reached {float(sampler.rows[-1]['temperature_c']):.1f} °C, "
                    f"above the protocol limit of {thermal_limit_c:.1f} °C."
                )
        sampler.set_phase(f"thermal_cool_{cycle + 1}")
        end = time.monotonic() + cool_seconds
        while time.monotonic() < end:
            time.sleep(min(0.5, end - time.monotonic()))
            sampler.ensure_healthy()
            if sampler.rows and float(sampler.rows[-1]["temperature_c"]) > thermal_limit_c:
                raise ProfilingError(
                    f"Thermal cycle reached {float(sampler.rows[-1]['temperature_c']):.1f} °C, "
                    f"above the protocol limit of {thermal_limit_c:.1f} °C."
                )


def wait_for_thermal_cooldown(
    sampler: TelemetrySampler,
    *,
    requested_clock_mhz: int,
    thermal_limit_c: float,
    cooldown_clock_mhz: int,
    condition: BenchmarkCondition | None = None,
    condition_index: int | None = None,
    events: list[dict[str, Any]] | None = None,
    target_margin_c: float = SWEEP_COOLDOWN_MARGIN_C,
    stable_samples_required: int = SWEEP_COOLDOWN_STABLE_SAMPLES,
    timeout_s: float = SWEEP_COOLDOWN_TIMEOUT_S,
    poll_s: float = SWEEP_COOLDOWN_POLL_S,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Wait for a reproducible, safely cool starting state before one sweep.

    Only fresh telemetry carrying this cooldown's phase label contributes to
    the stability requirement.  A temperature above the experiment limit is
    always a hard failure, including the latest sample present when cooldown
    begins.  ``events`` is mutated before waiting so both successful and
    failed attempts remain available to the outer manifest finalizer.
    """

    if not math.isfinite(thermal_limit_c) or thermal_limit_c <= 0.0:
        raise ProfilingError("Sweep cooldown thermal limit must be positive and finite.")
    if not math.isfinite(target_margin_c) or target_margin_c <= 0.0:
        raise ProfilingError("Sweep cooldown target margin must be positive and finite.")
    target_temperature_c = thermal_limit_c - target_margin_c
    if target_temperature_c <= 0.0:
        raise ProfilingError(
            "Sweep cooldown requires a positive finite target temperature below the limit."
        )
    if stable_samples_required < 1:
        raise ProfilingError("Sweep cooldown requires at least one stable sample.")
    if not math.isfinite(timeout_s) or timeout_s <= 0.0:
        raise ProfilingError("Sweep cooldown timeout must be positive and finite.")
    if not math.isfinite(poll_s) or poll_s <= 0.0:
        raise ProfilingError("Sweep cooldown polling interval must be positive and finite.")

    if (condition is None) != (condition_index is None):
        raise ProfilingError(
            "Sweep cooldown condition and condition index must be supplied together."
        )
    if condition_index is not None and condition_index < 0:
        raise ProfilingError("Sweep cooldown condition index cannot be negative.")
    if condition is None:
        phase = f"cooldown_before_f{requested_clock_mhz}"
        scope = "clock"
    else:
        phase = (
            f"cooldown_before_condition_{condition_index:02d}_{condition.phase}_"
            f"p{condition.prompt_tokens}_o{condition.output_tokens}_"
            f"c{condition.concurrency}_f{requested_clock_mhz}"
        )
        scope = "condition"
    start = monotonic()
    start_row_index = len(sampler.rows)
    initial_temperature_c = (
        float(sampler.rows[-1]["temperature_c"]) if sampler.rows else None
    )
    event: dict[str, Any] = {
        "requested_clock_mhz": requested_clock_mhz,
        "cooldown_clock_mhz": cooldown_clock_mhz,
        "phase": phase,
        "scope": scope,
        "status": "waiting",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "target_temperature_c": target_temperature_c,
        "thermal_limit_c": thermal_limit_c,
        "stable_samples_required": stable_samples_required,
        "timeout_s": timeout_s,
        "initial_temperature_c": initial_temperature_c,
        "observed_samples": 0,
        "stable_samples_observed": 0,
    }
    if condition is not None:
        event["condition_index"] = condition_index
        event["condition"] = asdict(condition)
    if events is not None:
        events.append(event)
    sampler.set_phase(phase)

    observed_temperatures = (
        [initial_temperature_c] if initial_temperature_c is not None else []
    )
    consecutive_stable = 0
    next_row_index = start_row_index

    def finish(status: str, *, final_temperature_c: float | None = None) -> None:
        event["status"] = status
        event["completed_utc"] = datetime.now(timezone.utc).isoformat()
        event["duration_s"] = monotonic() - start
        event["stable_samples_observed"] = consecutive_stable
        event["final_temperature_c"] = final_temperature_c
        if observed_temperatures:
            event["minimum_temperature_c"] = min(observed_temperatures)
            event["maximum_temperature_c"] = max(observed_temperatures)

    if initial_temperature_c is not None and not math.isfinite(initial_temperature_c):
        finish("failed_invalid_telemetry")
        raise ProfilingError(
            f"Sweep cooldown for {requested_clock_mhz} MHz began with a "
            "non-finite temperature."
        )
    if initial_temperature_c is not None and initial_temperature_c > thermal_limit_c:
        finish("failed_over_limit", final_temperature_c=initial_temperature_c)
        raise ProfilingError(
            f"Sweep cooldown for {requested_clock_mhz} MHz began at "
            f"{initial_temperature_c:.1f} °C, above the protocol limit of "
            f"{thermal_limit_c:.1f} °C."
        )

    while True:
        sampler.ensure_healthy()
        current_rows = list(sampler.rows)
        new_rows = current_rows[next_row_index:]
        next_row_index = len(current_rows)
        for row in new_rows:
            if str(row.get("phase", "")) != phase:
                continue
            temperature_c = float(row["temperature_c"])
            if not math.isfinite(temperature_c):
                finish("failed_invalid_telemetry")
                raise ProfilingError(
                    f"Sweep cooldown for {requested_clock_mhz} MHz received a "
                    "non-finite temperature."
                )
            observed_temperatures.append(temperature_c)
            event["observed_samples"] = int(event["observed_samples"]) + 1
            if temperature_c > thermal_limit_c:
                finish("failed_over_limit", final_temperature_c=temperature_c)
                raise ProfilingError(
                    f"Sweep cooldown for {requested_clock_mhz} MHz reached "
                    f"{temperature_c:.1f} °C, above the protocol limit of "
                    f"{thermal_limit_c:.1f} °C."
                )
            if temperature_c <= target_temperature_c:
                consecutive_stable += 1
            else:
                consecutive_stable = 0
            event["stable_samples_observed"] = consecutive_stable
            if consecutive_stable >= stable_samples_required:
                finish("complete", final_temperature_c=temperature_c)
                return event

        elapsed_s = monotonic() - start
        if elapsed_s >= timeout_s:
            final_temperature_c = (
                observed_temperatures[-1] if observed_temperatures else None
            )
            finish("failed_timeout", final_temperature_c=final_temperature_c)
            last_temperature = (
                "no fresh telemetry"
                if final_temperature_c is None
                else f"{final_temperature_c:.1f} °C"
            )
            raise ProfilingError(
                f"Sweep cooldown for {requested_clock_mhz} MHz timed out after "
                f"{timeout_s:.1f} s at {last_temperature}; required "
                f"{stable_samples_required} consecutive samples at or below "
                f"{target_temperature_c:.1f} °C."
            )
        sleep(min(poll_s, timeout_s - elapsed_s))


def checkpoint_condition_progress(
    *,
    raw_profile_path: Path,
    telemetry_path: Path,
    profile_path: Path,
    manifest_path: Path,
    rows: Sequence[Mapping[str, Any]],
    telemetry_rows: Sequence[Mapping[str, Any]],
    completed_clocks_mhz: Sequence[int],
    manifest: dict[str, Any],
    requested_clock_mhz: int,
    clock_index: int,
    condition: BenchmarkCondition,
    condition_index: int,
) -> list[dict[str, float]]:
    """Atomically checkpoint one completed benchmark condition.

    The aggregate contains only clocks whose entire condition matrix has
    completed.  Before the first complete clock, the aggregate checkpoint is a
    schema-only CSV rather than an estimate from an incomplete matrix.
    """

    aggregate_rows = (
        aggregate_profile(rows, telemetry_rows, completed_clocks_mhz)
        if completed_clocks_mhz
        else []
    )
    _write_csv(raw_profile_path, rows, RAW_PROFILE_COLUMNS)
    _write_csv(telemetry_path, telemetry_rows, TELEMETRY_COLUMNS)
    _write_csv(profile_path, aggregate_rows, PROFILE_COLUMNS)
    checkpoint = {
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "requested_clock_mhz": requested_clock_mhz,
        "clock_index": clock_index,
        "condition_index": condition_index,
        "condition": asdict(condition),
        "profile_rows": len(rows),
        "telemetry_rows": len(telemetry_rows),
        "aggregate_complete_clocks_mhz": list(completed_clocks_mhz),
    }
    manifest.setdefault("condition_checkpoints", []).append(checkpoint)
    manifest["latest_condition_checkpoint"] = checkpoint
    _write_json(manifest_path, manifest)
    return aggregate_rows


def run_thermal_identification(arguments: argparse.Namespace) -> dict[str, Any]:
    """Acquire one prespecified thermal protocol without a clock sweep."""

    output_directory = arguments.output_directory.resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    phase_followup = arguments.mode == "thermal-phase-identification"
    if phase_followup:
        sequences = thermal_phase_identification_sequences()
        telemetry_path = output_directory / "l4_thermal_phase_telemetry.csv"
        requests_path = output_directory / "l4_thermal_phase_requests.csv"
        manifest_path = output_directory / "thermal_phase_manifest.json"
        vllm_log_path = output_directory / "thermal_phase_vllm.log"
        complete_path = output_directory / "thermal_phase.complete"
        failed_path = output_directory / "thermal_phase.failed"
        marker_prefix = "thermal-phase-block"
        protocol = "cold-start-phase-pairs-v1"
        required_maximum_power_w = 61.0
        training_power_limits_w = [46, 61]
        training_duration_ceilings_s = {46: 75.0, 61: 45.0}
        validation_power_limits_w = [55]
        protocol_rationale = {
            "discovery_bundle_is_not_confirmatory_data": True,
            "discovery_observation": (
                "At matched 55 W and 60 s, prefill and decode measured 54.979 W "
                "and 55.186 W time-weighted mean power, but the sustained "
                "temperature gap averaged 2.61 C and the raw peak rises were "
                "16 C and 14 C."
            ),
            "design_response": (
                "Counterbalance matched prefill/decode cold-start pulses at "
                "46 W and 61 W for fitting, then evaluate one untouched 55 W pair."
            ),
        }
        fit_protocol = {
            "acquisition_only": True,
            "training_sequences": ["phase_training"],
            "untouched_validation_sequence": "phase_validation",
            "validation_evaluation_passes": 1,
            "fit_input": "measured power rather than requested cap",
            "prespecified_phase_model": "P_eff = P * (1 + beta * I_prefill)",
            "primary_validation_metrics": [
                "free_run_rmse_c",
                "prefill_minus_decode_peak_temperature_rise_c",
            ],
            "temperature_fit": (
                "continuous-time trajectory fit; include pre-pulse cooldown "
                "telemetry when initializing hidden thermal state"
            ),
        }
    else:
        sequences = thermal_identification_sequences()
        telemetry_path = output_directory / "l4_thermal_telemetry.csv"
        requests_path = output_directory / "l4_thermal_requests.csv"
        manifest_path = output_directory / "thermal_manifest.json"
        vllm_log_path = output_directory / "thermal_vllm.log"
        complete_path = output_directory / "thermal.complete"
        failed_path = output_directory / "thermal.failed"
        marker_prefix = "thermal-block"
        protocol = "cold-start-pulses-v2"
        required_maximum_power_w = 64.0
        training_power_limits_w = [40, 46, 52, 58, 64]
        training_duration_ceilings_s = _THERMAL_TRAINING_DURATION_CEILINGS_S
        validation_power_limits_w = [43, 49, 55, 61]
        protocol_rationale = {
            "preserved_prior_trial_is_not_part_of_this_dataset": True,
            "prior_trial_observation": (
                "The cumulative v1 trial safely stopped at 77 C, 376.8 s into "
                "its first nominal 40 W block. It began at 49 C, read back a "
                "40 W cap, measured 47.20 W mean power and a 210 MHz realized "
                "clock, and rose to 77 C. Step diagnostics suggested a "
                "134-210 s time constant."
            ),
            "design_response": (
                "Use a separately cooled cold-start pulse before every bounded "
                "excitation; remove cumulative sequences and long plateaus."
            ),
        }
        fit_protocol = {
            "acquisition_only": True,
            "training_sequences": ["training_a", "training_b"],
            "untouched_validation_sequence": "validation",
            "fit_input": "measured power rather than requested cap",
            "temperature_fit": (
                "continuous-time trajectory fit; do not finite-difference the "
                "integer-valued temperature samples"
            ),
        }
    if manifest_path.exists() and not arguments.overwrite:
        raise ProfilingError(
            f"{manifest_path} already exists. Preserve it or pass --overwrite explicitly."
        )
    if phase_followup and not arguments.overwrite:
        other_manifests = [
            path
            for path in output_directory.glob("*manifest.json")
            if path != manifest_path
        ]
        if other_manifests:
            raise ProfilingError(
                "The phase follow-up requires a separate output directory; found "
                + ", ".join(path.name for path in other_manifests)
            )

    metadata = query_gpu_metadata(arguments.gpu_index)
    supported = query_supported_clocks(arguments.gpu_index)
    memory_clock = max(int(value) for value in supported)
    available_graphics_clocks = {
        int(value) for value in supported[memory_clock]
    }
    if THERMAL_REQUESTED_CLOCK_MHZ not in available_graphics_clocks:
        raise ProfilingError(
            f"The thermal protocol requires {THERMAL_REQUESTED_CLOCK_MHZ} MHz at "
            f"{memory_clock} MHz memory; supported graphics clocks are "
            f"{sorted(available_graphics_clocks)}."
        )
    cooldown_clock = min(available_graphics_clocks)
    if metadata.minimum_power_limit_w > THERMAL_MINIMUM_POWER_W + 1e-9:
        raise ProfilingError(
            f"The L4 minimum power limit is {metadata.minimum_power_limit_w:.1f} W; "
            f"the fixed protocol requires {THERMAL_MINIMUM_POWER_W:.1f} W."
        )
    if metadata.maximum_power_limit_w < required_maximum_power_w - 1e-9:
        raise ProfilingError(
            f"The L4 maximum power limit is {metadata.maximum_power_limit_w:.1f} W; "
            f"the fixed protocol requires {required_maximum_power_w:.1f} W."
        )

    profile_start = time.monotonic()
    request_rows: list[dict[str, Any]] = []
    request_rows_lock = threading.Lock()
    sampler: TelemetrySampler | None = None
    manifest: dict[str, Any] = {
        "schema_version": 2,
        "mode": arguments.mode,
        "protocol": protocol,
        "status": "initializing",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision": arguments.source_git_revision or _git_revision(),
        "source_worktree_state": arguments.source_worktree_state,
        "launcher_script_sha256": arguments.launcher_script_sha256,
        "profiler_script_sha256": _sha256(Path(__file__).resolve()),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "model": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "served_model_name": SERVED_MODEL_NAME,
        "vllm_image": arguments.vllm_image,
        "vllm_prefix_caching_enabled": VLLM_PREFIX_CACHING_ENABLED,
        "cloud": {
            "project": arguments.cloud_project,
            "zone": arguments.cloud_zone,
            "provisioning_model": arguments.provisioning_model,
            "machine_type": arguments.machine_type,
        },
        "gpu": asdict(metadata),
        "selected_memory_clock_mhz": memory_clock,
        "requested_graphics_clock_mhz": THERMAL_REQUESTED_CLOCK_MHZ,
        "cooldown_graphics_clock_mhz": cooldown_clock,
        "telemetry_period_s": arguments.telemetry_period_s,
        "telemetry_columns": THERMAL_TELEMETRY_COLUMNS,
        "request_columns": THERMAL_REQUEST_COLUMNS,
        "schedule": [
            {
                "sequence": sequence_name,
                "split": blocks[0].split,
                "requires_cooldown_before_every_pulse": True,
                "blocks": [asdict(block) for block in blocks],
            }
            for sequence_name, blocks in sequences
        ],
        "schedule_total_s": sum(
            block.duration_s for _, blocks in sequences for block in blocks
        ),
        "training_power_limits_w": training_power_limits_w,
        "training_duration_ceilings_s": training_duration_ceilings_s,
        "validation_power_limits_w": validation_power_limits_w,
        "cooldown_protocol": {
            "before_every_pulse": True,
            "power_limit_w": THERMAL_MINIMUM_POWER_W,
            "graphics_clock_mhz": cooldown_clock,
            "workload": "idle",
            "target_temperature_c": THERMAL_COOLDOWN_TARGET_C,
            "stability_band_c": THERMAL_COOLDOWN_STABILITY_C,
            "stability_window_s": THERMAL_COOLDOWN_STABILITY_S,
            "timeout_s": THERMAL_COOLDOWN_TIMEOUT_S,
        },
        "protocol_rationale": protocol_rationale,
        "safety_protocol": {
            "independent_of_request_loop": True,
            "safe_down_temperature_c": THERMAL_SAFE_DOWN_C,
            "abort_temperature_c": THERMAL_ABORT_C,
            "stale_telemetry_s": THERMAL_STALE_TELEMETRY_S,
            "safe_power_limit_w": THERMAL_MINIMUM_POWER_W,
            "safe_graphics_clock_mhz": cooldown_clock,
        },
        "fit_protocol": fit_protocol,
        "cooldown_events": [],
        "block_checkpoints": [],
        "completed_block_ids": [],
    }
    _write_json(manifest_path, manifest)

    try:
        with managed_gpu_controls(
            metadata,
            memory_clock,
            gpu_index=arguments.gpu_index,
        ) as (controls, applied):
            manifest["gpu_controls"] = applied
            with TelemetrySampler(
                profile_start,
                gpu_index=arguments.gpu_index,
                period_s=arguments.telemetry_period_s,
            ) as active_sampler:
                sampler = active_sampler
                with ThermalSafetyWatchdog(
                    sampler,
                    controls,
                    profile_start=profile_start,
                    safe_clock_mhz=cooldown_clock,
                ) as watchdog:
                    with VllmServer(
                        server_url=arguments.server_url,
                        image=arguments.vllm_image,
                        model=MODEL_ID,
                        revision=MODEL_REVISION,
                        served_name=SERVED_MODEL_NAME,
                        log_path=vllm_log_path,
                        launch=not arguments.external_server,
                        readiness_timeout_s=arguments.readiness_timeout_s,
                    ) as server:
                        manifest["vllm_image_digest"] = server.image_digest
                        manifest["vllm_server_version"] = server.server_version
                        manifest["status"] = "acquiring"
                        _write_json(manifest_path, manifest)
                        for _, blocks in sequences:
                            run_thermal_identification_sequence(
                                blocks,
                                controls=controls,
                                sampler=sampler,
                                watchdog=watchdog,
                                server_url=server.server_url,
                                served_model=SERVED_MODEL_NAME,
                                profile_start=profile_start,
                                timeout_s=arguments.request_timeout_s,
                                output_directory=output_directory,
                                telemetry_path=telemetry_path,
                                requests_path=requests_path,
                                manifest_path=manifest_path,
                                request_rows=request_rows,
                                request_rows_lock=request_rows_lock,
                                manifest=manifest,
                                cooldown_clock_mhz=cooldown_clock,
                                cooldown_events=manifest["cooldown_events"],
                                marker_prefix=marker_prefix,
                            )
                        final_context_prefix = (
                            "thermal_phase_identification"
                            if phase_followup
                            else "thermal_identification"
                        )
                        sampler.set_context(
                            f"{final_context_prefix}_final_safe_down",
                            split="conditioning",
                            sequence="finalization",
                            block_id="final_safe_down",
                            block_role="safe_down",
                            requested_power_limit_w=THERMAL_MINIMUM_POWER_W,
                            requested_clock_mhz=cooldown_clock,
                            workload_phase="idle",
                        )
                        safe_down_errors = controls.emergency_safe_down(
                            power_limit_w=THERMAL_MINIMUM_POWER_W,
                            graphics_clock_mhz=cooldown_clock,
                        )
                        if safe_down_errors:
                            raise ProfilingError(
                                "Final thermal safe-down failed: "
                                + "; ".join(safe_down_errors)
                            )

        if sampler is None:
            raise ProfilingError("Thermal telemetry sampler was not initialized.")
        final_telemetry = sampler.snapshot()
        with request_rows_lock:
            final_requests = [dict(row) for row in request_rows]
        _write_csv(telemetry_path, final_telemetry, THERMAL_TELEMETRY_COLUMNS)
        _write_csv(requests_path, final_requests, THERMAL_REQUEST_COLUMNS)
        manifest["status"] = "complete"
        manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
        manifest["elapsed_s"] = time.monotonic() - profile_start
        manifest["telemetry_row_count"] = len(final_telemetry)
        manifest["request_row_count"] = len(final_requests)
        manifest["sha256"] = {
            telemetry_path.name: _sha256(telemetry_path),
            requests_path.name: _sha256(requests_path),
        }
        if vllm_log_path.exists():
            manifest["sha256"][vllm_log_path.name] = _sha256(vllm_log_path)
        _write_json(manifest_path, manifest)
        complete_path.write_text(
            datetime.now(timezone.utc).isoformat() + "\n", encoding="utf-8"
        )
        return manifest
    except Exception as error:
        if sampler is not None:
            try:
                _write_csv(
                    telemetry_path,
                    sampler.snapshot(),
                    THERMAL_TELEMETRY_COLUMNS,
                )
            except OSError:
                pass
        with request_rows_lock:
            saved_requests = [dict(row) for row in request_rows]
        try:
            _write_csv(requests_path, saved_requests, THERMAL_REQUEST_COLUMNS)
        except OSError:
            pass
        manifest["status"] = "failed"
        manifest["failed_utc"] = datetime.now(timezone.utc).isoformat()
        manifest["elapsed_s"] = time.monotonic() - profile_start
        manifest["error"] = f"{type(error).__name__}: {error}"
        manifest["request_rows_before_failure"] = len(saved_requests)
        _write_json(manifest_path, manifest)
        failed_path.write_text(manifest["error"] + "\n", encoding="utf-8")
        raise


def run_profile(arguments: argparse.Namespace) -> dict[str, Any]:
    output_directory = arguments.output_directory.resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    profile_path = output_directory / "l4_profile.csv"
    full_profile_path = output_directory / "l4_profile_all_requested.csv"
    raw_profile_path = output_directory / "l4_profile_raw.csv"
    telemetry_path = output_directory / "l4_telemetry.csv"
    manifest_path = output_directory / "profile_manifest.json"
    if profile_path.exists() and not arguments.overwrite:
        raise ProfilingError(
            f"{profile_path} already exists. Use --overwrite only after preserving the prior run."
        )

    metadata = query_gpu_metadata(arguments.gpu_index)
    supported = query_supported_clocks(arguments.gpu_index)
    memory_clock, graphics_clocks = select_clock_levels(supported)
    conditions = profile_conditions()
    # Some L4/driver combinations expose current temperature but no slowdown
    # threshold. In that case the only defined member of the protocol's
    # min(80 C, slowdown - 5 C) rule is its explicit 80 C ceiling. Record the
    # missing metadata rather than inventing a hardware threshold.
    thermal_limit, thermal_limit_source = select_thermal_limit(
        metadata.slowdown_temperature_c
    )
    if thermal_limit <= 0:
        raise ProfilingError(f"Invalid derived thermal limit: {thermal_limit} °C.")

    profile_start = time.monotonic()
    rows: list[dict[str, Any]] = []
    telemetry_fallback_batches = 0
    manifest: dict[str, Any] = {
        "schema_version": PROFILE_SCHEMA_VERSION,
        "status": "initializing",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision": arguments.source_git_revision or _git_revision(),
        "source_worktree_state": arguments.source_worktree_state,
        "launcher_script_sha256": arguments.launcher_script_sha256,
        "profiler_script_sha256": _sha256(Path(__file__).resolve()),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "model": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "served_model_name": SERVED_MODEL_NAME,
        "vllm_image": arguments.vllm_image,
        "vllm_prefix_caching_enabled": VLLM_PREFIX_CACHING_ENABLED,
        "vllm_server_arguments": ["--no-enable-prefix-caching"],
        "container_gpu_smoke_test": "pinned vLLM image: nvidia-smi -L",
        "cloud": {
            "project": arguments.cloud_project,
            "zone": arguments.cloud_zone,
            "provisioning_model": arguments.provisioning_model,
            "machine_type": arguments.machine_type,
        },
        "gpu": asdict(metadata),
        "supported_clocks_mhz": {str(key): value for key, value in supported.items()},
        "selected_memory_clock_mhz": memory_clock,
        "selected_graphics_clocks_mhz": graphics_clocks,
        "thermal_limit_c": thermal_limit,
        "thermal_limit_source": thermal_limit_source,
        "telemetry_period_s": arguments.telemetry_period_s,
        "thermal_cycles": 2,
        "thermal_load_seconds": arguments.thermal_load_seconds,
        "thermal_cool_seconds": arguments.thermal_cool_seconds,
        "sweep_cooldown_protocol": {
            "applies_before_every_clock": True,
            "applies_before_every_condition": True,
            "cooldown_clock_mhz": min(graphics_clocks),
            "target_margin_below_thermal_limit_c": SWEEP_COOLDOWN_MARGIN_C,
            "target_temperature_c": thermal_limit - SWEEP_COOLDOWN_MARGIN_C,
            "stable_samples_required": SWEEP_COOLDOWN_STABLE_SAMPLES,
            "timeout_s": SWEEP_COOLDOWN_TIMEOUT_S,
            "poll_s": SWEEP_COOLDOWN_POLL_S,
            "stability_rule": (
                "consecutive fresh telemetry samples at or below the target"
            ),
            "condition_event_metadata": (
                "clock, condition index, phase, prompt tokens, output tokens, "
                "and concurrency"
            ),
        },
        "sweep_cooldown_events": [],
        "condition_checkpoints": [],
        "idle_sample_seconds": arguments.idle_sample_seconds,
        "warmup_batches_per_condition": 1,
        "measured_repeats_per_condition": arguments.repeats,
        "conditions": [asdict(condition) for condition in conditions],
        "profile_columns": PROFILE_COLUMNS,
        "raw_profile_columns": RAW_PROFILE_COLUMNS,
        "aggregation": (
            "First compute one throughput and mean-power observation per "
            "concurrent batch, then take the median across the fixed prompt, "
            "concurrency, and repeat matrix at each requested clock. Prefill "
            "throughput counts observed prompt tokens over total request time; "
            "decode throughput counts completion tokens over post-first-token time. "
            "The modeled CSV keeps the prespecified maximum-cardinality strictly "
            "increasing measured subset; the all-requested CSV and raw rows retain "
            "every profiled clock."
        ),
        "units": {
            "clock": "MHz",
            "time": "s",
            "energy": "J per concurrent request",
            "power": "W",
            "temperature": "degrees Celsius",
        },
    }
    _write_json(manifest_path, manifest)

    try:
        with managed_gpu_controls(
            metadata,
            memory_clock,
            gpu_index=arguments.gpu_index,
        ) as (controls, applied):
            manifest["gpu_controls"] = applied
            with VllmServer(
                server_url=arguments.server_url,
                image=arguments.vllm_image,
                model=MODEL_ID,
                revision=MODEL_REVISION,
                served_name=SERVED_MODEL_NAME,
                log_path=output_directory / "vllm.log",
                launch=not arguments.external_server,
                readiness_timeout_s=arguments.readiness_timeout_s,
            ) as server:
                manifest["vllm_image_digest"] = server.image_digest
                manifest["vllm_server_version"] = server.server_version
                manifest["status"] = "profiling"
                _write_json(manifest_path, manifest)
                with TelemetrySampler(
                    profile_start,
                    gpu_index=arguments.gpu_index,
                    period_s=arguments.telemetry_period_s,
                ) as sampler:
                    for clock_index, requested_clock in enumerate(graphics_clocks):
                        cooldown_clock = min(graphics_clocks)
                        controls.lock_graphics(cooldown_clock)
                        wait_for_thermal_cooldown(
                            sampler,
                            requested_clock_mhz=requested_clock,
                            thermal_limit_c=thermal_limit,
                            cooldown_clock_mhz=cooldown_clock,
                            events=manifest["sweep_cooldown_events"],
                        )
                        _write_json(manifest_path, manifest)
                        controls.lock_graphics(requested_clock)
                        sampler.set_phase(f"idle_f{requested_clock}")
                        idle_deadline = time.monotonic() + arguments.idle_sample_seconds
                        while time.monotonic() < idle_deadline:
                            time.sleep(min(0.25, idle_deadline - time.monotonic()))
                            sampler.ensure_healthy()
                        for condition_index, condition in enumerate(conditions):
                            controls.lock_graphics(cooldown_clock)
                            wait_for_thermal_cooldown(
                                sampler,
                                requested_clock_mhz=requested_clock,
                                thermal_limit_c=thermal_limit,
                                cooldown_clock_mhz=cooldown_clock,
                                condition=condition,
                                condition_index=condition_index,
                                events=manifest["sweep_cooldown_events"],
                            )
                            _write_json(manifest_path, manifest)
                            controls.lock_graphics(requested_clock)
                            phase_name = (
                                f"warmup_{condition.phase}_{condition.prompt_tokens}_"
                                f"{condition.output_tokens}_{condition.concurrency}_f{requested_clock}"
                            )
                            sampler.set_phase(phase_name)
                            run_concurrent_batch(
                                condition,
                                server_url=server.server_url,
                                served_model=SERVED_MODEL_NAME,
                                profile_start=profile_start,
                                timeout_s=arguments.request_timeout_s,
                            )
                            sampler.ensure_healthy()
                            for repeat in range(arguments.repeats):
                                phase_name = (
                                    f"measure_{condition.phase}_{condition.prompt_tokens}_"
                                    f"{condition.output_tokens}_{condition.concurrency}_"
                                    f"r{repeat}_f{requested_clock}"
                                )
                                sampler.set_phase(phase_name)
                                timings = run_concurrent_batch(
                                    condition,
                                    server_url=server.server_url,
                                    served_model=SERVED_MODEL_NAME,
                                    profile_start=profile_start,
                                    timeout_s=arguments.request_timeout_s,
                                )
                                sampler.ensure_healthy()
                                batch_start = min(item.start_elapsed_s for item in timings)
                                batch_end = max(item.end_elapsed_s for item in timings)
                                telemetry = sampler.between(batch_start, batch_end)
                                telemetry_fallback_used = not telemetry
                                # A sample just outside either edge represents the held
                                # power over the uncovered short interval.
                                if not telemetry and sampler.rows:
                                    telemetry_fallback_batches += 1
                                    telemetry = [
                                        min(
                                            sampler.rows,
                                            key=lambda row: abs(
                                                float(row["elapsed_s"])
                                                - 0.5 * (batch_start + batch_end)
                                            ),
                                        )
                                    ]
                                summary = _telemetry_summary(
                                    telemetry,
                                    start_elapsed_s=batch_start,
                                    end_elapsed_s=batch_end,
                                    concurrency=condition.concurrency,
                                )
                                if summary["peak_temp_c"] > thermal_limit:
                                    raise ProfilingError(
                                        f"Measured {summary['peak_temp_c']:.1f} °C, above the "
                                        f"protocol limit of {thermal_limit:.1f} °C."
                                    )
                                for timing in timings:
                                    rows.append(
                                        {
                                            "phase": condition.phase,
                                            "prompt_tokens": condition.prompt_tokens,
                                            "output_tokens": condition.output_tokens,
                                            "concurrency": condition.concurrency,
                                            "requested_clock_mhz": requested_clock,
                                            "realized_clock_mhz": summary["realized_clock_mhz"],
                                            "repeat": repeat,
                                            "ttft_s": timing.ttft_s,
                                            "tpot_s": timing.tpot_s,
                                            "total_s": timing.total_s,
                                            "energy_j": summary["energy_j"],
                                            "mean_power_w": summary["mean_power_w"],
                                            "peak_power_w": summary["peak_power_w"],
                                            "peak_temp_c": summary["peak_temp_c"],
                                            "telemetry_sample_count": len(telemetry),
                                            "telemetry_fallback_used": telemetry_fallback_used,
                                            "request_index": timing.request_index,
                                            "completion_tokens": timing.completion_tokens,
                                            "prompt_tokens_observed": timing.prompt_tokens_observed,
                                        }
                                    )
                            completed_clocks = graphics_clocks[:clock_index]
                            if condition_index == len(conditions) - 1:
                                completed_clocks = graphics_clocks[: clock_index + 1]
                            checkpoint_condition_progress(
                                raw_profile_path=raw_profile_path,
                                telemetry_path=telemetry_path,
                                profile_path=profile_path,
                                manifest_path=manifest_path,
                                rows=rows,
                                telemetry_rows=sampler.rows,
                                completed_clocks_mhz=completed_clocks,
                                manifest=manifest,
                                requested_clock_mhz=requested_clock,
                                clock_index=clock_index,
                                condition=condition,
                                condition_index=condition_index,
                            )
                        if clock_index == len(graphics_clocks) - 1:
                            _run_thermal_cycles(
                                sampler,
                                thermal_limit_c=thermal_limit,
                                cycles=2,
                                load_seconds=arguments.thermal_load_seconds,
                                cool_seconds=arguments.thermal_cool_seconds,
                                server_url=server.server_url,
                                served_model=SERVED_MODEL_NAME,
                                profile_start=profile_start,
                                timeout_s=arguments.request_timeout_s,
                            )
                        _write_csv(raw_profile_path, rows, RAW_PROFILE_COLUMNS)
                        _write_csv(telemetry_path, sampler.rows, TELEMETRY_COLUMNS)
                        aggregate_rows = aggregate_profile(
                            rows,
                            sampler.rows,
                            graphics_clocks[: clock_index + 1],
                        )
                        _write_csv(profile_path, aggregate_rows, PROFILE_COLUMNS)
                        marker = output_directory / f"sweep-{requested_clock}-mhz.done"
                        _write_json(
                            marker,
                            {
                                "requested_clock_mhz": requested_clock,
                                "completed_utc": datetime.now(timezone.utc).isoformat(),
                                "profile_rows": len(rows),
                                "telemetry_rows": len(sampler.rows),
                            },
                        )


            # The server context must close before the append-only vLLM log is
            # checksummed.  Shutdown writes a few final lines after the last
            # request; hashing inside the context creates a stale manifest.
            if not rows:
                raise ProfilingError("The profile matrix completed without a measurement row.")
            full_aggregate_rows = aggregate_profile(
                rows,
                sampler.rows,
                graphics_clocks,
            )
            modeled_clocks, clock_selection = select_usable_clock_levels(
                rows,
                full_aggregate_rows,
                graphics_clocks,
            )
            modeled_clock_set = set(modeled_clocks)
            modeled_aggregate_rows = [
                row
                for row in full_aggregate_rows
                if int(round(float(row["clock_mhz"]))) in modeled_clock_set
            ]
            _write_csv(full_profile_path, full_aggregate_rows, PROFILE_COLUMNS)
            _write_csv(profile_path, modeled_aggregate_rows, PROFILE_COLUMNS)
            manifest["modeled_graphics_clocks_mhz"] = modeled_clocks
            manifest["clock_profile_selection"] = clock_selection
            manifest["baseline_requested_clock_mhz"] = max(modeled_clocks)
            completed_utc = datetime.now(timezone.utc).isoformat()
            manifest.update(
                completed_profile_metadata(
                    rows,
                    sampler.rows,
                    maximum_clock_mhz=max(modeled_clocks),
                    metadata=metadata,
                    experiment_power_limit_w=float(applied["power_limit_w"]),
                    measured_on=completed_utc,
                )
            )
            manifest["status"] = "complete"
            manifest["completed_utc"] = completed_utc
            manifest["elapsed_s"] = time.monotonic() - profile_start
            manifest["row_count"] = len(rows)
            manifest["telemetry_row_count"] = sum(
                1 for _ in telemetry_path.open(encoding="utf-8")
            ) - 1
            manifest["telemetry_fallback_batch_count"] = telemetry_fallback_batches
            manifest["sha256"] = {
                profile_path.name: _sha256(profile_path),
                full_profile_path.name: _sha256(full_profile_path),
                raw_profile_path.name: _sha256(raw_profile_path),
                telemetry_path.name: _sha256(telemetry_path),
            }
            vllm_log = output_directory / "vllm.log"
            if vllm_log.exists():
                manifest["sha256"][vllm_log.name] = _sha256(vllm_log)
            _write_json(manifest_path, manifest)
            (output_directory / "profile.complete").write_text(
                datetime.now(timezone.utc).isoformat() + "\n", encoding="utf-8"
            )
            return manifest
    except Exception as error:
        manifest["status"] = "failed"
        manifest["failed_utc"] = datetime.now(timezone.utc).isoformat()
        manifest["error"] = f"{type(error).__name__}: {error}"
        manifest["profile_rows_before_failure"] = len(rows)
        _write_json(manifest_path, manifest)
        (output_directory / "profile.failed").write_text(
            manifest["error"] + "\n", encoding="utf-8"
        )
        raise


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=(
            "full-profile",
            "thermal-identification",
            "thermal-phase-identification",
        ),
        default="full-profile",
        help=(
            "run the original clock/profile matrix or the isolated thermal "
            "train/validation or workload-phase acquisition"
        ),
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("data/inference_serving"),
        help="directory for l4_profile.csv, raw telemetry, logs, and the manifest",
    )
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--server-url", default="http://127.0.0.1:8000")
    parser.add_argument("--vllm-image", default=VLLM_IMAGE)
    parser.add_argument(
        "--source-git-revision",
        default=os.environ.get("RLBOOK_GIT_REVISION"),
        help="source repository commit copied into this otherwise isolated VM",
    )
    parser.add_argument(
        "--source-worktree-state",
        choices=("clean", "dirty", "unknown"),
        default="unknown",
        help="whether the source checkout had changes outside the recorded git revision",
    )
    parser.add_argument("--launcher-script-sha256", default=None)
    parser.add_argument("--cloud-project", default=None)
    parser.add_argument("--cloud-zone", default=None)
    parser.add_argument(
        "--provisioning-model",
        choices=("SPOT", "STANDARD", "unknown"),
        default="unknown",
    )
    parser.add_argument("--machine-type", default=None)
    parser.add_argument(
        "--external-server",
        action="store_true",
        help="measure an already running compatible vLLM server instead of launching Docker",
    )
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--telemetry-period-s", type=float, default=TELEMETRY_PERIOD_S)
    parser.add_argument("--readiness-timeout-s", type=float, default=900.0)
    parser.add_argument("--request-timeout-s", type=float, default=600.0)
    parser.add_argument("--thermal-load-seconds", type=float, default=60.0)
    parser.add_argument("--thermal-cool-seconds", type=float, default=60.0)
    parser.add_argument("--idle-sample-seconds", type=float, default=5.0)
    parser.add_argument("--overwrite", action="store_true")
    arguments = parser.parse_args(argv)
    if arguments.mode == "full-profile" and arguments.repeats != 5:
        parser.error("the prespecified protocol requires exactly five measured repeats")
    if not math.isclose(arguments.telemetry_period_s, TELEMETRY_PERIOD_S):
        parser.error("the prespecified protocol requires 0.1 s telemetry")
    if arguments.mode == "full-profile" and (
        arguments.thermal_load_seconds <= 0
        or arguments.thermal_cool_seconds <= 0
        or arguments.idle_sample_seconds <= 0
    ):
        parser.error("thermal load, cool, and idle durations must be positive")
    return arguments


def main(argv: Sequence[str] | None = None) -> int:
    arguments: argparse.Namespace | None = None
    try:
        arguments = parse_arguments(argv)
        manifest = (
            run_thermal_identification(arguments)
            if arguments.mode in {
                "thermal-identification",
                "thermal-phase-identification",
            }
            else run_profile(arguments)
        )
    except (ProfilingError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        if arguments is not None:
            try:
                arguments.output_directory.mkdir(parents=True, exist_ok=True)
                marker_name = {
                    "thermal-identification": "thermal.failed",
                    "thermal-phase-identification": "thermal_phase.failed",
                }.get(arguments.mode, "profile.failed")
                (arguments.output_directory / marker_name).write_text(
                    f"{type(error).__name__}: {error}\n", encoding="utf-8"
                )
            except OSError:
                pass
        print(f"profiling failed: {error}", file=sys.stderr)
        return 2
    count = int(manifest.get("row_count", manifest.get("request_row_count", 0)))
    print(
        f"recorded {count} requests in "
        f"{manifest['elapsed_s'] / 60.0:.1f} minutes",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
