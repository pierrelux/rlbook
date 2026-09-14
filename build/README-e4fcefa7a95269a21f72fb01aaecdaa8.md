# Live L4 inference-control experiment

The experiment software and offline analyses are implemented. Fresh hardware
execution is pending Google Cloud reauthentication as of 4 September 2026. No
new controller-performance result is available and no new VM has been launched.
The original measured data, validation reports, and simulated comparisons remain
unchanged.

The primary question is which method uses the least GPU board energy while
meeting the same measured latency targets. The candidates are maximum clock, a
constant clock selected on development requests, an observable-state governor,
and MPC. A simple method can win this comparison. Thermal-model improvement is a
separate analysis and does not determine hardware safety.

## Frozen experiment design

| Setting | Value |
|---|---|
| Hardware | One NVIDIA L4 on a Standard GCP `g2-standard-8` |
| Approved zones, in order | `us-central1-b`, `us-central1-c` |
| Model | `Qwen/Qwen2.5-7B-Instruct` |
| Model revision | `acbd96531cda22292a3ceaa67e984955d3965282` |
| vLLM image | `vllm/vllm-openai@sha256:61fc8a896b0a4fbbbdc063bc4b0dbc25ce98e02b5050c24aeb7830ac02039b14` |
| Server settings | Context 8,192; dtype auto; memory utilization 0.9; prefix cache disabled; scheduler fixed |
| Power / memory | 64.8 W cap; 6,251 MHz requested memory clock |
| Graphics actions | Requested 210, 660, 1,125, 1,575, 2,040 MHz; retain all levels and actual clock readings |
| Development requests | Rows 0–127 of the committed Azure trace |
| Confirmation requests | Rows 128–255; remaining rows reserved |
| Arrival schedules | Steady and trace-shaped bursty, with the same work and arrival window |
| Capacity | Development measurement at maximum clock sets the window for approximately 70% load |
| Targets | Each target is 1.1 times the worse development p95 across the two maximum-clock schedules |
| Confirmation | Five paired repetitions per schedule, with counterbalanced method order |
| Controller period | One second |
| MPC | Ten steps, beam width 32, 0.8-second solver deadline, governor fallback |
| Telemetry | GPU queries every 100 ms; diagnostic server metrics every 250 ms |
| Independent watchdog | Safe-down at 77 C or stale telemetry; abort at 79 C |

The two latency targets are time to first token (TTFT) and each request's mean
inter-token latency (ITL), summarized by their request-level p95. The client
reproduces the trace's prompt and output lengths using deterministic synthetic
token-ID prompts and forced output lengths; it does not reconstruct request text.
The client
records SSE receipt times and token IDs; streamed chunks are not asserted to be
individual token-generation timestamps. Final server request metrics supply mean
ITL. Missing usage or latency fields fail the smoke test before the main sweep.

The controller sees submitted requests, received tokens, elapsed waiting time,
recent arrivals and completions, and available GPU readings. It does not receive
future arrivals, unfinished requests' final output lengths, or the server's true
prefill/decode assignment. Remaining output work uses an empirical conditional
length estimate fitted on development requests.

The pilot checks representative batches at all five clocks, sweeps all fixed
clocks on both schedules, selects the lowest-energy feasible fixed clock, and
fits a mixed-serving transition model. Whole-run validation checks causal
one-, five-, and ten-second predictions before MPC may use that model. A failed
model check triggers the recorded governor fallback. Confirmation verifies hashes
of the frozen workload, calibration, runner, and controller before dispatch.

Energy integrates measured board power from the first scheduled arrival through
the final request completion, including drain time. Missing telemetry brackets,
large telemetry gaps, request failures, unfinished work, watchdog events, or more
than 1% of requests dispatched over 100 ms late invalidate a run. Invalid trials
remain in the output. No energy-efficiency claim is eligible without all required
paired trials and both frozen latency targets. Bootstrap intervals resample
entire repetition blocks (10,000 samples, seed 20260904), not individual requests.

## Offline checks and artifacts

Run from the repository root with its existing Python environment:

```bash
.venv/bin/python -m pytest -q \
  tests/test_inference_live_runner.py tests/test_inference_live_control.py \
  tests/test_inference_live_launcher.py tests/test_inference_live_artifacts.py \
  tests/test_inference_live_trial.py tests/test_inference_thermal_history.py

.venv/bin/python scripts/run_inference_live.py --stage pilot --prepare-only \
  --output-directory /private/tmp/rlbook-live-preflight

MPLCONFIGDIR=/private/tmp/rlbook-live-mpl \
  .venv/bin/python scripts/build_inference_live_artifacts.py
```

The artifact builder produces `batch_calibration.csv`, `batch_calibration.json`,
`batch_calibration.md`, and `live_status.md` here, and the batch-calibration SVG,
PDF, and PNG under `_static/inference_serving/`. These reanalyse the original
profile while retaining phase, prompt context, concurrency, and requested clock.
They do not replace the old simulator or invent a live comparison.

To rebuild the continuous-history thermal analysis, choose a new output filename
(the fitter refuses to overwrite existing reports):

```bash
.venv/bin/python scripts/fit_inference_thermal_history.py develop \
  --data-root data/inference_serving \
  --output /private/tmp/thermal-history-development.json
```

The committed `thermal_history_development.json` compares one- and two-state RC
models, with and without phase gain, using complete acquisition histories and
whole-run development splits. Earlier validation and failed acquisitions are
explicitly development for this new analysis. Reported future-measured-power
errors are offline diagnostics. The current selection rule retains the simpler
one-state model; no candidate is accepted for hardware constraints.

## Cloud execution and budget

The new authorized budget is US$20: $5 for the pilot, $12 for confirmation, and
$3 contingency. The launcher reserves the maximum exposure before creating a VM,
enforces one VM at a time, retains a local ledger, and does not release ambiguous
reservations. Pilot and confirmation provider lifetimes are four and six hours;
the local monitor begins cleanup ten minutes earlier. Source bundles and periodic
checkpoints are retained, and cleanup verifies absence of the VM and boot disk.

First restore the approved account's interactive authentication:

```bash
gcloud auth login pierreluc@carbonforge.ai --force
```

Then verify the current Standard Iowa `g2-standard-8` price on Google's official
accelerator-optimized pricing page. The launcher requires an official source URL,
a positive hourly USD quote, and its verification timestamp within the last
24 hours. The 4 September 2026 check found $0.853624312/hour, before separately
reserved disk, IP, and transfer costs; this is an archived quotation, not a
permanent default.

The following command only prepares a local launch plan. Substitute the freshly
verified values and use a new output directory:

```bash
.venv/bin/python scripts/run_inference_live_gcp.py \
  --mode prepare --stage pilot \
  --output-directory artifacts/inference_serving/live_control_development/pilot-SESSION \
  --hourly-rate-usd CURRENT_HOURLY_USD \
  --price-source https://cloud.google.com/products/compute/pricing/accelerator-optimized \
  --price-checked-at CURRENT_VERIFICATION_ISO_TIMESTAMP
```

Changing `--mode prepare` to `--mode launch` performs the bounded experiment.
After a completed pilot, launch confirmation with the same quote arguments, a
new output directory, `--stage confirmation`, and
`--protocol PILOT_OUTPUT/protocol.json`. The default confirmation runs all five
repetitions; `--repetition-start` and `--repetitions` can partition them without
changing the frozen protocol. The budget ledger rejects overlapping repetition
reservations. A changed runner or controller requires a new development pilot,
not a revised confirmation run.

After collecting complete sessions, aggregate the actual measurements:

```bash
MPLCONFIGDIR=/private/tmp/rlbook-live-mpl \
  .venv/bin/python scripts/build_inference_live_artifacts.py \
  --sessions PILOT_OUTPUT CONFIRMATION_OUTPUT

.venv/bin/python scripts/fit_inference_thermal_history.py evaluate \
  --model artifacts/inference_serving/live_control_development/thermal_history_development.json \
  --bundle CONFIRMATION_OUTPUT \
  --output /private/tmp/thermal-history-confirmation.json
```

Live aggregation adds `live_metrics.csv` and `live_results.json`. Thermal
evaluation preserves the recorded power-forecast time axis and measures forecast
lead time from decision availability. Horizons without future coverage are
unavailable, not extrapolated. Neither the launcher nor these builders publishes
the book.
