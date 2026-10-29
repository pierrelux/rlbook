# Live L4 inference-control experiment

The revised development pilot, `pilot-v3-20260905T0038Z`, launched at 00:38 UTC
on 5 September 2026 on one Standard L4 in `us-central1-b`. Its runner and
controller are frozen for this acquisition. No confirmation run or validated
controller comparison is available yet. All earlier measurements, failed
acquisitions, calibration reports, and simulated comparisons remain preserved.

All three earlier pilots are closed with VM and boot-disk cleanup verified.
Their outcomes remain development evidence:

| Acquisition | Outcome |
|---|---|
| `pilot-20260904T2250Z` | Stopped before inference trials because of a Python 3.10 timestamp parsing incompatibility |
| `pilot-20260904T2303Z` | 33 trials, 32 valid; the steady trial stopped at 77 C after 95.728 seconds, with 54 of 128 requests complete |
| `pilot-v2-20260905T0005Z` | Four trials, three valid; bursty completed all 64 requests with a 73 C peak, while steady stopped at 77 C after 148.693 seconds, with 39 of 64 complete; cleanup verified at 00:33:49 UTC |

The short capacity tests established throughput under batching without
establishing sustainable thermal operation. The original failed steady window's
instantaneous and averaged power integrals differed by only 0.196%.

The experiment asks which method uses the least GPU board energy while meeting
the same measured latency targets. Candidates are maximum clock, a constant
clock selected on development requests, an observable-state governor, and MPC.
A simple method can win. Thermal-model evaluation remains separate from the
independent hardware watchdog.

| Current v3 setting | Value |
|---|---|
| Hardware | One NVIDIA L4 on Standard GCP `g2-standard-8`; approved zones `us-central1-b`, then `us-central1-c` |
| Model | `Qwen/Qwen2.5-7B-Instruct` |
| Model revision | `acbd96531cda22292a3ceaa67e984955d3965282` |
| vLLM image | `vllm/vllm-openai@sha256:61fc8a896b0a4fbbbdc063bc4b0dbc25ce98e02b5050c24aeb7830ac02039b14` |
| Server | Context 8,192; dtype auto; memory utilization 0.9; prefix cache disabled; scheduler fixed |
| Power / memory | 64.8 W cap; 6,251 MHz requested memory clock |
| Graphics actions | Requested 210, 660, 1,125, 1,575, 2,040 MHz; all levels and realized readings retained |
| Development requests | 32 committed Azure trace rows, zero-based interval `[0, 32)` |
| Confirmation requests | 32 rows, `[128, 160)`; other confirmation rows remain reserved |
| Arrival schedules | Steady and trace-shaped bursty; identical requests and arrival window |
| Arrival window | At least 240 seconds; otherwise the measured 32-request capacity duration, scaled by the larger total prompt/output work ratio and divided by 0.35 |
| Screening order | Smoke and capacity; maximum-clock steady, then bursty; only after both pass, the 30 representative batch checks |
| Latency targets | Each target is 1.1 times the worse development p95 across the two maximum-clock schedules |
| Controller | One-second decisions; MPC horizon 10, beam width 32, 0.8-second deadline, governor fallback |
| Telemetry | GPU queries every 100 ms; diagnostic server metrics every 250 ms |
| Watchdog | Safe-down at 77 C or stale telemetry; abort at 79 C |

The nominal fraction 0.35 scales a measured batch-throughput estimate. It does
not specify measured GPU utilization or establish thermal headroom. Batching
changes service efficiency, and idle time after a burst does not reduce the
heating already caused by its dense arrival cluster. Both maximum-clock
schedules must complete without watchdog events before the batch sweep begins.
Their success qualifies those development workloads on that acquisition. The
smaller cohorts and rescaled arrival patterns narrow the eventual conclusion
to these 32-request workloads; success would not establish performance or
thermal feasibility for the earlier 64- or 128-request trials.

The client reproduces trace prompt and output lengths using deterministic
synthetic token-ID prompts and forced output lengths. It records SSE receipt
times and token IDs; final server metrics supply each request's mean inter-token
latency (ITL). Stream chunks do not identify individual token-generation times.
Missing usage or latency fields invalidate the smoke test. The latency targets
use request-level p95 time to first token (TTFT) and p95 mean ITL.

Controllers observe submitted requests, received tokens, elapsed waiting time,
recent arrivals and completions, and available GPU readings. Future arrivals,
unfinished requests' final output lengths, and true server prefill/decode
assignments remain unavailable. An empirical conditional estimate of remaining
output length uses the development subset. After screening and batch checks,
the pilot evaluates the remaining fixed clocks on both schedules, selects the
lowest-energy feasible constant clock, and fits the mixed-serving model.
Whole-run validation tests causal one-, five-, and ten-second forecasts. MPC
falls back to the governor when its model or solver checks fail.

Valid energy windows extend from the first scheduled arrival through final
completion, including drain. Failed or incomplete requests, watchdog events,
missing telemetry brackets, excessive telemetry gaps, or more than 1% of
dispatches over 100 ms late invalidate a trial. Aborted energy remains a
measurement of truncated work. It cannot support an efficiency claim for the
full workload. Reports retain invalid trials, MPC fallback counts, and whether
the optimizer actually supplied actions.

Confirmation still plans five paired repetitions per schedule, with
counterbalanced method order across three independent acquisitions: repetitions
0–1, 2–3, and 4. Session time caps remain provisional until the actual v3 pilot
runtime and conditioning costs are known. Confirmation verifies hashes of the
frozen workload, calibration, runner, and controller. Comparisons require the
complete paired trials and both frozen latency targets; uncertainty resamples
whole repetition blocks, with 10,000 bootstrap samples and seed 20260904.

The authorized budget remains US$20. The schema-2 ledger in `budget.json`
records conservative exposure estimates, rather than provider billing:

| Allocation or exposure | USD |
|---|---:|
| Original pilot allocation | 5.000000 |
| Three closed pilots, conservatively accounted | 3.367014 |
| Current v3 maximum reservation | 4.101484 |
| Current reservation funded from remaining pilot allocation | 1.632986 |
| Original contingency allocation | 3.000000 |
| Current reservation funded from contingency | 2.468498 |
| Remaining unreserved contingency | 0.531502 |
| Confirmation allocation, untouched | 12.000000 |

The current launch uses `--allow-contingency-overflow`, which consumes the
remaining pilot allocation before the stated contingency amount. The ledger
stores both allocations explicitly and rejects overlapping or unresolved VM
reservations. The current pilot has a four-hour provider lifetime; the local
monitor begins cleanup ten minutes earlier. Cleanup must verify absence of both
VM and boot disk before another reservation. Unverified cleanup retains its
full exposure reservation.

Future launches require a freshly verified official hourly price and timestamp
within 24 hours. The current reservation uses the archived 4 September quote
of US$0.853624312 per hour, plus reserved disk, IP, and transfer costs. A launch
plan can be prepared locally with `scripts/run_inference_live_gcp.py --mode
prepare`; changing to `--mode launch` creates the bounded acquisition. A
confirmation launch additionally requires `--stage confirmation` and
`--protocol PILOT_OUTPUT/protocol.json`. Its `--repetition-start` and
`--repetitions` flags partition the frozen repetition indices. A changed runner
or controller requires a new development pilot.

The development evidence includes:

| Artifact | Scope |
|---|---|
| `sensor_pilot_audit.json` | Frozen early snapshot: observed query cadence, repeated sensor values, and short-window power-channel sensitivity |
| `fresh_batch_audit.json` | All 30 fresh cells and 130 requests; preserved context/concurrency axes and clock saturation observations |
| `mixed_sensor_audit.json` | Failed steady window, completed capacity comparison, and limits of the original workload normalization |
| `batch_calibration.csv`, `.json`, `.md` | Reaggregation of the original profile, retaining phase, context, concurrency, and requested clock |
| [Original thermal report](thermal_history_development.json) | Preserved continuous-history development comparison |
| [Frozen thermal report](thermal_history_development_live_v2.json) | Selected `two_state_rc` model for fresh evaluation; neither advisory nor hardware-safety acceptance has been established |
| [Thermal confirmation freeze](thermal_confirmation_freeze.json) | Model report hash, unchanged evaluation criteria, and permitted development acquisitions |

The thermal model was frozen at 00:32:39 UTC on 5 September, before v3 launched.
Despite the report's `live_v2` filename, its fit used only the five historical
acquisitions and the completed `pilot-20260904T2303Z` acquisition. It used no
v2 failure, active v3, or confirmation measurements. Fresh evaluation retains
the frozen one-, five-, and ten-second criteria, missing coverage, and
underprediction; it permits no post-confirmation refitting or threshold changes.

The audit files record exact source hashes and JSONL prefix byte counts. The
earlier snapshot audits remain unchanged as later acquisition files grow.
Offline checks run from the repository root:

```bash
.venv/bin/python -m pytest -q \
  tests/test_inference_live_runner.py tests/test_inference_live_control.py \
  tests/test_inference_live_launcher.py tests/test_inference_live_artifacts.py \
  tests/test_inference_live_trial.py tests/test_inference_thermal_history.py

.venv/bin/python scripts/run_inference_live.py --stage pilot --prepare-only \
  --output-directory /private/tmp/rlbook-live-preflight
```

After complete confirmation acquisitions, pass all three directories to
`scripts/build_inference_live_artifacts.py --sessions ...`. The builder exports
metrics, paired results, figures, and provenance under the frozen protocol hash.
Thermal evaluation uses the frozen report:

```bash
.venv/bin/python scripts/fit_inference_thermal_history.py evaluate \
  --model artifacts/inference_serving/live_control_development/thermal_history_development_live_v2.json \
  --bundle CONFIRMATION_OUTPUT \
  --output /private/tmp/thermal-history-confirmation.json
```

The evaluator preserves each
power forecast's original time axis and measures lead time from decision
availability. The extra logged terminal-clock interval is explicitly marked as
an assumption; it does not extend the ten-step optimizer. Uncovered horizons
remain unavailable. These tools do not publish the book.
