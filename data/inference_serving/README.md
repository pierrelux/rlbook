# Inference-serving data

`azure_code_animation.csv` and `azure_code_evaluation.csv` are unmodified row
subsets of the Azure 2023 LLM inference code trace. The first file contains the
first 20 requests. The second contains requests arriving during the first five
minutes, relative to the first timestamp. Timestamps and token counts are kept
exactly as published.

Source: [Azure Public Dataset, Azure LLM Inference Dataset
2023](https://github.com/Azure/AzurePublicDataset/blob/master/AzureLLMInferenceDataset2023.md)

License: CC BY 4.0. The trace was retrieved on 2026-09-01. The SHA-256 digest of
the complete downloaded code trace was
`54e9a6d2a4bd06ba1e060304b900abbc74cbea53de96506e60fe5bb4f2277fb6`.

The dataset's requested attribution is Pratyush Patel, Esha Choukse, Chaojie
Zhang, Aashaka Shah, Inigo Goiri, Saeed Maleki, and Ricardo Bianchini,
["Splitwise: Efficient generative LLM inference using phase
splitting"](https://www.microsoft.com/en-us/research/publication/splitwise-efficient-generative-llm-inference-using-phase-splitting/),
ISCA 2024.

`l4_profile.csv` is the deterministic aggregate of a completed NVIDIA L4
measurement run. The experiment used Qwen2.5-7B-Instruct at the revision and
vLLM image digest recorded in `profile_manifest.json`. It requested five GPU
clocks and retained all five in `l4_profile_all_requested.csv`; the modeled
profile uses the maximum-cardinality subset for which realized clock, prefill
rate, and decode rate are strictly increasing. The raw request observations,
100 ms GPU telemetry, requested-versus-realized clocks, conditions,
repetitions, checksums, and provenance are retained alongside the aggregate.

The measurements also expose limitations that matter for interpretation. The
upper requested clocks collapsed to a narrow realized-clock plateau under the
power cap, and recorded telemetry occasionally exceeded the requested 64.8 W
limit. Three very short batches required a one-sample telemetry fallback. The
one-state thermal fit has only `R^2 = 0.00201`, so it is a deliberately reduced
control model rather than evidence of a well-identified thermal plant. Consult
the manifest and raw telemetry before making hardware-performance claims.

No textbook build downloads the trace, contacts GCP, or rewrites these files.

## Thermal identification acquisition

The completed profile above remains immutable. A separate maintainer workflow
can collect richer thermal data into a new timestamped directory:

```bash
RLBOOK_RUN_THERMAL_IDENTIFICATION=YES \
  bash scripts/run_inference_thermal_gcp.sh
```

This command creates a guarded Standard `g2-standard-8` VM with one NVIDIA L4
and a four-hour provider deletion limit. Its cumulative cost guard includes a
conservative USD 0.40 estimate for the failed trial described below and remains
at most USD 4.80. The acquisition uses the pinned model and vLLM image from the
main profile, but it bypasses the full clock sweep.

The first thermal-identification trial provided direct evidence that a
cumulative schedule was unsafe. Its cooldown ended at 49 degrees C. During the
first nominal 40 W block, the cap readback remained 40 W, measured mean power
was 47.20 W, and the realized clock remained at the 210 MHz floor. Temperature
rose from 49 to 77 degrees C, where the watchdog safely stopped the trial after
376.8 seconds. One-state step diagnostics suggested a time constant between 134
and 210 seconds. That preserved trial is design evidence only. Its files are not
moved into or used as part of the replacement dataset.

The replacement protocol uses independent cold-start pulses. Before every
pulse, the workload is stopped and the GPU idles at 40 W and 210 MHz until a
60-second temperature window is at most 52 degrees C and varies by at most one
degree C. Failure to meet that condition within ten minutes aborts the run. Two
training repeats each visit 40, 46, 52, 58, and 64 W exactly once, in different
orders and with different bounded durations. The fixed training workload uses a
128-token prompt, 32 generated tokens, and concurrency eight. There are no long
plateaus or cumulative multilevel sequences. The held-out validation pulses use
43, 49, 55, and 61 W, followed by a duration- and cap-matched prefill/decode
pair at 55 W. No validation pulse exceeds 90 seconds.

The acquisition records the fixed split, pulse schedule, and every completed
cold-start event in `thermal_manifest.json`. It checkpoints telemetry, request
timing, and the manifest after every pulse. A watchdog independent of the
request loop forces
40 W and the minimum clock at 77 degrees C, and the run fails at 79 degrees C,
after stale telemetry, or after a failed power-limit readback. The launcher
copies and validates a complete bundle before deleting the VM. Partial files
remain in the reported staging directory after a failed run.

The acquisition script does not fit a model. Subsequent analysis must fit
continuous temperature trajectories against measured power, keep the validation
sequence untouched, and avoid finite-differencing the integer-valued temperature
sensor at the 100 ms telemetry rate.

## Confirmatory workload-phase acquisition

The completed cold-start dataset showed a workload-dependent discrepancy that
the power-only thermal model did not explain. In the duration- and cap-matched
55 W pulses, prefill and decode averaged 54.979 W and 55.186 W, respectively.
The sustained prefill-minus-decode temperature gap averaged 2.61 degrees C,
while raw peak temperature rises were 16 and 14 degrees C. This is an
exploratory observation from one prefill pulse and two decode pulses. It is not
yet a replicated workload-phase effect, and realized clocks differed between
the phases.

The separate confirmatory acquisition can be run with:

```bash
RLBOOK_RUN_THERMAL_PHASE_IDENTIFICATION=YES \
  bash scripts/run_inference_thermal_phase_gcp.sh
```

It uses six independent cold-start pulses. A first confirmatory attempt stopped
before any pulse because its memory clock remained pinned at 6251 MHz during
conditioning; on that host, the resulting 21.69 W idle state plateaued at 61
degrees C and could not meet the old 52 degree C gate. That failed bundle is
preserved separately and is not confirmatory data.

A second attempt completed one decode pulse, then exposed a different protocol
error. The one-second memory relock raised the next candidate start from 57 to
58 degrees C. Both post-relock starts lay inside the intended one-degree band,
but an extra check compared 58 degrees C against the original 56 degree C
pre-relock reference. That attempt also failed closed, and its partial bundle
is preserved separately.

The revised protocol releases the memory-clock lock during every idle cooldown,
requires a 120-second window no warmer than 58 degrees C with at most 1 degree C
variation, and checks the resulting temperature and idle power against the
first accepted conditioning state. It then re-locks memory at 6251 MHz for one
recorded second and verifies both clock and temperature before starting a
measured pulse. Pre-relock cooldown states are compared with one another;
post-relock pulse starts are compared with one another. The 58 degree C ceiling
is a safety constraint that preserves
19 degrees C of margin to the unchanged 77 degree C safe-down threshold; it is
not treated as a universal ambient temperature. The four training pulses are
ordered as 46 W decode for 75 seconds, 61 W prefill for 45 seconds, 46 W
prefill for 75 seconds, and 61 W decode for 45 seconds. The untouched
validation pair is 55 W decode followed by 55 W prefill, each for 60 seconds.
Every workload runs at a requested 2040 MHz clock and concurrency eight. Decode
uses 128 prompt tokens and 32 generated tokens; prefill uses 4096 prompt tokens
and one generated token. The counterbalanced training order separates workload
phase from cap and duration, while the validation pair is evaluated only after
the model and any phase correction have been fixed.

The six pulses provide 360 seconds of excitation. After the server is ready,
the conservative acquisition envelope is 96 minutes if every cooldown nearly
reaches its fifteen-minute timeout. The provider deletes the VM after 115
minutes, and local monitoring stops five minutes earlier to leave time for
cleanup. The hard cumulative exposure guard is USD 0.40 for the first failed
thermal trial, USD 1.60 for the completed cold-pulse run, USD 0.31 for each of
the two failed confirmatory attempts, USD 1.636113264666666666666666667 for at
most 115 new compute minutes, and USD 0.50 of disk and miscellaneous headroom.
The resulting USD 4.756113264666666666666666667 maximum remains below the USD
4.80 ceiling.

This workflow writes only `thermal_phase_manifest.json`,
`l4_thermal_phase_telemetry.csv`, `l4_thermal_phase_requests.csv`, and their
phase-specific logs and checkpoints in a new timestamped directory. Its local
validator requires the exact pulse order and split, six completed cooldowns,
six nonempty safe checkpoints, matching row counts and checksums, and telemetry
strictly below the 79 degree C abort threshold before promoting the bundle. The
workflow is acquisition-only. A later analysis may test a prespecified input
correction such as `P_eff = P * (1 + beta * I_prefill)`, but the untouched pair
must decide whether that added structure improves free-run prediction.

The completed confirmatory bundle is
`thermal-phase-identification-20260903T131518Z`. It contains 29,171 telemetry
rows, 640 completed requests, six stable cooldowns, and six complete pulses.
The four training starts and the untouched validation starts span 53 to 54
degrees C after memory relock. The two validation pulses drew 54.724 W for
decode and 54.591 W for prefill; their integrated energies differ by 6.66 J,
or about 0.2 percent.

The phase-gain model reduced validation RMSE from 1.427 to 0.870 degrees C, but
it failed the fixed acceptance rule. Its largest validation trajectory error
was 2.731 degrees C, and its error on the one-second aligned phase contrast was
1.431 degrees C. The fitted model is therefore not used for mixed-serving
thermal constraints or as a hardware safety certificate.

The launcher's first local promotion check treated the first row labeled with
the workload as the pulse-start measurement. On some pulses this asynchronous
sensor row arrived about 0.1 seconds after the verified memory-relock sample
and differed by up to 2 degrees C. The acquisition manifest already recorded
the six post-relock samples explicitly. The validator and fitter now use those
recorded samples for the start-state invariant and retain the first workload
row as ordinary trajectory data. No measured row or validation outcome was
changed.
