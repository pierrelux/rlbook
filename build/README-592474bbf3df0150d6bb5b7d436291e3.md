# Live L4 inference-control experiment

The `warm-reference-v1` acquisition qualified both requested settings, 210 and
660 MHz: all four designated steady and bursty trials completed their 32 requests
validly. Smoke and warm-up also completed validly. These are development
qualification results under a 40 W power cap. They
establish neither a common latency target nor an energy winner. The broader
fixed-clock, governor, and MPC comparison has not reached confirmation.

The acquisition is `fixed-warm-reference-20260905T0420Z`. Its final status is
complete; VM and boot-disk deletion was verified at
05:26:33.603683 UTC on 5 September 2026. Independent provider queries at
05:27:25.832331 UTC found no matching instances or disks, as recorded in
`warm_reference_v1_provider_cleanup.json`.
The immutable `warm_reference_v1_plan.json` records the approved protocol,
source hashes, exact launch command, and bounded reservation. No automatic retry,
controller study, or confirmation run follows from this qualification.

| Designated trial: requested setting / pattern | Completed | Energy / request | p95 TTFT | p95 mean ITL | Peak temperature |
|---|---:|---:|---:|---:|---:|
| 210 MHz / bursty | 32/32, valid | 239.0941 J | 33.7479 s | 2.70205 s | 60 C |
| 210 MHz / steady | 32/32, valid | 259.1936 J | 9.9828 s | 0.14001 s | 64 C |
| 660 MHz / steady | 32/32, valid | 263.4087 J | 5.6234 s | 0.09563 s | 65 C |
| 660 MHz / bursty | 32/32, valid | 221.4412 J | 12.8966 s | 1.54057 s | 60 C |

The smoke test and the separate 210 MHz steady warm-up completed validly but
cannot qualify a setting. Only the four designated trials count. Their complete,
invalid, and unstarted outcomes remain in `qualification.json`; a completed pair
remains recorded if a later trial fails. Independent request, latency, and energy
audits agree with the completed trials; the conditioning and clock audits also
pass. `warm_reference_v1_audit.json` records the final findings and their limits.

The labels 210 and 660 MHz name requested graphics-clock settings. In the
660 MHz trials, measured clocks during samples with nonzero GPU utilization
had time-weighted medians of 300 MHz for steady and 360 MHz for bursty; p95
was 405 MHz for both. Samples with zero utilization were predominantly 660 MHz.
Both power-limit and memory-clock readings remained at 40 W and 6,251 MHz.
These measurements describe operation under the cap; they do not establish the
cause of each clock transition or exact hardware residency. The 660 MHz result
must not be presented as sustained execution at 660 MHz.

The startup sequence ran the existing two-request 210 MHz smoke test, then a
32-request, 240-second 210 MHz steady warm-up without another conditioning call.
The subsequent selector commanded 210 MHz and accepted a trailing 300-second
window after 778.38 seconds. Its temperature range was 50–51 C; the two
150-second means differed by 0.2143 C. The final 60-second mean, 50.7813 C,
became the fixed acquisition reference. `warm_reference.json` preserves the
statistics and byte-prefix hashes; `warm_reference_window.json` preserves the
selected raw window.

Selection required all queries to begin after the cooling command completed,
at least 299.5 seconds of coverage, temperature range at most 1 C, maximum at
most 65 C, and half-window mean difference at most 0.25 C, within 900 seconds.
This is a stability criterion, not proof of thermal equilibrium. Each designated
trial then required its own 60-second window within 1 C of that reference,
with at least 59.5 seconds of coverage, range at most 1 C, maximum at most 65 C,
and a 900-second timeout. The order was 210 bursty, 210 steady, 660 steady,
then 660 bursty.

The warm-selection window is stored exactly. Ordinary trial conditioning did
not record each accepting poll timestamp, so its subsequently reconstructed
windows are diagnostic estimates with that limitation stated in the audit.

The server used `Qwen/Qwen2.5-7B-Instruct` at revision
`acbd96531cda22292a3ceaa67e984955d3965282`, with the pinned vLLM image recorded in
the plan and manifest, on one Standard GCP `g2-standard-8` L4. Each full workload
used the same committed Azure request rows `[0, 32)` and a 240-second arrival
window. The held-out rows remain unused. Prompt and output lengths were
reproduced using synthetic token-ID prompts and forced output lengths.
`warm_reference_v1_plan_metadata_correction.json` corrects the plan's unused
confirmation interval to `[128, 160)`. The original plan is preserved; the
bundled source, development workload `[0, 32)`, criteria, and budget did not change.

TTFT uses client stream-receipt timing; mean ITL comes from each request's final
server metrics. Stream chunks do not identify individual token-generation times.
Energy extends from the first scheduled arrival through the final completion,
including drainage. Failed requests, malformed required response fields,
watchdog events, missing telemetry brackets, excessive telemetry gaps, or more
than 1% of dispatches over 100 ms late invalidate a trial. The 77/79 C watchdogs
and stale-telemetry protection remained active. This acquisition performed no
capacity test, batch sweep, model fitting, controller optimization, power
forecasting, or confirmation protocol freeze.

All seven preceding acquisitions are closed with VM and boot-disk cleanup
verified. Their completed, invalid, and partial measurements and original software
bundles remain preserved:

| Acquisition | Outcome |
|---|---|
| `pilot-20260904T2250Z` | Stopped before inference trials because of a Python 3.10 timestamp parsing incompatibility |
| `pilot-20260904T2303Z` | 33 trials, 32 valid; steady stopped at 77 C after 95.728 seconds, with 54/128 requests complete |
| `pilot-v2-20260905T0005Z` | Four trials, three valid; bursty completed 64/64 at a 73 C peak; steady stopped at 77 C after 148.693 seconds, with 39/64 complete |
| `pilot-v3-20260905T0038Z` | Three trials, two valid; steady stopped at 77 C after 131.835 seconds, with 17/32 complete |
| `pilot-v4-40w-20260905T0115Z` | Three trials, two valid; steady stopped at 77 C after 201.412 seconds, with 26/32 complete |
| `fixed-feasibility-20260905T0210Z` | Smoke and 210 steady valid; 600-second conditioning timed out before bursty, leaving three designated trials unstarted |
| `fixed-feasibility-v2-20260905T0300Z` | Smoke and 210 steady valid; 900-second conditioning also timed out before bursty, leaving three designated trials unstarted |

The first qualification missed its allowed conditioning mean by about 0.15 C
near the 600-second deadline. The extended-wait retry's last 60-second diagnostic
window was at 57 C against an upper mean of 50.6995 C. Those failures
motivated the separately approved warm-reference protocol; their original
criteria and outcomes were not changed. Post-cleanup fixes also excluded
pre-command temperature samples and rejected malformed latency values without
silently dropping failed request records. The acquired source bundles remain
immutable. The retry's slow-mirror recovery preserved all 12 package versions,
both attempts, and the original lifetime and budget in
`fixed_feasibility_v2_bootstrap_recovery.json`. Warm-mode bootstrap uses that
mirror from startup and records exact installed-version verification in
`bootstrap_packages.json` with source and pin-file snapshots.

The current authorization is US$20: US$7 development, US$10 confirmation, and
US$3 contingency. The approved migration in
`budget_reallocation_20260905T0411Z.json` moved US$2 from confirmation to
development without changing any of the seven historical session records or
the reconciliation audit chain. The ledger records conservative exposure
estimates, not a provider invoice.

| Closed exposure or remaining authorization | USD |
|---|---:|
| Five earlier pilots after storage/IP reconciliation | 4.606422 |
| First fixed-clock qualification | 1.079176 |
| Extended-wait qualification | 1.276650 |
| Seven preceding acquisitions, total | 6.962248 |
| Warm-reference acquisition, closed exposure | 1.595865 |
| Total closed development exposure | 8.558113 |
| Development allocation remaining | 0.404135 |
| Contingency remaining | 1.037752 |
| Confirmation allocation, unused | 10.000000 |
| Total remaining authorization | 11.441887 |

The earlier storage/IP reconciliation released US$0.796330 while retaining
compute charges and full transfer allowances, original reservations, prior
accounting, and hashed cleanup evidence. Warm-reference reserved US$2.854054
before creation: US$2 development and US$0.854054 contingency, leaving
US$0.183698 of development and contingency unreserved at peak reservation.
Verified cleanup released US$1.258189 from that reservation. Its provider cap
was 9,360 seconds, with local cleanup after 8,760 seconds. The reservation used
the official [Google Cloud Iowa G2 price](https://cloud.google.com/products/compute/pricing/accelerator-optimized?hl=en)
of US$0.853624312 per hour, verified at 04:14 UTC on 5 September 2026, plus
disk, IP, and transfer allowances. Unused compute funds
can be released only after both VM and disk absence are verified; unresolved
cleanup retains the full reservation. No subsequent paid acquisition is
authorized by the remaining balance alone.

The broader action set remains 210, 660, 1,125, 1,575, and 2,040 MHz. The
five-action constrained study and the two-clock governor alternative remain
unapproved proposals. A two-clock comparison would explicitly narrow that
scope. Qualification alone supplies no learned service/power calibration,
common latency target, repeated method comparison, or confirmation result.

The separate `two_state_rc` thermal candidate remains frozen by
`thermal_confirmation_freeze.json`. Its fit used only the five historical
acquisitions and `pilot-20260904T2303Z`; it used no later development or
confirmation measurements. Fresh evaluation retains the frozen one-, five-,
and ten-second criteria, coverage requirements, and underprediction checks.
Neither advisory nor hardware-safety acceptance has been established. These
qualification results do not permit refitting or changing acceptance thresholds.

`warm_reference_v1_plan.json`, the acquisition's manifests and raw journals,
`fixed_feasibility_v1_audit.json`, `fixed_feasibility_v2_audit.json`, `budget.json`,
and the frozen thermal files retain the detailed provenance. Before launch,
502 inference tests and the strict executed HTML book build passed. The focused
runner, launcher, and budget-migration suites passed 87, 109, and 27 tests,
respectively. The final reports are `warm_reference_v1_audit_energy.json`,
`warm_reference_v1_audit_requests.json`, `warm_reference_v1_audit_latency.json`,
`warm_reference_v1_audit_conditioning.json`, `warm_reference_v1_audit_clocks.json`,
and `warm_reference_v1_audit_provenance_budget.json`. The last passed 179 checks
against 67 hashed input files, including bundle and manifest hashes, exact
package verification, cleanup records, and closed accounting, with no warnings
or failures. The source bundles and frozen thermal hashes remain unchanged.
