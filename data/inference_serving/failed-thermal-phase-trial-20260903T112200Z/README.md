# Failed workload-phase confirmation run

This bundle preserves the partial output from the prespecified six-pulse NVIDIA L4 workload-phase confirmation attempted on 2026-09-03 in `us-central1-b`.

The acquisition stopped before the first pulse because the initial conditioning interval did not satisfy the fixed cold-start criterion within 600 seconds. The GPU temperature fell from 70 C to 61 C and remained at 61 C throughout the final 60 seconds, above the required 52 C threshold. During that final minute the GPU was idle at a realized 210 MHz graphics clock and 6251 MHz memory clock, with mean measured power of approximately 21.69 W. The stability criterion was met, but the temperature criterion was not.

No training or validation pulse was executed, and the request log contains only its header. These data must not be used to fit or validate either thermal model. The profiler failed closed, retained the 77 C safe-down and 79 C abort thresholds, and did not alter the protocol after observing the failure.

The launcher deleted the VM and disk after the failure. A subsequent cloud inventory query confirmed that no instance or disk named `pierreluc-l4-rlbook-thermal-phase` remained. No automatic relaunch was performed.

See `thermal_phase_manifest.json` for the full prescribed schedule, environment metadata, and recorded failure; `l4_thermal_phase_telemetry.csv` contains the initialization and failed conditioning trajectory.
