# Reproducing the inference-serving artifacts

The book consumes `textbook_results.json` and the four `metrics_*.csv` files as
committed, read-only artifacts. A normal MyST build does not execute the GPU
profiler, download a model, access GCP, train fitted Q iteration, or rewrite any
tracked artifact.

The current `data/inference_serving/l4_profile.csv` is a validated aggregate of
a measured NVIDIA L4 run. Its adjacent manifest and raw files retain the exact
hardware and software provenance, request-level observations, telemetry,
requested-versus-realized clocks, aggregation rule, and checksums. The upper
clock settings form a measured power-capped plateau, and the weak one-state
thermal fit is carried only as a caveated reduced model. Regenerate the full
artifacts from the repository root with:

```bash
uv run python scripts/build_inference_artifacts.py
```

To verify and ingest a fresh copy of the official Azure trace before rebuilding:

```bash
uv run python scripts/build_inference_artifacts.py \
  --azure-source /absolute/path/to/AzureLLMInferenceTrace_code.csv
```

For software smoke testing only, the reduced protocol is:

```bash
uv run python scripts/build_inference_artifacts.py --quick
```

Quick artifacts must not be published. The `metadata.artifact_protocol` field
distinguishes them from the fixed 50,000-transition, 50-sweep, 200-tree FQI
protocol used by the textbook.

The dedicated L4 phase-confirmation experiment is kept separate from the
serving simulator because its candidate thermal model failed the prespecified
validation rule. Its textbook figure, held-out trajectories, generated result,
and provenance manifest are reproduced without fitting or network access by:

```bash
uv run python scripts/build_inference_thermal_validation_artifacts.py
```

The builder reads the immutable acquisition and training-only fit in
`data/inference_serving/thermal-phase-identification-20260903T131518Z`. It
reconstructs the two fixed model predictions on the untouched validation pair
and checks them against the stored metrics. It does not modify the inference
plant or promote the rejected fit into MPC constraints.
