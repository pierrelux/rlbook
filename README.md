# RL & Control

Source for *Building Up RL: From Dynamics and Control to Learning*, built with Jupyter Book 2 and MyST.

## Set up

The Python environment and lockfile are managed by `uv`:

```bash
uv sync
```

The browser lab uses `jupyterlite-xeus`, which needs `micromamba` while assembling its WebAssembly environment. Install it with your package manager before building the lab (for example, `brew install micromamba` on macOS).

## Build and preview

The production-equivalent book build executes every MyST code cell and treats warnings as failures:

```bash
BASE_URL=/rlbook uv run jupyter-book build --html --execute --strict
```

Build the browser notebooks into the same site:

```bash
uv run jupyter lite build --lite-dir lab --contents notebooks --output-dir _build/html/lab
```

For local authoring, use `uv run jupyter-book start --execute --port 3000`. The browser lab can be served separately with `uv run jupyter lite serve --lite-dir lab --contents notebooks`.

Keep `--execute` on builds as well as previews: it includes notebook outputs
even when execution results are cached. A build without it can leave only the
captions of generated figures and animations. Stop the preview before running
a separate build, since both commands write to `_build/site`, then restart it
with the authoring command above.

Use **Present** on any chapter in the local preview or published book. It opens
the spotlight recorder over the current chapter; **Exit** returns to the same
reading position. The presenter launcher keeps the recorder on the chapter's
origin, including when MyST serves assets on a separate development port.

`publish.sh` performs both strict builds and publishes the assembled `_build/html` directory to `gh-pages` with `ghp-import`.

Rebuild the single-file GPU inference demo gallery with:

```bash
uv run python scripts/build_gpu_demo_gallery.py
```

## Authoring conventions

- `pyproject.toml` is the dependency source of truth; `requirements.txt` is only a pip-compatible entry point.
- Build-time code cells must be deterministic and must not write generated data back into tracked source files.
- Short checks use native `{exercise}` and `{solution}` directives. Solutions carry `:class: dropdown` and stable labels such as `ex-dp-check-1`.
- Altair is the default for compact browser-side analytical interactions. Expensive solver results remain precomputed.
- Reactive marimo components are deliberately limited to focused conceptual islands and must include a static fallback.
- `interactive/` contains standalone HTML demonstrations copied verbatim into the site. `lab/` contains the xeus environment and curated JupyterLite notebooks.

For prose-first executed examples, put figure metadata on the MyST code-cell
directive and remove the input from the rendered page:

````markdown
```{code-cell} python
:tags: [remove-input]
:label: fig-example
:caption: A concise caption that states what the computation shows.

figure = make_figure(results)
display(figure)
```
````

Use `remove-cell` for imports and simulation setup that must execute without
leaving a notebook block. Use `hide-input` only when a visible **Source**
disclosure is intentional. After changing an imported Python module, run
`uv run jupyter-book clean --execute -y`; MyST's execution cache does not track
changes inside imported files.

Notebooks 01–06 are generated from their source definitions. To regenerate those notebooks, run:

```bash
uv run python lab/generate_notebooks.py
```

Notebooks 07 and 08 are authored directly in `lab/notebooks/`; the generator does not
overwrite them. Notebook 07 uses SymPy in Colab or local Jupyter. Notebook 08,
`08_collocation_from_nodes.ipynb`, is a short, self-contained NumPy/SciPy/Matplotlib
demonstration of direct collocation for Colab, local Jupyter, and the browser lab,
saved with its outputs. It is also a page of the book, listed in the `myst.yml`
table of contents under the Demos part and served at `/collocation-from-nodes`;
the book build executes it like any other page. The collocation chapter links to
its section headings, for example
`lab/notebooks/08_collocation_from_nodes.ipynb#step-5-solve-and-check`, so a renamed
heading must be renamed in the chapter too; the notebook tests check these anchors.
It is stored in Colab's notebook layout: format 4.0, a
Python 3 kernel, and text cells that each begin with their heading. Its five figure and
animation cells start with `# @title` and carry the `hide-input` tag, which folds their code in
Colab and in the book. Its formulas
avoid a backslash followed by punctuation, such as `\,` and `\\`, so that they
render the same way in Colab and in Jupyter. `tests/test_collocation_demo_notebook.py`
executes its code cells and checks the results. See [the lab guide](labs.md) for a
description. After editing the notebook, execute it from a fresh Python 3 kernel
before saving; in the browser lab, use Python (XPython).

The modeling chapter reads committed trajectories and figures so that an
ordinary book build does not rerun long experiments. Regenerate its domain
artifacts with:

```bash
uv run python scripts/build_swing_modeling_artifacts.py
uv run python scripts/build_bixi_artifacts.py --seeds 512
uv run python scripts/build_gimbal_artifacts.py
uv run --group artifacts python scripts/build_battery_artifacts.py
uv run python scripts/build_cubesat_artifacts.py
```

The battery builder uses the optional, lockfile-pinned PyBaMM dependency. A
normal book build reads its committed trajectories and does not solve the cell
model.

The BIXI builder consumes the small, checksum-pinned derived data committed in
`data/bixi/`; it does not download the original archive. Recreating those
derived inputs from official source files is documented in
`data/bixi/README.md`.

## Semi-truck alley dock with single shooting

The [numerical trajectory optimization chapter](numerical-trajectory-optimization.md)
parks a tractor-semitrailer by single shooting: the 320 speed and steering
values are the only decision variables, and L-BFGS-B minimizes the cost of
one differentiable rollout. Rebuild its two scenarios, static figures,
results tables, and browser replay data with:

```bash
uv run python scripts/build_truck_parking_artifacts.py
```

The replay shows checkpointed optimizer iterates and a fixed plan's future
rig poses. The experiment's [artifact notes](artifacts/truck_parking/README.md)
document the kinematic model, the data format, and the finer-replay arrival
checks, including the steering-amplification diagnostic that compares
reverse-mode derivatives with the product of per-interval hitch Jacobians.

## Boat docking with iLQR and DDP

The [iLQR and DDP chapter](iterative-trajectory-optimization.md) follows
numerical trajectory optimization and derives the methods through successive
approximation and backward elimination. Rebuild its two docking scenarios,
four optimizer runs, static figures, and interactive replay data with:

```bash
uv run python scripts/build_boat_docking_artifacts.py
```

Regenerate the same chapter's thermoacoustic pull-down results and figures with:

```bash
uv run python scripts/build_thermoacoustic_pulldown_artifacts.py
```

The browser replay separates optimizer iteration from simulation time. It
shows a fixed plan's remaining trajectory and future boat poses. Ordinary
book builds read the saved artifacts and do not run the optimizers. The
experiment's [artifact notes](artifacts/boat_docking/README.md) document the
model, data format, and finer-integration arrival and hull-clearance checks.
The second builder records the thermoacoustic refrigerator's pull-down solves
and static figures; its [artifact notes](artifacts/thermoacoustic_pulldown/README.md)
describe the synthetic model and checks.

## MPPI experiments

The [Model Predictive Path Integral Control chapter](model-predictive-path-integral-control.md) follows continuous-time transcription and collocation.

The MPPI chapter reads the precomputed aircraft experiment. The separate
[Path-Integral Stochastic Control chapter](path-integral-stochastic-control.md)
contains the HJB derivation and Brownian-passage experiment. Regenerate the
numerical results, static figures, and standalone browser replays with:

```bash
uv run python scripts/build_brownian_mppi_artifacts.py
uv run python scripts/build_aircraft_mppi_artifacts.py
```

The aircraft experiment uses numerical OpenAP performance models and the
committed ERA5 wind extraction in `data/aircraft/`. Recreating that extraction
from the original local GRIB file requires a temporary ecCodes environment:

```bash
uv run --no-project --with eccodes==2.48.0 --with numpy==2.5.3 python scripts/prepare_aircraft_wind.py
```

Neither ordinary book builds nor browser replay fetch weather data or rerun
the optimizers. The aircraft's synthetic gust process is separate from the
historical mean-wind snapshot. Controller comparisons keep planner random
numbers independent from the realized physical disturbances.

## Recorded spotlight presentations

The **Present** action opens either a frozen presentation for the chapter or a
live recorder when no frozen presentation exists. In recording mode, drag a
rectangle around visible textbook content. The presenter snaps to stable
document elements, focuses them, and records the interaction as one cue.

Use **Review / Freeze** after the lecture to reorder or delete cues and download
`<chapter>-presentation.json`. Install that recording as the chapter's
authoritative presentation with:

```bash
python3 tools/presentation_cues.py import ~/Downloads/modeling-controlled-systems-presentation.json
```

The importer validates the recording, writes `_present/<chapter>.json`, and
embeds all installed decks in `_static/presenter.html`. Rebuild the book after
importing. To refresh the embedded registry without importing a new file, run
`python3 tools/presentation_cues.py bundle`.

Unfinished recordings are autosaved in browser storage and can be resumed or
discarded the next time the same chapter is opened. Frozen cue files remain the
authoritative, version-controlled representation.
