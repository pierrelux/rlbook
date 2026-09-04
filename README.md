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

Build the six browser notebooks into the same site:

```bash
uv run jupyter lite build --lite-dir lab --contents notebooks --output-dir _build/html/lab
```

For local authoring, use `uv run jupyter-book start --execute --port 3000`. The browser lab can be served separately with `uv run jupyter lite serve --lite-dir lab --contents notebooks`.

The spotlight presenter must share an origin with the rendered chapter. Preview
that workflow from a root-based static build rather than MyST's split-port
development server:

```bash
BASE_URL='' uv run jupyter-book build --html
python3 -m http.server 8000 --bind 127.0.0.1 --directory _build/html
```

Then open `http://127.0.0.1:8000/modeling-controlled-systems/` and use **Present**.

`publish.sh` performs both strict builds and publishes the assembled `_build/html` directory to `gh-pages` with `ghp-import`.

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

To regenerate the checked-in notebook JSON after editing their source definitions, run:

```bash
uv run python lab/generate_notebooks.py
```

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
