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

## Semantic presentation chunks

The **Present** action can use an offline map to focus several adjacent rendered
blocks as one teaching beat. Build the book first, then run:

```bash
python3 tools/chunk_presentations.py dynamics.md
```

The script validates full block coverage, writes `_present/<chapter>.json`, and
embeds the maps in `_static/presenter.html`. Rebuild afterward. If a chapter's
block structure changes, regenerate its map; the presenter otherwise falls back
to focusing one rendered block at a time.
