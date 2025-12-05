# rlbook

## Notes on executable content

This project is built with the new MyST “book-theme” site generator. Unlike the
old Sphinx-based pipeline, the static HTML **does not** automatically embed the
last outputs that were present in your `.ipynb` / MyST source. A few rules keep
plots and widgets visible both on `localhost:3000` and on GitHub Pages:

1. Every code cell that produces a figure should follow this pattern:

   ````
   ```{code-cell} python
   :tags: [hide-input]

   #  label: fig-my-figure-id
   #  caption: Short description of the figure.

   %config InlineBackend.figure_format = 'retina'
   import matplotlib.pyplot as plt
   ...
   plt.tight_layout()
   ```
   ````

   **Notes:**
   - Use regular Python comments (`#  label:`) for metadata, not MyST directives (`#|`)
   - Do **not** add a `:::{figure}` embed after the cell—the figure displays
     directly from the cell output
   - Do **not** call `plt.show()`; end with `plt.tight_layout()` instead
   - The `%config InlineBackend.figure_format = 'retina'` ensures high-DPI output

2. Always build with notebook execution enabled. The helper script already
   enforces this:

   ```bash
   source publish.sh  # runs BASE_URL=/rlbook uv run jupyter-book build --html --execute
   ```

   Skipping `--execute` will re-use whatever cache happens to exist (or nothing,
   if you changed the cell), which can lead to empty output. 
   
3. When iterating locally, `uv run jupyter-book start --execute --port 3000`
   reproduces exactly what GitHub Pages will host, including cached PNG outputs.

Following those three steps prevents regressions where the site appears correct
only when a live kernel is attached.
