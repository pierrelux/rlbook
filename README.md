# rlbook

## Notes on executable content

This project is built with the new MyST “book-theme” site generator. Unlike the
old Sphinx-based pipeline, the static HTML **does not** automatically embed the
last outputs that were present in your `.ipynb` / MyST source. A few rules keep
plots and widgets visible both on `localhost:3000` and on GitHub Pages:

1. Every code cell that should expose a figure needs a label inside the cell:

   ````
   ```{code-cell} python
   :tags: [hide-input]

   #| label: my-figure-id
   %config InlineBackend.figure_format = 'retina'
   import matplotlib.pyplot as plt
   ...
   plt.show()
   ```
   ````

   Immediately after the cell, embed the output with `{figure}` (or `![](#id)`
   for quick prototypes):

   ````
   :::{figure} #my-figure-id
   Short caption explaining what the plot shows.
   :::
   ````

   Without the label/embed pair, the static build only shows the console text.

2. Always build with notebook execution enabled. The helper script already
   enforces this:

   ```bash
   source publish.sh  # runs BASE_URL=/rlbook uv run jupyter-book build --html --execute
   ```

   Skipping `--execute` will re-use whatever cache happens to exist (or nothing,
   if you changed the cell), which is why we previously saw the HTTP retrier
   section printing only text.

3. When iterating locally, `uv run jupyter-book start --execute --port 3000`
   reproduces exactly what GitHub Pages will host, including cached PNG outputs.

Following those three steps prevents regressions where the site appears correct
only when a live kernel is attached.
