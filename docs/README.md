# Maintaining these docs

This folder is a [Jupyter Book 2](https://next.jupyterbook.org/) site. Jupyter Book 2
is a rename/reconfiguration of [mystmd](https://mystmd.org) and has nothing in common
with Jupyter Book 1 — there is no `_config.yml`, no `_toc.yml`, no `conf.py`, and no
Sphinx. Everything is configured in a single [`myst.yml`](myst.yml), and the builder is
a Node.js toolchain.

## Build locally

```bash
pip install "jupyter-book>=2"     # or: npm install -g jupyter-book@2
cd docs
jupyter-book build --html         # static site -> docs/_build/html
jupyter-book start                # live-reloading preview on localhost:3000
```

The first build downloads Node.js and the web theme, so it takes a minute. Later
builds take a few seconds.

### Previewing over SSH: forward *two* ports

`jupyter-book start` runs two servers — the site on **3000** and a separate content
server for images and data on **3100** — and it writes image sources as absolute URLs
pointing at the second one:

```html
<img src="http://localhost:3100/clt-f6978a0f374e9f8007ae02fe375305a6.png">
```

So forwarding only port 3000 gives you a page where **every image is broken**. Forward
3100 as well (both ports appear in the VS Code *PORTS* panel). The ports are
configurable with `--port` and `--server-port`.

If forwarding two ports is inconvenient, preview the static build instead — it uses
same-origin relative paths and needs only one port:

```bash
jupyter-book build --html
python -m http.server 8000 --directory _build/html
```

This is a development-server quirk only. The published site is a static build, so nothing
about it depends on port 3100.

## Notebooks are NOT executed at build time

This is the one rule that matters. ScisTreeCNA requires an NVIDIA GPU —
`scistreecna/__init__.py` raises at *import* time if CuPy or a CUDA device is
missing — and neither Read the Docs nor GitHub Actions provides one. So the
notebooks under `tutorials/` are committed **with their outputs stored**, and the
build just renders those stored outputs.

`jupyter-book build` does not execute notebooks unless you pass `--execute`, so the
default behaviour is already what we want. Nothing needs to be configured to opt out.

The consequence: **after editing any notebook, you must re-run it on a GPU machine
and commit the result**, or the published page will show stale output.

```bash
conda activate scistreecna
cd docs
jupyter-book build --html --execute      # re-runs every notebook, needs a GPU
```

To re-run a single notebook instead:

```bash
jupyter nbconvert --to notebook --inplace --execute tutorials/usage.ipynb
```

### Always clean the output afterwards

ScisTreeCNA reports progress through `rich`, which drives a live spinner with ANSI
escapes and carriage returns. That looks right in a terminal and becomes unreadable noise
in a static HTML page, so executed notebooks must be run through the cleaner before being
committed:

```bash
python tools/clean_notebook_output.py tutorials/*.ipynb
```

It replays the control sequences the way a terminal would and writes back plain text. It
is idempotent, so running it on an already-clean notebook does nothing.

The full loop after editing a notebook is therefore:

```bash
conda activate scistreecna
jupyter nbconvert --to notebook --inplace --execute tutorials/usage.ipynb
python tools/clean_notebook_output.py tutorials/usage.ipynb
jupyter-book build --html
```

## Publishing

Read the Docs builds the repository's **default branch** (`main`) using
[`../.readthedocs.yaml`](../.readthedocs.yaml), which runs the mystmd build and copies
`docs/_build/html` into `$READTHEDOCS_OUTPUT/html`. Merging to `main` is all that is
needed; there is no separate docs repository and no manual push step.

## The logo

The theme's stock arrangement: `site.options.logo` in [`myst.yml`](myst.yml) puts the logo
in the top navigation bar, rendered at a fixed 36px height.

It points at [`imgs/logo-flat.png`](imgs/logo-flat.png), a horizontal lockup (1294×197, so
about 236px wide at 36px tall). Because that image already contains the "scistreecna"
wordmark, **`logo_text` is deliberately not set** — it would print the name a second time
next to the logo.

If you swap in a different image, keep it wide rather than square: the theme constrains
the height, so a tall logo ends up tiny. `imgs/logo.png` is a square, icon-only crop kept
around for that reason; it is not referenced by the site.

### Why there are two logo files

The artwork is drawn for a dark background. Its amber wordmark is `#efba68`, which gives

| background | contrast | |
| :--- | ---: | :--- |
| dark theme `#121216` | 10.58:1 | fine |
| light theme `#ffffff` | 1.77:1 | fails — WCAG wants at least 3:1 even for large text |

so on the light theme the logo looks washed out. The theme supports one image per colour
scheme, and `myst.yml` uses both:

- `logo: imgs/logo-flat-light.png` — a darkened copy (wordmark ≈ `#8e6628`, 5.11:1 on
  white), used by the light theme
- `logo_dark: imgs/logo-flat.png` — the original, used by the dark theme

`logo-flat-light.png` was produced mechanically, by scaling HSV value to 0.60 and
saturation to 1.25 on every non-transparent pixel. **It is a stopgap** — it shifts the
brand colour rather than being drawn that way. If a proper dark-ink version of the logo
exists, replace `logo-flat-light.png` with it and nothing else needs to change.

## Figures

`imgs/` currently holds generated placeholders. See [`imgs/README.md`](imgs/README.md)
for the list of figures that still need real artwork. Replacing a placeholder is a
drop-in file overwrite — no Markdown changes required.
