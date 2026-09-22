# PharmaGraph Dashboard

An interactive Shiny dashboard for exploring the PrimeKG-CL benchmark:
temporal graph evolution (t0 → t1), drug–disease relationships, and a
network view around a selected disease. Built on top of the raw benchmark
snapshots via a Python data-access layer, exposed to R through `reticulate`.
Try it 

cd dashboard   
Rscript Run_App.R

## Architecture

The app is split into four files, each with one job:

| File | Responsibility |
|---|---|
| `dependencies.R` | Loads R libraries, wires up `reticulate` to the project's Python environment, sources `src/dashboard_data.py`, and loads the small precomputed summary files (`statistics.json`, `diffs/diff_t0_t1.json`) once at startup. |
| `UI.R` | Layout only — five tabs (Overview, Temporal Evolution, Drug–Disease, Network Analysis, Benchmarking) inside a `shinydashboard` shell. No logic. |
| `Server.R` | All reactive logic. Renders value boxes, plots, tables, and the network graph by reading either the small JSON summaries (loaded once) or calling into the Python layer (for anything that needs the raw snapshots). |
| `Run_App.R` | Entry point — sources the three files above and calls `shinyApp()`. |

The data-heavy work does **not** live in R. `kg_t0.csv` and `kg_t1.csv` are
8.1M and 13M rows respectively — reading them inside a Shiny reactive would
be slow and memory-heavy on every session. Instead:

- **Small, precomputed files** (`statistics.json`, `diffs/diff_t0_t1.json`)
  are loaded once in `dependencies.R` and shared across all sessions.
- **Anything that needs the raw snapshots** (disease list, drug–disease
  filtering, subgraph extraction) is implemented in
  [`src/dashboard_data.py`](../src/dashboard_data.py) using pandas, and
  called from `Server.R` through `reticulate`. The first call to each of
  these functions scans the relevant CSV and caches the result to
  `output/cache/`; every call after that (including app restarts) reads
  the cache instead.

## Tabs

- **Overview** — headline counts (triples per snapshot, total drugs,
  total diseases) plus a bar chart of added / removed / persistent triples
  between t0 and t1.
- **Temporal Evolution** — the same added/removed/persistent breakdown,
  but per relation type (`indication`, `protein_protein`, etc.), as both a
  stacked bar chart and a sortable table.
- **Drug–Disease** — pick a disease from the dropdown; see every
  drug/disease-relevant edge touching it, tagged as `persistent`,
  `added (t1)`, or `removed (t0 only)`.
- **Network Analysis** — a 1-hop `visNetwork` graph around the selected
  disease, built from the t1 snapshot.
- **Benchmarking** — currently a placeholder table. Will be wired up to
  actual model run results (Hits@K, MRR, AUPRC, Precision@K) once
  training/evaluation is complete — see the main project README for
  methodology status.

## Data dependencies

The app expects, relative to the project root (one level above this
folder):
```
data/benchmark/statistics.json
data/benchmark/diffs/diff_t0_t1.json
data/benchmark/snapshots/kg_t0.csv
data/benchmark/snapshots/kg_t1.csv
src/dashboard_data.py
```

## Running it

From inside this folder:
```r
Rscript -e "shiny::runApp('.')"
```
Requires R packages `shiny`, `shinydashboard`, `reticulate`, `plotly`,
`DT`, `jsonlite`, `visNetwork`, and a Python environment (resolved via
`Sys.which("python")`) with `pandas` installed.

**Note:** the first time you open the Drug–Disease or Network tab, expect
a delay while `src/dashboard_data.py` builds its cache from the full
snapshot files. Subsequent loads are fast.

## Known limitations

- Benchmark tab is a placeholder — no model results wired in yet.
- Network tab is limited to 1-hop neighborhoods (`max_neighbors = 75`) to
  keep the graph readable; not configurable from the UI yet.