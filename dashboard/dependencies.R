library(shiny)
library(shinydashboard)
library(reticulate)
library(plotly)
library(DT)
library(jsonlite)
library(visNetwork)

# Python environment
use_python(Sys.which("python"), required = TRUE)

# Add project root to Python path
root_dir <- normalizePath("..")
py_path <- file.path(root_dir)

py_sys <- import("sys")
if (!py_path %in% py_sys$path) {
  py_sys$path <- c(py_path, py_sys$path)
}

# ---------------------------------------------------------------
# Python data-access layer (src/dashboard_data.py). This is where
# every query that has to touch the raw 8M/13M-row snapshots lives —
# never inside a reactive in Server.R. See that file for caching.
# ---------------------------------------------------------------
source_python(file.path(root_dir, "src", "dashboard_data.py"))

# ---------------------------------------------------------------
# Precomputed, small summary files — safe to load once at app
# startup and share across all sessions.
# TODO: confirm these field names against your actual JSON files
# and adjust the $ accessors used in Server.R if they differ.
# ---------------------------------------------------------------
app_stats <- fromJSON(file.path(root_dir, "data", "benchmark", "statistics.json"))
app_diff  <- fromJSON(file.path(root_dir, "data", "benchmark", "diffs", "diff_t0_t1.json"))
# Disease dropdown choices — computed once (cached to disk by the
# Python side after the very first run), not per-session.
disease_choices <- get_disease_list()