library(shiny)
library(shinydashboard)
library(reticulate)
library(plotly)
library(DT)

# Python environment
use_python(Sys.which("python"), required = TRUE)

# Add project root to Python path
root_dir <- normalizePath("..")
py_path <- file.path(root_dir)

if (!py_path %in% sys.path()) {
  sys.path$append(py_path)
}