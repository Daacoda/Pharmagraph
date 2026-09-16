source("dependencies.R")
source("UI.R")
source("server.R")

shinyApp(
  ui = ui,
  server = server
)