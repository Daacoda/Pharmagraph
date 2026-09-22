source("dependencies.R")
source("UI.R")
source("Server.R")

shinyApp(
  ui = ui,
  server = server
)