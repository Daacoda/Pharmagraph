ui <- dashboardPage(

  dashboardHeader(
    title = "PrimeKG Temporal Analysis"
  ),

  dashboardSidebar(

    sidebarMenu(

      menuItem(
        "Overview",
        tabName = "overview",
        icon = icon("dashboard")
      ),

      menuItem(
        "Temporal Evolution",
        tabName = "temporal",
        icon = icon("clock")
      ),

      menuItem(
        "Drug–Disease",
        tabName = "drug_disease",
        icon = icon("pills")
      ),

      menuItem(
        "Network Analysis",
        tabName = "network",
        icon = icon("project-diagram")
      ),

      menuItem(
        "Benchmarking",
        tabName = "benchmark",
        icon = icon("chart-line")
      )

    )
  ),

  dashboardBody(

    tabItems(

      # -------------------------
      # OVERVIEW
      # -------------------------

      tabItem(
        tabName = "overview",

        h2("PrimeKG Temporal Analysis"),

        fluidRow(

          valueBoxOutput(
            "nodes_t0",
            width = 3
          ),

          valueBoxOutput(
            "nodes_t1",
            width = 3
          ),

          valueBoxOutput(
            "edges_t0",
            width = 3
          ),

          valueBoxOutput(
            "edges_t1",
            width = 3
          )

        ),

        fluidRow(

          box(
            title = "Temporal Changes",
            width = 12,

            plotlyOutput(
              "temporal_summary",
              height = "400px"
            )
          )

        )
      ),


      # -------------------------
      # TEMPORAL
      # -------------------------

      tabItem(
        tabName = "temporal",

        h2("Temporal Evolution"),

        fluidRow(

          box(
            title = "Edge Evolution",
            width = 12,

            plotlyOutput(
              "edge_evolution",
              height = "450px"
            )
          )

        ),

        fluidRow(

          box(
            title = "Relation Changes",
            width = 12,

            DTOutput(
              "relation_changes"
            )
          )

        )
      ),


      # -------------------------
      # DRUG-DISEASE
      # -------------------------

      tabItem(
        tabName = "drug_disease",

        h2("Drug–Disease Relationships"),

        fluidRow(

          box(
            title = "Disease",
            width = 4,

            selectInput(
              "disease",
              "Select Disease",
              choices = NULL
            )
          )

        ),

        fluidRow(

          box(
            title = "Relationship Evolution",
            width = 12,

            DTOutput(
              "drug_disease_table"
            )
          )

        )
      ),


      # -------------------------
      # NETWORK
      # -------------------------

      tabItem(
        tabName = "network",

        h2("Network Analysis"),

        fluidRow(

          box(
            title = "Network Statistics",
            width = 12,

            visNetworkOutput(
              "network_plot",
              height = "500px"
            )
          )

        )
      ),


      # -------------------------
      # BENCHMARK
      # -------------------------

      tabItem(
        tabName = "benchmark",

        h2("Model Benchmarking"),

        fluidRow(

          box(
            title = "Model Performance",
            width = 12,

            DTOutput(
              "benchmark_table"
            )
          )

        )
      )

    )
  )
)