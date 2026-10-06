server <- function(input, output, session) {

  # ===============================================================
  # OVERVIEW — value boxes read straight from statistics.json
  # ===============================================================

  output$nodes_t0 <- renderValueBox({
    valueBox(
      value = format(app_stats$snapshots$t0$num_triples, big.mark = ","),
      subtitle = "Triples — June 2021",
      icon = icon("circle-nodes"),
      color = "blue"
    )
  })

  output$nodes_t1 <- renderValueBox({
    valueBox(
      value = format(app_stats$snapshots$t1$num_triples, big.mark = ","),
      subtitle = "Triples — July 2023",
      icon = icon("circle-nodes"),
      color = "aqua"
    )
  })

  output$edges_t0 <- renderValueBox({
    valueBox(
      value = format(app_stats$features$num_drugs, big.mark = ","),
      subtitle = "Total Drugs",
      icon = icon("pills"),
      color = "blue"
    )
  })

  output$edges_t1 <- renderValueBox({
    valueBox(
      value = format(app_stats$features$num_diseases, big.mark = ","),
      subtitle = "Total Diseases",
      icon = icon("virus"),
      color = "aqua"
    )
  })

  # Overview: added/removed/persistent totals from statistics.json / diff
  output$temporal_summary <- renderPlotly({
    diff_df <- data.frame(
      category = c("Added", "Removed", "Persistent"),
      count = c(
        app_stats$diff$added_triples,
        app_stats$diff$removed_triples,
        app_stats$diff$persistent_triples
      )
    )

    plot_ly(
      diff_df,
      x = ~category,
      y = ~count,
      type = "bar",
      marker = list(color = c("#00A65A", "#DD4B39", "#3C8DBC"))
    ) %>%
      layout(
        title = "Triple Changes: t0 (Jun 2021) → t1 (Jul 2023)",
        yaxis = list(title = "Number of Triples")
      )
  })


  # ===============================================================
  # TEMPORAL — per-relation breakdown from diff_t0_t1.json
  # ===============================================================

  relation_diff <- reactive({
    if (!is.null(app_diff$by_relation)) {
      as.data.frame(app_diff$by_relation)
    } else {
      get_relation_diff()
    }
  })

  output$edge_evolution <- renderPlotly({
    rd <- relation_diff()

    plot_ly(rd, x = ~relation, y = ~added, type = "bar", name = "Added") %>%
      add_trace(y = ~removed, name = "Removed") %>%
      add_trace(y = ~persistent, name = "Persistent") %>%
      layout(
        barmode = "stack",
        xaxis = list(title = "", tickangle = -45),
        yaxis = list(title = "Triples")
      )
  })

  output$relation_changes <- renderDT({
    datatable(relation_diff(), options = list(pageLength = 15))
  })


  # ===============================================================
  # DRUG–DISEASE — dropdown + table driven by the Python helper
  # ===============================================================

  updateSelectInput(session, "disease", choices = disease_choices)

  output$drug_disease_table <- renderDT({
    req(input$disease)
    df <- get_drug_disease_subset(input$disease)  # -> pandas.DataFrame via reticulate
    datatable(df, options = list(pageLength = 10))
  })


  # ===============================================================
  # NETWORK — subgraph around the selected disease (t1 snapshot)
  # ===============================================================

  output$network_plot <- renderVisNetwork({
    req(input$disease)
    g <- get_subgraph(input$disease, hops = 1L, snapshot = "t1")

    nodes <- as.data.frame(g$nodes)
    edges <- as.data.frame(g$edges)

    validate(need(nrow(nodes) > 0, "No neighbors found for this entity."))

    visNetwork(nodes, edges) %>%
      visOptions(highlightNearest = TRUE, nodesIdSelection = TRUE) %>%
      visPhysics(stabilization = FALSE) %>%
      visGroups() %>%
      visLegend()
  })


  # ===============================================================
  # BENCHMARK — populated directly from model run results via Python bridge
  # ===============================================================

  output$benchmark_table <- renderDT({
    bench_df <- get_benchmark_table()
    datatable(bench_df, options = list(pageLength = 10, dom = "t"))
  })

}