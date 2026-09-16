server <- function(input, output, session) {

  output$nodes_t0 <- renderValueBox({

    valueBox(
      value = "Loading...",
      subtitle = "Nodes — June 2021"
    )

  })


  output$nodes_t1 <- renderValueBox({

    valueBox(
      value = "Loading...",
      subtitle = "Nodes — July 2023"
    )

  })


  output$edges_t0 <- renderValueBox({

    valueBox(
      value = "Loading...",
      subtitle = "Edges — June 2021"
    )

  })


  output$edges_t1 <- renderValueBox({

    valueBox(
      value = "Loading...",
      subtitle = "Edges — July 2023"
    )

  })

}