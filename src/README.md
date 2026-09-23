src/
├── __init__.py                # Initializes the directory as a Python package
├── config.py                  # Central configuration (paths, hyperparameters, model selection flags)
├── data_loader.py             # Loads t0 and t1 PrimeKG-CL snapshots and builds PyG HeteroData objects
├── preprocessing.py           # Handles data cleaning, deduplication, and feature encoding
├── models/
│   ├── __init__.py            # Exposes model architectures
│   ├── graphsage.py           # Heterogeneous GraphSAGE implementation
│   └── rgcn.py                # Relational Graph Convolutional Network (R-GCN) implementation
├── trainer.py                 # Unified training loops, loss functions, and negative sampling
├── evaluator.py               # Stratified temporal evaluation on t1 (persistent, added, removed edges)
├── evidence_scorer.py         # Integrates external evidence layers (Open Targets, PPI, GO)
├── explainer.py               # Extracts multi-hop biological paths for explanations
└── dashboard_data.py          # Python bridge layer to supply data to the R Shiny dashboard via reticulate

Module Breakdown & Responsibilities
config.py: Centralizes file paths, feature dimensions, and GNN hyperparameters to maintain consistency across scripts.
data_loader.py: Manages the ingestion of PrimeKG-CL data, cleaning, node indexing, and constructs the PyTorch Geometric HeteroData object spanning all node types and relation-direction pairs.
models/graphsage.py: Implements the heterogeneous GraphSAGE architecture (optimized for compute efficiency across multiple relation types).
models/rgcn.py: Implements the R-GCN architecture (handling distinct weight matrices per relation type across the 53 relation-direction pairs).
trainer.py: Handles model optimization, training loops, and margin-based ranking loss incorporating hard negatives from contraindication edges.
evaluator.py: Evaluates model predictions exclusively on the held-out $t_1$ temporal snapshot, calculating metrics such as Hits@K, MRR, and AUPRC stratified by edge change types.
evidence_scorer.py: Applies post-hoc scoring adjustments using external evidence features like Open Targets association scores and protein-protein interactions.
explainer.py: Extracts multi-hop paths (e.g., Drug $\rightarrow$ Target $\rightarrow$ Pathway $\rightarrow$ Disease) to generate transparent explanations for top candidate predictions.
dashboard_data.py: Precomputes and queries lightweight summaries and dataset views to feed the R Shiny application via reticulate.
