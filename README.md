# PharmaGraph — Continual Graph Learning for Drug Repurposing on Evolving Biomedical Knowledge Graphs
<p align="center">
  <img src="figures/pharmagraph_team_logo.png" alt="PharmaGraph Team Logo" width="400"/>
</p>

**Status:** 🟢 Completed · 🔵 In Progress · ⚫ Planned
## 1. Introduction

Drug repurposing — finding new therapeutic uses for existing drugs — depends
heavily on biomedical knowledge graphs (KGs) that link drugs, targets,
diseases, and phenotypes. Most graph-learning benchmarks treat these KGs as
static snapshots, but real curated databases are never static: associations
get added as new evidence arrives, and previously accepted links get
retracted as ontologies are corrected. A model trained once on a fixed graph
silently drifts out of date, and standard continual-learning (CL) benchmarks
don't capture this — they simulate "new tasks" by artificially splitting a
single static graph, not by tracking how a real graph actually changed over
time.

**PharmaGraph** addresses this gap directly. It is built on **PrimeKG-CL**, a
benchmark constructed from two genuine temporal snapshots of PrimeKG — June
2021 and July 2023 — with a real, measured diff between them (what was
added, what was removed, what persisted), rather than a synthetic split.

The goal of this project is to build and evaluate a drug-disease link
prediction system that:
- generalizes correctly across **genuine** graph evolution (not just an
  arbitrary train/test split),
- produces predictions that are **evidence-grounded and explainable**
  rather than an opaque score, so results are usable for translational
  research triage, not just leaderboard ranking.

Full dataset documentation (schema, provenance, task splits, feature
coverage) lives in [`data/benchmark/README.md`](data/benchmark/README.md); this file focuses on
*what we are doing with it and why*.

![our project workflow](figures/workflow_overview.png)

```mermaid
graph TD
    A["1. Data Ingestion & Preprocessing"] -->|Python, Pandas, PrimeKG| B["2. Temporal Edge Dynamics"]
    B -->|PyTorch Geometric, NumPy| C["3. Graph Machine Learning"]
    C -->|PyTorch, GNNs, DREAMwalk| D["4. Scoring & Evidence"]
    D -->|Open Targets, PPI, GO| E["5. Interactive Dashboard"]
    E -->|R, R Shiny, JSON| F["6. Rigorous Evaluation"]
    F -->|Stratified Metrics| G["7. Manuscript & Workflow"]
    G -->|Overleaf, LaTeX, Git| H["Deployment & Final Output"]
```
## 2. Data (summary)

PharmaGraph uses the **PrimeKG-CL Benchmark v1.0**, providing a genuinely temporal evaluation framework rather than a synthetic split:

| Snapshot / Layer | Role & Description | Triples | Nodes | Relations |
| :--- | :--- | :---: | :---: | :---: |
| **$t_0$ — June 2021** | Historical training and candidate generation baseline. | 8,100,498 | 129,375 | 30 |
| **$t_1$ — July 2023** | Temporal evaluation target held out exclusively for testing. | 13,001,666 | 134,211 | 25 |
| **External Evidence** | Complementary evidence layer (Open Targets, PPI, GO), separate from PrimeKG-CL. | — | — | — |

**Temporal Evolution & Diff Breakdown:**
Between $t_0$ and $t_1$, **7,208,624** triples persisted unchanged, **5,760,234** were added (including GO annotations, disease-phenotype refinement, and drug-target updates), and **888,848** were removed (due to retracted associations and ontology corrections). This structural evolution forms the exact foundation for the stratified evaluation (persistent / added / removed edges) used in Step 9.

### **Core Data Pipeline Flow**
```mermaid
flowchart LR

    T0["🟢 2021<br/><b>PrimeKG t0</b><br/>June snapshot<br/><br/>8.1M triples<br/>129K nodes"]

    P["🔵 Temporal Learning<br/><br/>Historical knowledge<br/>Drug–Disease prediction<br/>Candidate ranking"]

    T1["🟣 2023<br/><b>PrimeKG-CL t1</b><br/>July reconstruction<br/><br/>13.0M triples<br/>134K nodes"]

    E["🟠 External Evidence<br/><br/>Open Targets<br/>PPI / GO<br/>Pathway & literature context"]

    V["⚫ Temporal Evaluation<br/><br/>Persistent<br/>Added<br/>Removed"]

    T0 --> P
    P --> T1
    P --> E
    T1 --> V
    E --> V

    classDef t0 fill:#00a65a,stroke:#00753d,color:#ffffff,stroke-width:2px
    classDef process fill:#3c8dbc,stroke:#2c6a8f,color:#ffffff,stroke-width:2px
    classDef t1 fill:#7E57C2,stroke:#512DA8,color:#ffffff,stroke-width:2px
    classDef evidence fill:#EF6C00,stroke:#E65100,color:#ffffff,stroke-width:2px
    classDef eval fill:#5f6368,stroke:#3f4245,color:#ffffff,stroke-width:2px

    class T0 t0
    class P process
    class T1 t1
    class E evidence
    class V eval
```

See `data/benchmark/README.md` for the full breakdown: task definitions, multimodal
feature coverage, and citation information.

## 3. Methodology

The pipeline below reflects our current working plan. Status markers show
where we actually are, not where the plan assumes we'll end up — this will
be updated as steps complete.

[ 🟢 Data Ingestion & Prep ] ──> [ 🔵 Heterogeneous GNN ] ──> [ ⏳ Evidence Scoring ] ──> [ ⏳ GraphRAG Explanation ] ──> [ ⏳ Temporal Evaluation ]


| Step | Description | Status |
|---|---|---|
| 1. Foundation | Load PrimeKG-CL t0 (Jun 2021) — 8.1M triples, 10 node types, 30 relations. t1 (Jul 2023) is held out entirely for temporal testing. | ✅ Done |
| 2. Preprocessing | Clean (dedupe, verify no missing values / self-loops), build node index, encode edges, construct the full graph as a PyTorch Geometric `HeteroData` object (10 node types, 53 relation-direction pairs). | ✅ Done |
| 3. Task definition | Drug–disease link prediction. Positive labels: `indication` (and possibly `off-label use`) edges. `contraindication` edges are reserved as hard negatives, not just noise. | 🔄 In progress |
| 4. Baseline model | 2-layer heterogeneous GraphSAGE, chosen over R-GCN for compute efficiency (R-GCN needs a separate weight matrix per relation type; we have 53). Node features: molecular/protein embeddings (in progress). | 🔄 In progress |
| 5. Loss & negative sampling | Margin-based ranking loss, mixing random negatives with hard negatives drawn from `contraindication` edges. | ⏳ Planned |
| 6. Evidence scoring | Layer in Open Targets association scores plus supplementary features (PPI, GO similarity) on top of raw model predictions. | ⏳ Planned |
| 7. Evidence-aware ranking | Final score = model prediction + evidence features + calibration status. | ⏳ Planned |
| 8. Path explanation | Extract multi-hop biological paths (Drug → Target → Pathway → Disease) for top candidates; generate GraphRAG-style explanations. | ⏳ Planned |
| 9. Evaluation | Test exclusively on the t1 (Jul 2023) snapshot, stratified by persistent / added / removed edges. Metrics: Hits@K, MRR, AUPRC, Precision@K, evidence support rate. | ⏳ Planned |

![PharmaGraph Pipeline](figures/pharmagraph_pipeline.png)
**Current status:** Steps 1–2 complete; moving into Steps 3–4.

4. Interactive Dashboard

Alongside the modeling pipeline, PharmaGraph includes an R Shiny dashboard for exploring the benchmark itself: how the graph changed between snapshots, which drug–disease edges were added/removed/persisted, and a network view around any selected disease.

Overview — headline stats and an added/removed/persistent triple breakdown between t0 and t1.
Temporal Evolution — the same breakdown, per relation type.
Drug–Disease — per-disease view of every relevant edge, tagged by which snapshot(s) it appears in.
Network Analysis — a 1-hop interactive graph around a selected disease (t1 snapshot).
Benchmarking — reserved for model results (Hits@K, MRR, AUPRC, Precision@K) once evaluation (Step 9 above) is complete; currently a placeholder.

The app reads small precomputed summaries (statistics.json, diffs/diff_t0_t1.json) directly, and queries the raw 8.1M/13M-row snapshots through a cached Python layer (src/dashboard_data.py) exposed to R via reticulate 

full architecture, data dependencies, and run instructions are documented in [dashboard/README.md](dashboard/README.md).
## 4. Repository layout

See the top-level directory listing for the full structure (`data/`,
`src/`, `dashboard/`, `notebooks/`, `output/`, etc.). The dataset itself and
its full schema/metadata are documented separately in `data/README.md` to
keep this file focused on project narrative rather than data mechanics.