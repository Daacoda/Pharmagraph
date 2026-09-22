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
time.لهف 

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
<p align="center">
  <img src="figures/workflow_overview.png" alt="our project workflow" width="220"/>
</p>

## 2. Data (summary)

PharmaGraph uses the **PrimeKG-CL Benchmark v1.0**:

| | t0 (Jun 2021) | t1 (Jul 2023) |
|---|---|---|
| Triples | 8,100,498 | 13,001,666 |
| Nodes | 129,375 | 134,211 |
| Relations | 30 | 25 |

Between t0 and t1: **5,760,234** triples were added (GO annotations,
disease-phenotype refinement, drug-target updates), **888,848** were removed
(retracted associations, ontology corrections), and **7,208,624** persisted
unchanged. This diff is what makes the benchmark genuinely temporal rather
than synthetically split, and it's the basis for the stratified evaluation
(persistent / added / removed) used in Step 9 below.

See `data/benchmark/README.md` for the full breakdown: task definitions, multimodal
feature coverage, and citation information.

## 3. Methodology

The pipeline below reflects our current working plan. Status markers show
where we actually are, not where the plan assumes we'll end up — this will
be updated as steps complete.

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