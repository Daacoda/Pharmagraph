# PharmaGraph Source Architecture (`src/`)

> **Active Working Branch**: `simon-wandera`  
> **Target Framework**: PyTorch 2.14.0+cpu / PyTorch Geometric 2.8.0.post1  
> **System Architecture Specifications**: Grounded in [`external_docs/SAD/PharmaGraph_Formal_Architecture_Resolution.md`](file:///home/nothim/HIM/projectWork/external_docs/SAD/PharmaGraph_Formal_Architecture_Resolution.md) and [`external_docs/SAD/PharmaGraph_Technical_Architecture_Research.md`](file:///home/nothim/HIM/projectWork/external_docs/SAD/PharmaGraph_Technical_Architecture_Research.md).

---

## 1. Overview & Research Objectives

The code in `src/` implements **PharmaGraph**—an end-to-end framework for continual graph learning and drug repurposing across evolving heterogeneous biomedical knowledge graphs (PrimeKG-CL). 

PharmaGraph directly addresses the **two fundamental flaws** of traditional GNN link prediction in computational pharmacology:
1. **Uncalibrated Probability Failure (Defect 1)**: Raw GNN unnormalized logits produce overconfident, miscalibrated probability estimates that collapse when tested under distribution shifts. We resolve this via **validation-set Temperature-Scaled Platt Scaling** coupled with **rank-damped orthogonal biological evidence fusion** (Open Targets, PPI, GO).
2. **Post-Hoc Attribution Hallucination (Defect 2)**: Heuristic subgraph explainers (e.g., GNNExplainer) frequently identify ungrounded or topologically disconnected artifacts. We resolve this via **mechanistic bidirectional breadth-first search (Drug $\to$ Target $\to$ Pathway $\to$ Disease)** ranked strictly by **Degree-Weighted Path Constrained (DWPC)** metrics ($w=0.4$) and **Normalized Pointwise Mutual Information (NPMI)**.

---

## 2. Directory Structure & Module Responsibilities

```
src/
├── __init__.py                 # Package initializer
├── config.py                   # Central immutable dataclass configs (Hardware, Model, Sampling, Calibration, Paths)
├── data_loader.py              # HeteroData ingestion & snapshot caching for PrimeKG-CL (t0, t1)
├── preprocessing.py            # Deduplication, entity indexing, and feature generation
├── models/
│   ├── __init__.py             # Model package exports
│   ├── graphsage.py            # Heterogeneous GraphSAGE with entity-specific projection & DistMult/dot decoder
│   └── rgcn.py                 # Relational Graph Convolutional Network (R-GCN) benchmark
├── sampler.py                  # Disjoint negative edge sampler (contraindication hard negatives + random unobserved)
├── trainer.py                  # Multi-objective trainer (BCEWithLogits + Margin Ranking with contraindication loss)
├── calibration.py              # Temperature-Scaled Platt Scaling via validation L-BFGS optimization
├── evidence_scorer.py          # Orthogonal evidence layer integration & convex harmonic fusion
├── explainer.py                # Bidirectional BFS mechanistic path discovery (DWPC hub damping & NPMI edge scoring)
├── evaluator.py                # Stratified temporal evaluation on t1 (Persistent, Added, Removed edges)
└── dashboard_data.py           # Precomputed serialization bridge for interactive R Shiny UI
```

### Module Breakdown

| Module | Status | Primary Responsibilities & Mathematical Formulations |
| :--- | :---: | :--- |
| [`config.py`](file:///home/nothim/HIM/projectWork/pharmagraph/src/config.py) | **Active / Complete** | Immutable frozen dataclasses: `HardwareConfig` (CPU 8 threads, RAM limits), `ModelConfig` (hidden_dim=64, 2 layers, dropout=0.2), `SamplingConfig` (2:1 ratio, 50% hard contraindications), `CalibrationConfig` (Platt L-BFGS, $\alpha=0.65$), and `PathConfig`. |
| `data_loader.py` | **Refactoring** | Reads `processed/pharmagraph_hetero_t0.pt` (129,312 nodes across 10 biological types, 53 relation pairs, ~468 MiB memory footprint) and streams batches via `HGTLoader` / `NeighborLoader`. |
| `models/graphsage.py` | **Phase 2 In-Progress** | 2-layer Heterogeneous GraphSAGE. Implements linear projection $W_{\tau} \in \mathbb{R}^{d_{in} \times d}$ for heterogeneous features, mean neighborhood aggregation per relation, and inner product link decoding: $\hat{y}_{(u, v)} = \mathbf{z}_u^\top \mathbf{z}_v$. |
| `models/rgcn.py` | **Phase 2 In-Progress** | Relational GCN with basis-decomposition regularizing the 53 relational weight matrices to prevent parameter explosion. |
| `sampler.py` | **Phase 2 In-Progress** | Mines hard negative candidate edges: $(\text{Drug}, \text{contraindication}, \text{Disease}) \setminus (\text{Drug}, \text{indication}, \text{Disease})$. Enforces a strict 2:1 ratio (50% contraindication, 50% random unobserved) and bidirectional masking of validation/test edges during message passing. |
| `trainer.py` | **Phase 2 In-Progress** | Optimizes combined objective: $\mathcal{L} = \mathcal{L}_{\text{BCE}}(y, \hat{y}) + \lambda_{\text{margin}} \max(0, \gamma - \hat{y}_{\text{pos}} + \hat{y}_{\text{contra}})$. Includes early stopping on validation AUROC/AUPRC. |
| `calibration.py` | **Phase 3 Planned** | Fits temperature $T^*$ and bias $b^*$ on held-out validation logits using L-BFGS: $P_{\text{cal}} = \sigma(z / T^* + b^*)$, minimizing negative log-likelihood (NLL) and Expected Calibration Error (ECE). |
| `evidence_scorer.py` | **Phase 3 Planned** | Computes rank-damped harmonic orthogonal score $E_{\text{ortho}}$ across Open Targets, PPI, and GO, then computes final composite score: $S_{\text{composite}} = \alpha \cdot P_{\text{cal}} + (1 - \alpha) \cdot E_{\text{ortho}}$ ($\alpha = 0.65$). |
| `explainer.py` | **Phase 4 Planned** | Extracts top-$K$ ($K=5$) mechanistic biological paths ($\text{Drug} \to \text{Gene} \to \text{Pathway} \to \text{Disease}$) via bidirectional BFS ($B_g=30, B_p=50$) scored with DWPC: $\text{DWPC}(p) = \prod_{i=0}^{l} (\deg(v_i))^{-w}$ ($w=0.4$) and NPMI edge co-occurrence weights. |
| `evaluator.py` | **Phase 5 Planned** | Measures temporal generalization on snapshot $t_1$, stratifying edge evaluation across Persistent, Added, and Removed sets to quantify inductive vs. transductive drift. Computes AUROC, AUPRC, Hits@10, MRR, Brier Score, and ECE. |
| `dashboard_data.py` | **Phase 5 Planned** | Serializes precomputed prediction tables, calibrated probabilities, mechanistic graph traces, and evaluation metrics into fast feather/json formats for seamless consumption by the R Shiny frontend. |

---

## 3. Engineering Guardrails & Hardware Constraints

Execution on this branch is strictly tailored for commodity execution without GPU acceleration:
- **Processor**: Intel Core i5-8350U (4 Cores / 8 Threads @ 1.70 GHz base, 3.60 GHz turbo).
- **Execution Target**: PyTorch CPU-only (`torch.device('cpu')`), multi-threaded with `torch.set_num_threads(8)`.
- **Memory Footprint**: Heterogeneous graph stored in 32-bit float tensors in `processed/pharmagraph_hetero_t0.pt` (~468.61 MiB in memory), safely staying well beneath the available 3.9 GiB free RAM.
- **Disk Protection**: Raw LFS snapshots (`kg_t0.csv`, `kg_t1.csv`) are tracked via Git LFS pointers to prevent multi-gigabyte disk bloat on the 95% full disk partition.

---

## 4. Phase-Gate Implementation Roadmap

| Phase | Milestone | Acceptance Criteria | Current Status |
| :---: | :--- | :--- | :---: |
| **Phase 1** | **Foundation & Configuration** | Centralized immutable configs; reconstructed PyG `HeteroData` graph verified for all 10 node types, 53 relation pairs, and 129,312 nodes; automated test suite passing. | **COMPLETE**  *(Verified: `tests/test_phase1_foundation.py` 4/4 passing)* |
| **Phase 2** | **Heterogeneous GNN Modularization & Training** | Modularized `GraphSAGE`, `sampler.py`, and `trainer.py`; real data training runs on CPU; Test AUROC $\ge 0.90$, Test AUPRC $\ge 0.50$, Twin Bias $\Delta\text{AUROC} \le 0.03$. | **ACTIVE** |
| **Phase 3** | **Validation Platt Scaling & Evidence Fusion** | L-BFGS calibrated temperature scaling; composite score combining GNN logits and orthogonal evidence ($\alpha=0.65$); ECE $\le 0.05$. | Pending Phase 2 |
| **Phase 4** | **Mechanistic GraphRAG Path Explainer** | Bidirectional BFS path finder; DWPC hub damping ($w=0.4$) + NPMI edge scoring; path extraction latency $< 2.5\text{s}$ per drug-disease pair. | Pending Phase 3 |
| **Phase 5** | **Temporal Generalization & Dashboard Bridge** | Snapshot $t_1$ stratified temporal evaluation (Persistent vs Added vs Removed edges); automated data pipeline for R Shiny interactive dashboard. | Pending Phase 4 |

---

## 5. Running Tests & Verification

All code modifications are subject to strict test verification on real data prior to advancing between phases.

To run the foundation test suite:
```bash
# Activate virtual environment
source .venv/bin/activate

# Run Phase 1 Foundation verification
pytest -v tests/test_phase1_foundation.py

# Run Phase 2 GNN verification (once implemented)
pytest -v tests/test_phase2_gnn.py
```
