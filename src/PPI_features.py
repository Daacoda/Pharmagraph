"""
python src/PPI_features.py
Protein protein interaction (PPI) and Gene Ontology (GO) similarity features for drug-disease pairs.
Stage 2 - PPI and GO-similarity evidence features, computed directly from
your own graph (no external API/download needed - unlike Open Targets).

For every (drug_idx, disease_idx) pair in processed/candidate_pairs.csv:
  - ppi_min_hops: shortest path (BFS, capped) in the protein-protein network
    between any of the drug's target proteins and any of the disease's
    associated proteins. Lower = more biologically connected.
  - ppi_common_neighbors: size of the overlap between the drug-targets' PPI
    neighborhood and the disease-proteins' PPI neighborhood.
  - go_jaccard_bp / go_jaccard_mf / go_jaccard_cc: best (max) Jaccard overlap
    of GO term sets (biological_process / molecular_function /
    cellular_component) between any drug-target protein and any
    disease-associated protein.

Run from the project root, after export_candidate_pairs.py:
    python src/ppi_go_features.py
"""
import os
from collections import defaultdict, deque

import numpy as np
import pandas as pd
import torch

processed_dir = "processed"
MAX_HOPS = 3  # cap BFS depth - anything further isn't meaningfully "close"

pairs = pd.read_csv(os.path.join(processed_dir, "candidate_pairs.csv"))
data = torch.load(os.path.join(processed_dir, "pharmagraph_hetero_t0.pt"), weights_only=False)


def edge_index_to_adj(edge_index, n_src, symmetric=False):
    adj = defaultdict(set)
    src, dst = edge_index[0].tolist(), edge_index[1].tolist()
    for s, d in zip(src, dst):
        adj[s].add(d)
        if symmetric:
            adj[d].add(s)
    return adj


# --- protein-protein network (already undirected in the data, both directions present) ---
ppi_adj = edge_index_to_adj(data["gene/protein", "protein_protein", "gene/protein"].edge_index,
                             data["gene/protein"].num_nodes)
print(f"PPI network: {len(ppi_adj)} proteins with at least one interaction")

# --- drug -> target proteins, disease -> associated proteins ---
drug_targets = edge_index_to_adj(data["drug", "drug_protein", "gene/protein"].edge_index,
                                  data["drug"].num_nodes)
disease_proteins = edge_index_to_adj(data["disease", "disease_protein", "gene/protein"].edge_index,
                                      data["disease"].num_nodes)
print(f"Drugs with >=1 target protein: {len(drug_targets)} / {data['drug'].num_nodes}")
print(f"Diseases with >=1 associated protein: {len(disease_proteins)} / {data['disease'].num_nodes}")

# --- protein -> GO term sets, one dict per GO category ---
go_edges = {
    "bp": ("gene/protein", "bioprocess_protein", "biological_process"),
    "mf": ("gene/protein", "molfunc_protein", "molecular_function"),
    "cc": ("gene/protein", "cellcomp_protein", "cellular_component"),
}
protein_go = {}
for cat, (src_type, rel, dst_type) in go_edges.items():
    et = (dst_type, rel, src_type) if (dst_type, rel, src_type) in data.edge_types else (src_type, rel, dst_type)
    ei = data[et].edge_index
    if et[0] == "gene/protein":
        protein_go[cat] = edge_index_to_adj(ei, data["gene/protein"].num_nodes)
    else:
        protein_go[cat] = edge_index_to_adj(torch.stack([ei[1], ei[0]]), data["gene/protein"].num_nodes)
    print(f"GO category '{cat}': {len(protein_go[cat])} proteins annotated")


def bfs_min_hops(sources, targets, adj, max_hops=MAX_HOPS):
    """Shortest hop count from ANY source node to ANY target node."""
    if not sources or not targets:
        return None
    targets = set(targets)
    if sources & targets:
        return 0
    frontier = set(sources)
    visited = set(sources)
    for hop in range(1, max_hops + 1):
        nxt = set()
        for node in frontier:
            nxt |= adj.get(node, set())
        nxt -= visited
        if nxt & targets:
            return hop
        visited |= nxt
        frontier = nxt
        if not frontier:
            break
    return None  # beyond max_hops, or unreachable


def jaccard_best(set_list_a, set_list_b):
    """Max pairwise Jaccard between any set in a and any set in b."""
    best = 0.0
    for a in set_list_a:
        if not a:
            continue
        for b in set_list_b:
            if not b:
                continue
            inter = len(a & b)
            if inter == 0:
                continue
            j = inter / len(a | b)
            if j > best:
                best = j
    return best


rows = []
for drug_idx, disease_idx in pairs[["drug_idx", "disease_idx"]].drop_duplicates().itertuples(index=False):
    targets = drug_targets.get(drug_idx, set())
    d_proteins = disease_proteins.get(disease_idx, set())

    ppi_neighbors_targets = set()
    for t in targets:
        ppi_neighbors_targets |= ppi_adj.get(t, set())
    ppi_neighbors_disease = set()
    for p in d_proteins:
        ppi_neighbors_disease |= ppi_adj.get(p, set())
    common_neighbors = len(ppi_neighbors_targets & ppi_neighbors_disease)

    min_hops = bfs_min_hops(targets, d_proteins, ppi_adj)

    go_scores = {}
    for cat in ("bp", "mf", "cc"):
        a_sets = [protein_go[cat].get(t, set()) for t in targets]
        b_sets = [protein_go[cat].get(p, set()) for p in d_proteins]
        go_scores[cat] = jaccard_best(a_sets, b_sets)

    rows.append((drug_idx, disease_idx,
                 min_hops if min_hops is not None else MAX_HOPS + 1,
                 common_neighbors,
                 go_scores["bp"], go_scores["mf"], go_scores["cc"]))

out = pd.DataFrame(rows, columns=[
    "drug_idx", "disease_idx", "ppi_min_hops", "ppi_common_neighbors",
    "go_jaccard_bp", "go_jaccard_mf", "go_jaccard_cc",
])
out.to_parquet(os.path.join(processed_dir, "ppi_go_features.parquet"), index=False)
print(f"\nSaved processed/ppi_go_features.parquet: {len(out)} pairs")
print(out.describe())
