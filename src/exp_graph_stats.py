"""
Graph statistics in the style of KG-bench Table 3 / Fig. 1 (read-only).
The heterogeneous graph is projected to a simple undirected graph (no self loops,
duplicate edges and reverse relations collapsed).

    py src/exp_graph_stats.py                 # t0 graph, full + drug-disease subgraph
    py src/exp_graph_stats.py --sample 20000  # approximate clustering on 20k random nodes
"""
import argparse
import os

import numpy as np
import pandas as pd
import scipy.sparse as sp

PROCESSED = "processed"


def build_adj(src, dst, n):
    a = sp.coo_matrix((np.ones(len(src), dtype=np.float32), (src, dst)), shape=(n, n)).tocsr()
    a = a + a.T
    a.setdiag(0)
    a.eliminate_zeros()
    a.data[:] = 1
    return a.tocsr()


def stats(a, sample, seed=42):
    deg = np.asarray(a.sum(1)).ravel()
    coo = sp.triu(a, 1).tocoo()
    u, v = coo.row, coo.col
    # degree assortativity = Pearson r of degrees at both ends of every edge
    x = np.concatenate([deg[u], deg[v]]); y = np.concatenate([deg[v], deg[u]])
    assort = np.corrcoef(x, y)[0, 1] if x.std() > 0 else float("nan")
    # average local clustering (nodes with degree < 2 count as 0, networkx convention)
    rng = np.random.default_rng(seed)
    nodes = np.arange(a.shape[0])
    if sample and sample < len(nodes):
        nodes = rng.choice(nodes, sample, replace=False)
    cc = np.zeros(len(nodes))
    for i, n in enumerate(nodes):
        nb = a.indices[a.indptr[n]:a.indptr[n + 1]]
        k = len(nb)
        if k >= 2:
            cc[i] = a[nb][:, nb].nnz / (k * (k - 1))   # nnz counts each triangle edge twice
    return dict(nodes=a.shape[0], edges=int(a.nnz // 2), avg_degree=deg.mean(), std_degree=deg.std(),
                median_degree=float(np.median(deg)), max_degree=int(deg.max()),
                pct_deg_le_2=float((deg <= 2).mean() * 100),
                top1pct_edge_share=float(np.sort(deg)[::-1][:max(1, len(deg) // 100)].sum() / deg.sum() * 100),
                assortativity=assort, avg_clustering=cc.mean(), clustering_sampled=bool(sample))


def main():
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", default=os.path.join(PROCESSED, "pharmagraph_hetero_t0.pt"))
    ap.add_argument("--sample", type=int, default=0, help="0 = exact clustering on all nodes")
    ap.add_argument("--out", default=os.path.join("output", "graph_stats.csv"))
    args = ap.parse_args()

    data = torch.load(args.graph, weights_only=False)
    ntypes = list(data.node_types)
    off, tot = {}, 0
    for t in ntypes:
        off[t] = tot; tot += data[t].num_nodes
    S, D, DD_S, DD_D = [], [], [], []
    for et in data.edge_types:
        s, _, d = et
        ei = data[et].edge_index.numpy()
        S.append(ei[0] + off[s]); D.append(ei[1] + off[d])
        if {s, d} == {"drug", "disease"}:
            DD_S.append(ei[0] + off[s]); DD_D.append(ei[1] + off[d])
    rows = {"full_graph": stats(build_adj(np.concatenate(S), np.concatenate(D), tot), args.sample)}
    if DD_S:
        rows["drug_disease_only"] = stats(build_adj(np.concatenate(DD_S), np.concatenate(DD_D), tot), args.sample)
        # drop isolated nodes' effect: report on connected nodes only
        a = build_adj(np.concatenate(DD_S), np.concatenate(DD_D), tot)
        keep = np.asarray(a.sum(1)).ravel() > 0
        rows["drug_disease_connected"] = stats(a[keep][:, keep], args.sample)
    df = pd.DataFrame(rows).T
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out)
    pd.set_option("display.width", 200)
    print(df.round(4).T.to_string())
    print("\nKG-bench (Open Targets KG): 13,004 nodes, 106,793 edges, avg degree 16.42, std 76.93,"
          " assortativity -0.133, avg clustering 0.011")
    print("Saved", args.out)


if __name__ == "__main__":
    main()