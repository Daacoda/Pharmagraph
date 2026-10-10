"""
t1_task_graphsage.py
====================
TEMPORAL link-prediction task on PrimeKG-CL t0 -> t1 for a non-drug relation
(default: GO annotations, e.g. molfunc_protein), GraphSAGE only.

Protocol
--------
* Train on the t0 graph only. The relation's t0 edges are split into
      message-passing (80%)  |  supervised training (15%)  |  validation (5%)
  supervision and validation edges are HIDDEN from message passing (both
  directions) - the model is never allowed to "see" the edge it is asked to score.
* Test = edges that exist in t1 but not in t0 (processed/t1_tasks/<rel>_added.parquet,
  made by build_t1_tasks.py), both endpoints already present at t0. At test time the
  FULL t0 graph is used (--test-graph full, the realistic setting).
* Same architecture/recipe as the drug-disease v3 model: 2-layer HeteroConv(SAGEConv)
  sum aggregation, 128 dims, dot-product score, MarginRankingLoss(1.0), Adam 1e-3,
  weight decay 1e-4, dropout 0.3 on embeddings, early stopping on validation AUPRC.
  Node features are the constant 1 (as in the saved graph), so sum aggregation
  gives the model each node's degree directly - popularity is easy for it to learn.
* Training negatives (--neg):
      uniform : random term for each positive protein
      degree  : term drawn from the positives' own term distribution (popularity-
                matched), so "popular term" is no longer a free shortcut.

Evaluation (the rigorous protocol from Experiments 2-3)
-------------------------------------------------------
  A. PER-PROTEIN RANKING (main result): for each protein with new edges, rank all
     terms of the other type; known t0 terms and the protein's other new terms are
     filtered out. Hits@10/50/100, MRR, median rank, within-protein AUC
     (0.5 = random). Compared with no-learning baselines:
         term_popularity   (term degree in t0)
         ontology_neighbor (# of the protein's t0 terms adjacent to the candidate term
                            in the ontology graph; captures "parent/child of a term
                            the protein already has")
         random
     Stratified by protein t0 degree (0 / 1-4 / 5-19 / 20+) and term degree.
  B. PAIRWISE AUROC/AUPRC at 1:1 and 1:10 against random and popularity-matched
     negatives, vs degree_sum baseline.

Outputs (new files only): output/t1task_<relation>_{ranking,summary,pairwise}.csv

Run from the project root (build_t1_tasks.py first):
    py src\\t1_task_graphsage.py --relation molfunc_protein
    py src\\t1_task_graphsage.py --relation bioprocess_protein --neg degree
Speed: add --drop-relations anatomy_protein_present anatomy_protein_absent to remove the
millions of expression edges (changes the graph vs the drug model; report it if used).
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

PROCESSED = "processed"
PROT = "gene/protein"
P_BUCKETS = [(-1, 0, "0"), (0, 4, "1-4"), (4, 19, "5-19"), (19, 10**9, "20+")]
T_BUCKETS = [(-1, 0, "0"), (0, 9, "1-9"), (9, 99, "10-99"), (99, 10**9, "100+")]


def bucket(deg, spec):
    for lo, hi, name in spec:
        if lo < deg <= hi:
            return name
    return spec[0][2]


# ------------------------------------------------------------------ evaluation
def ranks_for_query(scores, excluded_mask, targets):
    """Average-tie rank of each target among {non-excluded candidates} U {target}."""
    base = np.sort(scores[~excluded_mask])
    st = scores[targets]
    left = np.searchsorted(base, st, side="left")
    right = np.searchsorted(base, st, side="right")
    return 1.0 + (len(base) - right) + 0.5 * (right - left), len(base) + 1


def summarise(q):
    r, n = q["rank"].values, q["n_cand"].values
    return {"n_queries": len(q), "n_proteins": q["query_idx"].nunique(),
            "hits@10": float((r <= 10).mean()), "hits@50": float((r <= 50).mean()),
            "hits@100": float((r <= 100).mean()), "mrr": float((1.0 / r).mean()),
            "median_rank": float(np.median(r)),
            "within_query_auc": float(((n - r) / np.maximum(n - 1, 1)).mean())}


def summarise_all(q):
    rows = []
    for by in ("query_bucket", "term_bucket"):
        for (sc, b), g in q.groupby(["scorer", by]):
            rows.append({"scorer": sc, "stratum": by, "bucket": b, **summarise(g)})
        for sc, g in q.groupby("scorer"):
            rows.append({"scorer": sc, "stratum": by, "bucket": "ALL", **summarise(g)})
    return pd.DataFrame(rows).drop_duplicates(subset=["scorer", "stratum", "bucket"])


def per_query_ranking(queries, targets_by_q, known_by_q, n_cand, score_chunk_fn,
                      deg_q, deg_t, chunk=512, seed=42):
    """queries: array of query-node ids. score_chunk_fn(dict name->(len(chunk), n_cand) matrix) per chunk."""
    rng = np.random.default_rng(seed)
    out = []
    for s in range(0, len(queries), chunk):
        qs = queries[s:s + chunk]
        mats = score_chunk_fn(qs)
        mats["random"] = rng.random((len(qs), n_cand))
        for i, q in enumerate(qs):
            targets = np.array(sorted(targets_by_q[q]))
            excl = np.zeros(n_cand, bool)
            excl[list(known_by_q.get(q, ()))] = True
            excl[targets] = True
            qb = bucket(deg_q[q], P_BUCKETS)
            for name, M in mats.items():
                ranks, nc = ranks_for_query(M[i], excl, targets)
                for t, rk in zip(targets, ranks):
                    out.append((name, int(q), int(t), float(rk), nc, qb, bucket(deg_t[t], T_BUCKETS)))
        print(f"    ranked {min(s + chunk, len(queries))}/{len(queries)} proteins")
    return pd.DataFrame(out, columns=["scorer", "query_idx", "term_idx", "rank", "n_cand",
                                      "query_bucket", "term_bucket"])


def pairwise_eval(pos_q, pos_t, n_q, n_t, excluded_keys, scorers, ratios=(1, 10), repeats=3, seed=42):
    rows = []
    n_pos = len(pos_q)
    pos_sc = {k: f(pos_q, pos_t) for k, f in scorers.items()}
    for rep in range(repeats):
        rng = np.random.default_rng(seed + rep)
        for mode in ("random", "popularity_matched"):
            need = max(ratios) * n_pos
            keys = np.empty(0, dtype=np.int64)
            for _ in range(100):
                if len(keys) >= need:
                    break
                m = int(need * 1.4) + 1000
                if mode == "random":
                    q = rng.integers(0, n_q, m); t = rng.integers(0, n_t, m)
                else:
                    q = rng.choice(pos_q, m); t = rng.choice(pos_t, m)
                k = q.astype(np.int64) * n_t + t
                k = k[~np.isin(k, excluded_keys)]
                keys = np.unique(np.concatenate([keys, k]))
            keys = rng.permutation(keys)[:need]
            nq, nt = keys // n_t, keys % n_t
            neg_sc = {k: f(nq, nt) for k, f in scorers.items()}
            for r in ratios:
                nn_ = min(r * n_pos, len(keys))
                y = np.concatenate([np.ones(n_pos), np.zeros(nn_)]).astype(int)
                for k in scorers:
                    s = np.concatenate([pos_sc[k], neg_sc[k][:nn_]])
                    rows.append({"neg_mode": mode, "scorer": k, "ratio": f"1:{r}", "repeat": rep,
                                 "n_pos": n_pos, "n_neg": nn_,
                                 "auroc": float(roc_auc_score(y, s)),
                                 "auprc": float(average_precision_score(y, s))})
    df = pd.DataFrame(rows)
    return df.groupby(["neg_mode", "scorer", "ratio"], as_index=False)[["auroc", "auprc", "n_pos", "n_neg"]].mean()


# -------------------------------------------------------------------- training
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--relation", default="molfunc_protein")
    ap.add_argument("--tasks-dir", default=os.path.join(PROCESSED, "t1_tasks"))
    ap.add_argument("--graph", default=os.path.join(PROCESSED, "pharmagraph_hetero_t0.pt"))
    ap.add_argument("--neg", choices=["uniform", "degree"], default="uniform")
    ap.add_argument("--neg-per-pos", type=int, default=2)
    ap.add_argument("--sup-frac", type=float, default=0.15)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--test-graph", choices=["full", "train"], default="full")
    ap.add_argument("--max-epochs", type=int, default=100)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--patience", type=int, default=4, help="in units of eval checks")
    ap.add_argument("--max-test-queries", type=int, default=None, help="subsample proteins for speed")
    ap.add_argument("--drop-relations", nargs="*", default=[])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="output")
    args = ap.parse_args()

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import scipy.sparse as sp
    from torch_geometric.nn import HeteroConv, SAGEConv

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rel = args.relation

    t0 = pd.read_parquet(os.path.join(args.tasks_dir, f"{rel}_t0.parquet"))
    added = pd.read_parquet(os.path.join(args.tasks_dir, f"{rel}_added.parquet"))
    X, Y = t0["x_type"].iloc[0], t0["y_type"].iloc[0]
    if X == Y:
        raise SystemExit("same-type relations are not supported by this script")
    # query side = proteins (rank terms), term side = the other type
    q_type, t_type = (X, Y) if X == PROT else (Y, X)
    qcol, tcol = ("x_local", "y_local") if X == q_type else ("y_local", "x_local")
    print(f"Relation {rel}: query type '{q_type}', candidate type '{t_type}' | "
          f"t0 edges {len(t0)}, new (test) edges {len(added)}")

    data = torch.load(args.graph, weights_only=False)
    for et in list(data.edge_types):
        if et[1] in args.drop_relations:
            del data[et]
    n_q, n_t = data[q_type].num_nodes, data[t_type].num_nodes
    fwd, rev = (q_type, rel, t_type), (t_type, rel, q_type)
    if fwd not in data.edge_types or rev not in data.edge_types:
        raise SystemExit(f"edge types {fwd}/{rev} not in graph: {data.edge_types[:5]} ...")

    E = t0[[qcol, tcol]].drop_duplicates().values.astype(np.int64)
    perm = rng.permutation(len(E))
    n_val = int(len(E) * args.val_frac)
    n_sup = int(len(E) * args.sup_frac)
    val_E, sup_E = E[perm[:n_val]], E[perm[n_val:n_val + n_sup]]
    key = lambda a: a[:, 0] * n_t + a[:, 1]
    hide_keys = np.concatenate([key(val_E), key(sup_E)])
    all_t0_keys = np.unique(key(E))
    test_E = added[[qcol, tcol]].drop_duplicates().values.astype(np.int64)
    test_keys = np.unique(key(test_E))
    print(f"split of t0 edges: message {len(E) - n_val - n_sup} | supervised {n_sup} | val {n_val}")

    def edge_keys(et, q_first):
        ei = data[et].edge_index.numpy()
        return (ei[0] * n_t + ei[1]) if q_first else (ei[1] * n_t + ei[0])

    def masked_graph(hide):
        d = data.clone()
        for et, qf in ((fwd, True), (rev, False)):
            ei = data[et].edge_index
            keep = ~np.isin(edge_keys(et, qf), hide)
            d[et].edge_index = ei[:, torch.from_numpy(keep)]
        return d.to(device)

    train_g = masked_graph(hide_keys)
    full_g = data.to(device)
    test_g = full_g if args.test_graph == "full" else train_g

    class Net(nn.Module):
        def __init__(self, ets):
            super().__init__()
            self.c1 = HeteroConv({e: SAGEConv((-1, -1), 128) for e in ets}, aggr="sum")
            self.c2 = HeteroConv({e: SAGEConv((-1, -1), 128) for e in ets}, aggr="sum")

        def forward(self, x, ei):
            x = {k: F.relu(v) for k, v in self.c1(x, ei).items()}
            return self.c2(x, ei)

    model = Net(train_g.edge_types).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.MarginRankingLoss(margin=1.0)
    drop = nn.Dropout(0.3)
    with torch.no_grad():
        model(train_g.x_dict, train_g.edge_index_dict)           # materialise lazy layers

    sup_q, sup_t = sup_E[:, 0], sup_E[:, 1]
    sup_q_t = torch.tensor(sup_q, device=device)
    sup_t_t = torch.tensor(sup_t, device=device)

    def sample_neg(k, q_nodes, excluded):
        negs = np.empty((k, len(q_nodes)), dtype=np.int64)
        for j in range(k):
            t = rng.integers(0, n_t, len(q_nodes)) if args.neg == "uniform" else rng.choice(sup_t, len(q_nodes))
            for _ in range(5):
                bad = np.isin(q_nodes * n_t + t, excluded)
                if not bad.any():
                    break
                t[bad] = (rng.integers(0, n_t, bad.sum()) if args.neg == "uniform"
                          else rng.choice(sup_t, bad.sum()))
            negs[j] = t
        return negs

    def embed(g, train):
        out = model(g.x_dict, g.edge_index_dict)
        eq, et_ = out[q_type], out[t_type]
        return (drop(eq), drop(et_)) if train else (eq, et_)

    val_neg = sample_neg(1, val_E[:, 0], all_t0_keys)[0]
    best, best_state, bad_checks = -1.0, None, 0
    print(f"\nTraining (neg={args.neg}, device={device}) ...")
    for ep in range(1, args.max_epochs + 1):
        t_start = time.time()
        model.train(); opt.zero_grad()
        eq, et_ = embed(train_g, True)
        negs = sample_neg(args.neg_per_pos, sup_q, all_t0_keys)
        pos = (eq[sup_q_t] * et_[sup_t_t]).sum(-1)
        neg_t = torch.tensor(negs.reshape(-1), device=device)
        neg = (eq[sup_q_t.repeat(args.neg_per_pos)] * et_[neg_t]).sum(-1)
        loss = loss_fn(pos.repeat(args.neg_per_pos), neg, torch.ones_like(neg))
        loss.backward(); opt.step()
        if ep % args.eval_every == 0:
            model.eval()
            with torch.no_grad():
                eq, et_ = embed(train_g, False)
                vp = (eq[torch.tensor(val_E[:, 0], device=device)] * et_[torch.tensor(val_E[:, 1], device=device)]).sum(-1)
                vn = (eq[torch.tensor(val_E[:, 0], device=device)] * et_[torch.tensor(val_neg, device=device)]).sum(-1)
            y = np.r_[np.ones(len(vp)), np.zeros(len(vn))]
            s = torch.cat([vp, vn]).cpu().numpy()
            auprc = average_precision_score(y, s)
            print(f"  epoch {ep:3d} | loss {loss.item():.4f} | val AUPRC {auprc:.4f} | {time.time() - t_start:.1f}s/epoch")
            if auprc > best:
                best, bad_checks = auprc, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                bad_checks += 1
                if bad_checks >= args.patience:
                    print("  early stopping")
                    break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        eq, et_ = embed(test_g, False)
    Q, T = eq.cpu(), et_.cpu()
    print(f"best val AUPRC {best:.4f}")

    # ---------------------------------------------------------- baselines/inputs
    deg_q = np.bincount(E[:, 0], minlength=n_q)
    deg_t = np.bincount(E[:, 1], minlength=n_t)
    A = sp.csr_matrix((np.ones(len(E)), (E[:, 0], E[:, 1])), shape=(n_q, n_t))
    O = None
    tt = [et for et in data.edge_types if et[0] == t_type and et[2] == t_type]
    if len(tt) == 1:
        oi = data[tt[0]].edge_index.numpy()
        O = sp.csr_matrix((np.ones(oi.shape[1]), (oi[0], oi[1])), shape=(n_t, n_t))
        O.data[:] = 1.0
        print(f"ontology edges used for ontology_neighbor baseline: {tt[0]} ({oi.shape[1]})")
    else:
        print(f"ontology_neighbor baseline skipped (term-term edge types found: {tt})")
    pop_vec = np.log1p(deg_t).astype(float)

    targets_by_q, known_by_q = {}, {}
    for a, b in test_E:
        targets_by_q.setdefault(int(a), set()).add(int(b))
    for a, b in E:
        known_by_q.setdefault(int(a), set()).add(int(b))
    queries = np.array(sorted(targets_by_q))
    if args.max_test_queries and len(queries) > args.max_test_queries:
        queries = np.sort(rng.choice(queries, args.max_test_queries, replace=False))
    print(f"\nPer-protein ranking: {len(queries)} proteins, {sum(len(targets_by_q[q]) for q in queries)} new edges, "
          f"{n_t} candidate terms")

    def score_chunk(qs):
        mats = {"model": (Q[qs] @ T.T).numpy().astype(float),
                "term_popularity": np.tile(pop_vec, (len(qs), 1))}
        if O is not None:
            mats["ontology_neighbor"] = (A[qs] @ O).toarray()
        return mats

    q_df = per_query_ranking(queries, targets_by_q, known_by_q, n_t, score_chunk, deg_q, deg_t, seed=args.seed)
    S = summarise_all(q_df)
    os.makedirs(args.out_dir, exist_ok=True)
    q_df.to_csv(os.path.join(args.out_dir, f"t1task_{rel}_ranking.csv"), index=False)
    S.to_csv(os.path.join(args.out_dir, f"t1task_{rel}_summary.csv"), index=False)

    cols = ["scorer", "bucket", "n_queries", "n_proteins", "hits@10", "hits@100", "mrr",
            "median_rank", "within_query_auc"]
    for by in ("query_bucket", "term_bucket"):
        print("\n" + "=" * 105)
        print(f"{rel}: per-protein ranking by {by}")
        print("=" * 105)
        print(S[S.stratum == by][cols].sort_values(["bucket", "scorer"]).to_string(
            index=False, float_format=lambda v: f"{v:.4f}"))

    # pairwise
    Qn, Tn = Q.numpy(), T.numpy()
    scorers = {
        "model": lambda a, b: (Qn[a] * Tn[b]).sum(1).astype(float),
        "degree_sum": lambda a, b: np.log1p(deg_q[a]) + np.log1p(deg_t[b]),
    }
    excl = np.unique(np.concatenate([all_t0_keys, test_keys]))
    P = pairwise_eval(test_E[:, 0], test_E[:, 1], n_q, n_t, excl, scorers, seed=args.seed)
    P.to_csv(os.path.join(args.out_dir, f"t1task_{rel}_pairwise.csv"), index=False)
    print("\n" + "=" * 105)
    print(f"{rel}: pairwise AUROC/AUPRC (3 draws)")
    print("=" * 105)
    print(P.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nSaved output/t1task_{rel}_ranking.csv, _summary.csv, _pairwise.csv")


if __name__ == "__main__":
    main()