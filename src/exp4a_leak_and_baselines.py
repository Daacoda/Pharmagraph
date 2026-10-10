"""
exp4a_leak_and_baselines.py
===========================
EXPERIMENT 4a - validity checks for the drug-disease results. EVALUATION ONLY: loads
the trained v3 checkpoint, trains nothing, overwrites nothing (writes output/exp4a_*.csv).

Part 1 - contraindication-edge leak test (internal split)
    The hard negatives are contraindication edges. Training/evaluation hid the val/test
    INDICATION edges from message passing but left every contraindication edge visible,
    so for a hard-negative test pair the edge itself was in the graph the model reads.
    A model with a separate weight per edge type can learn "an edge of type
    contraindication links this pair -> negative". Here the val+test hard-negative
    contraindication edges are hidden (same rule the indications already follow), and
    the test AUROC is recomputed. A large drop on hard negatives = the 0.90 was inflated.

Part 2 - stronger no-learning baselines on the temporal per-drug ranking
    (positives from output/exp2_positives.csv, full t0 graph, filtered ranking as in exp3)
        model | disease_popularity | fingerprint_knn | shared_protein_AA | random
    fingerprint_knn   : disease score = sum over the K most Tanimoto-similar drugs (that
                        have t0 indications) of similarity * [that drug is indicated for the disease]
    shared_protein_AA : disease score = Adamic-Adar weighted count of proteins that the
                        drug targets and the disease is associated with (2-hop path)
    If a simple similarity baseline matches the GNN on cold drugs, the GNN adds little there.

Part 3 - drug-level cluster bootstrap (1000 resamples) for the main comparisons: within-drug
    AUC, Hits@100, MRR, with 95% CIs and the paired difference  model - baseline.

Run from the project root (after exp2 and exp3):
    py src\\exp4a_leak_and_baselines.py
    py src\\exp4a_leak_and_baselines.py --skip-leak        # only baselines + bootstrap
"""
import argparse
import os

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

PROCESSED = "processed"
FP_SIZE = 256
BUCKETS = [(-1, 0, "0"), (0, 2, "1-2"), (2, 10**9, "3+")]


def bucket(deg):
    for lo, hi, name in BUCKETS:
        if lo < deg <= hi:
            return name
    return "0"


def pair_keys(a, b, n_b):
    return np.asarray(a, dtype=np.int64) * int(n_b) + np.asarray(b, dtype=np.int64)


# --------------------------------------------------------------- scorer builder
def build_scorer(checkpoint, cand, hide_ind_split=("val", "test"), hide_contra=None, device_str=None):
    """Rebuild the message-passing graph and return score_fn(drug_idx, dis_idx).
    hide_ind_split : candidate splits whose positive indication edges are hidden ((): hide none)
    hide_contra    : DataFrame(drug_idx, disease_idx) whose contraindication edges are hidden"""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn import HeteroConv, SAGEConv

    class HeteroGraphSAGE(nn.Module):
        def __init__(self, ets):
            super().__init__()
            self.conv1 = HeteroConv({e: SAGEConv((-1, -1), 128) for e in ets}, aggr="sum")
            self.conv2 = HeteroConv({e: SAGEConv((-1, -1), 128) for e in ets}, aggr="sum")

        def forward(self, x, ei):
            x = {k: F.relu(v) for k, v in self.conv1(x, ei).items()}
            return self.conv2(x, ei)

    device = torch.device(device_str or ("cuda" if torch.cuda.is_available() else "cpu"))
    ent = pd.read_csv(os.path.join(PROCESSED, "entities.csv"), dtype={"node_id": str})
    fps = np.load(os.path.join(PROCESSED, "drug_fingerprints.npy"), allow_pickle=True).item()
    data = torch.load(os.path.join(PROCESSED, "pharmagraph_hetero_t0.pt"), weights_only=False)
    dn = ent[ent.node_type == "drug"].sort_values("local_idx")
    X = np.zeros((len(dn), FP_SIZE), dtype=np.float32)
    for nid, li in zip(dn["node_id"], dn["local_idx"]):
        if nid in fps:
            X[li] = fps[nid]
    data["drug"].x = torch.tensor(X)
    n_drugs, n_dis = data["drug"].num_nodes, data["disease"].num_nodes

    def mask(et, hide_keys, drug_first):
        ei = data[et].edge_index
        k = pair_keys(ei[0].numpy(), ei[1].numpy(), n_dis) if drug_first else pair_keys(ei[1].numpy(), ei[0].numpy(), n_dis)
        data[et].edge_index = ei[:, torch.from_numpy(~np.isin(k, hide_keys))]

    pos = cand[(cand.label == 1) & cand.split.isin(list(hide_ind_split))]
    if len(pos):
        hk = np.unique(pair_keys(pos.drug_idx.values, pos.disease_idx.values, n_dis))
        mask(("drug", "indication", "disease"), hk, True)
        mask(("disease", "indication", "drug"), hk, False)
    n_hidden_contra = 0
    if hide_contra is not None and len(hide_contra):
        hk = np.unique(pair_keys(hide_contra.drug_idx.values, hide_contra.disease_idx.values, n_dis))
        before = data["drug", "contraindication", "disease"].edge_index.shape[1]
        mask(("drug", "contraindication", "disease"), hk, True)
        if ("disease", "contraindication", "drug") in data.edge_types:
            mask(("disease", "contraindication", "drug"), hk, False)
        n_hidden_contra = before - data["drug", "contraindication", "disease"].edge_index.shape[1]
    print(f"  graph built: indication edges {data['drug', 'indication', 'disease'].edge_index.shape[1]}, "
          f"contraindication edges hidden {n_hidden_contra}")

    data = data.to(device)
    model = HeteroGraphSAGE(data.edge_types).to(device)
    with torch.no_grad():
        model(data.x_dict, data.edge_index_dict)
    ck = torch.load(checkpoint, weights_only=False)
    model.load_state_dict(ck["model_state"])
    model.eval()
    with torch.no_grad():
        out = model(data.x_dict, data.edge_index_dict)
        de, se = out["drug"], out["disease"]

    def score_fn(d, s):
        with torch.no_grad():
            dd = torch.as_tensor(np.array(d), dtype=torch.long, device=device)
            ss = torch.as_tensor(np.array(s), dtype=torch.long, device=device)
            return (de[dd] * se[ss]).sum(-1).cpu().numpy()

    return score_fn, n_drugs, n_dis


# ------------------------------------------------------------------- Part 1
def subset_auroc(test, score):
    y = test.label.values
    pos = y == 1
    hard = test.is_hard_negative.values == 1
    out = {}
    for name, m in (("all_negatives", np.ones(len(y), bool)),
                    ("hard_negatives_only", pos | (~pos & hard)),
                    ("random_negatives_only", pos | (~pos & ~hard))):
        out[name] = (float(roc_auc_score(y[m], score[m])), float(average_precision_score(y[m], score[m])))
    return out


def leak_test(args, cand):
    print("Part 1 - contraindication-edge leak test")
    test = cand[cand.split == "test"].reset_index(drop=True)
    hide_c = cand[(cand.is_hard_negative == 1) & cand.split.isin(["val", "test"])]
    print(" original setup (contraindication edges visible):")
    f0, _, _ = build_scorer(args.checkpoint, cand)
    print(" contraindication edges of val+test hard negatives hidden:")
    f1, _, _ = build_scorer(args.checkpoint, cand, hide_contra=hide_c)
    r0 = subset_auroc(test, f0(test.drug_idx.values, test.disease_idx.values))
    r1 = subset_auroc(test, f1(test.drug_idx.values, test.disease_idx.values))
    rows = []
    for k in r0:
        rows.append({"subset": k, "auroc_visible": r0[k][0], "auroc_hidden": r1[k][0],
                     "auroc_drop": r0[k][0] - r1[k][0], "auprc_visible": r0[k][1], "auprc_hidden": r1[k][1]})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out_dir, "exp4a_leak_test.csv"), index=False)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    d = float(df.loc[df.subset == "hard_negatives_only", "auroc_drop"].iloc[0])
    print(f"\n -> hard-negative AUROC change when the contraindication edge is hidden: {-d:+.4f}")
    print("    (about 0 = no leak; a large negative number = the edge itself was a shortcut)\n")


# ------------------------------------------------------------------- Part 2
def tanimoto_row(Xb, cnt, d):
    inter = Xb @ Xb[d]
    return inter / np.maximum(cnt + cnt[d] - inter, 1e-9)


def ranks_for_drug(scores, excl, targets):
    base = np.sort(scores[~excl])
    st = scores[targets]
    left = np.searchsorted(base, st, side="left")
    right = np.searchsorted(base, st, side="right")
    return 1.0 + (len(base) - right) + 0.5 * (right - left), len(base) + 1


def make_baselines(data, n_drugs, n_dis, ei, X_fp, knn_k):
    import scipy.sparse as sp
    ones = np.ones(ei.shape[1])
    Ind = sp.csr_matrix((ones, (ei[0], ei[1])), shape=(n_drugs, n_dis))
    Ind.data[:] = 1.0
    deg_d = np.asarray(Ind.sum(1)).ravel()
    deg_s = np.asarray(Ind.sum(0)).ravel()
    has_ind = deg_d > 0
    Xb = (X_fp > 0).astype(np.float32)
    cnt = Xb.sum(1)
    pop = np.log1p(deg_s).astype(float)

    def knn(d):
        sims = tanimoto_row(Xb, cnt, d)
        sims[d] = -1
        sims[~has_ind] = -1
        top = np.argpartition(-sims, min(knn_k, len(sims) - 1))[:knn_k]
        top = top[sims[top] > 0]
        if len(top) == 0:
            return np.zeros(n_dis)
        return np.asarray(Ind[top].T @ sims[top]).ravel()

    # shared-protein Adamic-Adar
    sp_fn = None
    def find(rel, a):
        for et in data.edge_types:
            if et[1] == rel and et[0] == a and et[2] == "gene/protein":
                return et
        return None
    e_dp, e_sp = find("drug_protein", "drug"), find("disease_protein", "disease")
    if e_dp and e_sp:
        n_p = data["gene/protein"].num_nodes
        a = data[e_dp].edge_index.numpy(); b = data[e_sp].edge_index.numpy()
        Adp = sp.csr_matrix((np.ones(a.shape[1]), (a[0], a[1])), shape=(n_drugs, n_p)); Adp.data[:] = 1.0
        Asp = sp.csr_matrix((np.ones(b.shape[1]), (b[0], b[1])), shape=(n_dis, n_p)); Asp.data[:] = 1.0
        w = 1.0 / np.log(2.0 + np.asarray(Asp.sum(0)).ravel())
        AspW = (Asp @ sp.diags(w)).tocsr()
        def sp_fn(d):
            return np.asarray((Adp[d] @ AspW.T).todense()).ravel()
    else:
        print("  shared_protein_AA skipped (drug_protein / disease_protein edge types not found)")
    return pop, knn, sp_fn, deg_d, deg_s, Ind


def run_ranking(pos, score_model, pop, knn, sp_fn, deg_d, deg_s, Ind, n_dis, seed):
    rng = np.random.default_rng(seed)
    union = pos.groupby("drug_idx")["disease_idx"].apply(lambda x: set(x.tolist())).to_dict()
    rows = []
    drugs = sorted(pos.drug_idx.unique())
    for i, d in enumerate(drugs, 1):
        t0_set = set(Ind[d].indices.tolist())
        excl = np.zeros(n_dis, bool)
        excl[list(t0_set | union[d])] = True
        scorers = {"model": score_model(d), "disease_popularity": pop, "fingerprint_knn": knn(d),
                   "random": rng.random(n_dis)}
        if sp_fn is not None:
            scorers["shared_protein_AA"] = sp_fn(d)
        for cfg, g in pos[pos.drug_idx == d].groupby("config"):
            tg = g.disease_idx.values
            for name, sc in scorers.items():
                rk, nc = ranks_for_drug(sc, excl, tg)
                for t, r in zip(tg, rk):
                    rows.append((cfg, name, int(d), int(t), float(r), nc, bucket(deg_d[d])))
        if i % 200 == 0:
            print(f"    ranked {i}/{len(drugs)} drugs")
    return pd.DataFrame(rows, columns=["config", "scorer", "drug_idx", "disease_idx", "rank", "n_cand", "drug_bucket"])


# ------------------------------------------------------------------- Part 3
def metric_arrays(q):
    r, n = q["rank"].values, q["n_cand"].values
    return {"within_drug_auc": (n - r) / np.maximum(n - 1, 1), "hits@100": (r <= 100).astype(float),
            "mrr": 1.0 / r}


def bootstrap(q, scorers_vs, B=1000, seed=0):
    """Cluster bootstrap over drugs. q: per-query frame for ONE config/stratum, all scorers.
    Returns rows: scorer, metric, mean, lo, hi and paired diff model - scorer."""
    rng = np.random.default_rng(seed)
    drugs = np.sort(q.drug_idx.unique())
    D = len(drugs)
    pos_of = {d: i for i, d in enumerate(drugs)}
    out = []
    agg = {}
    for sc, g in q.groupby("scorer"):
        idx = g.drug_idx.map(pos_of).values
        n_per = np.bincount(idx, minlength=D).astype(float)
        sums = {m: np.bincount(idx, weights=v, minlength=D) for m, v in metric_arrays(g).items()}
        agg[sc] = (n_per, sums)
    draws = rng.integers(0, D, size=(B, D))
    boot = {}
    for sc, (n_per, sums) in agg.items():
        den = n_per[draws].sum(1)
        for m, s in sums.items():
            boot[(sc, m)] = s[draws].sum(1) / np.maximum(den, 1)
    for (sc, m), v in boot.items():
        pt = agg[sc][1][m].sum() / max(agg[sc][0].sum(), 1)
        out.append({"scorer": sc, "metric": m, "mean": pt, "lo": np.percentile(v, 2.5), "hi": np.percentile(v, 97.5),
                    "diff_model_minus": np.nan, "diff_lo": np.nan, "diff_hi": np.nan})
    for sc in scorers_vs:
        if sc == "model" or ("model", "mrr") not in boot or (sc, "mrr") not in boot:
            continue
        for m in ("within_drug_auc", "hits@100", "mrr"):
            dv = boot[("model", m)] - boot[(sc, m)]
            pt = (agg["model"][1][m].sum() / agg["model"][0].sum()) - (agg[sc][1][m].sum() / agg[sc][0].sum())
            out.append({"scorer": sc, "metric": m + " (model - scorer)", "mean": np.nan, "lo": np.nan, "hi": np.nan,
                        "diff_model_minus": pt, "diff_lo": np.percentile(dv, 2.5), "diff_hi": np.percentile(dv, 97.5)})
    return pd.DataFrame(out)


# ---------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default=os.path.join("output", "models", "graphsage_v3_real_features.pt"))
    ap.add_argument("--candidates", default=os.path.join(PROCESSED, "candidate_pairs.csv"))
    ap.add_argument("--positives", default=os.path.join("output", "exp2_positives.csv"))
    ap.add_argument("--configs", nargs="+", default=["phase>=1 | new_vs_ot2111", "phase>=4 | new_vs_ot2111"])
    ap.add_argument("--knn-k", type=int, default=20)
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--skip-leak", action="store_true")
    ap.add_argument("--skip-baselines", action="store_true")
    ap.add_argument("--out-dir", default="output")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    cand = pd.read_csv(args.candidates)

    if not args.skip_leak:
        leak_test(args, cand)
    if args.skip_baselines:
        return

    import torch
    print("Part 2 - stronger baselines on temporal per-drug ranking")
    pos = pd.read_csv(args.positives)
    pos = pos[pos.config.isin(args.configs)].reset_index(drop=True)
    if pos.empty:
        raise SystemExit(f"no positives for configs {args.configs}; available: see {args.positives}")
    score_fn, n_drugs, n_dis = build_scorer(args.checkpoint, cand, hide_ind_split=())
    data = torch.load(os.path.join(PROCESSED, "pharmagraph_hetero_t0.pt"), weights_only=False)
    ei = data["drug", "indication", "disease"].edge_index.numpy()
    ent = pd.read_csv(os.path.join(PROCESSED, "entities.csv"), dtype={"node_id": str})
    fps = np.load(os.path.join(PROCESSED, "drug_fingerprints.npy"), allow_pickle=True).item()
    dn = ent[ent.node_type == "drug"].sort_values("local_idx")
    X_fp = np.zeros((n_drugs, FP_SIZE), dtype=np.float32)
    for nid, li in zip(dn["node_id"], dn["local_idx"]):
        if nid in fps:
            X_fp[li] = fps[nid]
    pop, knn, sp_fn, deg_d, deg_s, Ind = make_baselines(data, n_drugs, n_dis, ei, X_fp, args.knn_k)
    all_dis = np.arange(n_dis)
    q = run_ranking(pos, lambda d: score_fn(np.full(n_dis, d), all_dis).astype(float),
                    pop, knn, sp_fn, deg_d, deg_s, Ind, n_dis, args.seed)
    q.to_csv(os.path.join(args.out_dir, "exp4a_per_drug_ranking.csv"), index=False)

    print("\nPart 3 - bootstrap CIs (drug-level cluster bootstrap)")
    rows = []
    for cfg in args.configs:
        for stratum in ["ALL", "0", "1-2", "3+"]:
            sub = q[q.config == cfg] if stratum == "ALL" else q[(q.config == cfg) & (q.drug_bucket == stratum)]
            if sub.drug_idx.nunique() < 20:
                continue
            b = bootstrap(sub, [s for s in sub.scorer.unique() if s != "model"], args.bootstrap, args.seed)
            b.insert(0, "drug_bucket", stratum); b.insert(0, "config", cfg)
            b["n_queries"] = int((sub.scorer == "model").sum()); b["n_drugs"] = sub.drug_idx.nunique()
            rows.append(b)
    B = pd.concat(rows, ignore_index=True)
    B.to_csv(os.path.join(args.out_dir, "exp4a_bootstrap.csv"), index=False)

    pd.set_option("display.width", 220)
    for cfg in args.configs:
        for stratum in ["ALL", "0"]:
            s = B[(B.config == cfg) & (B.drug_bucket == stratum) & B["mean"].notna()]
            if s.empty:
                continue
            print("\n" + "=" * 100)
            print(f"{cfg} | drug bucket {stratum} | n_queries={int(s.n_queries.iloc[0])} n_drugs={int(s.n_drugs.iloc[0])}")
            print("=" * 100)
            t = s.pivot(index="scorer", columns="metric", values="mean")
            lo = s.pivot(index="scorer", columns="metric", values="lo")
            hi = s.pivot(index="scorer", columns="metric", values="hi")
            for sc in t.index:
                print(f"  {sc:<20}" + "  ".join(
                    f"{m}: {t.loc[sc, m]:.4f} [{lo.loc[sc, m]:.4f}, {hi.loc[sc, m]:.4f}]" for m in ["within_drug_auc", "hits@100", "mrr"]))
            d = B[(B.config == cfg) & (B.drug_bucket == stratum) & B.diff_model_minus.notna()]
            print("  paired difference, model minus baseline (95% CI; CI containing 0 = no real difference):")
            for _, r in d.iterrows():
                print(f"    vs {r.scorer:<20} {r.metric:<30} {r.diff_model_minus:+.4f} [{r.diff_lo:+.4f}, {r.diff_hi:+.4f}]")
    print("\nSaved output/exp4a_leak_test.csv, exp4a_per_drug_ranking.csv, exp4a_bootstrap.csv")


if __name__ == "__main__":
    main()