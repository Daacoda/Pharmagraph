"""
ablation_drug_disease.py
========================
ABLATION STUDY for the GraphSAGE drug-disease model (v3 recipe), evaluated with the
rigorous protocol from Experiments 2-4a. Trains NEW models (never touches the saved
v3 checkpoint or any existing file); writes output/ablation_*.csv.

Every variant is trained from scratch with identical splits (processed/candidate_pairs.csv:
fixed train/val/test), the same masked message-passing graph rule (val+test indication
edges hidden) and the same optimiser, changing ONE component:

  full                   v3 recipe: real Morgan fingerprints, dropout 0.3, weight decay 1e-4,
                         2 layers, hard (contraindication) + random negatives
  no_fingerprints        drug features = constant 1 (no chemistry)
  placeholder_features   random fixed binary vectors instead of fingerprints (control: do
                         ANY drug features + regularisation help, or the chemistry itself?)
  no_regularization      dropout 0, weight decay 0
  random_negatives_only  hard negatives in the training set replaced by fresh random pairs
  one_layer              a single message-passing layer
  no_contra_edges        contraindication edges removed from the message-passing graph
  no_protein_edges       every edge type touching gene/protein removed
  no_ontology_context    edges touching GO/anatomy/pathway/exposure/phenotype nodes removed
  no_indication_edges    NO indication edges in message passing (supervision only), also at test

Metrics per run (all on RAW scores, no sigmoid ties):
  internal test split : AUROC overall / vs hard negatives only / vs random negatives only
  temporal (Open Targets positives from output/exp2_positives.csv, full t0 graph, filtered
  per-drug ranking as in exp3/exp4a): within-drug AUC, Hits@100, MRR, and within-drug AUC
  for cold drugs (0 t0 indications), for the two strictly-new configs.

Run from the project root (needs exp2_positives.csv from Experiment 2):
    py src\\ablation_drug_disease.py --seeds 0 1 2
    py src\\ablation_drug_disease.py --variants full no_fingerprints --seeds 0     # quick start
Time: one run is roughly the time of a normal v3 training (minutes on CPU); the full default
grid is 10 variants x 3 seeds = 30 runs. Results are appended to output/ablation_runs.csv after
EVERY run, so it can be stopped and resumed (finished variant/seed pairs are skipped).
"""
import argparse
import gc
import os
import time

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

PROCESSED = "processed"
FP_SIZE = 256
CONTEXT_TYPES = {"biological_process", "molecular_function", "cellular_component", "anatomy",
                 "pathway", "exposure", "effect/phenotype"}
VARIANTS = {
    "full": {},
    "no_fingerprints": {"features": "ones"},
    "placeholder_features": {"features": "random"},
    "no_regularization": {"dropout": 0.0, "weight_decay": 0.0},
    "random_negatives_only": {"neg": "random_only"},
    "one_layer": {"layers": 1},
    "no_contra_edges": {"drop": "contra"},
    "no_protein_edges": {"drop": "protein"},
    "no_ontology_context": {"drop": "context"},
    "no_indication_edges": {"drop": "indication"},
}
DEFAULTS = {"features": "real", "dropout": 0.3, "weight_decay": 1e-4, "neg": "as_is", "layers": 2, "drop": None}
CFGS = {"p1": "phase>=1 | new_vs_ot2111", "p4": "phase>=4 | new_vs_ot2111"}


def pair_keys(a, b, n_b):
    return np.asarray(a, dtype=np.int64) * int(n_b) + np.asarray(b, dtype=np.int64)


def types_to_drop(edge_types, kind):
    if kind == "contra":
        return [e for e in edge_types if e[1] == "contraindication"]
    if kind == "indication":
        return [e for e in edge_types if e[1] == "indication"]
    if kind == "protein":
        return [e for e in edge_types if "gene/protein" in (e[0], e[2])]
    if kind == "context":
        return [e for e in edge_types if e[0] in CONTEXT_TYPES or e[2] in CONTEXT_TYPES]
    return []


# ----------------------------------------------------------- ranking (as exp4a)
def ranks_for_drug(scores, excl, targets):
    base = np.sort(scores[~excl])
    st = scores[targets]
    left = np.searchsorted(base, st, side="left")
    right = np.searchsorted(base, st, side="right")
    return 1.0 + (len(base) - right) + 0.5 * (right - left), len(base) + 1


def temporal_metrics(score_all, pos, t0_by_drug, deg_d, n_dis):
    """pos: DataFrame(config, drug_idx, disease_idx) already filtered to the evaluated configs."""
    union = pos.groupby("drug_idx")["disease_idx"].apply(lambda x: set(x.tolist())).to_dict()
    recs = []
    for d in sorted(union):
        excl = np.zeros(n_dis, bool)
        excl[list(t0_by_drug.get(d, set()) | union[d])] = True
        sc = score_all(d)
        for cfg, g in pos[pos.drug_idx == d].groupby("config"):
            rk, nc = ranks_for_drug(sc, excl, g.disease_idx.values)
            for r in rk:
                recs.append((cfg, r, nc, deg_d[d] == 0))
    q = pd.DataFrame(recs, columns=["config", "rank", "n_cand", "cold"])
    out = {}
    for short, cfg in CFGS.items():
        s = q[q.config == cfg]
        if s.empty:
            continue
        out[f"{short}_within_auc"] = float(((s.n_cand - s["rank"]) / (s.n_cand - 1)).mean())
        out[f"{short}_hits100"] = float((s["rank"] <= 100).mean())
        out[f"{short}_mrr"] = float((1.0 / s["rank"]).mean())
        c = s[s.cold]
        out[f"{short}_cold_within_auc"] = float(((c.n_cand - c["rank"]) / (c.n_cand - 1)).mean()) if len(c) else np.nan
    return out


# ----------------------------------------------------------------- one training run
def run_variant(name, seed, ctx, max_epochs, eval_every, patience):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn import HeteroConv, SAGEConv

    cfg = {**DEFAULTS, **VARIANTS[name]}
    device = ctx["device"]
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    cand, n_drugs, n_dis = ctx["cand"], ctx["n_drugs"], ctx["n_dis"]

    # ---- graph for this variant
    data = ctx["data"].clone()
    if cfg["features"] == "real":
        data["drug"].x = torch.tensor(ctx["X_fp"])
    elif cfg["features"] == "ones":
        data["drug"].x = torch.ones((n_drugs, 1))
    else:
        data["drug"].x = torch.tensor((rng.random((n_drugs, FP_SIZE)) < 0.1).astype(np.float32))
    for et in types_to_drop(list(data.edge_types), cfg["drop"]):
        del data[et]

    hide = cand[(cand.label == 1) & cand.split.isin(["val", "test"])]
    hide_keys = np.unique(pair_keys(hide.drug_idx.values, hide.disease_idx.values, n_dis))
    train_g = data.clone()
    for et, drug_first in ((("drug", "indication", "disease"), True), (("disease", "indication", "drug"), False)):
        if et in train_g.edge_types:
            ei = train_g[et].edge_index
            k = pair_keys(ei[0].numpy(), ei[1].numpy(), n_dis) if drug_first else pair_keys(ei[1].numpy(), ei[0].numpy(), n_dis)
            train_g[et].edge_index = ei[:, torch.from_numpy(~np.isin(k, hide_keys))]
    train_g, full_g = train_g.to(device), data.to(device)

    class Net(nn.Module):
        def __init__(self, ets, layers):
            super().__init__()
            self.layers = layers
            self.c1 = HeteroConv({e: SAGEConv((-1, -1), 128) for e in ets}, aggr="sum")
            if layers == 2:
                self.c2 = HeteroConv({e: SAGEConv((-1, -1), 128) for e in ets}, aggr="sum")

        def forward(self, x, ei):
            x = self.c1(x, ei)
            if self.layers == 2:
                x = {k: F.relu(v) for k, v in x.items()}
                x = self.c2(x, ei)
            return x

    model = Net(train_g.edge_types, cfg["layers"]).to(device)
    with torch.no_grad():
        model(train_g.x_dict, train_g.edge_index_dict)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=cfg["weight_decay"])
    loss_fn = nn.MarginRankingLoss(margin=1.0)
    drop = nn.Dropout(cfg["dropout"])

    tr, va, te = (cand[cand.split == s].reset_index(drop=True) for s in ("train", "val", "test"))
    pos_tr, neg_tr = tr[tr.label == 1], tr[tr.label == 0].copy()
    if cfg["neg"] == "random_only":                      # replace hard negatives by fresh random pairs
        hard = neg_tr.is_hard_negative.values == 1
        n_fill = int(hard.sum())
        got = np.empty(0, dtype=np.int64)
        while len(got) < n_fill:
            k = rng.integers(0, n_drugs, n_fill * 2).astype(np.int64) * n_dis + rng.integers(0, n_dis, n_fill * 2)
            got = np.unique(np.concatenate([got, k[~np.isin(k, ctx["known_keys"])]]))
        got = rng.permutation(got)[:n_fill]
        neg_tr.loc[hard, "drug_idx"], neg_tr.loc[hard, "disease_idx"] = got // n_dis, got % n_dis

    T = lambda a: torch.tensor(np.asarray(a), dtype=torch.long, device=device)
    pd_, ps_, nd_, ns_ = T(pos_tr.drug_idx), T(pos_tr.disease_idx), T(neg_tr.drug_idx), T(neg_tr.disease_idx)
    vd, vs = T(va.drug_idx), T(va.disease_idx)
    rep = len(nd_) // len(pd_) + 1

    best, best_state, bad, ep_done = -1.0, None, 0, 0
    for ep in range(1, max_epochs + 1):
        model.train(); opt.zero_grad()
        out = model(train_g.x_dict, train_g.edge_index_dict)
        de, se = drop(out["drug"]), drop(out["disease"])
        pos = (de[pd_] * se[ps_]).sum(-1)
        neg = (de[nd_] * se[ns_]).sum(-1)
        loss = loss_fn(pos.repeat(rep)[:len(neg)], neg, torch.ones_like(neg))
        loss.backward(); opt.step(); ep_done = ep
        if ep % eval_every == 0:
            model.eval()
            with torch.no_grad():
                o = model(train_g.x_dict, train_g.edge_index_dict)
                s = (o["drug"][vd] * o["disease"][vs]).sum(-1).cpu().numpy()
            auprc = average_precision_score(va.label.values, s)
            if auprc > best:
                best, bad = auprc, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= patience:
                    break
    model.load_state_dict(best_state)
    model.eval()

    res = {"variant": name, "seed": seed, "epochs": ep_done, "best_val_auprc": best}
    # ---- internal test (masked graph)
    with torch.no_grad():
        o = model(train_g.x_dict, train_g.edge_index_dict)
        s = (o["drug"][T(te.drug_idx)] * o["disease"][T(te.disease_idx)]).sum(-1).cpu().numpy()
    y = te.label.values
    posm, hardm = y == 1, te.is_hard_negative.values == 1
    for nm, m in (("all", np.ones(len(y), bool)), ("hard", posm | (~posm & hardm)), ("random", posm | (~posm & ~hardm))):
        res[f"test_auroc_{nm}"] = float(roc_auc_score(y[m], s[m]))
    res["test_auprc_all"] = float(average_precision_score(y, s))
    # ---- temporal (full graph)
    with torch.no_grad():
        o = model(full_g.x_dict, full_g.edge_index_dict)
        de, se = o["drug"], o["disease"]
    all_dis = torch.arange(n_dis, device=device)

    def score_all(d):
        with torch.no_grad():
            return (de[d].unsqueeze(0) * se[all_dis]).sum(-1).cpu().numpy().astype(float)

    res.update(temporal_metrics(score_all, ctx["pos"], ctx["t0_by_drug"], ctx["deg_d"], n_dis))
    del model, train_g, full_g, data, o
    gc.collect()
    return res


# ------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--candidates", default=os.path.join(PROCESSED, "candidate_pairs.csv"))
    ap.add_argument("--positives", default=os.path.join("output", "exp2_positives.csv"))
    ap.add_argument("--max-epochs", type=int, default=100)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--out-dir", default="output")
    args = ap.parse_args()
    unknown = [v for v in args.variants if v not in VARIANTS]
    if unknown:
        raise SystemExit(f"unknown variants {unknown}; choose from {list(VARIANTS)}")

    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)
    cand = pd.read_csv(args.candidates)
    pos = pd.read_csv(args.positives)
    pos = pos[pos.config.isin(CFGS.values())].reset_index(drop=True)
    if pos.empty:
        raise SystemExit(f"{args.positives} has none of {list(CFGS.values())}")

    ent = pd.read_csv(os.path.join(PROCESSED, "entities.csv"), dtype={"node_id": str})
    fps = np.load(os.path.join(PROCESSED, "drug_fingerprints.npy"), allow_pickle=True).item()
    data = torch.load(os.path.join(PROCESSED, "pharmagraph_hetero_t0.pt"), weights_only=False)
    n_drugs, n_dis = data["drug"].num_nodes, data["disease"].num_nodes
    dn = ent[ent.node_type == "drug"].sort_values("local_idx")
    X_fp = np.zeros((n_drugs, FP_SIZE), dtype=np.float32)
    for nid, li in zip(dn["node_id"], dn["local_idx"]):
        if nid in fps:
            X_fp[li] = fps[nid]
    ei = data["drug", "indication", "disease"].edge_index.numpy()
    ci = data["drug", "contraindication", "disease"].edge_index.numpy()
    t0_by_drug = {}
    for a, b in zip(ei[0].tolist(), ei[1].tolist()):
        t0_by_drug.setdefault(a, set()).add(b)
    ctx = {"device": device, "cand": cand, "n_drugs": n_drugs, "n_dis": n_dis, "data": data, "X_fp": X_fp,
           "pos": pos, "t0_by_drug": t0_by_drug, "deg_d": np.bincount(ei[0], minlength=n_drugs),
           "known_keys": np.unique(np.concatenate([pair_keys(ei[0], ei[1], n_dis), pair_keys(ci[0], ci[1], n_dis),
                                                   pair_keys(cand.drug_idx, cand.disease_idx, n_dis)]))}

    runs_path = os.path.join(args.out_dir, "ablation_runs.csv")
    done = pd.read_csv(runs_path) if os.path.exists(runs_path) else pd.DataFrame()
    done_keys = set(zip(done.variant, done.seed)) if len(done) else set()
    for seed in args.seeds:
        for v in args.variants:
            if (v, seed) in done_keys:
                print(f"skip {v} seed {seed} (already in {runs_path})")
                continue
            t = time.time()
            print(f"\n=== variant {v} | seed {seed} ===")
            r = run_variant(v, seed, ctx, args.max_epochs, args.eval_every, args.patience)
            print(f"  epochs {r['epochs']} | val AUPRC {r['best_val_auprc']:.4f} | test AUROC all/hard/random "
                  f"{r['test_auroc_all']:.4f}/{r['test_auroc_hard']:.4f}/{r['test_auroc_random']:.4f} | "
                  f"temporal within-drug AUC {r.get('p1_within_auc', float('nan')):.4f} | {time.time() - t:.0f}s")
            done = pd.concat([done, pd.DataFrame([r])], ignore_index=True)
            done.to_csv(runs_path, index=False)

    # ---- summary
    metrics = [c for c in done.columns if c not in ("variant", "seed", "epochs")]
    order = [v for v in VARIANTS if v in set(done.variant)]
    g = done.groupby("variant")[metrics]
    mean, std = g.mean().loc[order], g.std().loc[order]
    summ = mean.add_suffix("_mean").join(std.add_suffix("_std"))
    summ.insert(0, "n_seeds", done.groupby("variant").size().loc[order])
    summ.to_csv(os.path.join(args.out_dir, "ablation_summary.csv"))
    show = ["test_auroc_all", "test_auroc_hard", "test_auroc_random", "p1_within_auc", "p1_hits100",
            "p1_cold_within_auc", "p4_within_auc"]
    show = [m for m in show if m in mean.columns]
    print("\n" + "=" * 120)
    print("ABLATION (mean +- std over seeds)   [difference vs 'full' in brackets]")
    print("=" * 120)
    for v in order:
        parts = []
        for m in show:
            d = mean.loc[v, m] - mean.loc["full", m] if "full" in mean.index else np.nan
            parts.append(f"{m}={mean.loc[v, m]:.4f}+-{std.loc[v, m]:.4f}" + (f"[{d:+.4f}]" if v != "full" and d == d else ""))
        print(f"{v:<24} n={int(summ.loc[v, 'n_seeds'])} " + " ".join(parts))
    print("\nA component matters only if its removal moves a metric by more than the seed-to-seed std.")
    print(f"Saved {runs_path} and ablation_summary.csv")


if __name__ == "__main__":
    main()