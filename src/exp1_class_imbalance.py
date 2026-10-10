"""
exp1_class_imbalance.py
=======================
EXPERIMENT 1 - class-imbalance robustness (KG-Bench-style), EVALUATION ONLY.

Reference concept: Wei et al., "KG-Bench: benchmarking graph neural network
algorithms for drug repurposing", Bioinformatics 42(5), btag159, 2026 - they
train/validate at 1:1 and evaluate the test set at 1:1, 1:10 and 1:100
(positive:negative), reporting AUC and F1.

What this script does
---------------------
* Loads the ALREADY-TRAINED v3 checkpoint. No training, no weight updates.
* Rebuilds the exact leak-free message-passing graph used at train/test time
  (val + test indication edges masked out, both directions) so every score is
  produced the same way as in training. A sanity check compares the AUROC on the
  full held-out test split against the AUROC recorded inside the checkpoint.
* Keeps the SAME held-out test positives for every ratio and only changes the
  number/composition of negatives.
* Writes NEW files only (output/exp1_class_imbalance_*.csv). Nothing existing is
  modified or overwritten.

Two negative-pool modes (both reported)
---------------------------------------
random    : negatives are fresh random drug-disease pairs that are NOT known
            indications, NOT known contraindications and NOT any pair that already
            appears in candidate_pairs.csv (train/val/test). This is the closest
            analogue of KG-Bench's random non-existent-edge negatives.
protocol  : PharmaGraph's own test negatives first (half contraindication hard
            negatives + half random, as in the 1:2 protocol). The held-out test
            split only has ~2.8k negatives, so for 1:10 and 1:100 the remainder
            is filled with fresh random pairs (same exclusions as above).
            Hard negatives are NEVER taken from train/val, because the model was
            trained on them (that would be leakage). The n_hard_neg /
            n_random_neg columns record the exact composition of every row.

Metrics
-------
AUROC and AUPRC are threshold-free. Precision/recall/F1 need a threshold, so two
are reported:
  *_t05  : fixed decision threshold sigmoid(score) >= 0.5  (i.e. raw score >= 0),
           the usual fixed-threshold convention. Note our training used a margin
           ranking loss, so raw scores are NOT calibrated probabilities.
  *_tval : the F1-optimal threshold chosen ONCE on the validation split
           (1:2 composition) and then frozen for every ratio - this shows what
           happens to precision when prevalence drops but the threshold does not.
random_auprc_baseline = positive prevalence (what a random ranker scores).

Run from the project root (needs processed/ and output/models/ from earlier steps):
    python src/exp1_class_imbalance.py
    python src/exp1_class_imbalance.py --repeats 10 --seed 7
"""
import argparse
import hashlib
import os

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, precision_recall_curve,
                             roc_auc_score)

PROCESSED = "processed"
FP_SIZE = 256
RATIOS = [1, 2, 10, 100]          # negatives per positive  ->  1:1, 1:2, 1:10, 1:100


# --------------------------------------------------------------------------- utils
def pair_keys(drug, dis, n_dis):
    """Encode (drug, disease) pairs as single int64 keys for fast set operations."""
    return np.asarray(drug, dtype=np.int64) * int(n_dis) + np.asarray(dis, dtype=np.int64)


def sample_random_pairs(n, n_drugs, n_dis, excluded_keys, rng):
    """n unique uniformly-random (drug, disease) keys that are not in excluded_keys."""
    got = np.empty(0, dtype=np.int64)
    while len(got) < n:
        need = int((n - len(got)) * 1.3) + 1000
        keys = rng.integers(0, n_drugs, size=need).astype(np.int64) * n_dis \
            + rng.integers(0, n_dis, size=need).astype(np.int64)
        keys = keys[~np.isin(keys, excluded_keys)]
        got = np.unique(np.concatenate([got, keys]))
    return rng.permutation(got)[:n]          # shuffle: np.unique sorts, keep it random


def set_hash(drug, dis, label):
    """Order-independent fingerprint of an evaluation set (drug, disease, label)."""
    arr = np.stack([np.asarray(drug), np.asarray(dis), np.asarray(label)], axis=1).astype(np.int64)
    arr = arr[np.lexsort((arr[:, 2], arr[:, 1], arr[:, 0]))]
    return hashlib.sha256(arr.tobytes()).hexdigest()[:12]


def threshold_metrics(y, s, thr):
    pred = s >= thr
    tp = int((pred & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return prec, rec, f1


def tune_threshold(y, s):
    """Score threshold that maximises F1 on (y, s). Used on validation only."""
    p, r, t = precision_recall_curve(y, s)
    f1 = 2 * p[:-1] * r[:-1] / np.clip(p[:-1] + r[:-1], 1e-12, None)
    return float(t[int(np.argmax(f1))])


def evaluate_set(y, s, thr_val):
    out = {
        "prevalence": float(y.mean()),
        "auroc": float(roc_auc_score(y, s)),
        "auprc": float(average_precision_score(y, s)),
    }
    out["random_auprc_baseline"] = out["prevalence"]
    out["precision_t05"], out["recall_t05"], out["f1_t05"] = threshold_metrics(y, s, 0.0)
    out["thr_val"] = thr_val
    out["precision_tval"], out["recall_tval"], out["f1_tval"] = threshold_metrics(y, s, thr_val)
    return out


# ------------------------------------------------------------------- the experiment
def run_experiment(test, score_fn, n_drugs, n_dis, excluded_keys, thr_val,
                   repeats, seed, ratios=RATIOS):
    """test: DataFrame (drug_idx, disease_idx, label, is_hard_negative, score).
    score_fn(drug_idx_array, disease_idx_array) -> raw scores (same graph/model as test)."""
    pos = test[test.label == 1]
    neg_test = test[test.label == 0].reset_index(drop=True)
    n_pos = len(pos)
    max_needed = int(round(max(ratios) * n_pos))
    rows = []

    for rep in range(repeats):
        rng = np.random.default_rng(seed + rep)
        pool = sample_random_pairs(max_needed, n_drugs, n_dis, excluded_keys, rng)
        pool_d, pool_s = pool // n_dis, pool % n_dis
        pool_score = score_fn(pool_d, pool_s)
        neg_perm = rng.permutation(len(neg_test))

        for r in ratios:
            needed = int(round(r * n_pos))
            for mode in ("random", "protocol"):
                if mode == "random":
                    nd, ns, nsc = pool_d[:needed], pool_s[:needed], pool_score[:needed]
                    n_hard, n_fresh = 0, needed
                else:
                    if needed <= 1.01 * len(neg_test):          # test split covers it
                        take = neg_test.iloc[neg_perm[:min(needed, len(neg_test))]]
                        extra = 0
                    else:                                       # top up with fresh randoms
                        take = neg_test
                        extra = needed - len(neg_test)
                    nd = np.concatenate([take.drug_idx.values, pool_d[:extra]])
                    ns = np.concatenate([take.disease_idx.values, pool_s[:extra]])
                    nsc = np.concatenate([take.score.values, pool_score[:extra]])
                    n_hard, n_fresh = int(take.is_hard_negative.sum()), extra

                d_all = np.concatenate([pos.drug_idx.values, nd])
                s_all = np.concatenate([pos.disease_idx.values, ns])
                y = np.concatenate([np.ones(n_pos), np.zeros(len(nd))]).astype(int)
                sc = np.concatenate([pos.score.values, nsc])

                row = {
                    "neg_mode": mode, "ratio_label": f"1:{r}", "ratio": r,
                    "repeat": rep, "seed": seed + rep,
                    "n_pos": n_pos, "n_neg": len(nd),
                    "n_hard_neg": n_hard, "n_random_neg": len(nd) - n_hard,
                    "n_fresh_random_neg": n_fresh,
                    "test_hash": set_hash(d_all, s_all, y),
                }
                row.update(evaluate_set(y, sc, thr_val))
                rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------- model / graph (torch imported lazily)
def load_scorer(args, cand):
    """Rebuild the masked message-passing graph, load the checkpoint, return a scorer."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn import HeteroConv, SAGEConv

    class HeteroGraphSAGE(nn.Module):          # identical to G_SAGE_model_training.py
        def __init__(self, node_types, edge_types, hidden_channels=128, out_channels=128):
            super().__init__()
            self.conv1 = HeteroConv({et: SAGEConv((-1, -1), hidden_channels) for et in edge_types}, aggr="sum")
            self.conv2 = HeteroConv({et: SAGEConv((-1, -1), out_channels) for et in edge_types}, aggr="sum")

        def forward(self, x_dict, edge_index_dict):
            x_dict = self.conv1(x_dict, edge_index_dict)
            x_dict = {k: F.relu(v) for k, v in x_dict.items()}
            return self.conv2(x_dict, edge_index_dict)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    entities = pd.read_csv(os.path.join(PROCESSED, "entities.csv"), dtype={"node_id": str})
    fps = np.load(os.path.join(PROCESSED, "drug_fingerprints.npy"), allow_pickle=True).item()
    data = torch.load(os.path.join(PROCESSED, "pharmagraph_hetero_t0.pt"), weights_only=False)

    drug_nodes = entities[entities.node_type == "drug"].sort_values("local_idx")
    X = np.zeros((len(drug_nodes), FP_SIZE), dtype=np.float32)
    for node_id, local_idx in zip(drug_nodes["node_id"], drug_nodes["local_idx"]):
        if node_id in fps:
            X[local_idx] = fps[node_id]
    data["drug"].x = torch.tensor(X, dtype=torch.float)

    n_drugs, n_dis = data["drug"].num_nodes, data["disease"].num_nodes

    # contraindication pairs (for excluding them from the fresh random negatives)
    c_ei = data["drug", "contraindication", "disease"].edge_index
    contra_keys = pair_keys(c_ei[0].numpy(), c_ei[1].numpy(), n_dis)

    # leak-free graph: hide val + test positive indication edges, both directions
    hide = cand[(cand.label == 1) & cand.split.isin(["val", "test"])]
    hide_keys = np.unique(pair_keys(hide.drug_idx.values, hide.disease_idx.values, n_dis))
    f_ei = data["drug", "indication", "disease"].edge_index
    r_ei = data["disease", "indication", "drug"].edge_index
    keep_f = ~np.isin(pair_keys(f_ei[0].numpy(), f_ei[1].numpy(), n_dis), hide_keys)
    keep_r = ~np.isin(pair_keys(r_ei[1].numpy(), r_ei[0].numpy(), n_dis), hide_keys)
    data["drug", "indication", "disease"].edge_index = f_ei[:, torch.from_numpy(keep_f)]
    data["disease", "indication", "drug"].edge_index = r_ei[:, torch.from_numpy(keep_r)]
    print(f"Indication edges kept in message-passing graph: {int(keep_f.sum())} "
          f"(training log reported 6571)")

    data = data.to(device)
    model = HeteroGraphSAGE(data.node_types, data.edge_types).to(device)
    ckpt = torch.load(args.checkpoint, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    with torch.no_grad():
        out = model(data.x_dict, data.edge_index_dict)
        drug_emb, dis_emb = out["drug"], out["disease"]

    def score_fn(drug_idx, dis_idx):
        with torch.no_grad():
            d = torch.as_tensor(np.asarray(drug_idx), dtype=torch.long, device=device)
            s = torch.as_tensor(np.asarray(dis_idx), dtype=torch.long, device=device)
            return (drug_emb[d] * dis_emb[s]).sum(dim=-1).cpu().numpy()

    return score_fn, n_drugs, n_dis, contra_keys, ckpt


# ------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default=os.path.join("output", "models", "graphsage_v3_real_features.pt"))
    ap.add_argument("--candidates", default=os.path.join(PROCESSED, "candidate_pairs.csv"))
    ap.add_argument("--out-dir", default="output")
    ap.add_argument("--repeats", type=int, default=5, help="independent negative draws per ratio")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cand = pd.read_csv(args.candidates)
    needed_cols = {"drug_idx", "disease_idx", "label", "is_hard_negative", "split"}
    if not needed_cols <= set(cand.columns):
        raise SystemExit(f"{args.candidates} must contain columns {sorted(needed_cols)}")

    score_fn, n_drugs, n_dis, contra_keys, ckpt = load_scorer(args, cand)

    cand = cand.copy()
    cand["score"] = score_fn(cand.drug_idx.values, cand.disease_idx.values)
    val = cand[cand.split == "val"]
    test = cand[cand.split == "test"].reset_index(drop=True)

    import torch
    s32 = torch.sigmoid(torch.tensor(test.score.values, dtype=torch.float32)).numpy()
    print("AUROC raw            :", roc_auc_score(test.label, test.score))
    print("AUROC sigmoid float32:", roc_auc_score(test.label, s32))
    print("scores == 1.0 after sigmoid:", int((s32 == 1.0).sum()), "of", len(s32))
    print("raw score range:", test.score.min(), test.score.max())

    # sanity check: reproduce the AUROC stored in the checkpoint on the full test split
    auroc_full = roc_auc_score(test.label, test.score)
    rec = ckpt.get("test_auroc")
    print(f"\nSanity check - full held-out test split ({int((test.label==1).sum())} pos / "
          f"{int((test.label==0).sum())} neg):")
    print(f"  AUROC recomputed here : {auroc_full:.4f}")
    print(f"  AUROC in checkpoint   : {rec if rec is None else round(float(rec), 4)}")
    if rec is not None and abs(auroc_full - float(rec)) > 0.005:
        print("  WARNING: difference > 0.005 - check that this is the final v3 checkpoint and that "
              "candidate_pairs.csv / entities.csv / drug_fingerprints.npy are the ones used in training.")

    thr_val = tune_threshold(val.label.values, val.score.values)
    print(f"  F1-optimal threshold tuned on validation (raw score): {thr_val:.3f}\n")

    excluded = np.unique(np.concatenate([
        pair_keys(cand.drug_idx.values, cand.disease_idx.values, n_dis),   # every pair in train/val/test
        contra_keys,                                                       # all known contraindications
    ]))

    long_df = run_experiment(test, score_fn, n_drugs, n_dis, excluded, thr_val,
                             args.repeats, args.seed)

    os.makedirs(args.out_dir, exist_ok=True)
    long_path = os.path.join(args.out_dir, "exp1_class_imbalance_long.csv")
    long_df.to_csv(long_path, index=False)

    metric_cols = ["auroc", "auprc", "precision_t05", "recall_t05", "f1_t05",
                   "precision_tval", "recall_tval", "f1_tval"]
    g = long_df.groupby(["neg_mode", "ratio", "ratio_label"], sort=True)
    summary = g[metric_cols].agg(["mean", "std"])
    summary.columns = [f"{m}_{s}" for m, s in summary.columns]
    summary = summary.join(g.agg(n_pos=("n_pos", "first"), n_neg=("n_neg", "first"),
                                 n_hard_neg=("n_hard_neg", "first"),
                                 n_random_neg=("n_random_neg", "first"),
                                 prevalence=("prevalence", "mean"),
                                 random_auprc_baseline=("random_auprc_baseline", "mean"),
                                 test_hash_repeat0=("test_hash", "first"))).reset_index()
    summary_path = os.path.join(args.out_dir, "exp1_class_imbalance_summary.csv")
    summary.to_csv(summary_path, index=False)

    show = summary[["neg_mode", "ratio_label", "n_pos", "n_neg", "n_hard_neg", "prevalence",
                    "auroc_mean", "auroc_std", "auprc_mean", "random_auprc_baseline",
                    "f1_t05_mean", "precision_t05_mean", "recall_t05_mean",
                    "f1_tval_mean", "test_hash_repeat0"]]
    print("=" * 100)
    print(f"EXPERIMENT 1 - class imbalance (same checkpoint, no retraining) - mean over {args.repeats} draws")
    print("=" * 100)
    print(show.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nSaved {long_path}\nSaved {summary_path}")
    print("\nReading guide: AUROC should stay roughly flat as imbalance grows; AUPRC and F1 should fall "
          "toward the random baseline (= prevalence). 'random' mode is the KG-Bench-comparable one; "
          "'protocol' keeps PharmaGraph's hard negatives wherever the held-out split has them.")


if __name__ == "__main__":
    main()