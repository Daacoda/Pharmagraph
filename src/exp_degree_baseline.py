"""
Degree-only baselines: how much of the GNN's AUROC/AUPRC can node degree alone explain?

Read-only: no GNN, no retraining, no existing file is modified. Degrees are computed on the
same leak-free message-passing graph the GNN sees (val/test indication edges hidden), and the
baselines are scored with exactly the same imbalance protocol as exp1_class_imbalance.py.

Baselines
  PA_indication : log(1+train indication degree of drug) + log(1+... of disease)   (unsupervised)
  PA_total      : same, but degree over all relations in the graph                 (unsupervised)
  HGB_degree    : gradient boosting on 6 degree features, fitted on the TRAIN pairs only

Run from the project root (after exp1_class_imbalance.py has produced its summary):
    py src/exp_degree_baseline.py
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

import exp1_class_imbalance as e1   # reuse the exact same evaluation harness

PROCESSED = "processed"


def degree_arrays(data, n_drug, n_dis):
    tot_drug = np.zeros(n_drug)
    tot_dis = np.zeros(n_dis)
    for et in data.edge_types:
        src, _, _ = et
        ei = data[et].edge_index[0].numpy()
        if src == "drug":
            tot_drug += np.bincount(ei, minlength=n_drug)
        elif src == "disease":
            tot_dis += np.bincount(ei, minlength=n_dis)
    ind = data["drug", "indication", "disease"].edge_index.numpy()
    con = data["drug", "contraindication", "disease"].edge_index.numpy()
    return dict(
        tot_drug=tot_drug, tot_dis=tot_dis,
        ind_drug=np.bincount(ind[0], minlength=n_drug), ind_dis=np.bincount(ind[1], minlength=n_dis),
        con_drug=np.bincount(con[0], minlength=n_drug), con_dis=np.bincount(con[1], minlength=n_dis))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default=os.path.join(PROCESSED, "candidate_pairs.csv"))
    ap.add_argument("--gnn-summary", default=os.path.join("output", "exp1_class_imbalance_summary.csv"))
    ap.add_argument("--out-dir", default="output")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cand = pd.read_csv(args.candidates)
    data = torch.load(os.path.join(PROCESSED, "pharmagraph_hetero_t0.pt"), weights_only=False)
    n_drug, n_dis = data["drug"].num_nodes, data["disease"].num_nodes

    # contraindication keys (excluded from fresh random negatives, same as exp1)
    c = data["drug", "contraindication", "disease"].edge_index.numpy()
    contra_keys = e1.pair_keys(c[0], c[1], n_dis)

    # hide val/test positive indication edges in both directions (same as training)
    held = cand[(cand.label == 1) & cand.split.isin(["val", "test"])]
    hide = np.unique(e1.pair_keys(held.drug_idx.values, held.disease_idx.values, n_dis))
    for et, drug_first in [(("drug", "indication", "disease"), True), (("disease", "indication", "drug"), False)]:
        ei = data[et].edge_index
        d, s = (ei[0], ei[1]) if drug_first else (ei[1], ei[0])
        keep = ~np.isin(e1.pair_keys(d.numpy(), s.numpy(), n_dis), hide)
        data[et].edge_index = ei[:, torch.from_numpy(keep)]
    print("Indication edges kept:", data["drug", "indication", "disease"].edge_index.shape[1], "(expect 6571)")

    deg = degree_arrays(data, n_drug, n_dis)
    L = np.log1p

    def pa_ind(d, s):
        return L(deg["ind_drug"][d]) + L(deg["ind_dis"][s])

    def pa_tot(d, s):
        return L(deg["tot_drug"][d]) + L(deg["tot_dis"][s])

    def feats(d, s):
        d, s = np.asarray(d), np.asarray(s)
        return np.column_stack([L(deg["tot_drug"][d]), L(deg["tot_dis"][s]),
                                L(deg["ind_drug"][d]), L(deg["ind_dis"][s]),
                                L(deg["con_drug"][d]), L(deg["con_dis"][s])])

    tr = cand[cand.split == "train"]
    hgb = HistGradientBoostingClassifier(max_iter=200, random_state=args.seed)
    hgb.fit(feats(tr.drug_idx.values, tr.disease_idx.values), tr.label.values)

    def hgb_score(d, s):
        p = np.clip(hgb.predict_proba(feats(d, s))[:, 1], 1e-12, 1 - 1e-12)
        return np.log(p / (1 - p))                      # logit: keeps ranking, avoids saturation

    excluded = np.unique(np.concatenate([
        e1.pair_keys(cand.drug_idx.values, cand.disease_idx.values, n_dis), contra_keys]))
    val = cand[cand.split == "val"]
    test_base = cand[cand.split == "test"].reset_index(drop=True)

    all_rows = []
    for name, fn in [("PA_indication", pa_ind), ("PA_total", pa_tot), ("HGB_degree", hgb_score)]:
        test = test_base.copy()
        test["score"] = fn(test.drug_idx.values, test.disease_idx.values)
        thr = e1.tune_threshold(val.label.values, fn(val.drug_idx.values, val.disease_idx.values))
        res = e1.run_experiment(test, fn, n_drug, n_dis, excluded, thr, args.repeats, args.seed)
        res["model"] = name
        all_rows.append(res)
        print(f"{name}: full 1:2 test AUROC = {roc_auc_score(test.label, test.score):.4f}")

    long_df = pd.concat(all_rows, ignore_index=True)
    os.makedirs(args.out_dir, exist_ok=True)
    long_df.to_csv(os.path.join(args.out_dir, "degree_baseline_long.csv"), index=False)
    summ = (long_df.groupby(["model", "neg_mode", "ratio", "ratio_label"])[["auroc", "auprc"]]
            .mean().reset_index().rename(columns={"auroc": "auroc_mean", "auprc": "auprc_mean"}))

    if os.path.exists(args.gnn_summary):               # put the GNN next to the baselines
        g = pd.read_csv(args.gnn_summary)[["neg_mode", "ratio", "ratio_label", "auroc_mean", "auprc_mean"]]
        g.insert(0, "model", "GNN_v3")
        summ = pd.concat([summ, g], ignore_index=True)
    else:
        print(f"(no GNN summary at {args.gnn_summary}; run exp1_class_imbalance.py first to compare)")

    summ.to_csv(os.path.join(args.out_dir, "degree_baseline_summary.csv"), index=False)
    for metric in ["auroc_mean", "auprc_mean"]:
        pv = summ.pivot_table(index=["neg_mode", "ratio"], columns="model", values=metric)
        print(f"\n{metric}")
        print(pv.round(4).to_string())
    print("\nSaved degree_baseline_long.csv and degree_baseline_summary.csv in", args.out_dir)


if __name__ == "__main__":
    main()