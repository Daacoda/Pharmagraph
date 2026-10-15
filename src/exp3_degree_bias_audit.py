"""
exp3_degree_bias_audit.py
=========================
EXPERIMENT 3 - degree-bias audit. EVALUATION ONLY: loads the trained v3
checkpoint, trains nothing, overwrites nothing (writes output/exp3_*.csv only).

Question: how much of PharmaGraph's performance is explained by node popularity
(how many indications a drug / disease already has), and does the model add
anything beyond it where popularity is uninformative?

Part A - internal test split (the 0.9375 AUROC number)
    Same masked graph and held-out test pairs as training. Compares the model with
    popularity-only scorers (drug degree, disease degree, their sum), all computed
    from the edges the model could see (val/test indications hidden).
    Reported on: all negatives / hard negatives only (contraindications) / random
    negatives only.

Part B - temporal Open Targets positives (from output/exp2_positives.csv)
    Per-drug ranking, the standard repurposing question: "given this drug, where
    does the new true disease rank among ALL diseases?" Filtered setting: the drug's
    other known/new positives are removed from the candidate list, so only unknown
    diseases compete. This removes the drug-popularity shortcut by construction
    (a drug-level score cancels inside one drug's ranking).
    Scorers: model | disease_popularity (disease degree in t0, no learning) | random.
    Metrics: Hits@10/50/100, MRR, median rank, mean within-drug AUC
             ((n_cand - rank) / (n_cand - 1); 0.5 = random).
    Stratified by the drug's t0 indication degree (0 / 1-2 / 3+) - drugs with 0
    indications are the genuine repurposing case - and by disease degree.
    Ties get the average rank (matters for the popularity baseline).

Run from project root, after exp2 (needs output/exp2_positives.csv):
    py src\\exp3_degree_bias_audit.py
"""
import argparse
import os

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

PROCESSED = "processed"
BUCKETS = [(-1, 0, "0"), (0, 2, "1-2"), (2, 10**9, "3+")]


def bucket(deg):
    for lo, hi, name in BUCKETS:
        if lo < deg <= hi:
            return name
    return "0"


# ------------------------------------------------------------------ Part A
def internal_audit(test, score, deg_d, deg_s):
    """test: DataFrame(drug_idx, disease_idx, label, is_hard_negative); score: raw model score."""
    d, s = test.drug_idx.values, test.disease_idx.values
    scorers = {
        "model": score,
        "drug_degree": np.log1p(deg_d[d]).astype(float),
        "disease_degree": np.log1p(deg_s[s]).astype(float),
        "degree_sum": (np.log1p(deg_d[d]) + np.log1p(deg_s[s])).astype(float),
    }
    rows = []
    pos = test.label.values == 1
    hard = test.is_hard_negative.values == 1
    subsets = {"all_negatives": np.ones(len(test), bool),
               "hard_negatives_only": pos | (~pos & hard),
               "random_negatives_only": pos | (~pos & ~hard)}
    for sub, mask in subsets.items():
        y = test.label.values[mask]
        for name, sc in scorers.items():
            rows.append({"subset": sub, "scorer": name, "n_pos": int(y.sum()),
                         "n_neg": int((1 - y).sum()),
                         "auroc": float(roc_auc_score(y, sc[mask])),
                         "auprc": float(average_precision_score(y, sc[mask])),
                         "prevalence": float(y.mean())})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ Part B
def ranks_for_drug(scores, excluded_mask, targets):
    """Average-tie rank of each target disease among {non-excluded diseases} U {target}.
    scores: (n_dis,) ; excluded_mask: (n_dis,) bool, True for diseases removed
    (includes the targets themselves); targets: int array.
    Returns (ranks, n_cand)."""
    base = np.sort(scores[~excluded_mask])
    st = scores[targets]
    left = np.searchsorted(base, st, side="left")
    right = np.searchsorted(base, st, side="right")
    greater = len(base) - right
    equal = right - left
    ranks = 1.0 + greater + 0.5 * equal
    return ranks, len(base) + 1


def summarise(q):
    """q: per-query DataFrame with rank, n_cand -> metric dict."""
    r = q["rank"].values
    n = q["n_cand"].values
    return {"n_queries": len(q), "n_drugs": q["drug_idx"].nunique(),
            "hits@10": float((r <= 10).mean()), "hits@50": float((r <= 50).mean()),
            "hits@100": float((r <= 100).mean()),
            "mrr": float((1.0 / r).mean()), "median_rank": float(np.median(r)),
            "within_drug_auc": float(((n - r) / np.maximum(n - 1, 1)).mean()),
            "median_n_cand": float(np.median(n))}


def temporal_ranking(positives, score_all_diseases, deg_d, deg_s, t0_pairs_by_drug,
                     n_dis, seed=42):
    """positives: DataFrame(config, drug_idx, disease_idx). Returns per-query DataFrame."""
    rng = np.random.default_rng(seed)
    union = positives.groupby("drug_idx")["disease_idx"].apply(lambda x: set(x.tolist())).to_dict()
    dis_pop = np.log1p(deg_s).astype(float)
    out = []
    drugs = sorted(positives.drug_idx.unique())
    for i, d in enumerate(drugs, 1):
        known = set(t0_pairs_by_drug.get(d, ())) | union[d]
        excl = np.zeros(n_dis, bool)
        excl[list(known)] = True
        model_s = score_all_diseases(d)
        scorers = {"model": model_s, "disease_popularity": dis_pop, "random": rng.random(n_dis)}
        sub = positives[positives.drug_idx == d]
        for cfg, g in sub.groupby("config"):
            targets = g.disease_idx.values
            for name, sc in scorers.items():
                ranks, n_cand = ranks_for_drug(sc, excl, targets)
                for t, rk in zip(targets, ranks):
                    out.append((cfg, name, d, int(t), float(rk), n_cand,
                                bucket(deg_d[d]), bucket(deg_s[t])))
        if i % 200 == 0:
            print(f"    ranked {i}/{len(drugs)} drugs")
    return pd.DataFrame(out, columns=["config", "scorer", "drug_idx", "disease_idx", "rank",
                                      "n_cand", "drug_bucket", "disease_bucket"])


def summarise_all(q):
    rows = []
    for by in ("drug_bucket", "disease_bucket"):
        for (cfg, sc, b), g in q.groupby(["config", "scorer", by]):
            rows.append({"config": cfg, "scorer": sc, "stratum": by, "bucket": b, **summarise(g)})
        for (cfg, sc), g in q.groupby(["config", "scorer"]):
            rows.append({"config": cfg, "scorer": sc, "stratum": by, "bucket": "ALL", **summarise(g)})
    return pd.DataFrame(rows).drop_duplicates(subset=["config", "scorer", "stratum", "bucket"])


# --------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default=os.path.join("output", "models", "graphsage_v3_real_features.pt"))
    ap.add_argument("--candidates", default=os.path.join(PROCESSED, "candidate_pairs.csv"))
    ap.add_argument("--positives", default=os.path.join("output", "exp2_positives.csv"))
    ap.add_argument("--out-dir", default="output")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import torch
    from exp1_class_imbalance import load_scorer, pair_keys

    cand = pd.read_csv(args.candidates)
    pos_all = pd.read_csv(args.positives)

    # ---------------- Part A: internal split, masked graph (as in training)
    print("Part A - internal test split vs popularity baselines")
    score_masked, n_drugs, n_dis, _, _ = load_scorer(args, cand)
    data = torch.load(os.path.join(PROCESSED, "pharmagraph_hetero_t0.pt"), weights_only=False)
    ei = data["drug", "indication", "disease"].edge_index.numpy()

    hide = cand[(cand.label == 1) & cand.split.isin(["val", "test"])]
    hide_keys = np.unique(pair_keys(hide.drug_idx.values, hide.disease_idx.values, n_dis))
    keep = ~np.isin(pair_keys(ei[0], ei[1], n_dis), hide_keys)
    deg_d_m = np.bincount(ei[0][keep], minlength=n_drugs)
    deg_s_m = np.bincount(ei[1][keep], minlength=n_dis)

    test = cand[cand.split == "test"].reset_index(drop=True)
    sc = score_masked(test.drug_idx.values, test.disease_idx.values)
    A = internal_audit(test, sc, deg_d_m, deg_s_m)
    os.makedirs(args.out_dir, exist_ok=True)
    A.to_csv(os.path.join(args.out_dir, "exp3_internal_baseline.csv"), index=False)
    print(A.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # ---------------- Part B: temporal per-drug ranking, FULL t0 graph
    print("\nPart B - temporal per-drug ranking (Open Targets positives, full t0 graph)")
    score_full, _, _, _, _ = load_scorer(args, cand.iloc[0:0])
    deg_d = np.bincount(ei[0], minlength=n_drugs)
    deg_s = np.bincount(ei[1], minlength=n_dis)
    t0_by_drug = {}
    for a, b in zip(ei[0].tolist(), ei[1].tolist()):
        t0_by_drug.setdefault(a, set()).add(b)
    all_dis = np.arange(n_dis)

    def score_all_diseases(d):
        return score_full(np.full(n_dis, d), all_dis).astype(float)

    q = temporal_ranking(pos_all, score_all_diseases, deg_d, deg_s, t0_by_drug, n_dis, args.seed)
    q.to_csv(os.path.join(args.out_dir, "exp3_per_drug_ranking.csv"), index=False)
    S = summarise_all(q)
    S.to_csv(os.path.join(args.out_dir, "exp3_summary.csv"), index=False)

    cols = ["config", "scorer", "bucket", "n_queries", "n_drugs", "hits@10", "hits@100",
            "mrr", "median_rank", "within_drug_auc"]
    for by in ("drug_bucket", "disease_bucket"):
        print("\n" + "=" * 110)
        print(f"Per-drug ranking, stratified by {by} (t0 indication degree: 0 / 1-2 / 3+)")
        print("=" * 110)
        show = S[S.stratum == by][cols].sort_values(["config", "bucket", "scorer"])
        print(show.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    print("\nReading guide: within_drug_auc 0.5 = random. disease_popularity is the no-learning baseline; "
          "the model adds value only where it beats it, especially in drug_bucket '0' (cold drugs).")
    print("Saved output/exp3_internal_baseline.csv, exp3_per_drug_ranking.csv, exp3_summary.csv")


if __name__ == "__main__":
    main()