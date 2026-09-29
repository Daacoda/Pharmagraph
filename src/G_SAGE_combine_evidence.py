"""
combine_evidence.py
Stage 3, Steps 2-5 - bridges Open Targets to drug-level, merges everything
(model score + OT + PPI/GO), builds BOTH ranking options, and evaluates all
of them against the model-alone baseline so you get a direct before/after
comparison.

Requires (all produced by earlier stages):
    processed/candidate_pairs.csv
    processed/model_scores.parquet (G_SAGE_model_predict.py)
    processed/ppi_go_features.parquet
    processed/ot_scores.parquet      (from fetch_open_targets.py)
    processed/entities.csv
    processed/pharmagraph_hetero_t0.pt

Run from the project root:
    python src/G_SAGE_combine_evidence.py
"""
import os

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score

processed_dir = "processed"
FEATURE_COLS = ["model_prob", "ot_score", "ppi_min_hops", "ppi_common_neighbors",
                 "go_jaccard_bp", "go_jaccard_mf", "go_jaccard_cc"]


def bridge_ot_to_drug_level():
    """Open Targets scores are (gene, disease). Bridges to (drug, disease) via
    each drug's known targets, taking the MAX score across all of a drug's
    targets - one strong signal should count, not get diluted by averaging
    with weak ones (same reasoning as the GO Jaccard aggregation)."""
    entities = pd.read_csv(os.path.join(processed_dir, "entities.csv"), dtype={"node_id": str})
    gene_map = dict(zip(entities.loc[entities.node_type == "gene/protein", "node_id"],
                         entities.loc[entities.node_type == "gene/protein", "local_idx"]))
    disease_map = dict(zip(entities.loc[entities.node_type == "disease", "node_id"],
                            entities.loc[entities.node_type == "disease", "local_idx"]))

    ot = pd.read_parquet(os.path.join(processed_dir, "ot_scores.parquet"))
    ot["gene_local_idx"] = ot["gene_node_id"].astype(str).map(gene_map)
    ot["disease_local_idx"] = ot["disease_node_id"].astype(str).map(disease_map)
    n_before = len(ot)
    ot = ot.dropna(subset=["gene_local_idx", "disease_local_idx"])
    ot["gene_local_idx"] = ot["gene_local_idx"].astype(int)
    ot["disease_local_idx"] = ot["disease_local_idx"].astype(int)
    print(f"OT scores mapped to local ids: {len(ot)}/{n_before} rows kept")

    data = torch.load(os.path.join(processed_dir, "pharmagraph_hetero_t0.pt"), weights_only=False)
    ei = data["drug", "drug_protein", "gene/protein"].edge_index
    drug_targets = {}
    for d, g in zip(ei[0].tolist(), ei[1].tolist()):
        drug_targets.setdefault(d, set()).add(g)

    ot_lookup = ot.groupby(["gene_local_idx", "disease_local_idx"])["ot_score"].max().to_dict()

    def bridged_score(drug_idx, disease_idx):
        best = np.nan
        for g in drug_targets.get(drug_idx, ()):
            s = ot_lookup.get((g, disease_idx))
            if s is not None and (np.isnan(best) or s > best):
                best = s
        return best

    return drug_targets, bridged_score


def evaluate(name, labels, scores, split_mask):
    l, s = labels[split_mask], scores[split_mask]
    if len(np.unique(l)) < 2:
        print(f"  {name}: skipped (only one class present)")
        return
    print(f"  {name}: AUROC={roc_auc_score(l, s):.4f}  AUPRC={average_precision_score(l, s):.4f}")


def main():
    pairs = pd.read_parquet(os.path.join(processed_dir, "model_scores.parquet"))
    ppi_go = pd.read_parquet(os.path.join(processed_dir, "ppi_go_features.parquet"))
    df = pairs.merge(ppi_go, on=["drug_idx", "disease_idx"], how="left")

    print("Bridging Open Targets to drug-level (max over each drug's known targets)...")
    _, bridged_score = bridge_ot_to_drug_level()
    df["ot_score"] = [bridged_score(d, dis) for d, dis in zip(df.drug_idx, df.disease_idx)]
    df["has_ot_score"] = df["ot_score"].notna().astype(int)
    print(f"Pairs with a real OT score: {df['has_ot_score'].sum()}/{len(df)} "
          f"({df['has_ot_score'].mean()*100:.1f}%)")

    # fill missing evidence with neutral values, keep the has_ot_score flag
    # as its own feature since "no evidence found" can itself be informative
    df["ot_score"] = df["ot_score"].fillna(0.0)
    df["ppi_min_hops"] = df["ppi_min_hops"].fillna(df["ppi_min_hops"].max())
    for col in ["ppi_common_neighbors", "go_jaccard_bp", "go_jaccard_mf", "go_jaccard_cc"]:
        df[col] = df[col].fillna(0.0)

    train_mask = (df.split == "train").values
    val_mask = (df.split == "val").values
    test_mask = (df.split == "test").values
    labels = df.label.values

    print("\n" + "=" * 60)
    print("BASELINE: model alone")
    print("=" * 60)
    evaluate("Val ", labels, df.model_prob.values, val_mask)
    evaluate("Test", labels, df.model_prob.values, test_mask)

    # ---------------- Option A: weighted formula ----------------
    print("\n" + "=" * 60)
    print("OPTION A: weighted formula")
    print("=" * 60)

    def normalize(col):
        v = df[col].values.astype(float)
        rng = v.max() - v.min()
        return (v - v.min()) / rng if rng > 0 else v * 0

    norm = {c: normalize(c) for c in FEATURE_COLS if c != "model_prob"}
    norm["model_prob"] = df["model_prob"].values  # already 0-1
    # ppi_min_hops: smaller is better, so invert after normalizing
    norm["ppi_min_hops"] = 1 - norm["ppi_min_hops"]

    best_weights, best_auprc = None, -1
    grid = [0.0, 0.1, 0.2, 0.3]
    for w_ot in grid:
        for w_ppi in grid:
            for w_go in grid:
                w_model = 1 - w_ot - w_ppi - w_go
                if w_model < 0.3:  # keep the model as the dominant signal
                    continue
                combined = (w_model * norm["model_prob"] + w_ot * norm["ot_score"]
                            + w_ppi * (norm["ppi_min_hops"] + norm["ppi_common_neighbors"]) / 2
                            + w_go * (norm["go_jaccard_bp"] + norm["go_jaccard_mf"] + norm["go_jaccard_cc"]) / 3)
                if len(np.unique(labels[val_mask])) < 2:
                    continue
                a = average_precision_score(labels[val_mask], combined[val_mask])
                if a > best_auprc:
                    best_auprc, best_weights = a, (w_model, w_ot, w_ppi, w_go)

    w_model, w_ot, w_ppi, w_go = best_weights
    print(f"Best weights on val (grid search): model={w_model:.2f} ot={w_ot:.2f} "
          f"ppi={w_ppi:.2f} go={w_go:.2f}")
    combined_a = (w_model * norm["model_prob"] + w_ot * norm["ot_score"]
                  + w_ppi * (norm["ppi_min_hops"] + norm["ppi_common_neighbors"]) / 2
                  + w_go * (norm["go_jaccard_bp"] + norm["go_jaccard_mf"] + norm["go_jaccard_cc"]) / 3)
    df["score_option_a"] = combined_a
    evaluate("Val ", labels, combined_a, val_mask)
    evaluate("Test", labels, combined_a, test_mask)

    # ---------------- Option B: learned re-ranker ----------------
    print("\n" + "=" * 60)
    print("OPTION B: logistic regression re-ranker")
    print("=" * 60)
    X = df[FEATURE_COLS].values
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X[train_mask], labels[train_mask])
    combined_b = clf.predict_proba(X)[:, 1]
    df["score_option_b"] = combined_b
    print("Learned feature weights:", dict(zip(FEATURE_COLS, clf.coef_[0].round(3))))
    evaluate("Val ", labels, combined_b, val_mask)
    evaluate("Test", labels, combined_b, test_mask)

    output_dir = "output"
    out_path = os.path.join(output_dir, "final_evidence_ranking.parquet")
    df.to_parquet(out_path, index=False)
    print(f"\nSaved {out_path}: {len(df)} rows, columns: {list(df.columns)}")


if __name__ == "__main__":
    main()
