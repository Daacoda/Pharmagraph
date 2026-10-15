"""
evaluate_t1_support.py
=======================
Temporal validation (methodology step 9): checks whether drug-disease pairs
the model ranked highly - but that were NOT known indications in the t0
(June 2021) snapshot - turn out to be real indications in the t1 (July 2023)
snapshot. This is the "evidence support rate" metric.

Why this matters: AUROC/AUPRC on the t0 test split only tell us the model is
good at recovering indications that already existed in 2021. This script
asks a stronger question: for pairs the model was NOT trained to know about,
did the real world (as captured by the t1 snapshot) later confirm them?

Requires (already produced by earlier stages):
    processed/entities.csv              (Data_laoder.py)
    processed/model_scores.parquet      (G_SAGE_model_predict.py)
    data/benchmark/snapshots/kg_t1.csv  (July 2023 snapshot)

Run from the project root:
    python src/evaluate_t1_support.py
"""
import argparse
import os

import numpy as np
import pandas as pd

PROCESSED = "processed"
OUTPUT = "output"


def load_t1_indication_pairs(t1_csv_path, chunksize=500_000):
    """Streams kg_t1.csv and returns the set of (drug_node_id, disease_node_id)
    pairs with an 'indication' relation, in either column order."""
    pairs = set()
    usecols = ["x_id", "x_type", "relation", "y_id", "y_type"]
    for chunk in pd.read_csv(t1_csv_path, sep=",", header=0, chunksize=chunksize,
                              dtype=str, usecols=usecols):
        ind = chunk[chunk["relation"] == "indication"]
        for _, row in ind.iterrows():
            if row["x_type"] == "drug" and row["y_type"] == "disease":
                pairs.add((row["x_id"], row["y_id"]))
            elif row["y_type"] == "drug" and row["x_type"] == "disease":
                pairs.add((row["y_id"], row["x_id"]))
    return pairs


def map_pairs_to_local_idx(node_id_pairs, entities):
    """Maps (drug_node_id, disease_node_id) string pairs to (drug_local_idx,
    disease_local_idx) using the same vocabulary the model was trained on.
    Pairs referencing an entity that didn't exist at t0 are dropped (the
    model has no embedding for a node it never saw)."""
    drug_map = dict(zip(
        entities.loc[entities.node_type == "drug", "node_id"],
        entities.loc[entities.node_type == "drug", "local_idx"],
    ))
    disease_map = dict(zip(
        entities.loc[entities.node_type == "disease", "node_id"],
        entities.loc[entities.node_type == "disease", "local_idx"],
    ))
    mapped = set()
    dropped = 0
    for drug_id, disease_id in node_id_pairs:
        d, dis = drug_map.get(drug_id), disease_map.get(disease_id)
        if d is None or dis is None:
            dropped += 1
            continue
        mapped.add((d, dis))
    return mapped, dropped


def evidence_support_rate(candidates, score_col, added_pairs, k_values):
    """For candidates ranked by score_col (descending), what fraction of the
    top-K are confirmed in added_pairs? Also reports the base rate (what
    fraction would be confirmed by chance, i.e. picking K pairs at random)
    so the result can be judged against a baseline."""
    ranked = candidates.sort_values(score_col, ascending=False).reset_index(drop=True)
    is_confirmed = np.array([
        (d, dis) in added_pairs for d, dis in zip(ranked["drug_idx"], ranked["disease_idx"])
    ])
    base_rate = is_confirmed.mean()

    results = {}
    for k in k_values:
        k = min(k, len(ranked))
        results[k] = {
            "support_rate_at_k": float(is_confirmed[:k].mean()) if k > 0 else float("nan"),
            "n_confirmed": int(is_confirmed[:k].sum()),
            "k": k,
        }
    results["base_rate_all_candidates"] = float(base_rate)
    results["n_candidates"] = len(ranked)
    results["n_added_pairs_total"] = len(added_pairs)
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--t1-csv", default=os.path.join("data", "benchmark", "snapshots", "kg_t1.csv"))
    ap.add_argument("--score-col", default="model_prob",
                     help="Column in model_scores.parquet to rank candidates by. "
                          "Use 'score_option_a' or 'score_option_b' to evaluate the "
                          "evidence-combined ranking instead of the model alone.")
    ap.add_argument("--scores-path", default=os.path.join(PROCESSED, "model_scores.parquet"))
    ap.add_argument("--k", type=int, nargs="+", default=[10, 50, 100, 500, 1000])
    args = ap.parse_args()

    os.makedirs(OUTPUT, exist_ok=True)

    print(f"Loading t0 entity vocabulary from {PROCESSED}/entities.csv ...")
    entities = pd.read_csv(os.path.join(PROCESSED, "entities.csv"), dtype={"node_id": str})

    print(f"Streaming t1 snapshot from {args.t1_csv} (this reads the full file once) ...")
    t1_pairs_raw = load_t1_indication_pairs(args.t1_csv)
    print(f"  Found {len(t1_pairs_raw)} drug-disease indication pairs in t1.")

    t1_pairs, dropped = map_pairs_to_local_idx(t1_pairs_raw, entities)
    print(f"  Mapped to t0 vocabulary: {len(t1_pairs)} kept, {dropped} dropped "
          f"(reference a drug/disease that didn't exist at t0).")

    print(f"\nLoading scored candidates from {args.scores_path} ...")
    scores = pd.read_parquet(args.scores_path)
    if args.score_col not in scores.columns:
        raise SystemExit(
            f"Column '{args.score_col}' not found in {args.scores_path}. "
            f"Available columns: {list(scores.columns)}"
        )

    # known_at_t0: pairs already labeled positive (label==1) are not a fair
    # test of temporal validation - the model already knew about them during
    # training/evaluation. We only care about candidates that were UNKNOWN
    # at t0 (label==0: hard negatives + random negatives + any unlabeled
    # candidate), since confirming one of those in t1 is a genuine new find.
    unknown_at_t0 = scores[scores["label"] == 0].copy()
    print(f"Candidates unknown at t0 (label=0): {len(unknown_at_t0)} / {len(scores)}")

    added_pairs = t1_pairs  # already excludes anything not in the t0 vocabulary;
    # pairs known at t0 are a separate (label==1) set in `scores` and are not
    # part of `unknown_at_t0`, so no further filtering is needed here.

    print(f"\nRanking by '{args.score_col}' and checking against t1-confirmed pairs ...")
    results = evidence_support_rate(unknown_at_t0, args.score_col, added_pairs, args.k)

    print("\n" + "=" * 60)
    print(f"TEMPORAL VALIDATION (evidence support rate) - ranked by {args.score_col}")
    print("=" * 60)
    print(f"Base rate (random K candidates): {results['base_rate_all_candidates']*100:.3f}%")
    for k in args.k:
        r = results[k]
        print(f"  Top-{r['k']:<5d}: {r['support_rate_at_k']*100:6.2f}% confirmed "
              f"({r['n_confirmed']} / {r['k']})")

    out_path = os.path.join(OUTPUT, f"t1_temporal_validation_{args.score_col}.json")
    import json
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()