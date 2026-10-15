"""
evaluate_discovery_t1_support.py
==================================
Temporal validation (methodology step 9), run against the broad discovery
candidate pool (processed/discovery_scores.parquet) instead of the training
negatives. This is the correct version of the check: every pair here was
genuinely unknown at t0 (neither a known indication nor a known
contraindication), so a t1 confirmation is a real discovery, not an
artifact of how the training negatives were constructed.

Requires:
    processed/entities.csv              (Data_laoder.py)
    processed/discovery_scores.parquet  (score_discovery_candidates.py)
    data/benchmark/snapshots/kg_t1.csv  (July 2023 snapshot)

Run from the project root:
    python src/evaluate_discovery_t1_support.py
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

PROCESSED = "processed"
OUTPUT = "output"


def load_t1_indication_pairs(t1_csv_path, chunksize=500_000):
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


def evidence_support_rate(candidates, score_col, confirmed_pairs, k_values):
    ranked = candidates.sort_values(score_col, ascending=False).reset_index(drop=True)
    is_confirmed = np.array([
        (d, dis) in confirmed_pairs for d, dis in zip(ranked["drug_idx"], ranked["disease_idx"])
    ])
    base_rate = is_confirmed.mean()

    results = {}
    for k in k_values:
        k = min(k, len(ranked))
        results[str(k)] = {
            "support_rate_at_k": float(is_confirmed[:k].mean()) if k > 0 else float("nan"),
            "n_confirmed": int(is_confirmed[:k].sum()),
            "k": k,
        }
    results["base_rate_all_candidates"] = float(base_rate)
    results["n_candidates"] = len(ranked)
    results["n_confirmed_total"] = int(is_confirmed.sum())
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--t1-csv", default=os.path.join("data", "benchmark", "snapshots", "kg_t1.csv"))
    ap.add_argument("--scores-path", default=os.path.join(PROCESSED, "discovery_scores.parquet"))
    ap.add_argument("--score-col", default="model_prob")
    ap.add_argument("--k", type=int, nargs="+", default=[10, 50, 100, 500, 1000])
    args = ap.parse_args()

    os.makedirs(OUTPUT, exist_ok=True)

    print(f"Loading t0 entity vocabulary from {PROCESSED}/entities.csv ...")
    entities = pd.read_csv(os.path.join(PROCESSED, "entities.csv"), dtype={"node_id": str})

    print(f"Streaming t1 snapshot from {args.t1_csv} (reads the full file once) ...")
    t1_pairs_raw = load_t1_indication_pairs(args.t1_csv)
    print(f"  Found {len(t1_pairs_raw)} drug-disease indication pairs in t1.")

    confirmed_pairs, dropped = map_pairs_to_local_idx(t1_pairs_raw, entities)
    print(f"  Mapped to t0 vocabulary: {len(confirmed_pairs)} kept, {dropped} dropped.")

    print(f"\nLoading discovery candidates from {args.scores_path} ...")
    candidates = pd.read_parquet(args.scores_path)
    print(f"  {len(candidates)} candidates, all genuinely unknown at t0 "
          f"(neither indication nor contraindication).")

    print(f"\nRanking by '{args.score_col}' and checking against t1-confirmed pairs ...")
    results = evidence_support_rate(candidates, args.score_col, confirmed_pairs, args.k)

    print("\n" + "=" * 60)
    print("TEMPORAL VALIDATION ON DISCOVERY CANDIDATES")
    print("=" * 60)
    print(f"Base rate (random candidate): {results['base_rate_all_candidates']*100:.4f}%")
    print(f"Total confirmed in t1 among all {results['n_candidates']} candidates: "
          f"{results['n_confirmed_total']}")
    for k in args.k:
        r = results[str(k)]
        print(f"  Top-{r['k']:<6d}: {r['support_rate_at_k']*100:6.3f}% confirmed "
              f"({r['n_confirmed']} / {r['k']})")

    out_path = os.path.join(OUTPUT, "t1_discovery_validation.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()