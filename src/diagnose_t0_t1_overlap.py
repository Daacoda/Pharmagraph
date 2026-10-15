"""
diagnose_t0_t1_overlap.py
===========================
Diagnostic only - compares the t0 and t1 indication edge sets directly (by
node_id, not by local_idx, so the comparison is exact) to determine whether
t1 genuinely contains new indications or is identical to t0.

Run from the project root:
    python src/diagnose_t0_t1_overlap.py
"""
import os

import pandas as pd

PROCESSED = "processed"


def load_indication_pairs(csv_path, label, chunksize=500_000):
    pairs = set()
    usecols = ["x_id", "x_type", "relation", "y_id", "y_type"]
    n_rows = 0
    for chunk in pd.read_csv(csv_path, sep=",", header=0, chunksize=chunksize,
                              dtype=str, usecols=usecols):
        n_rows += len(chunk)
        ind = chunk[chunk["relation"] == "indication"]
        for _, row in ind.iterrows():
            if row["x_type"] == "drug" and row["y_type"] == "disease":
                pairs.add((row["x_id"], row["y_id"]))
            elif row["y_type"] == "drug" and row["x_type"] == "disease":
                pairs.add((row["y_id"], row["x_id"]))
    print(f"{label}: {n_rows:,} total rows scanned, {len(pairs)} indication pairs found")
    return pairs


def main():
    t0_pairs = load_indication_pairs(
        os.path.join("data", "benchmark", "snapshots", "kg_t0.csv"), "t0 (June 2021)")
    t1_pairs = load_indication_pairs(
        os.path.join("data", "benchmark", "snapshots", "kg_t1.csv"), "t1 (July 2023)")

    only_t0 = t0_pairs - t1_pairs
    only_t1 = t1_pairs - t0_pairs
    shared = t0_pairs & t1_pairs

    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)
    print(f"t0 indication pairs: {len(t0_pairs)}")
    print(f"t1 indication pairs: {len(t1_pairs)}")
    print(f"Shared (identical in both): {len(shared)}")
    print(f"Only in t0 (removed by t1): {len(only_t0)}")
    print(f"Only in t1 (NEW, added since t0): {len(only_t1)}")
    print(f"\nAre the two sets IDENTICAL? {'YES' if t0_pairs == t1_pairs else 'NO'}")

    if only_t1:
        print(f"\nSample of 5 NEW pairs (only in t1):")
        for pair in list(only_t1)[:5]:
            print(f"  drug={pair[0]}  disease={pair[1]}")


if __name__ == "__main__":
    main()