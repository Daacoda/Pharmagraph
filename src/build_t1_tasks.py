"""
build_t1_tasks.py
=================
Data preparation ONLY (no model, no training) for temporal link-prediction tasks
on PrimeKG-CL t0 -> t1 for relations OTHER than drug-disease (those have zero
turnover by construction, see step9 write-up).

For every chosen relation it streams kg_t0.csv and kg_t1.csv and builds:
    t0 edges     : relation edges present at t0                (training side)
    added edges  : in t1 but NOT in t0 (orientation-insensitive for same-type
                   relations), restricted to edges whose BOTH endpoints already
                   exist as nodes at t0 (processed/entities.csv). Edges touching a
                   node that is new in t1 cannot be scored by a model trained on t0
                   and are only counted, not kept.
and prints a summary including a hub-concentration preview: what share of the
added edges have a head AND a tail that already had >=1 t0 edge of this relation
(if that share is very high, a popularity baseline will be strong - as it was for
drug-disease).

NOTE: PrimeKG stores each edge in both directions; edges are canonicalised to one
orientation first, so counts are real (undirected) edge counts.

Outputs (new files only), under processed/t1_tasks/:
    <relation>_t0.parquet, <relation>_added.parquet   columns: x_id,x_type,y_id,y_type,x_local,y_local
    t1_tasks_summary.csv

Run from the project root:
    py src\\build_t1_tasks.py
    py src\\build_t1_tasks.py --relations disease_phenotype_positive molfunc_protein
Large relations (anatomy_protein_present/absent: millions of rows) work but need RAM;
use --max-added to keep a random sample of the added edges.
"""
import argparse
import os

import numpy as np
import pandas as pd

PROCESSED = "processed"
DEFAULT_RELATIONS = ["disease_phenotype_positive", "bioprocess_protein", "molfunc_protein",
                     "cellcomp_protein", "exposure_protein", "disease_disease"]
COLS = ["x_id", "x_type", "relation", "y_id", "y_type"]


def stream_relations(csv_path, relations, chunksize=1_000_000):
    """-> {relation: DataFrame(x_id,x_type,y_id,y_type)} (deduplicated)."""
    parts = {r: [] for r in relations}
    for ch in pd.read_csv(csv_path, usecols=COLS, dtype=str, chunksize=chunksize):
        ch = ch[ch["relation"].isin(relations)]
        for r, g in ch.groupby("relation"):
            parts[r].append(g[["x_id", "x_type", "y_id", "y_type"]])
    out = {}
    for r in relations:
        df = (pd.concat(parts[r], ignore_index=True)
              if parts[r] else pd.DataFrame(columns=["x_id", "x_type", "y_id", "y_type"]))
        out[r] = canonicalize(df)
    return out


def canonicalize(df):
    """PrimeKG stores every edge in BOTH directions. Put each edge in one canonical
    orientation ((type, id) of the head <= (type, id) of the tail) and deduplicate, so
    one real edge is counted once - otherwise test edges would be double counted."""
    if df.empty:
        return df
    kx = df["x_type"] + "|" + df["x_id"]
    ky = df["y_type"] + "|" + df["y_id"]
    swap = kx > ky
    out = df.copy()
    out.loc[swap, ["x_id", "x_type", "y_id", "y_type"]] = df.loc[swap, ["y_id", "y_type", "x_id", "x_type"]].values
    return out.drop_duplicates().reset_index(drop=True)


def anti_join(t1, t0):
    """Rows of t1 whose (x_id,y_id) is not in t0 (both sides canonicalised)."""
    key = ["x_id", "y_id"]
    m = t1.merge(t0[key].drop_duplicates().assign(_in0=1), on=key, how="left")
    return m.loc[m["_in0"].isna(), ["x_id", "x_type", "y_id", "y_type"]].reset_index(drop=True)


def attach_local(df, ent):
    """Keep only rows whose both endpoints exist in the t0 entity table."""
    e = ent[["node_id", "node_type", "local_idx"]]
    d = df.merge(e.rename(columns={"node_id": "x_id", "node_type": "x_type", "local_idx": "x_local"}),
                 on=["x_id", "x_type"], how="inner")
    d = d.merge(e.rename(columns={"node_id": "y_id", "node_type": "y_type", "local_idx": "y_local"}),
                on=["y_id", "y_type"], how="inner")
    return d


def process_relation(rel, e0, e1, ent, max_added=None, seed=42):
    t0 = attach_local(e0, ent)
    added_all = anti_join(e1, e0)
    added = attach_local(added_all, ent)
    info = {"relation": rel, "t0_edges": len(e0), "t1_edges": len(e1),
            "added_total": len(added_all), "added_both_ends_in_t0": len(added),
            "share_usable": len(added) / max(len(added_all), 1)}
    if len(t0) and len(added):
        dh = t0["x_local"].value_counts()
        dt = t0["y_local"].value_counts()
        seen_h = added["x_local"].isin(dh.index)
        seen_t = added["y_local"].isin(dt.index)
        info["added_head_seen_in_t0"] = float(seen_h.mean())
        info["added_tail_seen_in_t0"] = float(seen_t.mean())
        info["added_both_seen_in_t0"] = float((seen_h & seen_t).mean())
        info["median_head_deg_added"] = float(added["x_local"].map(dh).fillna(0).median())
        info["median_head_deg_t0_edges"] = float(t0["x_local"].map(dh).median())
        info["n_unique_heads_added"] = int(added["x_local"].nunique())
        info["n_unique_tails_added"] = int(added["y_local"].nunique())
    if max_added and len(added) > max_added:
        added = added.sample(max_added, random_state=seed)
        info["added_kept_after_sampling"] = len(added)
    return t0, added, info


def diagnose(rel, e1, ent):
    """Why are (almost) no added edges usable? Look at the endpoints."""
    ids = set(ent["node_id"])
    typed = set(zip(ent["node_id"], ent["node_type"]))
    print(f"\n  --- diagnostic for {rel} ---")
    print("  sample t1 rows:")
    print(e1.head(5).to_string(index=False))
    print(f"  node types in t1 edges: x={sorted(e1['x_type'].unique())[:5]}  y={sorted(e1['y_type'].unique())[:5]}")
    for side in ("x", "y"):
        by_id = e1[f"{side}_id"].isin(ids).mean()
        by_typed = np.mean([(a, b) in typed for a, b in zip(e1[f"{side}_id"], e1[f"{side}_type"])])
        print(f"  {side}: id found in t0 entities (any type) {by_id:.3f} | (id,type) found {by_typed:.3f} "
              f"| unique {e1[f'{side}_id'].nunique()}")


def list_relations(path, chunksize=1_000_000):
    c = {}
    for ch in pd.read_csv(path, usecols=["relation"], dtype=str, chunksize=chunksize):
        for k, v in ch["relation"].value_counts().items():
            c[k] = c.get(k, 0) + v
    return pd.Series(c).sort_values(ascending=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--t0", default=os.path.join("data", "benchmark", "snapshots", "kg_t0.csv"))
    ap.add_argument("--t1", default=os.path.join("data", "benchmark", "snapshots", "kg_t1.csv"))
    ap.add_argument("--relations", nargs="+", default=DEFAULT_RELATIONS)
    ap.add_argument("--max-added", type=int, default=None)
    ap.add_argument("--out-dir", default=os.path.join(PROCESSED, "t1_tasks"))
    ap.add_argument("--list-relations", action="store_true",
                    help="print every relation name with row counts in t0 and t1, then exit")
    args = ap.parse_args()

    if args.list_relations:
        r0, r1 = list_relations(args.t0), list_relations(args.t1)
        both = pd.concat([r0.rename("t0_rows"), r1.rename("t1_rows")], axis=1).fillna(0).astype(int)
        print(both.sort_values("t1_rows", ascending=False).to_string())
        return

    os.makedirs(args.out_dir, exist_ok=True)
    ent = pd.read_csv(os.path.join(PROCESSED, "entities.csv"), dtype={"node_id": str})
    print(f"Streaming {args.t0} ...")
    s0 = stream_relations(args.t0, args.relations)
    print(f"Streaming {args.t1} ...")
    s1 = stream_relations(args.t1, args.relations)

    rows = []
    for rel in args.relations:
        if len(s0[rel]) == 0 and len(s1[rel]) == 0:
            print(f"  [{rel}] not found in either snapshot - check the relation name")
            continue
        t0, added, info = process_relation(rel, s0[rel], s1[rel], ent, args.max_added)
        t0.to_parquet(os.path.join(args.out_dir, f"{rel}_t0.parquet"), index=False)
        added.to_parquet(os.path.join(args.out_dir, f"{rel}_added.parquet"), index=False)
        rows.append(info)
        if info["share_usable"] < 0.5:
            diagnose(rel, s1[rel], ent)
        print(f"  [{rel}] t0={info['t0_edges']}  t1={info['t1_edges']}  added={info['added_total']}  "
              f"usable (both ends in t0)={info['added_both_ends_in_t0']}")

    summ = pd.DataFrame(rows)
    summ.to_csv(os.path.join(args.out_dir, "t1_tasks_summary.csv"), index=False)
    pd.set_option("display.width", 220, "display.max_columns", 30)
    print("\n" + "=" * 100)
    print("t0 -> t1 TASK CANDIDATES")
    print("=" * 100)
    print(summ.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\nReading guide: 'share_usable' near 1 = tasks are scoreable by a t0-trained model. "
          "'added_both_seen_in_t0' near 1 with a high median head degree = new edges concentrate "
          "on already-popular nodes, so expect a strong popularity baseline.")
    print(f"Saved to {args.out_dir}")


if __name__ == "__main__":
    main()