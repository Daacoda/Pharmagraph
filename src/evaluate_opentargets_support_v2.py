"""
evaluate_opentargets_support_v2.py
===================================
Full drug+disease level external validation against Open Targets'
knownDrugsAggregated, improving on evaluate_opentargets_support.py's
drug-only check by also matching the disease.

Bridge chain:
    Open Targets diseaseId (EFO_xxxxx)
        -> MONDO id, via data/diseases' dbXRefs column
        -> disease name, via data/diseases' name column
        -> our local disease node_id, via kg_t0.csv's x_name/y_name
           (name-matched, case-insensitive, since our node_id values are
           plain PrimeKG integers, not MONDO-formatted)
        -> our local_idx, via processed/entities.csv

Requires:
    data/known_drugs_aggregated/*.parquet
    data/diseases/*.parquet
    processed/chembl_to_drugbank.json   (built by the v1 script - reused, not rebuilt)
    processed/entities.csv
    processed/discovery_scores.parquet
    data/benchmark/snapshots/kg_t0.csv  (streamed once for disease id->name)

Run from the project root:
    python evaluate_opentargets_support_v2.py
"""
import argparse
import json
import os
import re

import numpy as np
import pandas as pd
from rapidfuzz import process, fuzz

PROCESSED = "processed"
OUTPUT = "output"
CHEMBL_TO_DRUGBANK_CACHE = os.path.join(PROCESSED, "chembl_to_drugbank.json")


def build_efo_to_mondo(diseases_df):
    """Parses dbXRefs to find a MONDO cross-reference for each EFO-indexed row,
    and also handles rows whose own `id` already IS a MONDO id."""
    efo_to_mondo = {}
    mondo_pattern = re.compile(r"MONDO[:_](\d+)")

    for _, row in diseases_df.iterrows():
        row_id = row["id"]
        xrefs = row["dbXRefs"] if isinstance(row["dbXRefs"], (list, np.ndarray)) else []

        mondo_id = None
        if row_id.startswith("MONDO"):
            m = mondo_pattern.search(row_id)
            if m:
                mondo_id = m.group(1)
        if mondo_id is None:
            for ref in xrefs:
                m = mondo_pattern.search(str(ref))
                if m:
                    mondo_id = m.group(1)
                    break

        if mondo_id is not None and row_id.startswith("EFO"):
            efo_to_mondo[row_id] = mondo_id

    return efo_to_mondo


def build_mondo_to_names(diseases_df):
    """MONDO numeric id -> SET of candidate names (primary name + all exact/
    related synonyms), from the same Open Targets diseases table. Using every
    synonym, not just the primary name, is what makes name-matching against a
    differently-curated source (PrimeKG) actually work in practice.

    IMPORTANT: a row's MONDO number can come from EITHER its own `id` (when
    the row IS a MONDO entry) OR its `dbXRefs` (when the row's primary id is
    EFO/DOID/HP/etc. but it cross-references a MONDO term). Earlier versions
    of this function only checked `id`, which silently dropped every
    EFO-primary row - exactly the rows build_efo_to_mondo() depends on -
    making the two maps draw from disjoint row sets and never intersect."""
    mondo_pattern = re.compile(r"MONDO[:_](\d+)")
    out = {}
    for _, row in diseases_df.iterrows():
        mondo_nums = set()

        m = mondo_pattern.search(row["id"])
        if m:
            mondo_nums.add(m.group(1))

        xrefs = row["dbXRefs"] if isinstance(row["dbXRefs"], (list, np.ndarray)) else []
        for ref in xrefs:
            m = mondo_pattern.search(str(ref))
            if m:
                mondo_nums.add(m.group(1))

        if not mondo_nums:
            continue

        names = {row["name"].strip().lower()}
        syn = row.get("synonyms")
        if isinstance(syn, dict):
            for key in ("hasExactSynonym", "hasRelatedSynonym"):
                values = syn.get(key)
                if values is not None and len(values) > 0:
                    for v in values:
                        names.add(str(v).strip().lower())

        for mondo_num in mondo_nums:
            out.setdefault(mondo_num, set()).update(names)
    return out


def stream_our_disease_names(kg_csv_path, chunksize=500_000):
    """our disease node_id -> lowercased name, from kg_t0.csv. A node_id field
    can contain multiple underscore-joined ids (e.g. '1200_1134_...'); each
    sub-id is registered separately, all pointing to the same name."""
    name_to_our_ids = {}
    usecols = ["x_id", "x_type", "x_name", "y_id", "y_type", "y_name"]
    for chunk in pd.read_csv(kg_csv_path, sep=",", header=0, chunksize=chunksize,
                              dtype=str, usecols=usecols):
        for side in ("x", "y"):
            sub = chunk[chunk[f"{side}_type"] == "disease"][[f"{side}_id", f"{side}_name"]]
            for _, row in sub.iterrows():
                raw_id, name = row[f"{side}_id"], row[f"{side}_name"]
                if pd.isna(name):
                    continue
                name_key = name.strip().lower()
                for sub_id in str(raw_id).split("_"):
                    name_to_our_ids.setdefault(name_key, set()).add(sub_id)
    return name_to_our_ids


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--kg-csv", default=os.path.join("data", "benchmark", "snapshots", "kg_t0.csv"))
    ap.add_argument("--scores-path", default=os.path.join(PROCESSED, "discovery_scores.parquet"))
    ap.add_argument("--score-col", default="model_prob")
    ap.add_argument("--k", type=int, nargs="+", default=[10, 50, 100, 500, 1000])
    args = ap.parse_args()

    os.makedirs(OUTPUT, exist_ok=True)

    print("Step 1: loading Open Targets knownDrugsAggregated + diseases ...")
    ot = pd.read_parquet("data/known_drugs_aggregated")
    diseases_df = pd.read_parquet("data/diseases")
    print(f"  {len(ot)} known-drug rows, {len(diseases_df)} Open Targets disease records.")

    print("\nStep 2: building EFO -> MONDO numeric id map ...")
    efo_to_mondo = build_efo_to_mondo(diseases_df)
    print(f"  {len(efo_to_mondo)} EFO ids resolved to a MONDO id.")

    print("\nStep 3: building MONDO numeric id -> candidate name set (incl. synonyms) ...")
    mondo_to_names = build_mondo_to_names(diseases_df)
    total_names = sum(len(v) for v in mondo_to_names.values())
    print(f"  {len(mondo_to_names)} MONDO ids, {total_names} total candidate names "
          f"(avg {total_names/max(1,len(mondo_to_names)):.1f} names/disease).")

    print(f"\nStep 4: streaming {args.kg_csv} once to map disease names -> our node_ids "
          f"(this reads the full file) ...")
    name_to_our_ids = stream_our_disease_names(args.kg_csv)
    print(f"  {len(name_to_our_ids)} distinct disease names found in our graph.")

    our_name_list = list(name_to_our_ids.keys())

    def resolve_name(name, cache={}):
        """Exact match first (fast path); fall back to fuzzy match (token-sort
        ratio, >=90 similarity) against our full name vocabulary. Fuzzy
        results are cached since the same OT name recurs across many rows."""
        if name in name_to_our_ids:
            return name_to_our_ids[name]
        if name in cache:
            return cache[name]
        match = process.extractOne(name, our_name_list, scorer=fuzz.token_sort_ratio,
                                    score_cutoff=90)
        result = name_to_our_ids[match[0]] if match else set()
        cache[name] = result
        return result

    print("\nStep 5: loading cached ChEMBL->DrugBank map (built by v1 script) ...")
    if not os.path.exists(CHEMBL_TO_DRUGBANK_CACHE):
        raise SystemExit(f"{CHEMBL_TO_DRUGBANK_CACHE} not found - run "
                          f"evaluate_opentargets_support.py (v1) first to build it.")
    with open(CHEMBL_TO_DRUGBANK_CACHE) as f:
        chembl_to_drugbank = json.load(f)
    n_mapped_drugs = sum(1 for v in chembl_to_drugbank.values() if v)
    print(f"  {n_mapped_drugs} ChEMBL drugs have a cached DrugBank id.")

    print("\nStep 6: building the full (drugbank_id, our_disease_node_id) known pair set ...")
    entities = pd.read_csv(os.path.join(PROCESSED, "entities.csv"), dtype={"node_id": str})
    drug_map = dict(zip(
        entities.loc[entities.node_type == "drug", "node_id"],
        entities.loc[entities.node_type == "drug", "local_idx"],
    ))
    disease_id_to_local = dict(zip(
        entities.loc[entities.node_type == "disease", "node_id"],
        entities.loc[entities.node_type == "disease", "local_idx"],
    ))

    known_pairs = set()
    unresolved_disease = 0
    for _, row in ot.iterrows():
        drugbank_id = chembl_to_drugbank.get(row["drugId"])
        if not drugbank_id or drugbank_id not in drug_map:
            continue
        mondo_num = efo_to_mondo.get(row["diseaseId"])
        if mondo_num is None:
            unresolved_disease += 1
            continue
        candidate_names = mondo_to_names.get(mondo_num)
        if not candidate_names:
            unresolved_disease += 1
            continue
        our_ids = set()
        for name in candidate_names:
            our_ids.update(resolve_name(name))
        if not our_ids:
            unresolved_disease += 1
            continue
        drug_local = drug_map[drugbank_id]
        for our_disease_id in our_ids:
            disease_local = disease_id_to_local.get(our_disease_id)
            if disease_local is not None:
                known_pairs.add((drug_local, disease_local))

    print(f"  {len(known_pairs)} full (drug, disease) pairs resolved end-to-end.")
    print(f"  ({unresolved_disease} OT rows had a disease we couldn't bridge to our vocabulary.)")

    print(f"\nLoading scored candidates from {args.scores_path} ...")
    candidates = pd.read_parquet(args.scores_path)
    ranked = candidates.sort_values(args.score_col, ascending=False).reset_index(drop=True)

    is_confirmed = np.array([
        (d, dis) in known_pairs for d, dis in zip(ranked["drug_idx"], ranked["disease_idx"])
    ])
    base_rate = float(is_confirmed.mean())

    results = {
        "n_candidates": len(ranked),
        "n_known_pairs_resolved": len(known_pairs),
        "base_rate_all_candidates": base_rate,
        "n_confirmed_total": int(is_confirmed.sum()),
    }

    print("\n" + "=" * 60)
    print(f"OPEN TARGETS EXTERNAL VALIDATION (full drug+disease pair) - ranked by {args.score_col}")
    print("=" * 60)
    print(f"Base rate (random candidate): {base_rate*100:.4f}%")
    for k in args.k:
        k = min(k, len(ranked))
        rate = float(is_confirmed[:k].mean()) if k > 0 else float("nan")
        n = int(is_confirmed[:k].sum())
        results[str(k)] = {"rate": rate, "n_confirmed": n, "k": k}
        print(f"  Top-{k:<6d}: {rate*100:6.3f}% confirmed as a known pair ({n} / {k})")

    out_path = os.path.join(OUTPUT, "opentargets_validation_full_pair.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()