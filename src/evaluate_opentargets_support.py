"""
evaluate_opentargets_support.py
================================
External validation (replaces the t1-based temporal check, which is
structurally impossible for drug-disease pairs - see diagnose_t0_t1_overlap.py).

Uses Open Targets' knownDrugsAggregated dataset (real-world approved/trialled
drug-disease associations, ChEMBL-indexed) as an independent "ground truth"
source, instead of PrimeKG's frozen t1 snapshot.

Steps:
    1. Map every unique ChEMBL drug ID in knownDrugsAggregated to a DrugBank ID
       via the UniChem REST API (cached to disk, resumable).
    2. Build the set of (DrugBank drug, disease) pairs Open Targets considers
       a known/approved/trialled association.
    3. Map those to our local_idx vocabulary (processed/entities.csv).
    4. Check what fraction of our top-ranked discovery_scores.parquet
       candidates are confirmed by this independent source.

Requires (already produced by earlier stages):
    data/known_drugs_aggregated/*.parquet   (downloaded from Open Targets FTP)
    processed/entities.csv                  (Data_laoder.py)
    processed/discovery_scores.parquet      (score_discovery_candidates.py)

Run from the project root:
    python evaluate_opentargets_support.py
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import requests

PROCESSED = "processed"
OUTPUT = "output"
CHEMBL_TO_DRUGBANK_CACHE = os.path.join(PROCESSED, "chembl_to_drugbank.json")

UNICHEM_URL = "https://www.ebi.ac.uk/unichem/api/v1/compounds"
DRUGBANK_SOURCE_ID = 2


def load_cache():
    if os.path.exists(CHEMBL_TO_DRUGBANK_CACHE):
        with open(CHEMBL_TO_DRUGBANK_CACHE) as f:
            return json.load(f)
    return {}


def save_cache(cache):
    with open(CHEMBL_TO_DRUGBANK_CACHE, "w") as f:
        json.dump(cache, f)


def chembl_to_drugbank(chembl_id, cache, sleep=0.1):
    """Looks up one ChEMBL ID via UniChem, returns a DrugBank ID or None.
    Resumable: results are cached to disk as we go."""
    if chembl_id in cache:
        return cache[chembl_id]
    try:
        r = requests.post(
            UNICHEM_URL,
            json={"type": "sourceID", "compound": chembl_id, "sourceID": 1},
            timeout=10,
        )
        data = r.json()
        drugbank_id = None
        if data.get("compounds"):
            sources = data["compounds"][0].get("sources", [])
            matches = [s for s in sources if s["id"] == DRUGBANK_SOURCE_ID]
            if matches:
                drugbank_id = matches[0]["compoundId"]
        cache[chembl_id] = drugbank_id
    except Exception:
        cache[chembl_id] = None
    time.sleep(sleep)
    return cache[chembl_id]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scores-path", default=os.path.join(PROCESSED, "discovery_scores.parquet"))
    ap.add_argument("--score-col", default="model_prob")
    ap.add_argument("--k", type=int, nargs="+", default=[10, 50, 100, 500, 1000])
    ap.add_argument("--save-every", type=int, default=100)
    args = ap.parse_args()

    os.makedirs(OUTPUT, exist_ok=True)

    print("Step 1: loading knownDrugsAggregated from Open Targets ...")
    ot = pd.read_parquet("data/known_drugs_aggregated")
    print(f"  {len(ot)} rows, {ot['drugId'].nunique()} unique ChEMBL drugs, "
          f"{ot['diseaseId'].nunique()} unique diseases.")

    unique_chembl = sorted(ot["drugId"].unique())
    cache = load_cache()
    print(f"\nStep 2: mapping {len(unique_chembl)} ChEMBL IDs to DrugBank via UniChem "
          f"({sum(1 for c in unique_chembl if c in cache)} already cached) ...")

    for i, chembl_id in enumerate(unique_chembl, 1):
        chembl_to_drugbank(chembl_id, cache)
        if i % args.save_every == 0:
            save_cache(cache)
            n_mapped = sum(1 for v in cache.values() if v)
            print(f"  [{i}/{len(unique_chembl)}] mapped so far: {n_mapped}")
    save_cache(cache)

    n_mapped = sum(1 for v in cache.values() if v)
    print(f"  Done. {n_mapped}/{len(unique_chembl)} ChEMBL drugs resolved to a DrugBank ID.")

    print("\nStep 3: building Open Targets known (drugbank_id, diseaseId) pairs ...")
    ot["drugbank_id"] = ot["drugId"].map(cache)
    ot_known = ot.dropna(subset=["drugbank_id"])
    known_pairs_raw = set(zip(ot_known["drugbank_id"], ot_known["diseaseId"]))
    print(f"  {len(known_pairs_raw)} unique (drugbank_id, EFO diseaseId) known pairs.")

    print("\nStep 4: mapping to our local_idx vocabulary ...")
    entities = pd.read_csv(os.path.join(PROCESSED, "entities.csv"), dtype={"node_id": str})
    drug_map = dict(zip(
        entities.loc[entities.node_type == "drug", "node_id"],
        entities.loc[entities.node_type == "drug", "local_idx"],
    ))
    # NOTE: our disease nodes are MONDO-numbered; Open Targets diseaseId here
    # is EFO-numbered. We match on drug only for this first pass and flag
    # this as a known simplification - see note in the printed summary.
    unique_known_drugbank_ids = {p[0] for p in known_pairs_raw}
    confirmed_drugs = {drug_map[d] for d in unique_known_drugbank_ids if d in drug_map}
    print(f"  {len(confirmed_drugs)}/{len(unique_known_drugbank_ids)} "
          f"drugbank_ids matched to our local drug vocabulary.")

    print(f"\nLoading scored candidates from {args.scores_path} ...")
    candidates = pd.read_parquet(args.scores_path)
    ranked = candidates.sort_values(args.score_col, ascending=False).reset_index(drop=True)

    # Drug-level check (since disease vocabularies differ, MONDO vs EFO):
    # does the CANDIDATE DRUG appear anywhere in Open Targets' known-drug list
    # at all? This is a coarser signal than a true pair-level match, reported
    # transparently as such.
    is_known_drug = ranked["drug_idx"].isin(confirmed_drugs).values

    results = {"n_candidates": len(ranked), "n_confirmed_drugs_total": int(is_known_drug.sum())}
    base_rate = float(is_known_drug.mean())
    results["base_rate_all_candidates"] = base_rate

    print("\n" + "=" * 60)
    print(f"OPEN TARGETS EXTERNAL VALIDATION (drug-level) - ranked by {args.score_col}")
    print("=" * 60)
    print("NOTE: this checks whether the CANDIDATE'S DRUG has ANY known Open")
    print("Targets association (disease-level match needs MONDO<->EFO mapping,")
    print("not yet implemented). Treat as a coarse signal, not a strict hit rate.")
    print(f"\nBase rate (random candidate): {base_rate*100:.3f}%")
    for k in args.k:
        k = min(k, len(ranked))
        rate = float(is_known_drug[:k].mean()) if k > 0 else float("nan")
        n = int(is_known_drug[:k].sum())
        results[str(k)] = {"rate": rate, "n_confirmed": n, "k": k}
        print(f"  Top-{k:<6d}: {rate*100:6.2f}% have a drug-level OT match ({n} / {k})")

    out_path = os.path.join(OUTPUT, "opentargets_validation.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()