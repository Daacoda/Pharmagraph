"""
build_discovery_candidates.py
==============================
Builds a broad pool of drug-disease "discovery candidates": pairs the model
was never told are true (indication) or false (contraindication) at t0.
This is the correct candidate pool for temporal validation - the training
negatives (contraindication + random negatives) used in G_SAGE_model_training.py
are NOT a fair discovery test, since contraindication pairs can never become
true, and the random negatives were already checked against known indications
by construction.

Output: processed/discovery_candidates.csv (drug_idx, disease_idx), a random
sample of the full drug x disease space with known t0 indication/contraindication
pairs removed.

Run from the project root, after Data_laoder.py has produced processed/entities.csv
and processed/pharmagraph_hetero_t0.pt:
    python src/build_discovery_candidates.py --n-samples 50000
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch

PROCESSED = "processed"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-samples", type=int, default=50_000,
                     help="How many random drug-disease pairs to sample as candidates.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print("Loading graph for known indication/contraindication pairs ...")
    data = torch.load(os.path.join(PROCESSED, "pharmagraph_hetero_t0.pt"), weights_only=False)

    ind_ei = data["drug", "indication", "disease"].edge_index
    known_indication = set(zip(ind_ei[0].tolist(), ind_ei[1].tolist()))

    contra_ei = data["drug", "contraindication", "disease"].edge_index
    known_contra = set(zip(contra_ei[0].tolist(), contra_ei[1].tolist()))

    excluded = known_indication | known_contra
    n_drugs = data["drug"].num_nodes
    n_diseases = data["disease"].num_nodes
    print(f"Graph: {n_drugs} drugs x {n_diseases} diseases "
          f"({n_drugs * n_diseases:,} possible pairs)")
    print(f"Excluding {len(known_indication)} known indication + "
          f"{len(known_contra)} known contraindication pairs "
          f"({len(excluded)} unique excluded pairs)")

    rng = np.random.default_rng(args.seed)
    candidates = set()
    # oversample a bit since some draws will hit excluded pairs or duplicates
    attempts = 0
    max_attempts = args.n_samples * 20
    while len(candidates) < args.n_samples and attempts < max_attempts:
        batch = args.n_samples - len(candidates)
        d = rng.integers(0, n_drugs, size=batch)
        dis = rng.integers(0, n_diseases, size=batch)
        for di, disi in zip(d.tolist(), dis.tolist()):
            if (di, disi) not in excluded:
                candidates.add((di, disi))
        attempts += batch

    if len(candidates) < args.n_samples:
        print(f"WARNING: only found {len(candidates)}/{args.n_samples} unique "
              f"candidates after {attempts} draws (space may be heavily excluded).")

    out = pd.DataFrame(sorted(candidates), columns=["drug_idx", "disease_idx"])
    out_path = os.path.join(PROCESSED, "discovery_candidates.csv")
    out.to_csv(out_path, index=False)
    print(f"\nSaved {out_path}: {len(out)} candidate pairs "
          f"({len(out) / (n_drugs * n_diseases) * 100:.3f}% of the full space)")


if __name__ == "__main__":
    main()