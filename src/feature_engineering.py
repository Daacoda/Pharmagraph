
"""
python src/feature_engineering.py
PharmaGraph - Feature Engineering
=================================
Fetches drug names from the PrimeKG-CL raw snapshot, resolves each name to a
SMILES string via PubChem, converts each SMILES to a Morgan (ECFP) fingerprint,
and saves the results to disk.

No Google Drive / Colab dependencies. All paths are configurable below or via
command-line arguments.

Requires: pandas, pubchempy, rdkit
    pip install pandas pubchempy rdkit

Usage:
    python feature_engineering.py \
        --kg-csv "https://huggingface.co/datasets/yradwan147/PrimeKGCL/resolve/main/benchmark/snapshots/kg_t0.csv" \
        --output-dir ./data \
        --fp-size 256
"""

import argparse
import json
import os
import time
from pathlib import Path

import pandas as pd
import pubchempy as pcp
from rdkit import Chem
from rdkit.Chem import AllChem


# ---------------------------------------------------------------------------
# Step 1: collect drug names from the raw KG snapshot
# ---------------------------------------------------------------------------
def collect_drug_names(kg_csv_path: str, chunksize: int = 500_000) -> dict:
    """Streams kg_t0.csv (works with a local path or an https URL) and
    returns {drugbank_id: drug_name} for every node of type 'drug'."""
    drug_names = {}
    usecols = ["x_id", "x_type", "x_name", "y_id", "y_type", "y_name"]
    for chunk in pd.read_csv(kg_csv_path, sep=",", header=0, chunksize=chunksize,
                              dtype=str, usecols=usecols):
        x_drugs = chunk[chunk["x_type"] == "drug"][["x_id", "x_name"]]
        y_drugs = chunk[chunk["y_type"] == "drug"][["y_id", "y_name"]]
        for _, row in x_drugs.iterrows():
            drug_names[row["x_id"]] = row["x_name"]
        for _, row in y_drugs.iterrows():
            drug_names[row["y_id"]] = row["y_name"]
    return drug_names


# ---------------------------------------------------------------------------
# Step 2: resolve drug names -> SMILES via PubChem (resumable, rate-limited)
# ---------------------------------------------------------------------------
def fetch_smiles(drug_names: dict, output_path: str,
                  sleep_between_calls: float = 1.0, save_every: int = 10) -> dict:
    """Looks up each drug name on PubChem and saves {drugbank_id: smiles}.
    Resumes from output_path if it already exists, so re-running after an
    interruption only fetches what is still missing."""
    if os.path.exists(output_path):
        with open(output_path, "r") as f:
            drug_smiles = json.load(f)
        print(f"Resuming: {len(drug_smiles)} drugs already resolved.")
    else:
        drug_smiles = {}

    remaining = [d for d in drug_names if d not in drug_smiles]
    print(f"Remaining to resolve: {len(remaining)}")

    failed = []
    for i, drug_id in enumerate(remaining):
        name = drug_names[drug_id]
        for attempt in range(3):
            try:
                compounds = pcp.get_compounds(name, "name")
                if compounds and compounds[0].connectivity_smiles:
                    drug_smiles[drug_id] = compounds[0].connectivity_smiles
                break
            except Exception:
                time.sleep(1.5)
        else:
            failed.append(drug_id)

        time.sleep(sleep_between_calls)  # stay under PubChem's rate limit

        if (i + 1) % save_every == 0:
            with open(output_path, "w") as f:
                json.dump(drug_smiles, f)
            print(f"[{i + 1}/{len(remaining)}] resolved={len(drug_smiles)} failed={len(failed)}")

    with open(output_path, "w") as f:
        json.dump(drug_smiles, f)

    print(f"Done. Resolved {len(drug_smiles)}/{len(drug_names)}. Failed: {len(failed)}")
    return drug_smiles


# ---------------------------------------------------------------------------
# Step 3: SMILES -> Morgan fingerprint
# ---------------------------------------------------------------------------
def build_fingerprints(drug_smiles: dict, fp_size: int, output_path: str) -> dict:
    """Converts every SMILES string to a Morgan fingerprint (radius=2) and
    saves {drugbank_id: np.ndarray} to output_path (.npy, pickled dict)."""
    import numpy as np

    fingerprints = {}
    failed = []
    for drug_id, smiles in drug_smiles.items():
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                failed.append(drug_id)
                continue
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=fp_size)
            fingerprints[drug_id] = np.array(fp, dtype=np.float32)
        except Exception:
            failed.append(drug_id)

    np.save(output_path, fingerprints, allow_pickle=True)
    print(f"Fingerprints built: {len(fingerprints)}/{len(drug_smiles)}. Failed: {len(failed)}")
    print(f"Saved to {output_path}")
    return fingerprints

def main():
    parser = argparse.ArgumentParser(description="Build drug molecular fingerprints for PharmaGraph.")
    parser.add_argument("--kg-csv", default=r"data\benchmark\snapshots\kg_t0.csv", 
                        help="Local path or URL to the raw PrimeKG-CL t0 snapshot CSV.")
    
    parser.add_argument("--output-dir", default=r"processed", 
                        help="Directory to save drug_smiles.json and drug_fingerprints.npy.")
    
    parser.add_argument("--fp-size", type=int, default=256, help="Morgan fingerprint bit length.")
    args = parser.parse_args()

    import torch
    pt_path = r"processed\pharmagraph_hetero_t0.pt"
    if os.path.exists(pt_path):
        print(f"Loading hetero graph data from: {pt_path}")
        data = torch.load(pt_path, weights_only=False)
        print("Graph loaded successfully!")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    smiles_path = os.path.join(args.output_dir, "drug_smiles.json")
    fingerprints_path = os.path.join(args.output_dir, "drug_fingerprints.npy")

    print("Step 1: collecting drug names from the raw KG snapshot...")
    drug_names = collect_drug_names(args.kg_csv)
    print(f"Found {len(drug_names)} drug nodes.")

    print("Step 2: resolving drug names to SMILES via PubChem (this can take a while)...")
    drug_smiles = fetch_smiles(drug_names, smiles_path)

    print("Step 3: converting SMILES to Morgan fingerprints...")
    build_fingerprints(drug_smiles, args.fp_size, fingerprints_path)


if __name__ == "__main__":
    main()
