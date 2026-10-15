"""
score_discovery_candidates.py
===============================
Scores every pair in processed/discovery_candidates.csv (built by
build_discovery_candidates.py) with the trained v3 model checkpoint.
Pure inference - no training, no label leakage, since these pairs were
never part of train/val/test.

This mirrors G_SAGE_model_predict.py's model-loading and scoring logic
exactly, but does NOT mask any indication edges out of the graph - the
full t0 graph is used for message passing, since none of these candidate
pairs were part of the original train/val/test split to begin with.

Output: processed/discovery_scores.parquet (drug_idx, disease_idx, model_prob)

Run from the project root, after build_discovery_candidates.py:
    python src/score_discovery_candidates.py
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv

PROCESSED = "processed"


class HeteroGraphSAGE(nn.Module):
    """Must match G_SAGE_model_training.py's architecture exactly."""
    def __init__(self, node_types, edge_types, hidden_channels=128, out_channels=128):
        super().__init__()
        self.conv1 = HeteroConv({et: SAGEConv((-1, -1), hidden_channels) for et in edge_types}, aggr='sum')
        self.conv2 = HeteroConv({et: SAGEConv((-1, -1), out_channels) for et in edge_types}, aggr='sum')

    def forward(self, x_dict, edge_index_dict):
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {k: F.relu(v) for k, v in x_dict.items()}
        x_dict = self.conv2(x_dict, edge_index_dict)
        return x_dict


def score_pairs(drug_emb, disease_emb, drug_idx, disease_idx):
    return (drug_emb[drug_idx] * disease_emb[disease_idx]).sum(dim=-1)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", default=os.path.join("output", "models", "graphsage_v3_real_features.pt"))
    ap.add_argument("--fp-size", type=int, default=256)
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("Loading entities, fingerprints, candidates, and graph ...")
    nodes_typed = pd.read_csv(os.path.join(PROCESSED, "entities.csv"))
    drug_fingerprints = np.load(os.path.join(PROCESSED, "drug_fingerprints.npy"), allow_pickle=True).item()
    candidates = pd.read_csv(os.path.join(PROCESSED, "discovery_candidates.csv"))
    data = torch.load(os.path.join(PROCESSED, "pharmagraph_hetero_t0.pt"), weights_only=False)

    drug_nodes = nodes_typed[nodes_typed["node_type"] == "drug"].sort_values("local_idx")
    drug_feature_matrix = np.zeros((len(drug_nodes), args.fp_size), dtype=np.float32)
    for _, row in drug_nodes.iterrows():
        if row["node_id"] in drug_fingerprints:
            drug_feature_matrix[row["local_idx"]] = drug_fingerprints[row["node_id"]]
    data["drug"].x = torch.tensor(drug_feature_matrix, dtype=torch.float)

    data = data.to(device)

    model = HeteroGraphSAGE(data.node_types, data.edge_types).to(device)
    checkpoint = torch.load(args.checkpoint, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    print(f"Loaded checkpoint: {args.checkpoint}")
    print(f"  (its own recorded test AUROC/AUPRC: "
          f"{checkpoint.get('test_auroc', '?')} / {checkpoint.get('test_auprc', '?')})")

    model.eval()
    with torch.no_grad():
        out = model(data.x_dict, data.edge_index_dict)
        drug_emb, disease_emb = out["drug"], out["disease"]

        d_idx = torch.tensor(candidates["drug_idx"].values, dtype=torch.long, device=device)
        dis_idx = torch.tensor(candidates["disease_idx"].values, dtype=torch.long, device=device)
        scores = score_pairs(drug_emb, disease_emb, d_idx, dis_idx)
        probs = torch.sigmoid(scores).cpu().numpy()

    candidates["model_prob"] = probs
    out_path = os.path.join(PROCESSED, "discovery_scores.parquet")
    candidates.to_parquet(out_path, index=False)
    print(f"\nSaved {out_path}: {len(candidates)} scored candidate pairs")
    print(f"Score range: min={probs.min():.4f} max={probs.max():.4f} mean={probs.mean():.4f}")


if __name__ == "__main__":
    main()