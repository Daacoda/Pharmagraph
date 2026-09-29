"""
G_SAGE_model_predict.py
Stage 3, Step 1 - scores every pair in processed/candidate_pairs.csv with the
ALREADY-TRAINED model checkpoint. No training happens here - pure inference.

Rebuilds train_graph with the exact same val/test masking used during
training, but derives which pairs to hide directly from candidate_pairs.csv
(the saved split) instead of re-running the random negative-sampling logic -
this guarantees it can never drift from what the model actually saw.

Uses the v3 checkpoint (output/models/graphsage_v3_real_features.pt) since
that's what's currently saved to disk. Swap --checkpoint once v3 is saved.

Run from the project root, after G_SAGE_model_training.py and
export_candidate_pairs.py (or the training script's own save) have run:
    python src/G_SAGE_model_predict.py
"""
import argparse
import os
import copy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv

processed_dir = "processed"


class HeteroGraphSAGE(nn.Module):
    """Must match G_SAGE_model_training.py's architecture exactly, since
    we're loading its saved weights into this class."""
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


def build_keep_mask(edge_index, hide_pairs, drug_first=True):
    mask = torch.ones(edge_index.shape[1], dtype=torch.bool)
    for i in range(edge_index.shape[1]):
        d = edge_index[0, i].item() if drug_first else edge_index[1, i].item()
        dis = edge_index[1, i].item() if drug_first else edge_index[0, i].item()
        if (d, dis) in hide_pairs:
            mask[i] = False
    return mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=os.path.join("output", "models", "graphsage_v3_real_features.pt"))
    ap.add_argument("--fp-size", type=int, default=256)
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # pandas/numpy first, torch.load last (see earlier crash notes)
    nodes_typed = pd.read_csv(os.path.join(processed_dir, "entities.csv"))
    drug_fingerprints = np.load(os.path.join(processed_dir, "drug_fingerprints.npy"), allow_pickle=True).item()
    pairs = pd.read_csv(os.path.join(processed_dir, "candidate_pairs.csv"))
    data = torch.load(os.path.join(processed_dir, "pharmagraph_hetero_t0.pt"), weights_only=False)

    drug_nodes = nodes_typed[nodes_typed["node_type"] == "drug"].sort_values("local_idx")
    drug_feature_matrix = np.zeros((len(drug_nodes), args.fp_size), dtype=np.float32)
    for _, row in drug_nodes.iterrows():
        if row["node_id"] in drug_fingerprints:
            drug_feature_matrix[row["local_idx"]] = drug_fingerprints[row["node_id"]]
    data["drug"].x = torch.tensor(drug_feature_matrix, dtype=torch.float)

    # rebuild the same val/test edge masking, straight from the saved split
    val_test_pairs = set(zip(
        pairs.loc[(pairs.label == 1) & (pairs.split != "train"), "drug_idx"],
        pairs.loc[(pairs.label == 1) & (pairs.split != "train"), "disease_idx"],
    ))

    train_graph = copy.copy(data)
    ei_fwd = data["drug", "indication", "disease"].edge_index
    train_graph["drug", "indication", "disease"].edge_index = ei_fwd[:, build_keep_mask(ei_fwd, val_test_pairs, True)]
    ei_rev = data["disease", "indication", "drug"].edge_index
    train_graph["disease", "indication", "drug"].edge_index = ei_rev[:, build_keep_mask(ei_rev, val_test_pairs, False)]

    train_graph = train_graph.to(device)

    model = HeteroGraphSAGE(train_graph.node_types, train_graph.edge_types).to(device)
    checkpoint = torch.load(args.checkpoint, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    print(f"Loaded checkpoint: {args.checkpoint}")
    print(f"  (its own recorded test AUROC/AUPRC: "
          f"{checkpoint.get('test_auroc', '?')} / {checkpoint.get('test_auprc', '?')})")

    model.eval()
    with torch.no_grad():
        out = model(train_graph.x_dict, train_graph.edge_index_dict)
        drug_emb, disease_emb = out["drug"], out["disease"]

        d_idx = torch.tensor(pairs["drug_idx"].values, dtype=torch.long, device=device)
        dis_idx = torch.tensor(pairs["disease_idx"].values, dtype=torch.long, device=device)
        scores = score_pairs(drug_emb, disease_emb, d_idx, dis_idx)
        probs = torch.sigmoid(scores).cpu().numpy()

    pairs["model_score"] = scores.cpu().numpy()
    pairs["model_prob"] = probs
    out_path = os.path.join(processed_dir, "model_scores.parquet")
    pairs.to_parquet(out_path, index=False)
    print(f"Saved {out_path}: {len(pairs)} scored pairs")


if __name__ == "__main__":
    main()
