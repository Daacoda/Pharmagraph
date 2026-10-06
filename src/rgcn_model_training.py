# src/rgcn_model_training.py
import os
import copy
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import roc_auc_score, average_precision_score
from torch_geometric.nn import RGCNConv

# 1. Setup Random Seeds for Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# 2. Load Preprocessed Data & Molecular Features
data_dir = "output/processed" if os.path.exists("output/processed") else "processed"
graph_path = os.path.join(data_dir, "pharmagraph_hetero_t0.pt")
entities_path = os.path.join(data_dir, "entities.csv")
fingerprints_path = os.path.join(data_dir, "drug_fingerprints.npy")

print(f"Loading heterogeneous graph from '{graph_path}'...")
data = torch.load(graph_path, weights_only=False)

entities_df = pd.read_csv(entities_path)
print(f"Loaded entities metadata with {len(entities_df)} total node mappings.")

# Attach real molecular features (Morgan fingerprints) to drug nodes
if os.path.exists(fingerprints_path):
    drug_fingerprints = np.load(fingerprints_path, allow_pickle=True).item()
    drug_nodes = entities_df[entities_df["node_type"] == "drug"].sort_values("local_idx")
    FP_SIZE = 256
    drug_feature_matrix = np.zeros((len(drug_nodes), FP_SIZE), dtype=np.float32)
    
    found = 0
    for _, row in drug_nodes.iterrows():
        if row["node_id"] in drug_fingerprints:
            drug_feature_matrix[row["local_idx"]] = drug_fingerprints[row["node_id"]]
            found += 1
            
    data["drug"].x = torch.tensor(drug_feature_matrix, dtype=torch.float, device=device)
    print(f"Attached real fingerprints for {found} / {len(drug_nodes)} drugs.")
else:
    raise FileNotFoundError(f"Could not find fingerprints at {fingerprints_path}")

data = data.to(device)

# 3. Load Candidate Pairs & Mask Validation/Test Edges (Prevent Leakage)
pairs_path = os.path.join(data_dir, "candidate_pairs.csv")
pairs_df = pd.read_csv(pairs_path)

train_pairs_df = pairs_df[pairs_df["split"] == "train"].reset_index(drop=True)
val_pairs_df = pairs_df[pairs_df["split"] == "val"].reset_index(drop=True)
test_pairs_df = pairs_df[pairs_df["split"] == "test"].reset_index(drop=True)

train_data = copy.deepcopy(data)
train_pos_df = train_pairs_df[train_pairs_df["label"] == 1]
train_drug_ids = torch.tensor(train_pos_df["drug_id"].values, dtype=torch.long, device=device)
train_disease_ids = torch.tensor(train_pos_df["disease_id"].values, dtype=torch.long, device=device)

train_data["drug", "indication", "disease"].edge_index = torch.stack([train_drug_ids, train_disease_ids], dim=0)
train_data["disease", "rev_indication", "drug"].edge_index = torch.stack([train_disease_ids, train_drug_ids], dim=0)
print("Data leakage check passed: Validation and test edges masked.")

# 4. Relational Graph Packing & Global Index Offsetting
node_types = list(train_data.node_types)
node_offsets = {}
curr_offset = 0

for n_type in node_types:
    node_offsets[n_type] = curr_offset
    curr_offset += train_data[n_type].num_nodes

total_nodes = curr_offset
edge_types = list(train_data.edge_types)
rel_to_id = {rel: i for i, rel in enumerate(edge_types)}

all_edges, all_type_ids = [], []
for rel in edge_types:
    src_type, _, dst_type = rel
    rel_id = rel_to_id[rel]
    edge_index = train_data[rel].edge_index

    if edge_index.numel() > 0:
        global_src = edge_index[0] + node_offsets[src_type]
        global_dst = edge_index[1] + node_offsets[dst_type]
        global_edge_index = torch.stack([global_src, global_dst], dim=0)
        type_ids = torch.full((global_edge_index.shape[1],), rel_id, dtype=torch.long, device=device)

        all_edges.append(global_edge_index)
        all_type_ids.append(type_ids)

unified_edge_index = torch.cat(all_edges, dim=1)
unified_edge_type = torch.cat(all_type_ids, dim=0)
num_relations = len(edge_types)

# 5. R-GCN Model Architecture Definition
class HeteroRGCN(nn.Module):
    def __init__(self, num_nodes, num_relations, in_dim=256, hidden_dim=128, out_dim=128, num_bases=30, dropout=0.3):
        super().__init__()
        self.node_emb = nn.Embedding(num_nodes, hidden_dim)
        nn.init.xavier_uniform_(self.node_emb.weight)
        self.drug_proj = nn.Linear(in_dim, hidden_dim)
        
        self.conv1 = RGCNConv(hidden_dim, hidden_dim, num_relations=num_relations, num_bases=num_bases)
        self.conv2 = RGCNConv(hidden_dim, out_dim, num_relations=num_relations, num_bases=num_bases)
        self.dropout = nn.Dropout(dropout)

    def forward(self, edge_index, edge_type, drug_x, drug_offset, num_drugs):
        x = self.node_emb.weight.clone()
        drug_feats = self.drug_proj(drug_x)
        x[drug_offset:drug_offset + num_drugs] = x[drug_offset:drug_offset + num_drugs] + drug_feats

        x = self.conv1(x, edge_index, edge_type)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index, edge_type)
        return x

model = HeteroRGCN(
    num_nodes=total_nodes,
    num_relations=num_relations,
    in_dim=256, hidden_dim=128, out_dim=128, num_bases=30, dropout=0.3
).to(device)

# 6. Evaluation and Loss Functions
def compute_pair_scores(embeddings, pair_df, node_offsets):
    drug_global_ids = torch.tensor(pair_df["drug_id"].values, dtype=torch.long, device=device) + node_offsets["drug"]
    disease_global_ids = torch.tensor(pair_df["disease_id"].values, dtype=torch.long, device=device) + node_offsets["disease"]
    
    drug_embs = embeddings[drug_global_ids]
    disease_embs = embeddings[disease_global_ids]
    return (drug_embs * disease_embs).sum(dim=1)

def evaluate_metrics(embeddings, pair_df, node_offsets):
    scores = compute_pair_scores(embeddings, pair_df, node_offsets).detach().cpu().numpy()
    labels = pair_df["label"].values
    return roc_auc_score(labels, scores), average_precision_score(labels, scores)

loss_fn = nn.MarginRankingLoss(margin=1.0)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

# 7. Training Loop
train_pos_df = train_pairs_df[train_pairs_df["label"] == 1].reset_index(drop=True)
train_neg_df = train_pairs_df[train_pairs_df["label"] == 0].reset_index(drop=True)
min_len = min(len(train_pos_df), len(train_neg_df))
train_pos_df = train_pos_df.iloc[:min_len]
train_neg_df = train_neg_df.iloc[:min_len]

EPOCHS = 100
EVAL_EVERY = 5
PATIENCE = 15
best_val_auroc = 0.0
patience_counter = 0
best_model_weights = None

drug_offset = node_offsets["drug"]
num_drugs = train_data["drug"].num_nodes

print("Starting R-GCN Training Loop...")
for epoch in range(1, EPOCHS + 1):
    model.train()
    optimizer.zero_dict if hasattr(optimizer, 'zero_dict') else optimizer.zero_grad()

    embeddings = model(unified_edge_index, unified_edge_type, train_data["drug"].x, drug_offset, num_drugs)
    pos_scores = compute_pair_scores(embeddings, train_pos_df, node_offsets)
    neg_scores = compute_pair_scores(embeddings, train_neg_df, node_offsets)

    target = torch.ones(pos_scores.shape[0], device=device)
    loss = loss_fn(pos_scores, neg_scores, target)

    loss.backward()
    optimizer.step()

    if epoch % EVAL_EVERY == 0 or epoch == 1:
        model.eval()
        with torch.no_grad():
            val_auroc, val_auprc = evaluate_metrics(embeddings, val_pairs_df, node_offsets)
        print(f"Epoch {epoch:03d}/{EPOCHS} | Loss: {loss.item():.4f} | Val AUROC: {val_auroc:.4f} | Val AUPRC: {val_auprc:.4f}")

        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            patience_counter = 0
            best_model_weights = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1
            if patience_counter >= (PATIENCE // EVAL_EVERY):
                print(f"Early stopping triggered at epoch {epoch}!")
                break

if best_model_weights is not None:
    model.load_state_dict(best_model_weights)

# 8. Test Evaluation & Saving Weights
model.eval()
with torch.no_grad():
    test_embeddings = model(unified_edge_index, unified_edge_type, train_data["drug"].x, drug_offset, num_drugs)
    test_auroc, test_auprc = evaluate_metrics(test_embeddings, test_pairs_df, node_offsets)

print("=" * 50)
print(f"Final Test AUROC : {test_auroc:.4f}")
print(f"Final Test AUPRC : {test_auprc:.4f}")
print("=" * 50)

os.makedirs("output/models", exist_ok=True)
save_path = "output/models/rgcn_v1_real_features.pt"
torch.save({
    "model_state_dict": model.state_dict(),
    "test_auroc": test_auroc,
    "test_auprc": test_auprc,
    "node_offsets": node_offsets,
    "num_relations": num_relations
}, save_path)
print(f"Model saved successfully to {save_path}!")