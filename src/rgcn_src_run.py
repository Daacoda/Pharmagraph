"""
R-GCN run:  GraphSAGE v3 script with ONLY the model swapped (HeteroGraphSAGE -> HeteroRGCN).
Everything else (features, pairs, split, masking, loss, dropout, optimizer, early stopping,
evaluation, twin audit) is copied unchanged from her script.

Edits vs GraphSAGE script (all marked "# EDIT"):
  1. import + use HeteroRGCN (from rgcn_model.py, same folder) instead of HeteroGraphSAGE
  2. dry-run forward before the optimizer (initializes lazy layers)
  3. checkpoint saved under a NEW name (does not overwrite her graphsage checkpoint)
  4. candidate_pairs.csv NOT overwritten (the split is regenerated identically in memory)
  5. her first no-dropout training block (v1) removed - only the v3 block, which she reports
  6. printed labels / hardcoded v1,v2 comparison table removed
  7. optional torch seed line added

UNTESTED. Report this run as R-GCN with num_bases=30 (a labeled variant).
"""
import os
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from rgcn_model import HeteroRGCN   # EDIT 1


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
processed_dir = "processed"
output_dir = "output"

data = torch.load(os.path.join(processed_dir, "pharmagraph_hetero_t0.pt"), weights_only=False)
nodes_typed = pd.read_csv(os.path.join(processed_dir, "entities.csv"))
drug_fingerprints = np.load(os.path.join(processed_dir, "drug_fingerprints.npy"), allow_pickle=True).item()

drug_nodes = nodes_typed[nodes_typed["node_type"] == "drug"].sort_values("local_idx")
FP_SIZE = 256
drug_feature_matrix = np.zeros((len(drug_nodes), FP_SIZE), dtype=np.float32)

found = 0
for _, row in drug_nodes.iterrows():
    if row["node_id"] in drug_fingerprints:
        drug_feature_matrix[row["local_idx"]] = drug_fingerprints[row["node_id"]]
        found += 1

print(f"Drugs with real molecular features: {found} / {len(drug_nodes)}")

data["drug"].x = torch.tensor(drug_feature_matrix, dtype=torch.float, device=device)

indication_edge_index = data["drug", "indication", "disease"].edge_index
drug_idx = indication_edge_index[0]
disease_idx = indication_edge_index[1]

contra_edge_index = data["drug", "contraindication", "disease"].edge_index

positive_pairs = set(zip(drug_idx.tolist(), disease_idx.tolist()))
hard_neg_pairs = set(zip(contra_edge_index[0].tolist(), contra_edge_index[1].tolist()))
clean_hard_neg_pairs = hard_neg_pairs - positive_pairs

n_drugs = data["drug"].num_nodes
n_diseases = data["disease"].num_nodes


def sample_random_negatives(n_samples, n_drugs, n_diseases, positive_pairs, seed=42):
    rng = np.random.default_rng(seed)
    negatives = []
    while len(negatives) < n_samples:
        d = rng.integers(0, n_drugs)
        dis = rng.integers(0, n_diseases)
        if (d, dis) not in positive_pairs:
            negatives.append((d, dis))
    return negatives


random_negatives = sample_random_negatives(len(positive_pairs), n_drugs, n_diseases, positive_pairs)

rng = np.random.default_rng(42)
NEG_RATIO = 2
n_positives = len(positive_pairs)
n_neg_target = n_positives * NEG_RATIO
n_hard_target = min(len(clean_hard_neg_pairs), n_neg_target // 2)
n_random_target = n_neg_target - n_hard_target

clean_hard_neg_list = list(clean_hard_neg_pairs)
sampled_hard_idx = rng.choice(len(clean_hard_neg_list), size=n_hard_target, replace=False)
sampled_hard = [clean_hard_neg_list[i] for i in sampled_hard_idx]
sampled_random = random_negatives[:n_random_target]

final_neg_drug = [p[0] for p in sampled_hard] + [p[0] for p in sampled_random]
final_neg_disease = [p[1] for p in sampled_hard] + [p[1] for p in sampled_random]
final_neg_is_hard = [True] * len(sampled_hard) + [False] * len(sampled_random)

print(f"Positives: {n_positives} | Negatives: {len(final_neg_drug)} (ratio {len(final_neg_drug)/n_positives:.1f}:1)")

from sklearn.model_selection import train_test_split
import copy

pos_drug_arr = np.array(list(zip(*positive_pairs))[0])
pos_disease_arr = np.array(list(zip(*positive_pairs))[1])

pos_train_idx, pos_temp_idx = train_test_split(np.arange(len(pos_drug_arr)), test_size=0.30, random_state=42)
pos_val_idx, pos_test_idx = train_test_split(pos_temp_idx, test_size=0.50, random_state=42)

neg_drug_arr = np.array(final_neg_drug)
neg_disease_arr = np.array(final_neg_disease)
neg_train_idx, neg_temp_idx = train_test_split(np.arange(len(neg_drug_arr)), test_size=0.30, random_state=42)
neg_is_hard_arr = np.array(final_neg_is_hard)
neg_val_idx, neg_test_idx = train_test_split(neg_temp_idx, test_size=0.50, random_state=42)

# EDIT 4: her script writes processed/candidate_pairs.csv here. Skipped so her file is not overwritten;
# the split above is regenerated identically from the same seeds.
print("Split built in memory (candidate_pairs.csv not rewritten).")

val_test_pairs = set()
for idx in np.concatenate([pos_val_idx, pos_test_idx]):
    val_test_pairs.add((pos_drug_arr[idx], pos_disease_arr[idx]))


def build_keep_mask(edge_index, hide_pairs, drug_first=True):
    mask = torch.ones(edge_index.shape[1], dtype=torch.bool)
    for i in range(edge_index.shape[1]):
        d = edge_index[0, i].item() if drug_first else edge_index[1, i].item()
        dis = edge_index[1, i].item() if drug_first else edge_index[0, i].item()
        if (d, dis) in hide_pairs:
            mask[i] = False
    return mask


train_graph = copy.copy(data)

ei_fwd = data["drug", "indication", "disease"].edge_index
mask_fwd = build_keep_mask(ei_fwd, val_test_pairs, drug_first=True)
train_graph["drug", "indication", "disease"].edge_index = ei_fwd[:, mask_fwd]

ei_rev = data["disease", "indication", "drug"].edge_index
mask_rev = build_keep_mask(ei_rev, val_test_pairs, drug_first=False)
train_graph["disease", "indication", "drug"].edge_index = ei_rev[:, mask_rev]

print("Indication edges after masking (train only):", train_graph["drug", "indication", "disease"].edge_index.shape[1])

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def score_pairs(drug_emb, disease_emb, drug_idx, disease_idx):
    d = drug_emb[drug_idx]
    dis = disease_emb[disease_idx]
    return (d * dis).sum(dim=-1)


pos_train_drug = torch.tensor(pos_drug_arr[pos_train_idx], dtype=torch.long, device=device)
pos_train_disease = torch.tensor(pos_disease_arr[pos_train_idx], dtype=torch.long, device=device)
neg_train_drug = torch.tensor(neg_drug_arr[neg_train_idx], dtype=torch.long, device=device)
neg_train_disease = torch.tensor(neg_disease_arr[neg_train_idx], dtype=torch.long, device=device)

val_drug = torch.tensor(np.concatenate([pos_drug_arr[pos_val_idx], neg_drug_arr[neg_val_idx]]), dtype=torch.long, device=device)
val_disease = torch.tensor(np.concatenate([pos_disease_arr[pos_val_idx], neg_disease_arr[neg_val_idx]]), dtype=torch.long, device=device)
val_labels = torch.tensor(np.concatenate([np.ones(len(pos_val_idx)), np.zeros(len(neg_val_idx))]), dtype=torch.float, device=device)

test_drug = torch.tensor(np.concatenate([pos_drug_arr[pos_test_idx], neg_drug_arr[neg_test_idx]]), dtype=torch.long, device=device)
test_disease = torch.tensor(np.concatenate([pos_disease_arr[pos_test_idx], neg_disease_arr[neg_test_idx]]), dtype=torch.long, device=device)
test_labels = torch.tensor(np.concatenate([np.ones(len(pos_test_idx)), np.zeros(len(neg_test_idx))]), dtype=torch.float, device=device)

from sklearn.metrics import roc_auc_score, average_precision_score

# ---------------- MODEL (the only intended difference) ----------------
train_graph = train_graph.to(device)
model = HeteroRGCN(train_graph.node_types, train_graph.edge_types,
                   hidden_channels=128, out_channels=128, num_bases=30).to(device)   # EDIT 1

with torch.no_grad():                                                   # EDIT 2: init lazy layers
    out = model(train_graph.x_dict, train_graph.edge_index_dict)
print("Drug embedding shape:", out["drug"].shape)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
margin_loss_fn = nn.MarginRankingLoss(margin=1.0)
dropout = nn.Dropout(p=0.3)


def train_epoch():
    model.train()
    optimizer.zero_grad()
    out = model(train_graph.x_dict, train_graph.edge_index_dict)
    drug_emb = dropout(out["drug"])
    disease_emb = dropout(out["disease"])

    pos_scores = score_pairs(drug_emb, disease_emb, pos_train_drug, pos_train_disease)
    n_neg = len(neg_train_drug)
    n_pos = len(pos_train_drug)
    repeat_factor = (n_neg // n_pos) + 1
    pos_scores_matched = pos_scores.repeat(repeat_factor)[:n_neg]

    neg_scores = score_pairs(drug_emb, disease_emb, neg_train_drug, neg_train_disease)
    target = torch.ones(n_neg, device=device)
    loss = margin_loss_fn(pos_scores_matched, neg_scores, target)

    loss.backward()
    optimizer.step()
    return loss.item()


def evaluate(split_drug, split_disease, split_labels):
    model.eval()
    with torch.no_grad():
        out = model(train_graph.x_dict, train_graph.edge_index_dict)
        drug_emb = out["drug"]
        disease_emb = out["disease"]
        scores = score_pairs(drug_emb, disease_emb, split_drug, split_disease)
        probs = torch.sigmoid(scores).cpu().numpy()
    labels = split_labels.cpu().numpy()
    return roc_auc_score(labels, probs), average_precision_score(labels, probs)


best_auprc = 0
best_state = None
patience = 10
epochs_no_improve = 0

for epoch in range(1, 101):
    loss = train_epoch()
    if epoch % 5 == 0 or epoch == 1:
        auroc, auprc = evaluate(val_drug, val_disease, val_labels)
        print(f"Epoch {epoch:3d} | Loss: {loss:.4f} | Val AUROC: {auroc:.4f} | Val AUPRC: {auprc:.4f}")
        if auprc > best_auprc:
            best_auprc = auprc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

model.load_state_dict(best_state)
print(f"\nBest Val AUPRC: {best_auprc:.4f}")

test_auroc_v3, test_auprc_v3 = evaluate(test_drug, test_disease, test_labels)

print("=" * 60)
print("FINAL RESULTS - R-GCN (num_bases=30), same pipeline as GraphSAGE v3")   # EDIT 6
print("=" * 60)
print(f"Test AUROC: {test_auroc_v3:.4f}")
print(f"Test AUPRC: {test_auprc_v3:.4f}")

# ---------------- CHEMICAL TWIN AUDIT (her code, unchanged) ----------------
from sklearn.metrics.pairwise import cosine_similarity


def audit_chemical_twin_bias(model, train_graph, drug_feature_matrix,
                              train_drug_ids, test_drug_ids,
                              test_pair_drug_ids, test_pair_disease_ids, test_pair_labels,
                              similarity_threshold=0.95):
    train_fps = drug_feature_matrix[train_drug_ids]
    unique_test_ids = np.unique(test_pair_drug_ids)
    sim_matrix = cosine_similarity(drug_feature_matrix[unique_test_ids], train_fps)
    max_sim = sim_matrix.max(axis=1)
    twin_lookup = dict(zip(unique_test_ids, max_sim > similarity_threshold))

    has_twin = np.array([twin_lookup[d] for d in test_pair_drug_ids])

    model.eval()
    with torch.no_grad():
        out = model(train_graph.x_dict, train_graph.edge_index_dict)
        scores = score_pairs(
            out["drug"], out["disease"],
            torch.tensor(test_pair_drug_ids, device=device),
            torch.tensor(test_pair_disease_ids, device=device),
        )
        probs = torch.sigmoid(scores).cpu().numpy()

    auroc_twin = roc_auc_score(test_pair_labels[has_twin], probs[has_twin])
    auroc_novel = roc_auc_score(test_pair_labels[~has_twin], probs[~has_twin])

    print(f"Test pairs with a chemical twin:   {has_twin.sum():5d} ({has_twin.mean()*100:.1f}%) | AUROC = {auroc_twin:.4f}")
    print(f"Test pairs without a twin (novel): {(~has_twin).sum():5d} ({(1-has_twin.mean())*100:.1f}%) | AUROC = {auroc_novel:.4f}")
    print(f"Gap (twin - novel): {auroc_twin - auroc_novel:+.4f}")

    return auroc_twin, auroc_novel


train_drug_ids = np.unique(np.concatenate([
    pos_drug_arr[pos_train_idx], neg_drug_arr[neg_train_idx]
]))
test_drug_ids = np.unique(np.concatenate([
    pos_drug_arr[pos_test_idx], neg_drug_arr[neg_test_idx]
]))

test_pair_drug_ids = np.concatenate([pos_drug_arr[pos_test_idx], neg_drug_arr[neg_test_idx]])
test_pair_disease_ids = np.concatenate([pos_disease_arr[pos_test_idx], neg_disease_arr[neg_test_idx]])
test_pair_labels = np.concatenate([np.ones(len(pos_test_idx)), np.zeros(len(neg_test_idx))])

audit_chemical_twin_bias(
    model, train_graph, drug_feature_matrix,
    train_drug_ids, test_drug_ids,
    test_pair_drug_ids, test_pair_disease_ids, test_pair_labels,
)

# extra diagnostic (not in her script): how many test pairs involve a drug seen in training?
seen = np.isin(test_pair_drug_ids, train_drug_ids)
print(f"Test pairs whose drug also appears in train pairs: {seen.mean()*100:.1f}%")
for name, m in (("drug seen in train", seen), ("drug NOT seen in train", ~seen)):
    try:
        model.eval()
        with torch.no_grad():
            o = model(train_graph.x_dict, train_graph.edge_index_dict)
            pr = torch.sigmoid(score_pairs(o["drug"], o["disease"],
                                           torch.tensor(test_pair_drug_ids, device=device),
                                           torch.tensor(test_pair_disease_ids, device=device))).cpu().numpy()
        print(f"AUROC ({name}, n={int(m.sum())}): {roc_auc_score(test_pair_labels[m], pr[m]):.4f}")
    except ValueError as e:
        print(f"AUROC ({name}) not computable: {e}")

# EDIT 3: new checkpoint name
save_dir = "output"
os.makedirs(os.path.join(save_dir, "models"), exist_ok=True)
torch.save({
    "model_state": best_state,
    "test_auroc": test_auroc_v3,
    "test_auprc": test_auprc_v3,
    "val_auprc": best_auprc,
    "notes": "R-GCN (num_bases=30) in GraphSAGE v3 pipeline: real Morgan fingerprints (256-dim), "
             "dropout(0.3) + weight_decay(1e-4), indication vs contraindication+random negatives (2:1)"
}, os.path.join(save_dir, "models", "rgcn_v3_numbases30.pt"))

print(f"\nFinal model saved to {save_dir}/models/rgcn_v3_numbases30.pt")
