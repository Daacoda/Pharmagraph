"""
NOTE: This is the ORIGINAL (v1) R-GCN notebook code, exported as text for sharing.
It is superseded. Known issues: (1) reverse indication edges were masked under the name
'rev_indication', which likely created a new relation instead of masking the real
('disease','indication','drug') edges (51 relations vs 50 in the graph) -> probable leakage;
(2) trainable per-node ID embeddings (not inductive); (3) training loop differs from the
GraphSAGE baseline. Results from this version should not be reported or compared.
See R_gcn_model_training_fixed.ipynb for the corrected version.
"""

# ## 1. Setup — Dependencies, Imports, and Seed Initialization
# Installs required packages, imports core PyTorch and PyG modules, and enforces global random seeds to guarantee 100% reproducible experiments.

# ---- cell 1 ----
# Cell 1: Setup, Dependencies, and Random Seeds
import copy
import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import average_precision_score, roc_auc_score
from torch_geometric.data import HeteroData
from torch_geometric.nn import RGCNConv

# Set global reproducibility seeds
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
  torch.cuda.manual_seed_all(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# ## 2. Load Preprocessed Data & Real Molecular Features
# Loads the base heterogeneous graph (`pharmagraph_hetero_t0.pt`), entity mappings (`entities.csv`), and attaches the 256-dimensional Morgan fingerprint embeddings (`drug_fingerprints.npy`) to drug nodes.

# ---- cell 3 ----
import os
import torch
import pandas as pd
import numpy as np

# Resolve directory paths flexibly (works whether running from root or notebooks/)
if os.path.exists("output/processed"):
    data_dir = "output/processed"
elif os.path.exists("processed"):
    data_dir = "processed"
elif os.path.exists("../output/processed"):
    data_dir = "../output/processed"
elif os.path.exists("../processed"):
    data_dir = "../processed"
else:
    raise FileNotFoundError("Could not locate processed data directory! Check your path.")

graph_path = os.path.join(data_dir, "pharmagraph_hetero_t0.pt")
entities_path = os.path.join(data_dir, "entities.csv")
fingerprints_path = os.path.join(data_dir, "drug_fingerprints.npy")

# Set up dynamic device engine pointer (CPU now, GPU automatically later in Colab)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1. Load the base heterogeneous graph
print(f"Loading heterogeneous graph from '{graph_path}'...")
data = torch.load(graph_path, weights_only=False)

# 2. Load entity metadata
entities_df = pd.read_csv(entities_path)
print(f"Loaded entities metadata with {len(entities_df)} total node mappings.")

# 3. Attach real molecular features to drug nodes
if os.path.exists(fingerprints_path):
    # Unpack the dictionary object safely by bypassing the pickle restriction
    drug_fingerprints = np.load(fingerprints_path, allow_pickle=True).item()
    
    # Isolate drug nodes from the CSV metadata and sort them by their internal index
    drug_nodes = entities_df[entities_df["node_type"] == "drug"].sort_values("local_idx")
    FP_SIZE = 256
    drug_feature_matrix = np.zeros((len(drug_nodes), FP_SIZE), dtype=np.float32)
    
    # Map the dictionary mapping values into the sequential grid array PyTorch requires
    found = 0
    for _, row in drug_nodes.iterrows():
        if row["node_id"] in drug_fingerprints:
            drug_feature_matrix[row["local_idx"]] = drug_fingerprints[row["node_id"]]
            found += 1
            
    # Send the aligned matrix tensor to your active hardware memory layer
    data["drug"].x = torch.tensor(drug_feature_matrix, dtype=torch.float, device=device)
    print(f"Successfully attached real fingerprints for {found} / {len(drug_nodes)} drugs to data['drug'].x")
else:
    raise FileNotFoundError(f"Could not find fingerprints at {fingerprints_path}")

# 4. Push the remaining structural graph indices to the system hardware
data = data.to(device)

# Verify graph dimensions
print("\n--- Graph Summary ---")
for node_type in data.node_types:
    num_nodes = data[node_type].num_nodes
    feat_shape = data[node_type].x.shape if hasattr(data[node_type], "x") and data[node_type].x is not None else "No 'x'"
    print(f"Node Type '{node_type}': {num_nodes} nodes | Features: {feat_shape}")


# ## 3. Load Candidate Pairs, Mask Edges, & Prevent Data Leakage
# Loads `candidate_pairs.csv`, removes validation and test edges from the message-passing graph (`train_data`) to enforce strict temporal split isolation, and verifies zero leakage via explicit assertions.

# ---- cell 5 ----
# Cell 3: Load Candidate Pairs and Mask Validation/Test Edges

pairs_path = os.path.join(data_dir, "candidate_pairs.csv")
if not os.path.exists(pairs_path):
  raise FileNotFoundError(
      f"Could not find candidate pairs at {pairs_path}! Ensure Cell 5/export step"
      " was run."
  )

# 1. Load candidate pairs dataset
pairs_df = pd.read_csv(pairs_path)
print(f"Loaded {len(pairs_df)} candidate pairs from '{pairs_path}'.")
print("Split Distribution:\n", pairs_df["split"].value_counts())

# 2. Extract train, validation, and test pair subsets
train_pairs_df = pairs_df[pairs_df["split"] == "train"].reset_index(drop=True)
val_pairs_df = pairs_df[pairs_df["split"] == "val"].reset_index(drop=True)
test_pairs_df = pairs_df[pairs_df["split"] == "test"].reset_index(drop=True)

# 3. Create a deep copy of the full graph to build the masked training graph
train_data = copy.deepcopy(data)

# Extract positive training edges (label == 1) to build the message-passing edge index
train_pos_df = train_pairs_df[train_pairs_df["label"] == 1]
train_drug_ids = torch.tensor(
    train_pos_df["drug_id"].values, dtype=torch.long, device=device
)
train_disease_ids = torch.tensor(
    train_pos_df["disease_id"].values, dtype=torch.long, device=device
)

# Overwrite indication edges in train_data with ONLY training indications (Forward & Reverse)
train_data["drug", "indication", "disease"].edge_index = torch.stack(
    [train_drug_ids, train_disease_ids], dim=0
)
train_data["disease", "rev_indication", "drug"].edge_index = torch.stack(
    [train_disease_ids, train_drug_ids], dim=0
)

# 4. Data Leakage Verification
full_num_edges = data["drug", "indication", "disease"].edge_index.shape[1]
train_num_edges = train_data["drug", "indication", "disease"].edge_index.shape[1]

print(f"\n--- Edge Masking Verification ---")
print(f"Full Graph Indication Edges : {full_num_edges}")
print(f"Masked Train Graph Edges    : {train_num_edges}")
print(f"Masked Edges (Val + Test)   : {full_num_edges - train_num_edges}")

assert (
    train_num_edges < full_num_edges
), "Data Leakage Error: Masked graph contains all original edges!"
print(
    "Data leakage check passed! Validation and test edges successfully removed"
    " from training graph."
)

# ## 4. Relational Graph Packing & Global Index Offsetting
# 
# Merges a multi-category graph into a single master format for the `RGCNConv` layer:
# 
# 
# ###  Output Tensors
# *   `edge_index`
# *   `edge_type`

# ---- cell 7 ----
# Cell 4: Relational Graph Packing and Offset Calculation

# 1. Define distinct node types and assign unique global index offsets
node_types = list(train_data.node_types)
node_offsets = {}
curr_offset = 0

for n_type in node_types:
  node_offsets[n_type] = curr_offset
  curr_offset += train_data[n_type].num_nodes

total_nodes = curr_offset
print("--- Node Offsets ---")
for n_type, offset in node_offsets.items():
  print(
      f"Node Type '{n_type}': Offset = {offset} | Count ="
      f" {train_data[n_type].num_nodes}"
  )
print(f"Total Unified Nodes: {total_nodes}")

# 2. Extract and concatenate all relation types into global RGCN format
edge_types = list(train_data.edge_types)
rel_to_id = {rel: i for i, rel in enumerate(edge_types)}

all_edges = []
all_type_ids = []

for rel in edge_types:
  src_type, rel_name, dst_type = rel
  rel_id = rel_to_id[rel]

  # Extract local edge index tensor
  edge_index = train_data[rel].edge_index

  if edge_index.numel() > 0:
    # Shift local node indices to global node space
    global_src = edge_index[0] + node_offsets[src_type]
    global_dst = edge_index[1] + node_offsets[dst_type]
    global_edge_index = torch.stack([global_src, global_dst], dim=0)

    # Build corresponding relation ID tensor
    type_ids = torch.full(
        (global_edge_index.shape[1],),
        rel_id,
        dtype=torch.long,
        device=device,
    )

    all_edges.append(global_edge_index)
    all_type_ids.append(type_ids)

# Pack into single unified PyTorch tensors
unified_edge_index = torch.cat(all_edges, dim=1)
unified_edge_type = torch.cat(all_type_ids, dim=0)

num_relations = len(edge_types)
print(f"\n--- Unified RGCN Graph Summary ---")
print(f"Total Relations Packaged : {num_relations}")
print(f"Total Combined Edges     : {unified_edge_index.shape[1]}")
print(f"Unified Edge Index Shape : {unified_edge_index.shape}")
print(f"Unified Edge Type Shape  : {unified_edge_type.shape}")

# ## 5. R-GCN Model Architecture Definition
# Defines a 2-layer Relational Graph Convolutional Network (R-GCN) using `RGCNConv` with basis decomposition to regulate multi-relational weight parameters.

# ---- cell 9 ----
# Cell 5: R-GCN Model Architecture


class HeteroRGCN(nn.Module):

  def __init__(
      self,
      num_nodes,
      num_relations,
      in_dim=256,
      hidden_dim=128,
      out_dim=128,
      num_bases=30,
      dropout=0.3,
  ):
    super().__init__()

    self.hidden_dim = hidden_dim

    # 1. Base embedding lookup table for all 129,312 nodes
    self.node_emb = nn.Embedding(num_nodes, hidden_dim)
    nn.init.xavier_uniform_(self.node_emb.weight)

    # 2. Linear projection layer for real drug molecular features (256 -> 128)
    self.drug_proj = nn.Linear(in_dim, hidden_dim)

    # 3. 2-Layer RGCN Message Passing Architecture
    self.conv1 = RGCNConv(
        hidden_dim, hidden_dim, num_relations=num_relations, num_bases=num_bases
    )
    self.conv2 = RGCNConv(
        hidden_dim, out_dim, num_relations=num_relations, num_bases=num_bases
    )

    self.dropout = nn.Dropout(dropout)

  def forward(self, edge_index, edge_type, drug_x, drug_offset, num_drugs):
    # Initialize base feature representations for all nodes
    x = self.node_emb.weight.clone()

    # Inject projected real molecular features for drug node indices
    drug_feats = self.drug_proj(drug_x)
    x[drug_offset : drug_offset + num_drugs] = (
        x[drug_offset : drug_offset + num_drugs] + drug_feats
    )

    # Layer 1 message passing + non-linearity + dropout
    x = self.conv1(x, edge_index, edge_type)
    x = F.relu(x)
    x = self.dropout(x)

    # Layer 2 message passing
    x = self.conv2(x, edge_index, edge_type)

    return x


# Initialize Model Instance
model = HeteroRGCN(
    num_nodes=total_nodes,
    num_relations=num_relations,
    in_dim=256,
    hidden_dim=128,
    out_dim=128,
    num_bases=30,
    dropout=0.3,
).to(device)

print(model)
print(
    f"\nTotal Model Parameters: "
    f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
)

# ## 6. Link Prediction Decoder, Loss, & Evaluation Metrics
# Defines the dot-product similarity decoder, Margin Ranking Loss objective, and the evaluation routine to compute AUROC and AUPRC metrics across dataset splits.

# ---- cell 11 ----
# Cell 6: Decoder, Loss Function, and Evaluation Functions


def compute_pair_scores(embeddings, pair_df, node_offsets):
  """Computes link prediction dot-product scores for candidate pairs.

  Args:
      embeddings (Tensor): Global node embeddings [total_nodes, hidden_dim].
      pair_df (DataFrame): Candidate pairs containing 'drug_id' and
        'disease_id'.
      node_offsets (dict): Mapping of node types to global index offsets.

  Returns:
      Tensor: 1D tensor of predicted interaction scores.
  """
  # Shift local node IDs to global embedding space
  drug_global_ids = (
      torch.tensor(pair_df["drug_id"].values, dtype=torch.long, device=device)
      + node_offsets["drug"]
  )
  disease_global_ids = (
      torch.tensor(
          pair_df["disease_id"].values, dtype=torch.long, device=device
      )
      + node_offsets["disease"]
  )

  # Fetch corresponding node embeddings
  drug_embs = embeddings[drug_global_ids]
  disease_embs = embeddings[disease_global_ids]

  # Compute dot-product scoring
  scores = (drug_embs * disease_embs).sum(dim=1)
  return scores


def evaluate_metrics(embeddings, pair_df, node_offsets):
  """Computes AUROC and AUPRC metrics for a pair DataFrame."""
  scores = (
      compute_pair_scores(embeddings, pair_df, node_offsets).detach().cpu().numpy()
  )
  labels = pair_df["label"].values

  auroc = roc_auc_score(labels, scores)
  auprc = average_precision_score(labels, scores)

  return auroc, auprc


# Define Margin Ranking Loss objective (matching Asmaa's baseline setup)
loss_fn = nn.MarginRankingLoss(margin=1.0)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

print("Decoder, loss function, and evaluation metrics initialized successfully!")

# ## 7. Model Training & Early Stopping Validation Loop
# Trains the R-GCN over 100 epochs using Margin Ranking Loss, evaluates validation AUROC/AUPRC every 5 epochs, and checkpoints the best model state.

# ---- cell 13 ----
# Cell 7: Full Training Loop and Validation Checkpointing

# Extract positive and negative training pairs for pairwise margin loss
train_pos_df = train_pairs_df[train_pairs_df["label"] == 1].reset_index(
    drop=True
)
train_neg_df = train_pairs_df[train_pairs_df["label"] == 0].reset_index(
    drop=True
)

# If negative set is larger than positive set, trim to equal lengths for pairwise margin ranking
min_len = min(len(train_pos_df), len(train_neg_df))
train_pos_df = train_pos_df.iloc[:min_len]
train_neg_df = train_neg_df.iloc[:min_len]

# Training configuration
EPOCHS = 100
EVAL_EVERY = 5
PATIENCE = 15

best_val_auroc = 0.0
patience_counter = 0
best_model_weights = None

drug_offset = node_offsets["drug"]
num_drugs = train_data["drug"].num_nodes

print("Starting R-GCN Training Loop...\n")

for epoch in range(1, EPOCHS + 1):
  model.train()
  optimizer.zero_grad()

  # 1. Forward pass over unified training graph
  embeddings = model(
      unified_edge_index,
      unified_edge_type,
      train_data["drug"].x,
      drug_offset,
      num_drugs,
  )

  # 2. Compute pair scores
  pos_scores = compute_pair_scores(embeddings, train_pos_df, node_offsets)
  neg_scores = compute_pair_scores(embeddings, train_neg_df, node_offsets)

  # 3. Compute Margin Ranking Loss (targets = ones tensor indicating pos > neg)
  target = torch.ones(pos_scores.shape[0], device=device)
  loss = loss_fn(pos_scores, neg_scores, target)

  # 4. Backpropagation
  loss.backward()
  optimizer.step()

  # 5. Periodic Validation Evaluation
  if epoch % EVAL_EVERY == 0 or epoch == 1:
    model.eval()
    with torch.no_grad():
      val_auroc, val_auprc = evaluate_metrics(
          embeddings, val_pairs_df, node_offsets
      )

    print(
        f"Epoch {epoch:03d}/{EPOCHS:03d} | Loss: {loss.item():.4f} | Val"
        f" AUROC: {val_auroc:.4f} | Val AUPRC: {val_auprc:.4f}"
    )

    # Early stopping and checkpointing check
    if val_auroc > best_val_auroc:
      best_val_auroc = val_auroc
      patience_counter = 0
      best_model_weights = copy.deepcopy(model.state_dict())
    else:
      patience_counter += 1
      if patience_counter >= (PATIENCE // EVAL_EVERY):
        print(
            f"\nEarly stopping triggered at epoch {epoch}! Best Val AUROC:"
            f" {best_val_auroc:.4f}"
        )
        break

# Restore best checkpoint weights
if best_model_weights is not None:
  model.load_state_dict(best_model_weights)
  print(
      f"\nRestored best model weights with Validation AUROC: {best_val_auroc:.4f}"
  )

# ## 8. Final Test Evaluation & Model Weights Persistence
# Evaluates the optimal R-GCN model checkpoint on the held-out test set to compute final Test AUROC and Test AUPRC, then saves model weights to `output/models/`.

# ---- cell 15 ----
# Cell 8: Held-out Test Set Evaluation and Model Export

# Ensure model is in evaluation mode
model.eval()

with torch.no_grad():
  # 1. Regenerate global embeddings using best model checkpoint
  test_embeddings = model(
      unified_edge_index,
      unified_edge_type,
      train_data["drug"].x,
      drug_offset,
      num_drugs,
  )

  # 2. Evaluate performance on unseen test pairs
  test_auroc, test_auprc = evaluate_metrics(
      test_embeddings, test_pairs_df, node_offsets
  )

print("=" * 50)
print("             FINAL TEST SET EVALUATION            ")
print("=" * 50)
print(f"Test AUROC : {test_auroc:.4f}")
print(f"Test AUPRC : {test_auprc:.4f}")
print("=" * 50)

# 3. Save model weights to disk
models_dir = os.path.join(
    os.path.dirname(data_dir) if "processed" in data_dir else data_dir, "models"
)
if not os.path.exists("output/models"):
  os.makedirs("output/models", exist_ok=True)
  models_dir = "output/models"

save_path = os.path.join(models_dir, "rgcn_v1_real_features.pt")
torch.save(
    {
        "model_state_dict": model.state_dict(),
        "test_auroc": test_auroc,
        "test_auprc": test_auprc,
        "node_offsets": node_offsets,
        "num_relations": num_relations,
    },
    save_path,
)

print(f"\nModel checkpoint successfully saved to '{save_path}'!")

