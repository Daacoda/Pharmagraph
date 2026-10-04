"""
PharmaGraph - Heterogeneous GraphSAGE Architecture
==================================================
2-layer Heterogeneous GraphSAGE model implemented via PyTorch Geometric.
Performs bipartite message passing across diverse biological relation types,
incorporates entity projection, dropout regularization, and unweighted inner
product link decoding.
"""

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv


class HeteroGraphSAGE(nn.Module):
    """2-layer Heterogeneous GraphSAGE with bipartite message passing."""

    def __init__(
        self,
        node_types: List[str],
        edge_types: List[Tuple[str, str, str]],
        in_channels_dict: Optional[Dict[str, int]] = None,
        hidden_channels: int = 64,
        out_channels: int = 64,
        dropout: float = 0.2,
        aggr: str = "mean",
    ) -> None:
        super().__init__()
        self.node_types = list(node_types)
        self.edge_types = list(edge_types)
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.dropout = dropout
        self.aggr = aggr

        # Optional linear projection for heterogeneous input channels
        self.proj: Optional[nn.ModuleDict] = None
        if in_channels_dict is not None:
            self.proj = nn.ModuleDict({
                nt: nn.Linear(in_dim, hidden_channels)
                for nt, in_dim in in_channels_dict.items()
            })
            conv1_in = (hidden_channels, hidden_channels)
        else:
            # Lazy channel initialization when input dimensions are unspecified
            conv1_in = (-1, -1)

        # Layer 1: HeteroConv across all relation pairs
        self.conv1 = HeteroConv(
            {et: SAGEConv(conv1_in, hidden_channels) for et in edge_types},
            aggr=aggr,
        )

        # Layer 2: HeteroConv producing output latent embeddings
        self.conv2 = HeteroConv(
            {et: SAGEConv((hidden_channels, hidden_channels), out_channels) for et in edge_types},
            aggr=aggr,
        )

    def forward(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Forward pass generating updated node embeddings for all entity types."""
        # Step 1: Optional input projection
        if self.proj is not None:
            h_dict = {}
            for nt, x in x_dict.items():
                if nt in self.proj:
                    h_dict[nt] = F.relu(self.proj[nt](x))
                else:
                    h_dict[nt] = x
        else:
            h_dict = x_dict

        # Step 2: Layer 1 convolution + ReLU + Dropout
        h_dict = self.conv1(h_dict, edge_index_dict)
        h_dict = {
            nt: F.dropout(F.relu(h), p=self.dropout, training=self.training)
            for nt, h in h_dict.items()
        }

        # Step 3: Layer 2 convolution
        out_dict = self.conv2(h_dict, edge_index_dict)
        return out_dict

    def score_pairs(
        self,
        drug_emb: torch.Tensor,
        disease_emb: torch.Tensor,
        drug_idx: torch.Tensor,
        disease_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Compute scalar inner-product margins: s(d, y) = <z_d, z_y>."""
        return (drug_emb[drug_idx] * disease_emb[disease_idx]).sum(dim=-1)

    def predict_proba(
        self,
        drug_emb: torch.Tensor,
        disease_emb: torch.Tensor,
        drug_idx: torch.Tensor,
        disease_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Compute raw sigmoid probabilities (uncalibrated baseline)."""
        scores = self.score_pairs(drug_emb, disease_emb, drug_idx, disease_idx)
        return torch.sigmoid(scores)
