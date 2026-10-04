"""
PharmaGraph - Relational Graph Convolutional Network (R-GCN) Architecture
========================================================================
2-layer R-GCN with basis-decomposition regularizing 53 biological relation types
into a shared basis subspace (num_bases=16) to prevent parameter explosion and
satisfy commodity CPU memory bounds.
"""

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv


class HeteroRGCN(nn.Module):
    """2-layer Relational GCN with basis-decomposition across 53 relation types."""

    def __init__(
        self,
        node_types: List[str],
        edge_types: List[Tuple[str, str, str]],
        in_channels_dict: Optional[Dict[str, int]] = None,
        hidden_channels: int = 64,
        out_channels: int = 64,
        num_bases: int = 16,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.node_types = list(node_types)
        self.edge_types = list(edge_types)
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_bases = num_bases
        self.dropout = dropout
        self.num_relations = len(edge_types)

        # Map each canonical (src, rel, dst) tuple to a unique relation index [0, num_relations - 1]
        self.edge_type_to_idx = {et: idx for idx, et in enumerate(edge_types)}

        # Entity-specific input projections
        self.proj: Optional[nn.ModuleDict] = None
        if in_channels_dict is not None:
            self.proj = nn.ModuleDict({
                nt: nn.Linear(in_dim, hidden_channels)
                for nt, in_dim in in_channels_dict.items()
            })
        else:
            # Fallback 1-dim projection if unspecified
            self.proj = nn.ModuleDict({
                nt: nn.Linear(1, hidden_channels)
                for nt in node_types
            })

        # Layer 1: Basis-decomposed RGCN
        self.conv1 = RGCNConv(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            num_relations=self.num_relations,
            num_bases=self.num_bases,
        )

        # Layer 2: Basis-decomposed RGCN
        self.conv2 = RGCNConv(
            in_channels=hidden_channels,
            out_channels=out_channels,
            num_relations=self.num_relations,
            num_bases=self.num_bases,
        )

    def forward(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Convert heterogeneous tensors into homogeneous space, run RGCN, and unflatten."""
        # 1. Project node features to hidden_channels and record offsets
        projected_x: List[torch.Tensor] = []
        node_offsets: Dict[str, int] = {}
        node_counts: Dict[str, int] = {}
        current_offset = 0

        for nt in self.node_types:
            x = x_dict[nt]
            if self.proj is not None and nt in self.proj:
                h = F.relu(self.proj[nt](x))
            else:
                h = x
            projected_x.append(h)
            node_offsets[nt] = current_offset
            node_counts[nt] = h.size(0)
            current_offset += h.size(0)

        # Global node representation tensor
        x_all = torch.cat(projected_x, dim=0)

        # 2. Shift edge indices by global node offsets and assign integer relation IDs
        edge_indices: List[torch.Tensor] = []
        edge_types: List[torch.Tensor] = []

        for et, ei in edge_index_dict.items():
            if et not in self.edge_type_to_idx:
                continue
            rel_idx = self.edge_type_to_idx[et]
            src_type, _, dst_type = et
            src_offset = node_offsets[src_type]
            dst_offset = node_offsets[dst_type]

            shifted_ei = torch.stack([
                ei[0] + src_offset,
                ei[1] + dst_offset,
            ], dim=0)

            edge_indices.append(shifted_ei)
            edge_types.append(torch.full((ei.size(1),), rel_idx, dtype=torch.long, device=ei.device))

        if len(edge_indices) > 0:
            edge_index_all = torch.cat(edge_indices, dim=1)
            edge_type_all = torch.cat(edge_types, dim=0)
        else:
            edge_index_all = torch.empty((2, 0), dtype=torch.long, device=x_all.device)
            edge_type_all = torch.empty((0,), dtype=torch.long, device=x_all.device)

        # 3. Layer 1 Convolution + ReLU + Dropout
        h1 = F.relu(self.conv1(x_all, edge_index_all, edge_type_all))
        h1 = F.dropout(h1, p=self.dropout, training=self.training)

        # 4. Layer 2 Convolution
        out = self.conv2(h1, edge_index_all, edge_type_all)

        # 5. Unflatten global node tensor back into heterogeneous dictionary
        out_dict: Dict[str, torch.Tensor] = {}
        for nt in self.node_types:
            start = node_offsets[nt]
            count = node_counts[nt]
            out_dict[nt] = out[start : start + count]

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
        """Compute raw sigmoid probabilities."""
        scores = self.score_pairs(drug_emb, disease_emb, drug_idx, disease_idx)
        return torch.sigmoid(scores)
