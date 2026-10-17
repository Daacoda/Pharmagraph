"""
Drop-in R-GCN encoder with the SAME interface as the HeteroGraphSAGE:

    model = HeteroRGCN(node_types, edge_types, hidden_channels=128, out_channels=128)
    out = model(x_dict, edge_index_dict)      # dict: node_type -> embedding
    out["drug"], out["disease"]               # used by her score_pairs / loss / eval

Everything else in the pipeline (features, split, masking, loss, dropout, optimizer,
early stopping, evaluation) stays unchanged. Dropout is NOT inside the model, because her
train_epoch applies it to the output embeddings.

Notes:
- Per-type LazyLinear projects each node type's features to a shared width, because
  RGCNConv needs a single feature width. (Her SAGEConv((-1,-1)) handles per-type widths lazily.)
- num_bases=30: full per-relation weights (None) ran out of memory in our environment.
  This makes the model differ from GraphSAGE in weight sharing too - state it in results.
- Lazy layers: do a dry run `with torch.no_grad(): model(x_dict, edge_index_dict)`
  before creating the optimizer, to be safe.
UNTESTED against the src code; check the call sites match.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv


class HeteroRGCN(nn.Module):
    def __init__(self, node_types, edge_types, hidden_channels=128,
                 out_channels=128, num_bases=30):
        super().__init__()
        self.node_types = list(node_types)
        self.edge_types = [tuple(e) for e in edge_types]
        self.rel_id = {et: i for i, et in enumerate(self.edge_types)}
        num_relations = len(self.edge_types)

        self.proj = nn.ModuleDict(
            {nt: nn.LazyLinear(hidden_channels) for nt in self.node_types}
        )
        self.conv1 = RGCNConv(hidden_channels, hidden_channels, num_relations,
                              num_bases=num_bases)
        self.conv2 = RGCNConv(hidden_channels, out_channels, num_relations,
                              num_bases=num_bases)

    def forward(self, x_dict, edge_index_dict):
        # node offsets: concatenate node types in a fixed order
        sizes = {nt: x_dict[nt].size(0) for nt in self.node_types}
        offsets, off = {}, 0
        for nt in self.node_types:
            offsets[nt] = off
            off += sizes[nt]

        x = torch.cat([self.proj[nt](x_dict[nt].float()) for nt in self.node_types], dim=0)

        # pack all relations into one edge_index + edge_type with global relation ids
        eis, ets = [], []
        for et, ei in edge_index_dict.items():
            s, _, d = et
            eis.append(torch.stack([ei[0] + offsets[s], ei[1] + offsets[d]]))
            ets.append(torch.full((ei.size(1),), self.rel_id[tuple(et)],
                                  dtype=torch.long, device=ei.device))
        edge_index = torch.cat(eis, dim=1)
        edge_type = torch.cat(ets)

        x = F.relu(self.conv1(x, edge_index, edge_type))
        x = self.conv2(x, edge_index, edge_type)

        return {nt: x[offsets[nt]: offsets[nt] + sizes[nt]] for nt in self.node_types}
