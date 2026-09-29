"""
PharmaGraph - Disjoint Negative Sampler & Bidirectional Masking
==============================================================
Mines clean clinical contraindication hard negatives disjoint from positive
indications, samples random unobserved non-edges with an exact 2:1 ratio,
and constructs message-passing graphs with bidirectional leakage masking.
"""

import copy
from typing import Dict, List, Optional, Set, Tuple
import numpy as np
import torch
from torch_geometric.data import HeteroData


class DisjointNegativeSampler:
    """Samples disjoint hard negatives from contraindications and unobserved random non-edges."""

    def __init__(
        self,
        num_drugs: int,
        num_diseases: int,
        positive_pairs: Set[Tuple[int, int]],
        contra_pairs: Set[Tuple[int, int]],
        seed: int = 42,
    ) -> None:
        self.num_drugs = num_drugs
        self.num_diseases = num_diseases
        self.positive_pairs = set(positive_pairs)
        # Clinical hard negatives strictly disjoint from positive indications
        self.clean_hard_neg_pairs = list(set(contra_pairs) - self.positive_pairs)
        self.rng = np.random.default_rng(seed)

    def sample_negatives(
        self,
        n_positives: int,
        ratio: float = 2.0,
        hard_fraction: float = 0.5,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample negatives with a specified ratio.
        Returns:
            (neg_drugs, neg_diseases, is_hard_flags)
        """
        n_neg_total = int(n_positives * ratio)
        n_hard = min(len(self.clean_hard_neg_pairs), int(n_neg_total * hard_fraction))
        n_rand = n_neg_total - n_hard

        # 1. Sample hard negatives from clean contraindications
        hard_indices = self.rng.choice(len(self.clean_hard_neg_pairs), size=n_hard, replace=False)
        hard_samples = [self.clean_hard_neg_pairs[i] for i in hard_indices]

        # 2. Sample random unobserved non-edges
        rand_samples: List[Tuple[int, int]] = []
        rand_set: Set[Tuple[int, int]] = set()

        while len(rand_samples) < n_rand:
            d = int(self.rng.integers(0, self.num_drugs))
            dis = int(self.rng.integers(0, self.num_diseases))
            pair = (d, dis)
            if pair not in self.positive_pairs and pair not in rand_set:
                rand_set.add(pair)
                rand_samples.append(pair)

        # 3. Combine and flag
        all_drugs = [p[0] for p in hard_samples] + [p[0] for p in rand_samples]
        all_diseases = [p[1] for p in hard_samples] + [p[1] for p in rand_samples]
        is_hard = np.array([True] * n_hard + [False] * n_rand, dtype=bool)

        # Permute uniformly
        perm = self.rng.permutation(len(all_drugs))
        return (
            np.array(all_drugs, dtype=np.int64)[perm],
            np.array(all_diseases, dtype=np.int64)[perm],
            is_hard[perm],
        )


def build_masked_train_graph(
    data: HeteroData,
    val_test_pairs: Set[Tuple[int, int]],
) -> HeteroData:
    """
    Builds a copy of the graph with validation and test indication edges masked
    out in BOTH forward and reverse directions to prevent message-passing data leakage.
    """
    train_graph = copy.copy(data)

    # 1. Forward masking: (drug, indication, disease)
    if ("drug", "indication", "disease") in data.edge_types:
        ei_fwd = data["drug", "indication", "disease"].edge_index
        fwd_mask = torch.ones(ei_fwd.size(1), dtype=torch.bool)
        for i in range(ei_fwd.size(1)):
            d = ei_fwd[0, i].item()
            dis = ei_fwd[1, i].item()
            if (d, dis) in val_test_pairs:
                fwd_mask[i] = False
        train_graph["drug", "indication", "disease"].edge_index = ei_fwd[:, fwd_mask]

    # 2. Reverse masking: (disease, indication, drug) or (disease, rev_indication, drug)
    rev_rel = None
    if ("disease", "indication", "drug") in data.edge_types:
        rev_rel = ("disease", "indication", "drug")
    elif ("disease", "rev_indication", "drug") in data.edge_types:
        rev_rel = ("disease", "rev_indication", "drug")

    if rev_rel is not None:
        ei_rev = data[rev_rel].edge_index
        rev_mask = torch.ones(ei_rev.size(1), dtype=torch.bool)
        for i in range(ei_rev.size(1)):
            dis = ei_rev[0, i].item()
            d = ei_rev[1, i].item()
            if (d, dis) in val_test_pairs:
                rev_mask[i] = False
        train_graph[rev_rel].edge_index = ei_rev[:, rev_mask]

    return train_graph


def create_train_val_test_splits(
    pos_drugs: np.ndarray,
    pos_diseases: np.ndarray,
    neg_drugs: np.ndarray,
    neg_diseases: np.ndarray,
    neg_is_hard: np.ndarray,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Splits positive and negative pairs into train, val, and test partitions."""
    rng = np.random.default_rng(seed)

    # Split positives
    n_pos = len(pos_drugs)
    pos_perm = rng.permutation(n_pos)
    n_pos_train = int(n_pos * train_ratio)
    n_pos_val = int(n_pos * val_ratio)

    pos_train_idx = pos_perm[:n_pos_train]
    pos_val_idx = pos_perm[n_pos_train : n_pos_train + n_pos_val]
    pos_test_idx = pos_perm[n_pos_train + n_pos_val :]

    # Split negatives
    n_neg = len(neg_drugs)
    neg_perm = rng.permutation(n_neg)
    n_neg_train = int(n_neg * train_ratio)
    n_neg_val = int(n_neg * val_ratio)

    neg_train_idx = neg_perm[:n_neg_train]
    neg_val_idx = neg_perm[n_neg_train : n_neg_train + n_neg_val]
    neg_test_idx = neg_perm[n_neg_train + n_neg_val :]

    def bundle_split(p_idx: np.ndarray, n_idx: np.ndarray) -> Dict[str, np.ndarray]:
        drugs = np.concatenate([pos_drugs[p_idx], neg_drugs[n_idx]])
        diseases = np.concatenate([pos_diseases[p_idx], neg_diseases[n_idx]])
        labels = np.concatenate([np.ones(len(p_idx), dtype=np.int64), np.zeros(len(n_idx), dtype=np.int64)])
        is_hard = np.concatenate([np.zeros(len(p_idx), dtype=bool), neg_is_hard[n_idx]])
        return {
            "drug_idx": drugs,
            "disease_idx": diseases,
            "label": labels,
            "is_hard_negative": is_hard,
        }

    return {
        "train": bundle_split(pos_train_idx, neg_train_idx),
        "val": bundle_split(pos_val_idx, neg_val_idx),
        "test": bundle_split(pos_test_idx, neg_test_idx),
    }
