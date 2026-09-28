"""Phase 1 Foundation & Data Contract Verification Suite.

Executes live validation on commodity CPU hardware, testing genuine data
files (edges.parquet, entities.csv, pharmagraph_hetero_t0.pt) and confirming
exact adherence to the System Architecture Document.
"""

from __future__ import annotations

import sys
import gc
from pathlib import Path

# Add project root to python path to enable src.* imports
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import pytest
import torch

from src.config import (
    HARDWARE,
    MODEL,
    SAMPLING,
    PATHS,
    HardwareConfig,
    ModelConfig,
    SamplingConfig,
)


class TestPhase1Foundation:
    """Live verification suite for Phase 1 acceptance gate."""

    def test_hardware_and_threading(self) -> None:
        """Verifies CPU thread binding and device enforcement."""
        HARDWARE.apply()
        assert HARDWARE.device.type == "cpu", "Device must be CPU-only."
        assert torch.get_num_threads() == HARDWARE.num_threads, (
            f"Expected {HARDWARE.num_threads} CPU threads, got {torch.get_num_threads()}"
        )
        assert not torch.cuda.is_available(), (
            "CUDA acceleration detected; platform strictly enforces CPU-only execution."
        )

    def test_path_resolution_and_hierarchy(self) -> None:
        """Verifies directory structure and path resolution."""
        PATHS.ensure_directories()
        assert PATHS.project_root.exists(), f"Project root missing: {PATHS.project_root}"
        assert PATHS.processed_dir.exists(), f"Processed dir missing: {PATHS.processed_dir}"
        assert PATHS.benchmark_dir.exists(), f"Benchmark dir missing: {PATHS.benchmark_dir}"
        assert PATHS.models_dir.exists() and PATHS.models_dir.is_dir()
        assert PATHS.cache_dir.exists() and PATHS.cache_dir.is_dir()

    def test_data_schema_and_counts(self) -> None:
        """Verifies Parquet and CSV schemas against Section 3.5.1 of SAD."""
        # 1. Validate entities.csv
        assert PATHS.entities_csv.exists(), f"entities.csv missing: {PATHS.entities_csv}"
        entities_df = pd.read_csv(PATHS.entities_csv)
        assert len(entities_df) == 129312, (
            f"Expected exactly 129,312 nodes in entities.csv, got {len(entities_df)}"
        )
        expected_entity_cols = ["node_id", "node_type", "node_idx", "local_idx"]
        assert list(entities_df.columns) == expected_entity_cols, (
            f"Entity columns mismatch. Expected {expected_entity_cols}, got {list(entities_df.columns)}"
        )
        assert entities_df["node_idx"].is_monotonic_increasing, "node_idx must be contiguous."
        assert entities_df["node_idx"].min() == 0 and entities_df["node_idx"].max() == 129311

        # 2. Validate edges.parquet
        assert PATHS.edges_parquet.exists(), f"edges.parquet missing: {PATHS.edges_parquet}"
        edges_df = pd.read_parquet(PATHS.edges_parquet)
        expected_edge_cols = ["head_idx", "relation", "tail_idx", "x_type", "y_type"]
        assert list(edges_df.columns) == expected_edge_cols, (
            f"Edge columns mismatch. Expected {expected_edge_cols}, got {list(edges_df.columns)}"
        )
        assert not edges_df.empty, "edges.parquet cannot be empty."

        # Verify key relations
        relations = set(edges_df["relation"].unique())
        assert "indication" in relations, "Missing required 'indication' relation."
        assert "contraindication" in relations, "Missing required 'contraindication' relation."
        assert "off-label use" in relations, "Missing required 'off-label use' relation."

        # Clean memory
        del entities_df, edges_df
        gc.collect()

    def test_hetero_graph_and_fingerprints(self) -> None:
        """Loads genuine HeteroData and Morgan fingerprints, verifying tensor structures."""
        # 1. Validate HeteroData graph object
        assert PATHS.hetero_graph_pt.exists(), f"hetero graph missing: {PATHS.hetero_graph_pt}"
        graph = torch.load(PATHS.hetero_graph_pt, weights_only=False)

        expected_node_types = {
            "gene/protein", "drug", "disease", "effect/phenotype",
            "biological_process", "molecular_function", "cellular_component",
            "exposure", "pathway", "anatomy"
        }
        actual_node_types = set(graph.node_types)
        assert actual_node_types == expected_node_types, (
            f"Node types mismatch. Expected {expected_node_types}, got {actual_node_types}"
        )

        # Total entity count check across HeteroData
        total_nodes = sum(graph[ntype].num_nodes for ntype in graph.node_types)
        assert total_nodes == 129312, f"Total HeteroData nodes ({total_nodes}) != 129,312"

        # Check drug-disease relation existence
        edge_types = graph.edge_types
        assert ("drug", "indication", "disease") in edge_types
        assert ("drug", "contraindication", "disease") in edge_types

        # Check indication edge count
        ind_edges = graph["drug", "indication", "disease"].edge_index
        assert ind_edges.shape[0] == 2, "Indication edge_index must have 2 rows [src, dst]."
        assert ind_edges.shape[1] > 0, "Indication edge_index must be populated."

        # 2. Validate drug fingerprints
        if PATHS.drug_fingerprints_npy.exists():
            fps = np.load(PATHS.drug_fingerprints_npy, allow_pickle=True).item()
            assert isinstance(fps, dict), "Fingerprints artifact must be a dictionary."
            assert len(fps) > 0, "Fingerprints dictionary cannot be empty."
            sample_fp = next(iter(fps.values()))
            assert isinstance(sample_fp, np.ndarray), "Fingerprint vector must be a numpy ndarray."
            assert sample_fp.shape[0] == 256, (
                f"Expected 256-bit Morgan fingerprint, got dimension {sample_fp.shape[0]}"
            )

        del graph
        gc.collect()
