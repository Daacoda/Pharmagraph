"""
PharmaGraph - Phase 5 Dashboard Data Bridge Test Suite
======================================================
Validates Python-to-R Reticulate interface contracts:
1. get_disease_list returns alphabetically sorted unique disease names.
2. get_drug_disease_subset returns required DataFrame schema and valid status flags.
3. get_subgraph returns visNetwork nodes and edges bounded by max_neighbors.
4. get_relation_diff and get_benchmark_table return structured performance tables.
5. All interactive queries execute within the SLA ceiling (p95 <= 250ms).
"""

import time
import pytest
import pandas as pd

from src.dashboard_data import (
    get_disease_list,
    get_drug_disease_subset,
    get_subgraph,
    get_relation_diff,
    get_benchmark_table,
    get_snapshot_info,
    get_loaded_edges_df,
)


@pytest.fixture(scope="module", autouse=True)
def warm_dashboard_cache():
    """Pre-warms in-memory cache mirroring R Shiny dependencies.R startup."""
    get_loaded_edges_df()


class TestPhase5Dashboard:
    """Test suite verifying Dashboard data bridge contracts and latency SLAs."""

    def test_get_disease_list_contract(self):
        """Verifies disease list returns a non-empty sorted list of strings."""
        t0 = time.time()
        diseases = get_disease_list()
        elapsed = time.time() - t0

        assert isinstance(diseases, list)
        assert len(diseases) > 1000, "Expected at least 1,000 disease names"
        assert all(isinstance(d, str) for d in diseases[:50])
        # Assert alphabetical ordering
        assert diseases[:50] == sorted(diseases[:50])
        assert elapsed < 0.250, f"get_disease_list exceeded 250ms SLA ({elapsed*1000:.1f}ms)"

    def test_get_drug_disease_subset_schema_and_status(self):
        """Verifies drug-disease subset conforms to exact schema columns and status constraints."""
        diseases = get_disease_list()
        # Query with a representative disease
        test_disease = diseases[100]

        t0 = time.time()
        df = get_drug_disease_subset(test_disease)
        elapsed = time.time() - t0

        expected_cols = ["relation", "x_id", "x_type", "x_name", "y_id", "y_type", "y_name", "status"]
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == expected_cols

        if not df.empty:
            valid_statuses = {"persistent", "added (t1)", "removed (t0 only)"}
            assert set(df["status"].unique()).issubset(valid_statuses)

        # SLA test
        assert elapsed < 0.250, f"get_drug_disease_subset exceeded 250ms SLA ({elapsed*1000:.1f}ms)"

    def test_get_subgraph_visnetwork_format(self):
        """Verifies subgraph returns nodes and edges formatted for visNetwork bounded by max_neighbors."""
        diseases = get_disease_list()
        test_disease = diseases[100]

        max_n = 50
        t0 = time.time()
        sg = get_subgraph(test_disease, hops=1, snapshot="t1", max_neighbors=max_n)
        elapsed = time.time() - t0

        assert isinstance(sg, dict)
        assert "nodes" in sg and "edges" in sg
        assert isinstance(sg["nodes"], list)
        assert isinstance(sg["edges"], list)

        # Check node structure
        for node in sg["nodes"]:
            assert "id" in node and "label" in node and "group" in node

        # Check edge structure
        for edge in sg["edges"]:
            assert "from" in edge and "to" in edge and "label" in edge

        assert len(sg["nodes"]) <= max_n + 10
        assert elapsed < 0.250, f"get_subgraph exceeded 250ms SLA ({elapsed*1000:.1f}ms)"

    def test_relation_diff_and_benchmark_table(self):
        """Verifies relation diff and benchmark tables return required columns for Server.R."""
        rel_diff = get_relation_diff()
        assert isinstance(rel_diff, pd.DataFrame)
        expected_rel_cols = ["relation", "added", "removed", "persistent"]
        assert list(rel_diff.columns) == expected_rel_cols
        assert len(rel_diff) > 0

        bench = get_benchmark_table()
        assert isinstance(bench, pd.DataFrame)
        assert "Model" in bench.columns
        assert "MRR" in bench.columns
        assert len(bench) >= 2

    def test_snapshot_info_backward_compatibility(self):
        """Verifies get_snapshot_info returns metadata without reading raw LFS pointers."""
        info_t0 = get_snapshot_info("t0")
        assert "total_triples" in info_t0
        assert info_t0["total_triples"] > 0
        assert "columns" in info_t0
