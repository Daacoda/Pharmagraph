"""
PharmaGraph - Phase 4 GraphRAG Path Explainer Test Suite
========================================================
Validates:
1. Path extraction on real heterogeneous knowledge graph topology.
2. Canonical metapath structure conformance.
3. DWPC hub dampening penalty.
4. Verbalization prompt synthesis for downstream LLMs.
"""

import pandas as pd
import pytest
import torch
from src.config import PathConfig
from src.explainer import GraphRAGExplainer


@pytest.fixture(scope="module")
def explainer_instance():
    """Initializes GraphRAGExplainer with real graph and entities."""
    paths = PathConfig()
    graph_path = paths.processed_dir / "pharmagraph_hetero_t0.pt"
    entities_path = paths.processed_dir / "entities.csv"

    data = torch.load(graph_path, weights_only=False)
    entities = pd.read_csv(entities_path) if entities_path.exists() else None
    return GraphRAGExplainer(data, entities_df=entities, dwpc_weight=0.4, top_k=5)


class TestPhase4Explainer:
    """Test suite verifying GraphRAG path extraction and hub dampening."""

    def test_path_extraction_conformance(self, explainer_instance):
        """Validates that extracted paths conform to canonical biological node types."""
        # Drug 5553 and Disease 12675 known to have target and pathway overlap
        paths = explainer_instance.extract_paths(5553, 12675)
        assert len(paths) > 0
        assert len(paths) <= 5

        for p in paths:
            assert p["type"] in ["direct_target", "pathway_mediated", "ppi_interactome"]
            assert p["score"] > 0.0
            nodes = p["nodes"]
            assert nodes[0]["type"] == "drug"
            assert nodes[-1]["type"] == "disease"
            assert len(p["summary"]) > 0

    def test_dwpc_hub_dampening_penalty(self, explainer_instance):
        """Validates that DWPC penalizes high-degree promiscuous hubs compared to specific targets."""
        # Penalty for high degree hub (deg=1000) vs specific target (deg=10)
        penalty_hub = explainer_instance._dwpc_penalty(1000)
        penalty_specific = explainer_instance._dwpc_penalty(10)

        # specific penalty should be significantly larger (meaning less penalized)
        assert penalty_specific > 5.0 * penalty_hub

    def test_verbalizer_prompt_synthesis(self, explainer_instance):
        """Validates that verbalize_prompt generates structured context for LLMs."""
        paths = explainer_instance.extract_paths(5553, 12675)
        prompt = explainer_instance.verbalize_prompt(5553, 12675, paths)

        assert "Candidate Repurposing Hypothesis:" in prompt
        assert "Verified Mechanistic Biological Evidence Paths:" in prompt
        assert "DWPC score:" in prompt
