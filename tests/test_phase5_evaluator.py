"""
PharmaGraph - Phase 5 Temporal Evaluator Test Suite
===================================================
Validates:
1. Hits@K and MRR computation correctness.
2. Stratified evaluation across candidate predictions.
3. Candidate Report Card synthesis with evidence and explanations.
"""

import numpy as np
import pandas as pd
import pytest
from src.evaluator import TemporalEvaluator


class TestPhase5Evaluator:
    """Test suite verifying Longitudinal Temporal Evaluator."""

    def test_ranking_metrics_math(self):
        """Validates that Hits@K and MRR compute exact mathematical results."""
        evaluator = TemporalEvaluator()

        # Ranks: [1, 5, 10, 20]
        ranks = np.array([1, 5, 10, 20])

        # Hits@10: 3 out of 4 are <= 10 -> 0.75
        assert evaluator.compute_hits_at_k(ranks, k=10) == 0.75

        # Hits@5: 2 out of 4 are <= 5 -> 0.50
        assert evaluator.compute_hits_at_k(ranks, k=5) == 0.50

        # Hits@1: 1 out of 4 is <= 1 -> 0.25
        assert evaluator.compute_hits_at_k(ranks, k=1) == 0.25

        # MRR: (1/1 + 1/5 + 1/10 + 1/20) / 4 = (1 + 0.2 + 0.1 + 0.05) / 4 = 1.35 / 4 = 0.3375
        expected_mrr = (1.0 + 0.2 + 0.1 + 0.05) / 4.0
        assert pytest.approx(evaluator.compute_mrr(ranks), abs=1e-5) == expected_mrr

    def test_stratified_performance_evaluation(self):
        """Validates evaluation grouping across disease candidates."""
        evaluator = TemporalEvaluator()

        # Mock scored dataframe for 2 diseases
        data = {
            "disease_idx": [1, 1, 1, 2, 2, 2],
            "drug_idx": [10, 20, 30, 10, 20, 30],
            "model_score": [0.9, 0.5, 0.1, 0.2, 0.8, 0.4],
            "label": [1, 0, 0, 0, 1, 0],
        }
        df = pd.DataFrame(data)

        results = evaluator.evaluate_stratified_performance(df, score_col="model_score")

        assert results["total_pairs_evaluated"] == 6
        assert results["unique_diseases"] == 2
        assert "positive_indication_metrics" in results

        pos_metrics = results["positive_indication_metrics"]
        assert pos_metrics["count"] == 2
        # Disease 1: drug 10 has score 0.9 (rank 1) -> positive
        # Disease 2: drug 20 has score 0.8 (rank 1) -> positive
        # Hits@10 should be 1.0, MRR should be 1.0
        assert pos_metrics["hits_at_10"] == 1.0
        assert pos_metrics["mrr"] == 1.0

    def test_candidate_report_card_synthesis(self):
        """Validates that Candidate Report Cards synthesize cleanly."""
        evaluator = TemporalEvaluator()

        data = {
            "disease_idx": [100, 100, 100],
            "drug_idx": [1, 2, 3],
            "model_score": [120.0, 80.0, 40.0],
            "calibrated_prob": [0.85, 0.65, 0.30],
            "ot_score": [0.8, 0.4, 0.0],
            "ppi_min_hops": [1.0, 2.0, 3.0],
            "ppi_common_neighbors": [5.0, 2.0, 0.0],
            "go_jaccard_bp": [0.4, 0.2, 0.0],
            "temporal_status": ["persistent", "added", "novel_hypothesis"],
        }
        df = pd.DataFrame(data)

        report = evaluator.build_candidate_report_card(100, df, explainer=None, top_n=2)

        assert report["disease_idx"] == 100
        assert report["total_candidates_scored"] == 3
        assert len(report["top_candidates"]) == 2
        top1 = report["top_candidates"][0]
        assert top1["drug_idx"] == 1
        assert top1["calibrated_prob"] == 0.85
        assert top1["temporal_status"] == "persistent"
