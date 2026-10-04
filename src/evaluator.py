"""
PharmaGraph - Longitudinal Temporal Evaluator (t0 -> t1)
========================================================
Resolves the static evaluation fallacy by benchmarking predictions against the
held-out July 2023 snapshot (t1), stratified into three distinct strata:
  1. Emergent Discoveries (Added Edges): Hits@K and MRR on newly discovered facts
  2. Persistent Retention (Persistent Edges): Stability rate on validated facts
  3. Retraction Rejection (Removed Edges): Rejection rate on deprecated/retracted facts

Also generates Candidate Report Cards combining calibrated probabilities,
orthogonal evidence, GraphRAG biological paths, and temporal validation status.
"""

import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import PathConfig


class TemporalEvaluator:
    """Evaluates link prediction and evidence ranking across temporal snapshots (t0 -> t1)."""

    def __init__(
        self,
        diff_path: Optional[Union[str, Path]] = None,
        stratification_path: Optional[Union[str, Path]] = None,
    ) -> None:
        paths = PathConfig()
        self.diff_path = Path(diff_path) if diff_path else paths.benchmark_dir / "diffs" / "diff_t0_t1.json"
        self.strat_path = Path(stratification_path) if stratification_path else paths.benchmark_dir / "test_stratification.json"

        # Load diff and stratification metadata if present
        self.diff_metadata: Dict[str, Any] = {}
        if self.diff_path.exists():
            self.diff_metadata = json.loads(self.diff_path.read_text())

        self.strat_metadata: Dict[str, Any] = {}
        if self.strat_path.exists():
            self.strat_metadata = json.loads(self.strat_path.read_text())

    def compute_hits_at_k(self, ranks: np.ndarray, k: int = 10) -> float:
        """Computes Hits@K = (1 / N) * sum(I(rank <= K))."""
        ranks_arr = np.asarray(ranks, dtype=np.int64)
        if len(ranks_arr) == 0:
            return 0.0
        return float(np.mean(ranks_arr <= k))

    def compute_mrr(self, ranks: np.ndarray) -> float:
        """Computes Mean Reciprocal Rank (MRR) = (1 / N) * sum(1 / rank)."""
        ranks_arr = np.asarray(ranks, dtype=np.float64)
        if len(ranks_arr) == 0:
            return 0.0
        return float(np.mean(1.0 / np.maximum(ranks_arr, 1.0)))

    def evaluate_stratified_performance(
        self,
        scored_pairs_df: pd.DataFrame,
        score_col: str = "model_score",
        k_values: Tuple[int, ...] = (10, 20, 50),
    ) -> Dict[str, Any]:
        """
        Evaluates ranking metrics stratified across Persistent, Added, and Removed sets.
        Accepts DataFrame with columns: ['drug_idx', 'disease_idx', score_col]
        and optional 'temporal_status' or 'label'.
        """
        df = scored_pairs_df.copy()
        if score_col not in df.columns:
            raise ValueError(f"Score column '{score_col}' not found in DataFrame")

        # Rank pairs per disease (group by disease_idx, rank descending by score)
        df["rank"] = df.groupby("disease_idx")[score_col].rank(ascending=False, method="min")

        results: Dict[str, Any] = {
            "total_pairs_evaluated": len(df),
            "unique_diseases": int(df["disease_idx"].nunique()),
            "score_column_evaluated": score_col,
            "overall_metrics": {
                "mean_score": float(df[score_col].mean()),
                "median_score": float(df[score_col].median()),
            },
        }

        # If temporal_status is present, evaluate stratified metrics
        if "temporal_status" in df.columns:
            for status in ["added", "persistent", "removed"]:
                sub = df[df["temporal_status"] == status]
                if len(sub) > 0:
                    status_ranks = sub["rank"].values
                    results[f"{status}_metrics"] = {
                        "count": len(sub),
                        "mrr": self.compute_mrr(status_ranks),
                        **{f"hits_at_{k}": self.compute_hits_at_k(status_ranks, k=k) for k in k_values},
                    }
        elif "label" in df.columns:
            # Positive vs Negative evaluation
            pos_ranks = df.loc[df["label"] == 1, "rank"].values
            results["positive_indication_metrics"] = {
                "count": len(pos_ranks),
                "mrr": self.compute_mrr(pos_ranks),
                **{f"hits_at_{k}": self.compute_hits_at_k(pos_ranks, k=k) for k in k_values},
            }

        return results

    def build_candidate_report_card(
        self,
        disease_idx: int,
        candidates_df: pd.DataFrame,
        explainer: Optional[Any] = None,
        top_n: int = 10,
        score_col: str = "calibrated_prob",
    ) -> Dict[str, Any]:
        """
        Synthesizes an actionable Candidate Report Card for a query disease,
        combining ranking, calibrated probabilities, external evidence, and GraphRAG paths.
        """
        sub = candidates_df[candidates_df["disease_idx"] == disease_idx].copy()
        if len(sub) == 0:
            return {"disease_idx": disease_idx, "candidates": []}

        # Sort descending by primary prioritization score
        sub = sub.sort_values(by=score_col if score_col in sub.columns else sub.columns[0], ascending=False)
        top_candidates = sub.head(top_n)

        candidate_list: List[Dict[str, Any]] = []
        for _, row in top_candidates.iterrows():
            d_idx = int(row["drug_idx"])
            record = {
                "drug_idx": d_idx,
                "model_score": float(row.get("model_score", 0.0)),
                "calibrated_prob": float(row.get("calibrated_prob", row.get("model_prob", 0.0))),
                "evidence": {
                    "ot_score": float(row.get("ot_score", 0.0)),
                    "ppi_min_hops": float(row.get("ppi_min_hops", 0.0)),
                    "ppi_common_neighbors": float(row.get("ppi_common_neighbors", 0.0)),
                    "go_jaccard_bp": float(row.get("go_jaccard_bp", 0.0)),
                },
                "composite_score": float(row.get("score_option_a", row.get("score_option_b", row.get("calibrated_prob", 0.0)))),
                "temporal_status": str(row.get("temporal_status", "novel_hypothesis")),
            }

            if explainer is not None:
                paths = explainer.extract_paths(d_idx, disease_idx)
                record["mechanistic_paths"] = paths
                record["explanation_prompt"] = explainer.verbalize_prompt(d_idx, disease_idx, paths)

            candidate_list.append(record)

        return {
            "disease_idx": disease_idx,
            "total_candidates_scored": len(sub),
            "top_candidates": candidate_list,
        }


def main() -> None:
    """CLI to evaluate candidate rankings and export temporal benchmark metrics."""
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate candidate predictions on temporal benchmark")
    parser.add_argument("--input", default="output/calibrated_evidence_ranking.parquet", help="Path to scored candidates")
    parser.add_argument("--output", default="output/benchmark_results.json", help="Path for benchmark results")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        input_path = Path("output/final_evidence_ranking.parquet")
    if not input_path.exists():
        print(f"No candidate ranking file found at {input_path}")
        return

    df = pd.read_parquet(input_path)
    score_col = "calibrated_prob" if "calibrated_prob" in df.columns else "model_score"
    print(f"Loaded {len(df)} candidate pairs. Evaluating using '{score_col}'...")

    evaluator = TemporalEvaluator()
    results = evaluator.evaluate_stratified_performance(df, score_col=score_col)

    print("\nBenchmark Evaluation Results:")
    print(json.dumps(results, indent=2))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved benchmark metrics to {out_path}")


if __name__ == "__main__":
    main()
