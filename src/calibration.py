"""
PharmaGraph - Temperature-Scaled Platt Scaling Calibrator
=========================================================
Resolves Defect 1 (Uncalibrated Probability Failure).
Transforms unnormalized latent inner-product GNN margins into true posterior
probabilities using parametric Temperature-Scaled Platt Scaling optimized via
L-BFGS on held-out validation sets.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import json
import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import brier_score_loss, log_loss


class PlattCalibrator:
    """Temperature-Scaled Platt Scaling Calibrator optimized via L-BFGS."""

    def __init__(self, temperature: float = 1.0, bias: float = 0.0) -> None:
        self.temperature = float(temperature)
        self.bias = float(bias)
        self.is_fitted = False

    def fit(
        self,
        val_logits: np.ndarray,
        val_labels: np.ndarray,
        bounds: Tuple[Tuple[float, float], Tuple[float, float]] = ((1e-3, 1000.0), (-500.0, 500.0)),
    ) -> "PlattCalibrator":
        """
        Fit temperature (T) and intercept shift (b) via L-BFGS to minimize
        negative log-likelihood (binary cross-entropy) over the validation split.
        """
        logits = np.asarray(val_logits, dtype=np.float64)
        labels = np.asarray(val_labels, dtype=np.float64)

        if len(logits) != len(labels):
            raise ValueError(f"Length mismatch: {len(logits)} logits vs {len(labels)} labels")

        def nll_objective(params: np.ndarray) -> float:
            t_val, b_val = params
            if t_val <= 0:
                return 1e12

            # Compute calibrated probability with numerical stability
            scaled = -(logits - b_val) / t_val
            # Prevent overflow in exp
            scaled = np.clip(scaled, -50.0, 50.0)
            p = 1.0 / (1.0 + np.exp(scaled))
            p = np.clip(p, 1e-12, 1.0 - 1e-12)
            return float(log_loss(labels, p))

        # Initial parameters: T=1.0, b=0.0
        init_params = np.array([1.0, 0.0], dtype=np.float64)

        res = minimize(
            nll_objective,
            init_params,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 100, "ftol": 1e-7},
        )

        self.temperature = float(res.x[0])
        self.bias = float(res.x[1])
        self.is_fitted = True
        return self

    def predict_proba(self, logits: np.ndarray) -> np.ndarray:
        """
        Transform raw margins into calibrated posterior probabilities.
        P(indication = 1 | s_GNN) = sigma((s - b*) / T*)
        """
        logits_arr = np.asarray(logits, dtype=np.float64)
        scaled = -(logits_arr - self.bias) / self.temperature
        scaled = np.clip(scaled, -50.0, 50.0)
        probs = 1.0 / (1.0 + np.exp(scaled))
        return np.clip(probs, 0.0, 1.0)

    @staticmethod
    def expected_calibration_error(
        probs: np.ndarray,
        labels: np.ndarray,
        n_bins: int = 10,
    ) -> float:
        """
        Calculates Expected Calibration Error (ECE) over equal-width confidence bins:
        ECE = sum_m (|B_m| / N) * |acc(B_m) - conf(B_m)|
        """
        probs_arr = np.asarray(probs, dtype=np.float64)
        labels_arr = np.asarray(labels, dtype=np.float64)

        bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
        ece = 0.0
        n_total = len(probs_arr)

        if n_total == 0:
            return 0.0

        for i in range(n_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]

            if i == n_bins - 1:
                in_bin = (probs_arr >= bin_lower) & (probs_arr <= bin_upper)
            else:
                in_bin = (probs_arr >= bin_lower) & (probs_arr < bin_upper)

            bin_size = np.sum(in_bin)
            if bin_size > 0:
                bin_acc = np.mean(labels_arr[in_bin])
                bin_conf = np.mean(probs_arr[in_bin])
                ece += (bin_size / n_total) * abs(bin_acc - bin_conf)

        return float(ece)

    def evaluate_calibration(
        self,
        logits: np.ndarray,
        labels: np.ndarray,
        n_bins: int = 10,
    ) -> Dict[str, float]:
        """Evaluates ECE, Brier score, and NLL before and after calibration."""
        logits_arr = np.asarray(logits, dtype=np.float64)
        labels_arr = np.asarray(labels, dtype=np.float64)

        # Uncalibrated baseline: standard sigmoid
        uncal_probs = 1.0 / (1.0 + np.exp(-np.clip(logits_arr, -50.0, 50.0)))
        cal_probs = self.predict_proba(logits_arr)

        ece_uncal = self.expected_calibration_error(uncal_probs, labels_arr, n_bins=n_bins)
        ece_cal = self.expected_calibration_error(cal_probs, labels_arr, n_bins=n_bins)

        brier_uncal = float(brier_score_loss(labels_arr, uncal_probs))
        brier_cal = float(brier_score_loss(labels_arr, cal_probs))

        return {
            "temperature": self.temperature,
            "bias": self.bias,
            "ece_uncalibrated": ece_uncal,
            "ece_calibrated": ece_cal,
            "brier_uncalibrated": brier_uncal,
            "brier_calibrated": brier_cal,
        }

    def save(self, path: Union[str, Path]) -> None:
        """Saves fitted calibration parameters to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "temperature": self.temperature,
            "bias": self.bias,
            "is_fitted": self.is_fitted,
        }
        path.write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: Union[str, Path]) -> "PlattCalibrator":
        """Loads fitted calibration parameters from JSON."""
        path = Path(path)
        data = json.loads(path.read_text())
        calibrator = cls(temperature=data["temperature"], bias=data["bias"])
        calibrator.is_fitted = data.get("is_fitted", True)
        return calibrator


def main() -> None:
    """CLI to fit and evaluate Platt calibrator on final evidence rankings."""
    import argparse
    import pandas as pd

    parser = argparse.ArgumentParser(description="Fit Platt Scaling Calibrator on validation predictions")
    parser.add_argument("--input", default="output/final_evidence_ranking.parquet", help="Path to evidence parquet")
    parser.add_argument("--output", default="output/calibrated_evidence_ranking.parquet", help="Path for output parquet")
    parser.add_argument("--save-model", default="output/models/platt_calibrator.json", help="Path to save calibrator")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Input file not found at {input_path}")
        return

    df = pd.read_parquet(input_path)
    val_mask = (df["split"] == "val").values
    test_mask = (df["split"] == "test").values

    print(f"Loaded {len(df)} rows from {input_path}")
    val_logits = df.loc[val_mask, "model_score"].values
    val_labels = df.loc[val_mask, "label"].values

    calibrator = PlattCalibrator()
    calibrator.fit(val_logits, val_labels)
    print(f"Fitted parameters: Temperature T*={calibrator.temperature:.4f}, Bias b*={calibrator.bias:.4f}")

    metrics = calibrator.evaluate_calibration(df.loc[test_mask, "model_score"].values, df.loc[test_mask, "label"].values)
    print("Test Set Calibration Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    df["calibrated_prob"] = calibrator.predict_proba(df["model_score"].values)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"Saved calibrated predictions to {out_path}")

    if args.save_model:
        calibrator.save(args.save_model)
        print(f"Saved calibrator checkpoint to {args.save_model}")


if __name__ == "__main__":
    main()
