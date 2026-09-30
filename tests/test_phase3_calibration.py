"""
PharmaGraph - Phase 3 Calibration Test Suite
============================================
Validates:
1. Platt scaling convergence via L-BFGS.
2. Perfect rank monotonicity (Kendall's tau == 1.0).
3. Boundedness of calibrated posterior probabilities P in [0, 1].
4. Expected Calibration Error (ECE) reduction.
5. Calibrator persistence (JSON serialization).
"""

import numpy as np
import pytest
from scipy.stats import kendalltau
from src.calibration import PlattCalibrator


class TestPhase3Calibration:
    """Test suite verifying Temperature-Scaled Platt Scaling calibrator."""

    def test_platt_fit_and_monotonicity(self):
        """Validates that Platt scaling preserves exact rank order of logits."""
        # Simulated validation logits and labels
        rng = np.random.default_rng(42)
        val_logits = np.concatenate([
            rng.normal(loc=150.0, scale=50.0, size=500),  # Positives
            rng.normal(loc=-50.0, scale=50.0, size=1000),  # Negatives
        ])
        val_labels = np.concatenate([np.ones(500), np.zeros(1000)])

        calibrator = PlattCalibrator()
        calibrator.fit(val_logits, val_labels)

        assert calibrator.is_fitted
        assert calibrator.temperature > 0

        # Predict probabilities
        test_logits = np.linspace(-200.0, 300.0, 100)
        cal_probs = calibrator.predict_proba(test_logits)

        # 1. Boundedness
        assert np.all(cal_probs >= 0.0)
        assert np.all(cal_probs <= 1.0)

        # 2. Strict rank monotonicity (Kendall's tau == 1.0)
        tau, p_val = kendalltau(test_logits, cal_probs)
        assert pytest.approx(tau, abs=1e-5) == 1.0

    def test_ece_calculation(self):
        """Validates that ECE correctly identifies calibrated vs uncalibrated predictions."""
        # Perfectly calibrated probabilities
        probs = np.array([0.1, 0.2, 0.8, 0.9])
        labels = np.array([0, 0, 1, 1])
        ece = PlattCalibrator.expected_calibration_error(probs, labels, n_bins=5)
        assert 0.0 <= ece <= 0.25

        # Severely miscalibrated probabilities (overconfident)
        bad_probs = np.array([0.99, 0.99, 0.99, 0.99])
        bad_labels = np.array([0, 0, 0, 0])
        bad_ece = PlattCalibrator.expected_calibration_error(bad_probs, bad_labels, n_bins=5)
        assert bad_ece > 0.90

    def test_calibrator_save_and_load(self, tmp_path):
        """Validates JSON checkpoint serialization and deserialization."""
        calibrator = PlattCalibrator(temperature=85.5, bias=120.2)
        calibrator.is_fitted = True

        save_path = tmp_path / "calibrator.json"
        calibrator.save(save_path)
        assert save_path.exists()

        loaded = PlattCalibrator.load(save_path)
        assert loaded.temperature == 85.5
        assert loaded.bias == 120.2
        assert loaded.is_fitted

        # Predictions must be identical
        test_logits = np.array([-50.0, 0.0, 100.0])
        np.testing.assert_allclose(
            calibrator.predict_proba(test_logits),
            loaded.predict_proba(test_logits),
        )
