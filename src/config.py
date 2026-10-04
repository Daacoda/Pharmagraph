"""PharmaGraph Configuration Module.

Single source of truth for all paths, model hyperparameters, sampling rules,
and hardware execution constraints across the PharmaGraph platform.
Adheres strictly to the System Architecture Document (SAD).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
import torch


@dataclass(frozen=True)
class HardwareConfig:
    """Hardware constraints and compute device settings.
    
    Verified for Intel Core i5-8350U (4 cores / 8 threads) with 14 GiB RAM.
    """
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))
    num_threads: int = 8
    max_memory_mib: float = 468.61

    def apply(self) -> None:
        """Enforces thread scheduling and runtime hardware policy."""
        torch.set_num_threads(self.num_threads)
        os.environ["OMP_NUM_THREADS"] = str(self.num_threads)
        os.environ["MKL_NUM_THREADS"] = str(self.num_threads)


@dataclass(frozen=True)
class ModelConfig:
    """Hyperparameters for HeteroGraphSAGE and pairwise margin ranking."""
    in_channels: int = -1  # Bipartite lazy parameter initialization
    hidden_channels: int = 128
    out_channels: int = 128
    num_layers: int = 2
    dropout: float = 0.3
    weight_decay: float = 1e-4
    learning_rate: float = 1e-3
    margin: float = 1.0  # Hinge margin gamma for MarginRankingLoss
    random_seed: int = 42
    patience: int = 10
    max_epochs: int = 100


@dataclass(frozen=True)
class SamplingConfig:
    """Data partitioning and balanced negative sampling ratios."""
    neg_to_pos_ratio: float = 2.0
    hard_neg_fraction: float = 0.50
    random_neg_fraction: float = 0.50
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15


@dataclass(frozen=True)
class CalibrationConfig:
    """Formal calibration and multimodal evidence fusion parameters."""
    solver_method: str = "L-BFGS-B"
    max_iterations: int = 50
    initial_temperature: float = 1.0
    initial_intercept: float = 0.0
    alpha: float = 0.65  # Convex weight: alpha * P_GNN + (1 - alpha) * E_ortho
    harmonic_weights: tuple[float, float, float] = (1.0, 0.25, 0.1111)  # 1/r^2 weights
    harmonic_normalization: float = 36.0 / 49.0  # 1 / sum(1/r^2)


@dataclass(frozen=True)
class ExplainerConfig:
    """Parameters for GraphRAG bidirectional beam search traversal."""
    beam_width_targets: int = 30     # B_g
    beam_width_pathways: int = 50    # B_p
    top_k_paths: int = 5             # K
    dwpc_damping_w: float = 0.4      # Degree damping exponent w
    max_intersections: int = 1500    # B_g * B_p ceiling
    latency_sla_ms: float = 5.0      # SLA target per pair


@dataclass(frozen=True)
class PathConfig:
    """Filesystem hierarchy and persistence paths."""
    project_root: Path = field(
        default_factory=lambda: Path(__file__).resolve().parent.parent
    )

    @property
    def data_dir(self) -> Path:
        return self.project_root / "data"

    @property
    def benchmark_dir(self) -> Path:
        return self.data_dir / "benchmark"

    @property
    def processed_dir(self) -> Path:
        return self.project_root / "processed"

    @property
    def output_dir(self) -> Path:
        return self.project_root / "output"

    @property
    def models_dir(self) -> Path:
        return self.output_dir / "models"

    @property
    def cache_dir(self) -> Path:
        return self.output_dir / "cache"

    @property
    def edges_parquet(self) -> Path:
        return self.processed_dir / "edges.parquet"

    @property
    def entities_csv(self) -> Path:
        return self.processed_dir / "entities.csv"

    @property
    def hetero_graph_pt(self) -> Path:
        return self.processed_dir / "pharmagraph_hetero_t0.pt"

    @property
    def drug_fingerprints_npy(self) -> Path:
        return self.processed_dir / "drug_fingerprints.npy"

    @property
    def drug_smiles_json(self) -> Path:
        return self.processed_dir / "drug_smiles.json"

    def ensure_directories(self) -> None:
        """Creates required mutable directories if they do not exist."""
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)


# Global instantiated configuration bundles
HARDWARE = HardwareConfig()
MODEL = ModelConfig()
SAMPLING = SamplingConfig()
CALIBRATION = CalibrationConfig()
EXPLAINER = ExplainerConfig()
PATHS = PathConfig()

# Apply hardware policies immediately upon import
HARDWARE.apply()
PATHS.ensure_directories()
