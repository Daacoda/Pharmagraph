"""
check_sigmoid_ties.py
=====================
Evaluation-only diagnostic. Explains why exp1_class_imbalance.py recomputed a test
AUROC of 0.9375 while the checkpoint stores 0.9286.

Hypothesis: training evaluated roc_auc_score on torch.sigmoid(raw_score) in float32.
The margin-ranking loss leaves raw scores unbounded (tens to hundreds), so float32
sigmoid saturates to exactly 1.0 for every score above ~17. Those pairs become
exact ties, and ties between positives and negatives count as 0.5 in AUROC.
Raw scores have no ties, so the rank metric is higher. Nothing is retrained or
modified; this only re-scores the held-out test split.

Run from the project root:   python src/check_sigmoid_ties.py
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def compare(y, raw, recorded_auroc=None, recorded_auprc=None):
    """AUROC/AUPRC on raw scores vs float32 sigmoid, plus tie statistics."""
    import torch
    raw = np.asarray(raw, dtype=np.float32)
    p32 = torch.sigmoid(torch.from_numpy(raw.copy())).numpy()
    sat = p32 == 1.0
    out = {
        "raw_min": float(raw.min()), "raw_max": float(raw.max()),
        "auroc_raw": float(roc_auc_score(y, raw)),
        "auprc_raw": float(average_precision_score(y, raw)),
        "auroc_sigmoid": float(roc_auc_score(y, p32)),
        "auprc_sigmoid": float(average_precision_score(y, p32)),
        "n_saturated": int(sat.sum()),
        "n_saturated_pos": int((sat & (y == 1)).sum()),
        "n_saturated_neg": int((sat & (y == 0)).sum()),
        "n_test": int(len(y)),
    }
    if recorded_auroc is not None:
        out["explained"] = abs(out["auroc_sigmoid"] - float(recorded_auroc)) < 0.002
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default=os.path.join("output", "models", "graphsage_v3_real_features.pt"))
    ap.add_argument("--candidates", default=os.path.join("processed", "candidate_pairs.csv"))
    args = ap.parse_args()

    from exp1_class_imbalance import load_scorer
    cand = pd.read_csv(args.candidates)
    score_fn, _, _, _, ckpt = load_scorer(args, cand)

    test = cand[cand.split == "test"]
    y = test.label.values.astype(int)
    raw = score_fn(test.drug_idx.values, test.disease_idx.values)
    r = compare(y, raw, ckpt.get("test_auroc"), ckpt.get("test_auprc"))

    print(f"\nTest split: {r['n_test']} pairs | raw score range [{r['raw_min']:.1f}, {r['raw_max']:.1f}]")
    print(f"Pairs whose float32 sigmoid is exactly 1.0: {r['n_saturated']} "
          f"({r['n_saturated_pos']} positives, {r['n_saturated_neg']} negatives)\n")
    print(f"{'':28s}{'AUROC':>10s}{'AUPRC':>10s}")
    print(f"{'raw scores (no ties)':28s}{r['auroc_raw']:>10.4f}{r['auprc_raw']:>10.4f}")
    print(f"{'float32 sigmoid (training)':28s}{r['auroc_sigmoid']:>10.4f}{r['auprc_sigmoid']:>10.4f}")
    print(f"{'stored in checkpoint':28s}{float(ckpt.get('test_auroc', float('nan'))):>10.4f}"
          f"{float(ckpt.get('test_auprc', float('nan'))):>10.4f}")
    if r.get("explained"):
        print("\nExplained: the sigmoid column reproduces the checkpoint value, so the gap is "
              "float32 saturation ties, not a different model or split.")
    else:
        print("\nNOT explained by sigmoid ties: the sigmoid column does not reproduce the checkpoint "
              "value. Send this output back - the checkpoint, candidate_pairs.csv or fingerprints may "
              "come from different runs.")


if __name__ == "__main__":
    main()