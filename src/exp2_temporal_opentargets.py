"""
exp2_temporal_opentargets.py
============================
EXPERIMENT 2 - temporal validation on Open Targets, evaluated at 1:1 / 1:2 /
1:10 / 1:100 imbalance (KG-Bench-style). EVALUATION ONLY: the v3 checkpoint is
loaded, nothing is trained, nothing existing is overwritten.

Idea
----
PrimeKG-CL t1 contains no new drug-disease edges (DrugBank was not re-queried),
so we take "the future" from Open Targets instead:

    OLD = Open Targets 21.11  (data/known_drugs_aggregated, data/diseases)
    NEW = a later release     (data/ot_24.06/known_drug, data/ot_24.06/diseases;
                               made by download_ot_release.py)

Positives = drug-disease pairs in NEW that the model could NOT have seen:
    stratum  "new_vs_t0"    : not an indication in the PrimeKG t0 graph
    stratum  "new_vs_ot2111": additionally absent from OT 21.11  (strictly new
                              knowledge between 2021 and the NEW release - the
                              closest analogue of KG-Bench's 2021 -> 2024 test)
Both strata are reported, for each --min-phase (1 = any trial, 4 = approved).

Negatives (ratio negatives per positive, 5 independent draws):
    random          uniformly random pairs
    degree_matched  drug and disease drawn from the positives' own marginal
                    distributions (random re-pairing). Removes the "hub" shortcut:
                    if positives are popular drugs/diseases, a model that only
                    learned popularity looks good against uniform negatives.
All negatives exclude: every t0 indication, every contraindication, every pair in
candidate_pairs.csv, and every OT-known pair (old or new, any phase).

Scorers compared
----------------
    model            PharmaGraph v3, scored on the FULL t0 graph (--graph full:
                     all t0 indication edges visible, the real deployment setting)
    degree_baseline  log(1+drug indication degree) + log(1+disease indication degree)
                     in t0 - a pure popularity heuristic with no learning.
The model has to beat this baseline for the result to mean anything.

F1 uses the threshold tuned once on the validation split of the ORIGINAL (masked)
setup and frozen; AUROC/AUPRC are threshold-free; AUPRC baseline = prevalence.

Outputs (new files only): output/exp2_temporal_long.csv, exp2_temporal_summary.csv,
                          output/exp2_positives.csv, processed/our_disease_name_to_ids.json
                          (cache), processed/chembl_to_drugbank.json (extended).

Run from project root:
    py src\\exp2_temporal_opentargets.py
    py src\\exp2_temporal_opentargets.py --new-dir data\\ot_25.03 --min-phase 1 4
"""
import argparse
import glob
import json
import os
import re
import time

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

PROCESSED = "processed"
UNICHEM_URL = "https://www.ebi.ac.uk/unichem/api/v1/compounds"
MONDO_RE = re.compile(r"MONDO[:_](\d+)")
RATIOS = [1, 2, 10, 100]


# ----------------------------------------------------------------- I/O helpers
def read_parquet_dir(path):
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "**", "*.parquet"), recursive=True))
        if not files:
            raise SystemExit(f"No parquet files under {path}")
        return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    return pd.read_parquet(path)


def load_known_pairs(path, label):
    """-> DataFrame(drugId, diseaseId, phase) with one row per pair (max phase)."""
    df = read_parquet_dir(path)
    for c in ("drugId", "diseaseId"):
        if c not in df.columns:
            raise SystemExit(f"{label}: column {c} not found. Columns: {list(df.columns)}")
    if "phase" in df.columns:
        df["phase"] = pd.to_numeric(df["phase"], errors="coerce").fillna(0)
    else:
        print(f"  [{label}] no 'phase' column - phase filter disabled for this table")
        df["phase"] = 4.0
    df = df.dropna(subset=["drugId", "diseaseId"])
    out = df.groupby(["drugId", "diseaseId"], as_index=False)["phase"].max()
    print(f"  [{label}] {len(df)} rows -> {len(out)} unique (drug, disease) pairs, "
          f"{out.drugId.nunique()} drugs, {out.diseaseId.nunique()} diseases")
    return out


# ------------------------------------------------------ identifier bridging
def mondo_nums_of(row_id, xrefs):
    nums = set()
    m = MONDO_RE.search(str(row_id))
    if m:
        nums.add(m.group(1))
    for ref in (xrefs if xrefs is not None else []):
        m = MONDO_RE.search(str(ref))
        if m:
            nums.add(m.group(1))
    return nums


def build_disease_bridge(disease_tables, mondo_efo_json=None):
    """OT disease id (EFO/MONDO/Orphanet/...) -> set(MONDO numbers);
    MONDO number -> set(lower-cased names incl. synonyms)."""
    id_to_mondo, mondo_to_names = {}, {}
    for df in disease_tables:
        for rid, name, xrefs, syn in zip(df["id"], df["name"],
                                         df["dbXRefs"] if "dbXRefs" in df else [None] * len(df),
                                         df["synonyms"] if "synonyms" in df else [None] * len(df)):
            xr = list(xrefs) if isinstance(xrefs, (list, np.ndarray)) else []
            nums = mondo_nums_of(rid, xr)
            if not nums:
                continue
            id_to_mondo.setdefault(rid, set()).update(nums)
            names = {str(name).strip().lower()}
            if isinstance(syn, dict):
                for k in ("hasExactSynonym", "hasRelatedSynonym"):
                    v = syn.get(k)
                    if v is not None and len(v) > 0:
                        names.update(str(x).strip().lower() for x in v)
            for n in nums:
                mondo_to_names.setdefault(n, set()).update(names)
    if mondo_efo_json and os.path.exists(mondo_efo_json):       # optional extra bridge
        with open(mondo_efo_json) as f:
            for mondo, efos in json.load(f).items():
                for efo in efos:
                    id_to_mondo.setdefault(f"EFO_{efo}", set()).add(mondo)
    return id_to_mondo, mondo_to_names


def our_disease_names(kg_csv, cache_path, chunksize=500_000):
    """lower-cased disease name -> set(our node_id strings). Cached to disk."""
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            return {k: set(v) for k, v in json.load(f).items()}
    print(f"  streaming {kg_csv} once for disease names (a few minutes; cached afterwards) ...")
    out = {}
    cols = ["x_id", "x_type", "x_name", "y_id", "y_type", "y_name"]
    for ch in pd.read_csv(kg_csv, chunksize=chunksize, dtype=str, usecols=cols):
        for side in ("x", "y"):
            sub = ch[ch[f"{side}_type"] == "disease"][[f"{side}_id", f"{side}_name"]].drop_duplicates()
            for rid, name in zip(sub[f"{side}_id"], sub[f"{side}_name"]):
                if pd.isna(name):
                    continue
                key = name.strip().lower()
                out.setdefault(key, set()).add(str(rid))
                for s in str(rid).split("_"):
                    out[key].add(s)
    with open(cache_path, "w") as f:
        json.dump({k: sorted(v) for k, v in out.items()}, f)
    return out


def chembl_to_drugbank(ids, cache_path, sleep=0.1):
    """Resumable UniChem lookup (same cache file as the earlier validation scripts)."""
    import requests
    cache = json.load(open(cache_path)) if os.path.exists(cache_path) else {}
    todo = [c for c in ids if c not in cache]
    print(f"  ChEMBL->DrugBank: {len(ids) - len(todo)} cached, {len(todo)} to query")
    for i, c in enumerate(todo, 1):
        try:
            r = requests.post(UNICHEM_URL, json={"type": "sourceID", "compound": c, "sourceID": 1},
                              timeout=15).json()
            db = None
            if r.get("compounds"):
                m = [s for s in r["compounds"][0].get("sources", []) if s["id"] == 2]
                db = m[0]["compoundId"] if m else None
            cache[c] = db
        except Exception:
            cache[c] = None
        time.sleep(sleep)
        if i % 100 == 0:
            json.dump(cache, open(cache_path, "w"))
            print(f"    [{i}/{len(todo)}]")
    json.dump(cache, open(cache_path, "w"))
    return cache


def map_pairs(pairs, drug_map_cb, disease_ids_cb):
    """pairs: DataFrame(drugId, diseaseId, phase) -> set of (drug_local, dis_local) with max phase."""
    best = {}
    for d, dis, ph in zip(pairs.drugId, pairs.diseaseId, pairs.phase):
        dl = drug_map_cb(d)
        if dl is None:
            continue
        for sl in disease_ids_cb(dis):
            k = (dl, sl)
            if ph > best.get(k, -1):
                best[k] = ph
    return best


# ------------------------------------------------------------ sampling / scoring
def sample_random(n, n_drugs, n_dis, excluded, rng):
    keys = np.empty(0, dtype=np.int64)
    for _ in range(100):
        if len(keys) >= n:
            break
        k = rng.integers(0, n_drugs, int(n * 1.3) + 1000).astype(np.int64) * n_dis \
            + rng.integers(0, n_dis, int(n * 1.3) + 1000).astype(np.int64)
        k = k[~np.isin(k, excluded)]
        keys = np.unique(np.concatenate([keys, k]))
    return rng.permutation(keys)[:n]


def sample_matched(n, pos_d, pos_s, n_dis, excluded, rng):
    """Drug ~ positives' drug marginal, disease ~ positives' disease marginal."""
    keys = np.empty(0, dtype=np.int64)
    for _ in range(100):
        if len(keys) >= n:
            break
        m = int(n * 1.5) + 1000
        k = rng.choice(pos_d, m).astype(np.int64) * n_dis + rng.choice(pos_s, m).astype(np.int64)
        k = k[~np.isin(k, excluded)]
        keys = np.unique(np.concatenate([keys, k]))
    return rng.permutation(keys)[:n]


def metrics(y, s, thr_val):
    out = {"prevalence": float(y.mean()), "auroc": float(roc_auc_score(y, s)),
           "auprc": float(average_precision_score(y, s))}
    for name, thr in (("t05", 0.0), ("tval", thr_val)):
        if thr is None:
            continue
        pred = s >= thr
        tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum()); fn = int((~pred & (y == 1)).sum())
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        out[f"precision_{name}"], out[f"recall_{name}"] = p, r
        out[f"f1_{name}"] = 2 * p * r / (p + r) if p + r else 0.0
    return out


def run_eval(pos_d, pos_s, scorers, n_drugs, n_dis, excluded, repeats, seed,
             ratios=RATIOS, config=""):
    """scorers: {name: (fn(drug_idx, dis_idx)->scores, thr_val or None)}"""
    n_pos = len(pos_d)
    pos_sc = {k: f(pos_d, pos_s) for k, (f, _) in scorers.items()}
    rows = []
    for rep in range(repeats):
        rng = np.random.default_rng(seed + rep)
        for mode in ("random", "degree_matched"):
            need = int(max(ratios) * n_pos)
            pool = (sample_random(need, n_drugs, n_dis, excluded, rng) if mode == "random"
                    else sample_matched(need, pos_d, pos_s, n_dis, excluded, rng))
            nd, ns = pool // n_dis, pool % n_dis
            neg_sc = {k: f(nd, ns) for k, (f, _) in scorers.items()}
            for r in ratios:
                n_neg = min(int(round(r * n_pos)), len(pool))
                y = np.concatenate([np.ones(n_pos), np.zeros(n_neg)]).astype(int)
                for k, (_, thr) in scorers.items():
                    sc = np.concatenate([pos_sc[k], neg_sc[k][:n_neg]])
                    row = {"config": config, "neg_mode": mode, "scorer": k, "ratio": r,
                           "ratio_label": f"1:{r}", "repeat": rep, "n_pos": n_pos, "n_neg": n_neg}
                    row.update(metrics(y, sc, thr))
                    rows.append(row)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--old-known", default=os.path.join("data", "known_drugs_aggregated"))
    ap.add_argument("--new-dir", default=os.path.join("data", "ot_24.06"))
    ap.add_argument("--old-diseases", default=os.path.join("data", "diseases"))
    ap.add_argument("--kg-csv", default=os.path.join("data", "benchmark", "snapshots", "kg_t0.csv"))
    ap.add_argument("--checkpoint", default=os.path.join("output", "models", "graphsage_v3_real_features.pt"))
    ap.add_argument("--candidates", default=os.path.join(PROCESSED, "candidate_pairs.csv"))
    ap.add_argument("--min-phase", type=int, nargs="+", default=[1, 4])
    ap.add_argument("--graph", choices=["full", "masked"], default="full")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="output")
    args = ap.parse_args()

    # lazy heavy imports (torch) so the helpers above stay importable/testable
    import torch
    from exp1_class_imbalance import load_scorer, tune_threshold, pair_keys

    print("Step 1: loading Open Targets tables ...")
    old = load_known_pairs(args.old_known, "OT 21.11")
    new = load_known_pairs(os.path.join(args.new_dir, "known_drug"), "OT new")
    dis_old = read_parquet_dir(args.old_diseases)
    dis_new = read_parquet_dir(os.path.join(args.new_dir, "diseases"))

    print("\nStep 2: bridging identifiers (ChEMBL->DrugBank, OT disease->MONDO->name->our node) ...")
    id_to_mondo, mondo_to_names = build_disease_bridge(
        [dis_old, dis_new], os.path.join(PROCESSED, "mondo_efo_map.json"))
    name_to_ids = our_disease_names(args.kg_csv, os.path.join(PROCESSED, "our_disease_name_to_ids.json"))
    chembl = chembl_to_drugbank(sorted(set(old.drugId) | set(new.drugId)),
                                os.path.join(PROCESSED, "chembl_to_drugbank.json"))

    ent = pd.read_csv(os.path.join(PROCESSED, "entities.csv"), dtype={"node_id": str})
    drug_local = dict(zip(ent.loc[ent.node_type == "drug", "node_id"], ent.loc[ent.node_type == "drug", "local_idx"]))
    dis_local = dict(zip(ent.loc[ent.node_type == "disease", "node_id"], ent.loc[ent.node_type == "disease", "local_idx"]))

    dis_cache = {}

    def disease_ids(ot_id):
        if ot_id not in dis_cache:
            locs = set()
            for m in id_to_mondo.get(ot_id, ()):
                for name in mondo_to_names.get(m, ()):
                    for nid in name_to_ids.get(name, ()):
                        if nid in dis_local:
                            locs.add(int(dis_local[nid]))
            dis_cache[ot_id] = locs
        return dis_cache[ot_id]

    def drug_id(chembl_id):
        db = chembl.get(chembl_id)
        return int(drug_local[db]) if db and db in drug_local else None

    old_map = map_pairs(old, drug_id, disease_ids)
    new_map = map_pairs(new, drug_id, disease_ids)
    print(f"  mapped pairs  OT 21.11: {len(old_map)}   OT new: {len(new_map)}")

    print("\nStep 3: loading model, t0 graph, scorers ...")
    cand = pd.read_csv(args.candidates)
    # thr_val from the ORIGINAL masked setup
    score_masked, n_drugs, n_dis, contra_keys, _ = load_scorer(args, cand)
    val = cand[cand.split == "val"]
    thr_val = tune_threshold(val.label.values, score_masked(val.drug_idx.values, val.disease_idx.values))
    if args.graph == "full":
        score_fn, _, _, _, _ = load_scorer(args, cand.iloc[0:0])      # hides nothing
    else:
        score_fn = score_masked
    data = torch.load(os.path.join(PROCESSED, "pharmagraph_hetero_t0.pt"), weights_only=False)
    ei = data["drug", "indication", "disease"].edge_index.numpy()
    t0_keys = np.unique(pair_keys(ei[0], ei[1], n_dis))
    t0_pairs = set(zip(ei[0].tolist(), ei[1].tolist()))
    deg_d = np.bincount(ei[0], minlength=n_drugs)
    deg_s = np.bincount(ei[1], minlength=n_dis)
    degree_fn = lambda d, s: np.log1p(deg_d[np.asarray(d)]) + np.log1p(deg_s[np.asarray(s)])
    print(f"  t0 indication pairs in graph: {len(t0_pairs)}   thr_val={thr_val:.3f}   graph={args.graph}")

    all_known = np.unique(np.concatenate([
        t0_keys, contra_keys,
        pair_keys(cand.drug_idx.values, cand.disease_idx.values, n_dis),
        pair_keys([k[0] for k in old_map], [k[1] for k in old_map], n_dis),
        pair_keys([k[0] for k in new_map], [k[1] for k in new_map], n_dis),
    ]))
    cand_neg = set(zip(cand[cand.label == 0].drug_idx, cand[cand.label == 0].disease_idx))

    scorers = {"model": (score_fn, thr_val), "degree_baseline": (degree_fn, None)}
    all_long, pos_rows = [], []
    print("\nStep 4: building positives and evaluating ...")
    for min_phase in args.min_phase:
        for stratum in ("new_vs_t0", "new_vs_ot2111"):
            pos = [k for k, ph in new_map.items()
                   if ph >= min_phase and k not in t0_pairs and (stratum == "new_vs_t0" or k not in old_map)]
            cfg = f"phase>={min_phase} | {stratum}"
            if len(pos) < 30:
                print(f"  [{cfg}] only {len(pos)} positives - skipped (too few for stable metrics)")
                continue
            pd_, ps_ = np.array([k[0] for k in pos]), np.array([k[1] for k in pos])
            in_cand_neg = sum(k in cand_neg for k in pos)
            print(f"  [{cfg}] n_pos={len(pos)} | drugs={len(set(pd_))} diseases={len(set(ps_))} | "
                  f"median drug-degree {np.median(deg_d[pd_]):.0f} vs all {np.median(deg_d):.0f} | "
                  f"{in_cand_neg} were training/val/test negatives")
            for k in pos:
                pos_rows.append({"config": cfg, "drug_idx": k[0], "disease_idx": k[1], "phase": new_map[k]})
            all_long.append(run_eval(pd_, ps_, scorers, n_drugs, n_dis, all_known,
                                     args.repeats, args.seed, config=cfg))

    if not all_long:
        raise SystemExit("No configuration had enough positives. Check mapping counts printed in Step 2.")
    long_df = pd.concat(all_long, ignore_index=True)
    os.makedirs(args.out_dir, exist_ok=True)
    long_df.to_csv(os.path.join(args.out_dir, "exp2_temporal_long.csv"), index=False)
    pd.DataFrame(pos_rows).to_csv(os.path.join(args.out_dir, "exp2_positives.csv"), index=False)

    mc = [c for c in ("auroc", "auprc", "f1_t05", "f1_tval", "precision_tval", "recall_tval") if c in long_df]
    g = long_df.groupby(["config", "neg_mode", "scorer", "ratio_label", "ratio"])
    summ = g[mc].mean().join(g[["auroc"]].std().rename(columns={"auroc": "auroc_std"})) \
        .join(g.agg(n_pos=("n_pos", "first"), n_neg=("n_neg", "first"), prevalence=("prevalence", "mean"))).reset_index()
    summ.to_csv(os.path.join(args.out_dir, "exp2_temporal_summary.csv"), index=False)

    show = summ.sort_values(["config", "neg_mode", "ratio", "scorer"]).drop(columns=["ratio", "precision_tval", "recall_tval"])
    print("\n" + "=" * 110)
    print("EXPERIMENT 2 - temporal validation vs Open Targets (mean over repeats)")
    print("=" * 110)
    print(show.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\nReading guide: the model only counts as evidence if it beats degree_baseline, "
          "especially in the degree_matched rows (hub shortcut removed).")
    print("Saved output/exp2_temporal_long.csv, exp2_temporal_summary.csv, exp2_positives.csv")


if __name__ == "__main__":
    main()