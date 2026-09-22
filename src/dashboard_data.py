from pathlib import Path
import pandas as pd


# ---------------------------------------------------------------
# Paths
# ---------------------------------------------------------------

# Project root (this file lives at <ROOT>/src/dashboard_data.py)
ROOT = Path(__file__).resolve().parent.parent

T0_PATH = ROOT / "data" / "benchmark" / "snapshots" / "kg_t0.csv"
T1_PATH = ROOT / "data" / "benchmark" / "snapshots" / "kg_t1.csv"

CACHE_DIR = ROOT / "output" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

DRUG_DISEASE_RELATIONS = {
    "indication", "contraindication", "off-label use",
    "disease_protein", "drug_protein", "disease_phenotype_positive",
    "disease_phenotype_negative",
}


# ---------------------------------------------------------------
# Schema / debug helper (yours, unchanged)
# ---------------------------------------------------------------

def get_snapshot_info(snapshot):

    if snapshot == "t0":
        path = T0_PATH
    elif snapshot == "t1":
        path = T1_PATH
    else:
        raise ValueError("Snapshot must be 't0' or 't1'")

    # Read only a small sample first
    df = pd.read_csv(path, nrows=5)

    return {
        "columns": df.columns.tolist(),
        "sample": df.to_dict(orient="records")
    }


# ---------------------------------------------------------------
# Internal helper for streaming large CSVs without loading them
# fully into memory
# ---------------------------------------------------------------

def _chunked_filter(path, mask_fn, usecols=None, chunksize=500_000):
    """Stream a large snapshot CSV and keep only rows matching mask_fn."""
    keep = []
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunksize):
        sub = chunk[mask_fn(chunk)]
        if not sub.empty:
            keep.append(sub)
    if not keep:
        return pd.DataFrame(columns=usecols)
    return pd.concat(keep, ignore_index=True)


# ---------------------------------------------------------------
# Query layer used by Server.R
# ---------------------------------------------------------------

def get_disease_list():
    """Unique disease names across both snapshots. Cached after first run."""
    cache_path = CACHE_DIR / "disease_list.csv"
    if cache_path.exists():
        return pd.read_csv(cache_path)["disease"].tolist()

    names = set()
    for path in (T0_PATH, T1_PATH):
        df = _chunked_filter(
            path,
            lambda c: (c["x_type"] == "disease") | (c["y_type"] == "disease"),
            usecols=["x_type", "x_name", "y_type", "y_name"],
        )
        names.update(df.loc[df["x_type"] == "disease", "x_name"])
        names.update(df.loc[df["y_type"] == "disease", "y_name"])

    result = sorted(names)
    pd.DataFrame({"disease": result}).to_csv(cache_path, index=False)
    return result


def get_drug_disease_subset(disease_name):
    """
    All drug/disease-relevant edges touching `disease_name`, tagged with
    which snapshot(s) they appear in (persistent / added (t1) / removed (t0 only)).
    """
    cols = ["relation", "x_id", "x_type", "x_name", "y_id", "y_type", "y_name"]

    def mask(c):
        touches = (c["x_name"] == disease_name) | (c["y_name"] == disease_name)
        return touches & c["relation"].isin(DRUG_DISEASE_RELATIONS)

    t0 = _chunked_filter(T0_PATH, mask, usecols=cols)
    t1 = _chunked_filter(T1_PATH, mask, usecols=cols)

    key = ["relation", "x_id", "y_id"]
    t0_keys = set(map(tuple, t0[key].values)) if not t0.empty else set()
    t1_keys = set(map(tuple, t1[key].values)) if not t1.empty else set()

    combined = pd.concat([t0, t1], ignore_index=True).drop_duplicates(subset=key)

    def status(row):
        k = tuple(row[k_] for k_ in key)
        in0, in1 = k in t0_keys, k in t1_keys
        if in0 and in1:
            return "persistent"
        return "added (t1)" if in1 else "removed (t0 only)"

    if not combined.empty:
        combined["status"] = combined.apply(status, axis=1)
    return combined


def get_subgraph(entity_name, hops=1, snapshot="t1", max_neighbors=75):
    """
    1-hop (or `hops`-hop) neighborhood of `entity_name` from the given
    snapshot, returned as node/edge dicts ready for visNetwork.
    """
    path = T1_PATH if snapshot == "t1" else T0_PATH
    cols = ["relation", "x_id", "x_type", "x_name", "y_id", "y_type", "y_name"]

    frontier = {entity_name}
    edges_all = []
    for _ in range(hops):
        mask = lambda c: c["x_name"].isin(frontier) | c["y_name"].isin(frontier)
        hop_edges = _chunked_filter(path, mask, usecols=cols)
        if hop_edges.empty:
            break
        hop_edges = hop_edges.head(max_neighbors)
        edges_all.append(hop_edges)
        frontier = set(hop_edges["x_name"]).union(hop_edges["y_name"])

    if not edges_all:
        return {"nodes": [], "edges": []}

    edges_df = pd.concat(edges_all, ignore_index=True).drop_duplicates()
    node_names = pd.unique(edges_df[["x_name", "y_name"]].values.ravel())
    node_types = {}
    for _, r in edges_df.iterrows():
        node_types[r["x_name"]] = r["x_type"]
        node_types[r["y_name"]] = r["y_type"]

    nodes = [
        {"id": n, "label": n, "group": node_types.get(n, "other")}
        for n in node_names
    ]
    edges = [
        {"from": r["x_name"], "to": r["y_name"], "label": r["relation"]}
        for _, r in edges_df.iterrows()
    ]
    return {"nodes": nodes, "edges": edges}