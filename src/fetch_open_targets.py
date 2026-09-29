"""
Stage 1 - Open Targets evidence lookup
======================================
Builds processed/ot_scores.parquet: one row per (gene node, disease node) with
an Open Targets association score, keyed by YOUR graph's node ids.

Why bulk download instead of the API: Open Targets documents the GraphQL API as
meant for single-entity queries and points to the data downloads / BigQuery for
systematic multi-target work. Get the association tables from the Downloads
page of the Open Targets Platform, then point --ot-dir at the parquet folder.

Leakage warning: the overall score includes the "known_drug" datatype, which
is built from approved drug indications - i.e. the very labels you are trying
to predict. By default this script uses the per-datatype table and drops
known_drug. Use --use-overall only for non-evaluation exploration.

Steps:
  1. NCBI Entrez gene id (your gene nodes) -> Ensembl id, via mygene.info (cached)
  2. Disease node id (MONDO number) -> Open Targets id like MONDO_0005147
  3. Read Open Targets scores filtered to those genes/diseases
  4. Save gene_node_id, disease_node_id, ot_score

Usage (from project root):
  python src/fetch_open_targets.py --spot-check ENSG00000105974
  python src/fetch_open_targets.py --ot-dir data/association_by_datatype_indirect
"""
import argparse
import json
import time
from pathlib import Path

import pandas as pd
import requests

PROCESSED = Path("processed")
OT_API = "https://api.platform.opentargets.org/api/v4/graphql"
MYGENE_URL = "https://mygene.info/v3/gene"
EXCLUDE_DATATYPES = {"known_drug"}


def disease_node_to_ot_ids(node_id):
    """PrimeKG disease ids are MONDO numbers (sometimes several joined by '_').
    Returns candidate Open Targets ids, e.g. '5147' -> ['MONDO_0005147']."""
    ids = []
    for tok in str(node_id).replace(",", "_").split("_"):
        tok = tok.strip()
        if tok.isdigit():
            ids.append(f"MONDO_{int(tok):07d}")
        elif tok.upper().startswith("MONDO:") and tok[6:].isdigit():
            ids.append(f"MONDO_{int(tok[6:]):07d}")
    return ids


def parse_mygene_hits(hits):
    """mygene.info returns a list of dicts; 'ensembl' may be a dict or a list."""
    mapping = {}
    for h in hits:
        if h.get("notfound"):
            continue
        ens = h.get("ensembl")
        if isinstance(ens, list):
            ens = ens[0] if ens else None
        if isinstance(ens, dict) and ens.get("gene"):
            mapping[str(h["query"])] = ens["gene"]
    return mapping


def map_entrez_to_ensembl(entrez_ids, cache_path, batch=500, sleep=0.5):
    """Resumable: results (including 'not found' = None) are cached on disk."""
    cache_path = Path(cache_path)
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    todo = [g for g in entrez_ids if g not in cache]
    print(f"Entrez->Ensembl: {len(cache)} cached, {len(todo)} to fetch")
    for i in range(0, len(todo), batch):
        chunk = todo[i:i + batch]
        for _ in range(3):
            try:
                r = requests.post(
                    MYGENE_URL,
                    data={"ids": ",".join(chunk), "fields": "ensembl.gene", "species": "human"},
                    timeout=60,
                )
                r.raise_for_status()
                found = parse_mygene_hits(r.json())
                break
            except Exception as e:
                print("  retrying after error:", e)
                time.sleep(2)
        else:
            raise RuntimeError("mygene.info failed 3 times; re-run to resume from cache")
        for g in chunk:
            cache[g] = found.get(g)
        cache_path.write_text(json.dumps(cache))
        print(f"  [{min(i + batch, len(todo))}/{len(todo)}]")
        time.sleep(sleep)
    return cache


def load_ot_scores(ot_dir, ensembl_ids, ot_disease_ids,
                   target_col="targetId", disease_col="diseaseId",
                   score_col="score", datatype_col="datatypeId", use_overall=False):
    """Reads Open Targets parquet, filtered on load so memory stays small."""
    df = pd.read_parquet(
        ot_dir,
        filters=[(target_col, "in", list(ensembl_ids)), (disease_col, "in", list(ot_disease_ids))],
    )
    print("Open Targets columns:", list(df.columns), "| rows after filtering:", len(df))
    if datatype_col in df.columns and not use_overall:
        n0 = len(df)
        df = df[~df[datatype_col].isin(EXCLUDE_DATATYPES)]
        print(f"Dropped {n0 - len(df)} rows from excluded datatypes {sorted(EXCLUDE_DATATYPES)}")
    elif use_overall:
        print("WARNING: --use-overall keeps known_drug evidence -> label leakage for evaluation")
    return (df.groupby([target_col, disease_col], as_index=False)[score_col].max()
              .rename(columns={score_col: "ot_score"}))


def spot_check(ensembl_id, n=5):
    """Confirms the API is reachable and shows what disease ids look like."""
    query = """query t($id: String!) {
      target(ensemblId: $id) { approvedSymbol
        associatedDiseases(page: {size: %d, index: 0}) { count rows { score disease { id name } } } } }""" % n
    r = requests.post(OT_API, json={"query": query, "variables": {"id": ensembl_id}}, timeout=60)
    r.raise_for_status()
    t = r.json()["data"]["target"]
    print(f"{t['approvedSymbol']}: {t['associatedDiseases']['count']} associated diseases. Top {n}:")
    for row in t["associatedDiseases"]["rows"]:
        print(f"  {row['disease']['id']:<16} {row['score']:.3f}  {row['disease']['name']}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--spot-check", help="Ensembl gene id to test the API, e.g. ENSG00000105974")
    p.add_argument("--ot-dir", help="Folder of downloaded Open Targets association parquet files")
    p.add_argument("--target-col", default="targetId")
    p.add_argument("--disease-col", default="diseaseId")
    p.add_argument("--score-col", default="score")
    p.add_argument("--datatype-col", default="datatypeId")
    p.add_argument("--use-overall", action="store_true")
    args = p.parse_args()

    if args.spot_check:
        spot_check(args.spot_check)
        return
    if not args.ot_dir:
        p.error("give --ot-dir (or --spot-check to test the API first)")

    entities = pd.read_csv(PROCESSED / "entities.csv", dtype={"node_id": str})
    genes = entities.loc[entities.node_type == "gene/protein", "node_id"].unique().tolist()
    diseases = entities.loc[entities.node_type == "disease", "node_id"].unique().tolist()
    print("Sample disease node ids (check these look like MONDO numbers):", diseases[:5])

    entrez2ens = map_entrez_to_ensembl(genes, PROCESSED / "entrez_to_ensembl.json")
    gene_rows = pd.DataFrame([(g, e) for g, e in entrez2ens.items() if e],
                             columns=["gene_node_id", "ensembl_id"])
    print(f"Genes mapped to Ensembl: {len(gene_rows)}/{len(genes)}")

    dis_rows = pd.DataFrame([(d, oid) for d in diseases for oid in disease_node_to_ot_ids(d)],
                            columns=["disease_node_id", "ot_disease_id"])
    print(f"Diseases with an Open Targets-style id: {dis_rows.disease_node_id.nunique()}/{len(diseases)}")

    ot = load_ot_scores(args.ot_dir, gene_rows.ensembl_id.unique(), dis_rows.ot_disease_id.unique(),
                        args.target_col, args.disease_col, args.score_col, args.datatype_col, args.use_overall)
    merged = (ot.merge(gene_rows, left_on=args.target_col, right_on="ensembl_id")
                .merge(dis_rows, left_on=args.disease_col, right_on="ot_disease_id"))
    out = (merged.groupby(["gene_node_id", "disease_node_id"], as_index=False)["ot_score"].max())
    out.to_parquet(PROCESSED / "ot_scores.parquet", index=False)
    print(f"Saved processed/ot_scores.parquet: {len(out)} gene-disease pairs, "
          f"{out.disease_node_id.nunique()} diseases with any score")


if __name__ == "__main__":
    main()
