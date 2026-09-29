# PrimeKG-CL Benchmark v1.0

A continual graph learning benchmark on a real biomedical knowledge graph
with genuine temporal evolution.

## Snapshots
- `snapshots/kg_t0.csv` — June 2021 PrimeKG release. 8,100,498 triples, 129,375 nodes, 30 relations.
- `snapshots/kg_t1.csv` — July 2023 reconstruction from nine upstream databases (Bgee, CTD, GO, Gene2GO, HPO, HPOA, MONDO, Uberon, HGNC). 13,001,666 triples, 134,211 nodes, 25 relations.
- `snapshots/t1_build_info.json` — provenance / database versions used for the t_1 reconstruction.

## Temporal diff (t_0 -> t_1)
- Added: 5,760,234 new triples (driven by GO annotations, disease-phenotype refinement, drug-target updates)
- Removed: 888,848 deprecated triples (retracted associations, ontology corrections)
- Persistent: 7,208,624 triples unchanged across both snapshots

Full breakdown in `diffs/diff_t0_t1.json`.

## Continual learning tasks
10 entity-type-grouped tasks (`tasks/task_*`). Each directory contains
`train.txt`, `valid.txt`, `test.txt`, one tab-separated triple per line
(`head_id<TAB>relation<TAB>tail_id`). 70/10/20 train/valid/test split.

```
task_0_base                    8,100,498 triples (full t_0)
task_1_disease_related            17,009
task_1_drug_related              125,343
task_2_disease_related           115,382
task_2_gene_protein            2,850,593
task_3_gene_protein               99,761
task_3_phenotype_related          47,997
task_4_biological_process        116,118
task_4_phenotype_related          57,390
task_5_anatomy_pathway         2,752,675
```

## Stratified evaluation
`test_stratification.json` — per-task partition of test triples into
**persistent** (in both snapshots), **added** (new in t_1), and
**removed** (only in t_0) strata, supporting stratified-MRR analysis of
correct retention vs. correct forgetting.

Task 0's test split: 1,443,243 persistent + 176,856 removed = 1,620,099 total.

## Multimodal features (`features/`)
- `text_embeddings.pt` — BiomedBERT [CLS] embeddings projected to 256-d.
  Coverage: 36% of entities (those with textual descriptions).
- `mol_features.pt` — Morgan fingerprints (radius 2, 1024 bits) for drugs
  with available SMILES. Coverage: 4% of entities.
- `edge_index.pt`, `edge_type.pt` — R-GCN message-passing tensors for the
  full t_0 graph (used by the structural encoder).
- `node_has_text.pt`, `node_has_mol.pt` — boolean coverage masks.
- `node_index_map.csv` — global entity_id <-> row_index map.
- `vocab_sizes.json` — entity / relation vocab sizes.

## Files
```
LICENSE                       MIT (code) + CC BY 4.0 (data)
croissant.json                MLCommons Croissant metadata
README.md                     this file
statistics.json               benchmark-level summary
test_stratification.json      per-task persistent/added/removed counts
diffs/diff_t0_t1.json         full t_0->t_1 diff with per-relation breakdown
snapshots/                    kg_t0.csv, kg_t1.csv, build provenance
tasks/                        10 task directories with train/valid/test splits
features/                     multimodal node features
```

## Citation

If you use this dataset, please cite:

```bibtex
@article{beecroft2022primekg,
  title={Building a knowledge graph to enable precision medicine},
  author={Chandak, Payal and Huang, Kexin and Zitnik, Marinka},
  journal={Scientific Data},
  volume={9},
  number={1},
  pages={667},
  year={2022},
  publisher={Nature Publishing Group}
}

@inproceedings{primekgcl2026,
  title={PrimeKG-CL: A Continual Graph Learning Benchmark on Evolving Biomedical Knowledge Graphs},
  author={Anonymous},
  booktitle={NeurIPS Datasets and Benchmarks Track},
  year={2026}
}
```

## License
- Code components: MIT.
- Data files: CC BY 4.0, inheriting PrimeKG's license; users must respect upstream-database licenses where applicable.
- See `LICENSE` for details.


🧬 Open Targets Data Description: association_by_datatype_indirect
This dataset includes extracted records of association_by_datatype_indirect from the Open Targets platform, which is used to study and evaluate indirect target-disease associations, support drug repurposing research, and analyze biomedical knowledge graphs.

📋 Data Schema & Components
These files represent the extracted biological and disease-related interaction data, comprising the following core elements:

Ensembl Gene IDs: The standard identifiers for the targeted genes (e.g., ENSG...).

EFO Disease IDs: Standardized disease classifications based on the Experimental Factor Ontology (e.g., EFO_00001836).

Association Scores & Evidence Strings: Numerical values and metrics associated with each gene-disease pair to evaluate the strength of scientific and biological evidence.

🛠️ Project Usage
This data is utilized and integrated within pipelines for:

Filtering and analyzing relationships between genetic targets and diseases.

Constructing and representing biomedical knowledge networks (Knowledge Graphs).

Training and evaluating machine learning models for drug-disease prediction tasks.

📥 Data Source & Download
The raw Parquet files for this dataset can be accessed and downloaded directly from the official Open Targets FTP server:

Repository Link: 
https://ftp.ebi.ac.uk/pub/databases/opentargets/platform/21.11/output/etl/parquet/
