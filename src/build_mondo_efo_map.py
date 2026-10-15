"""
build_mondo_efo_map.py
========================
Builds a MONDO <-> EFO mapping directly from the Mondo Disease Ontology's own
official release (data/mondo.obo), instead of relying on Open Targets'
frozen 2021 dbXRefs. Mondo is the authoritative, actively-maintained source
for these cross-references (curated as precise equivalence mappings), so
this should give substantially better coverage than the previous approach.

EFO ids appear inside a [Term] block either as a direct `xref: EFO:xxxxx`
line, or embedded in another xref's {source="EFO:xxxxx"} annotation (e.g.
`xref: DOID:4 {source="MONDO:equivalentTo", source="EFO:0000408"}`) - both
forms are captured.

Output: processed/mondo_efo_map.json  ({mondo_numeric_id: [efo_ids]})

Run from the project root:
    python build_mondo_efo_map.py
"""
import json
import os
import re

PROCESSED = "processed"
MONDO_OBO_PATH = os.path.join("data", "mondo.obo")

EFO_PATTERN = re.compile(r"EFO[:_](\d+)")
MONDO_ID_PATTERN = re.compile(r"^id:\s*MONDO:(\d+)")


def main():
    if not os.path.exists(MONDO_OBO_PATH):
        raise SystemExit(f"{MONDO_OBO_PATH} not found - download it first:\n"
                          f'  python -c "import urllib.request; '
                          f"urllib.request.urlretrieve('http://purl.obolibrary.org/obo/mondo.obo', "
                          f"'{MONDO_OBO_PATH}')\"")

    print(f"Reading {MONDO_OBO_PATH} ...")
    with open(MONDO_OBO_PATH, encoding="utf-8") as f:
        lines = f.readlines()
    print(f"  {len(lines)} lines.")

    mondo_to_efo = {}
    current_mondo = None
    in_term = False
    n_terms = 0
    n_obsolete = 0

    for line in lines:
        if line.startswith("[Term]"):
            in_term = True
            current_mondo = None
            n_terms += 1
            continue
        if not in_term:
            continue
        if line.strip() == "":
            in_term = False
            continue

        if current_mondo is None:
            m = MONDO_ID_PATTERN.match(line)
            if m:
                current_mondo = m.group(1)
            continue

        if line.startswith("is_obsolete: true"):
            n_obsolete += 1
            current_mondo = None  # stop collecting xrefs for obsolete terms
            continue

        efo_matches = EFO_PATTERN.findall(line)
        if efo_matches:
            mondo_to_efo.setdefault(current_mondo, set()).update(efo_matches)

    mondo_to_efo = {k: sorted(v) for k, v in mondo_to_efo.items()}

    print(f"\nParsed {n_terms} [Term] blocks ({n_obsolete} obsolete, skipped).")
    print(f"MONDO ids with at least one EFO cross-reference: {len(mondo_to_efo)}")
    total_efo_refs = sum(len(v) for v in mondo_to_efo.values())
    print(f"Total MONDO->EFO edges: {total_efo_refs}")

    os.makedirs(PROCESSED, exist_ok=True)
    out_path = os.path.join(PROCESSED, "mondo_efo_map.json")
    with open(out_path, "w") as f:
        json.dump(mondo_to_efo, f)
    print(f"\nSaved {out_path}")

    # quick sanity check against a known example from manual inspection
    sample = mondo_to_efo.get("0000001")
    print(f"\nSanity check - MONDO:0000001 EFO refs: {sample}")


if __name__ == "__main__":
    main()