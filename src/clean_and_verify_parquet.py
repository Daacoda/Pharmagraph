# -*- coding: utf-8 -*-
"""
Clean non-parquet and corrupted files from OT 24.06 directories
"""
import os
import pyarrow.parquet as pq

dirs = [
    os.path.join("data", "ot_24.06", "known_drug"),
    os.path.join("data", "ot_24.06", "diseases")
]

for d in dirs:
    print(f"\nChecking directory: {d}")
    if not os.path.exists(d):
        continue
    for fname in os.listdir(d):
        fpath = os.path.join(d, fname)
        # Remove non-parquet files like .json
        if not fname.endswith(".parquet"):
            print(f"  --> Removing non-parquet file: {fname}")
            os.remove(fpath)
            continue
        # Verify parquet readability
        try:
            pq.read_metadata(fpath)
        except Exception:
            print(f"  --> Removing corrupted parquet file: {fname}")
            os.remove(fpath)

print("\nCleanup complete!")