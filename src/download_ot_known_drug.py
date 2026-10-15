# -*- coding: utf-8 -*-
"""
Download missing knownDrugsAggregated and diseases parquet files for OT 24.06 with retry logic.
"""
import os
import time
from ftplib import FTP

HOST = "ftp.ebi.ac.uk"
BASE = "/pub/databases/opentargets/platform/24.06/output/etl/parquet"

TARGETS = {
    "knownDrugsAggregated": os.path.join("data", "ot_24.06", "known_drug"),
    "diseases": os.path.join("data", "ot_24.06", "diseases"),
}


def connect_ftp():
    ftp = FTP(HOST, timeout=120)
    ftp.login()
    return ftp


def download():
    ftp = connect_ftp()
    try:
        for remote_dir, local_dir in TARGETS.items():
            remote_path = f"{BASE}/{remote_dir}"
            print(f"\nChecking remote: {remote_path}")

            for _ in range(3):
                try:
                    ftp.cwd(remote_path)
                    break
                except Exception:
                    ftp = connect_ftp()

            files = [f for f in ftp.nlst() if f.lower().endswith(".parquet")]
            print(f"Found {len(files)} parquet files. Downloading to {local_dir}...")
            os.makedirs(local_dir, exist_ok=True)

            for i, fname in enumerate(files, 1):
                out_path = os.path.join(local_dir, fname)
                if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
                    print(f"[{i}/{len(files)}] Exists: {fname}")
                    continue

                for attempt in range(5):
                    try:
                        with open(out_path, "wb") as f:
                            ftp.retrbinary(f"RETR {fname}", f.write)
                        print(f"[{i}/{len(files)}] Downloaded: {fname}")
                        break
                    except Exception as e:
                        print(f"  --> Retry {attempt+1}/5 for {fname} due to connection drop...")
                        if os.path.exists(out_path):
                            try:
                                os.remove(out_path)
                            except OSError:
                                pass
                        time.sleep(2)
                        ftp = connect_ftp()
                        ftp.cwd(remote_path)
    finally:
        try:
            ftp.quit()
        except Exception:
            pass


if __name__ == "__main__":
    download()