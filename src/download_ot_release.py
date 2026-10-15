
# -*- coding: utf-8 -*-
"""
Download Open Targets indication Parquet files
for temporal evaluation.

Usage from project root:
    py src\download_ot_release.py
"""

import os
from ftplib import FTP

HOST = "ftp.ebi.ac.uk"
BASE = "/pub/databases/opentargets/platform"

RELEASES = ["23.06", "24.06"]


def main():
    ftp = FTP(HOST, timeout=120)
    ftp.login()

    try:
        for release in RELEASES:
            remote = (
                f"{BASE}/{release}/output/etl/parquet/indication"
            )
            local = os.path.join(
                "data", f"ot_{release}", "indication"
            )

            print(f"\nRelease: {release}")
            print(f"Remote: {remote}")

            ftp.cwd(remote)
            names = ftp.nlst()

            parquet_files = [
                name for name in names
                if name.lower().endswith(".parquet")
                and "/" not in name
            ]

            if not parquet_files:
                print("ERROR: No .parquet files found.")
                print("Remote directory entries:")
                for name in names[:20]:
                    print("  ", name)
                continue

            os.makedirs(local, exist_ok=True)
            print(f"Parquet files found: {len(parquet_files)}")
            print(f"Destination: {local}")

            for i, name in enumerate(parquet_files, 1):
                path = os.path.join(local, name)

                if os.path.isfile(path) and os.path.getsize(path) > 0:
                    print(f"[{i}/{len(parquet_files)}] Exists: {name}")
                    continue

                with open(path, "wb") as output:
                    ftp.retrbinary(
                        f"RETR {name}",
                        output.write
                    )

                print(f"[{i}/{len(parquet_files)}] Downloaded: {name}")

    finally:
        ftp.quit()

    print("\nDownload step finished.")


if __name__ == "__main__":
    main()

