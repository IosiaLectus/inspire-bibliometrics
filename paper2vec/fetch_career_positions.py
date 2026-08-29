#!/usr/bin/python3

################################################################################
# Q3: does paper-embedding position at PhD graduation predict time to
# faculty, and whether it happens at all?
#
# fetch_advisors.py already pulled each researcher's PHD-rank position
# (start/end year) but discarded the rest of their positions history. This
# fetches the FULL positions list (rank, start_date, end_date, institution)
# for every eligible researcher, needed to find if/when they first reached
# a faculty-level rank (JUNIOR/SENIOR/STAFF) after graduating.
################################################################################

import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

API = "https://inspirehep.net/api/authors"
HERE = os.path.dirname(os.path.abspath(__file__))
ELIGIBLE_PATH = os.path.join(HERE, "eligible_researchers.json")
OUT_PATH = os.path.join(HERE, "career_positions.json")
MAX_WORKERS = 8
CHECKPOINT_EVERY = 50


def fetch_positions(bai, retries=4):
    for attempt in range(retries):
        try:
            r = requests.get(
                API,
                params={"q": f"ids.value:{bai}", "size": 1, "fields": "positions"},
                timeout=30,
            )
            r.raise_for_status()
            hits = r.json().get("hits", {}).get("hits", [])
            break
        except Exception:
            if attempt == retries - 1:
                return bai, "ERROR"
    if not hits:
        return bai, None
    positions = hits[0]["metadata"].get("positions") or []
    cleaned = [
        {
            "rank": p.get("rank"),
            "start_date": p.get("start_date"),
            "end_date": p.get("end_date"),
            "institution": p.get("institution"),
            "current": p.get("current", False),
        }
        for p in positions
    ]
    return bai, cleaned


def main():
    with open(ELIGIBLE_PATH) as f:
        all_bais = [p["bai"] for p in json.load(f)]

    try:
        with open(OUT_PATH) as f:
            results = json.load(f)
    except FileNotFoundError:
        results = {}

    needed = [b for b in all_bais if b not in results]
    print(f"{len(needed)} of {len(all_bais)} still needed", flush=True)

    lock = threading.Lock()
    done = [0]

    def checkpoint(force=False):
        with lock:
            if force or done[0] % CHECKPOINT_EVERY == 0:
                with open(OUT_PATH, "w") as f:
                    json.dump(results, f)
                print(f"[{done[0]}/{len(needed)}] checkpoint saved", flush=True)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {ex.submit(fetch_positions, b): b for b in needed}
        for fut in as_completed(futs):
            bai, positions = fut.result()
            with lock:
                results[bai] = positions
                done[0] += 1
            checkpoint()

    checkpoint(force=True)
    n_err = sum(1 for v in results.values() if v == "ERROR")
    print(f"DONE: {len(results)}/{len(all_bais)} ({n_err} errors)", flush=True)


if __name__ == "__main__":
    main()
