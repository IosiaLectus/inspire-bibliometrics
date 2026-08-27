#!/usr/bin/python3

################################################################################
# Diagnostic: fetches each BAI's latest-paper year (sort=mostrecent, size=1),
# the mirror image of fetch_first_years.py. Used to explore alternative
# inclusion criteria based on career recency/length rather than a fixed
# first-paper-year window.
################################################################################

import json
import os
import sys
import time

import requests

API = "https://inspirehep.net/api/literature"
HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PATH = os.path.join(HERE, "latest_years.json")


def get_latest_year(bai, retries=4):
    for attempt in range(retries):
        try:
            r = requests.get(
                API,
                params={
                    "q": f"author:{bai}",
                    "size": 1,
                    "sort": "mostrecent",
                    "fields": "earliest_date",
                },
                timeout=30,
            )
            r.raise_for_status()
            data = r.json()
            break
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(1.5)
    hits = data["hits"]["hits"]
    if not hits:
        return None
    date = hits[0]["metadata"].get("earliest_date")
    if not date:
        return None
    return int(date[:4])


def main(bais):
    try:
        with open(OUTPUT_PATH) as f:
            results = json.load(f)
    except FileNotFoundError:
        results = {}

    total = len(bais)
    print(f"{len(results)}/{total} already fetched; resuming.", flush=True)
    for i, bai in enumerate(bais):
        if bai in results:
            continue
        results[bai] = get_latest_year(bai)
        if (i + 1) % 100 == 0 or i == total - 1:
            with open(OUTPUT_PATH, "w") as f:
                json.dump(results, f)
            print(f"[{i+1}/{total}] checkpoint saved, {len(results)} total done", flush=True)

    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f)
    print(f"DONE: {len(results)}/{total}", flush=True)


if __name__ == "__main__":
    with open(sys.argv[1]) as f:
        bais = json.load(f)
    main(bais)
