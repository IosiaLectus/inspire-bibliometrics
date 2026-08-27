#!/usr/bin/python3

################################################################################
# v2 of fetch_citers.py, pointed at eligible_authors_v2.json (the three-part
# seed-rule population) instead of the original eligible_authors.json.
# Same linear, resumable, stable-sort structure.
################################################################################

import json
import os
import time

import requests

API = "https://inspirehep.net/api/literature"
HERE = os.path.dirname(os.path.abspath(__file__))
ELIGIBLE_PATH = os.path.join(HERE, "eligible_authors_v2.json")
OUTPUT_PATH = os.path.join(HERE, "citing_papers_v2.json")

PAGE_SIZE = 250
MAX_RESULT_WINDOW = 10000


def get_citing_control_numbers(bai, retries=4):
    ids = []
    page = 1
    while True:
        for attempt in range(retries):
            try:
                r = requests.get(
                    API,
                    params={
                        "q": f"refersto:author:{bai}",
                        "size": PAGE_SIZE,
                        "page": page,
                        "sort": "mostrecent",
                        "fields": "control_number",
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
            break
        ids.extend(h["metadata"]["control_number"] for h in hits)
        if len(hits) < PAGE_SIZE:
            break
        page += 1
        if page * PAGE_SIZE > MAX_RESULT_WINDOW:
            print(f"  WARNING: {bai} exceeds INSPIRE's {MAX_RESULT_WINDOW}-result "
                  f"pagination cap; truncating (flagged for pairwise fallback).")
            break
    deduped = sorted(set(ids))
    if len(deduped) != len(ids):
        print(f"  NOTE: {bai} had {len(ids) - len(deduped)} duplicate records "
              f"even under stable sort; deduplicated.")
    return deduped


def main():
    with open(ELIGIBLE_PATH) as f:
        eligible = json.load(f)

    try:
        with open(OUTPUT_PATH) as f:
            results = json.load(f)
    except FileNotFoundError:
        results = {}

    total = len(eligible)
    done_before = len(results)
    print(f"{done_before}/{total} already fetched; resuming.", flush=True)

    capped = []
    for i, person in enumerate(eligible):
        bai = person["bai"]
        if bai in results:
            if len(results[bai]) >= MAX_RESULT_WINDOW:
                capped.append(bai)
            continue
        ids = get_citing_control_numbers(bai)
        results[bai] = ids
        if len(ids) >= MAX_RESULT_WINDOW:
            capped.append(bai)
        if (i + 1) % 20 == 0 or i == total - 1:
            with open(OUTPUT_PATH, "w") as f:
                json.dump(results, f)
            print(f"[{i+1}/{total}] {person['name']:30s} {len(ids):5d} citing papers "
                  f"(checkpoint saved, {len(results)} total done)", flush=True)

    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f)
    with open(os.path.join(HERE, "capped_bais_v2.json"), "w") as f:
        json.dump(capped, f)
    print(f"DONE: {len(results)}/{total}, {len(capped)} capped (need pairwise fallback)", flush=True)


if __name__ == "__main__":
    main()
