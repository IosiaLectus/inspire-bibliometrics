#!/usr/bin/python3

################################################################################
# O(N) x capped-count pairwise fallback for people whose citing-paper list hit
# INSPIRE's 10,000-result pagination cap in fetch_citers_v2.py (their
# citing_papers_v2.json entry is a truncated subsample, not the true list, so
# any co-citation count derived from intersecting it is biased -- measured at
# 5-15% undercount in the earlier study, worse when both endpoints are capped).
#
# For each capped person X, queries the EXACT co-citation count against every
# other eligible person Y directly (refersto:author:X and refersto:author:Y,
# size=1, just hits.total) rather than relying on set intersection of
# (possibly truncated) citing-paper lists. Resumable/checkpointed given the
# potentially large number of pairs.
################################################################################

import json
import os
import time

import requests

API = "https://inspirehep.net/api/literature"
HERE = os.path.dirname(os.path.abspath(__file__))
ELIGIBLE_PATH = os.path.join(HERE, "eligible_authors_v2.json")
CAPPED_PATH = os.path.join(HERE, "capped_bais_v2.json")
OUTPUT_PATH = os.path.join(HERE, "pairwise_fallback_counts.json")


def exact_cocitation_count(bai_a, bai_b, retries=4):
    for attempt in range(retries):
        try:
            r = requests.get(
                API,
                params={
                    "q": f"refersto:author:{bai_a} and refersto:author:{bai_b}",
                    "size": 1,
                    "fields": "control_number",
                },
                timeout=20,
            )
            r.raise_for_status()
            return r.json()["hits"]["total"]
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(1.5)


def main():
    with open(ELIGIBLE_PATH) as f:
        eligible = json.load(f)
    all_bais = [p["bai"] for p in eligible]

    with open(CAPPED_PATH) as f:
        capped = json.load(f)
    print(f"{len(capped)} capped people; computing exact pairwise counts against "
          f"{len(all_bais)-1} others each ({len(capped)*(len(all_bais)-1)} pairs total)", flush=True)

    try:
        with open(OUTPUT_PATH) as f:
            results = json.load(f)
    except FileNotFoundError:
        results = {}

    total_pairs_needed = 0
    for a in capped:
        for b in all_bais:
            if a == b:
                continue
            key = "|".join(sorted([a, b]))
            if key not in results:
                total_pairs_needed += 1

    done = 0
    checkpoint_every = 200
    for a in capped:
        for b in all_bais:
            if a == b:
                continue
            key = "|".join(sorted([a, b]))
            if key in results:
                continue
            results[key] = exact_cocitation_count(a, b)
            done += 1
            if done % checkpoint_every == 0:
                with open(OUTPUT_PATH, "w") as f:
                    json.dump(results, f)
                print(f"[{done}] checkpoint saved, {len(results)} pairs total cached", flush=True)

    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f)
    print(f"DONE: {len(results)} pairs cached", flush=True)


if __name__ == "__main__":
    main()
