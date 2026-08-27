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
# (possibly truncated) citing-paper lists.
#
# Parallelized with a thread pool -- these are independent, stateless GET
# requests, and a serial run over ~171k pairs was projected at 30+ hours.
# Resumable/checkpointed (thread-safe) given the large pair count.
################################################################################

import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

API = "https://inspirehep.net/api/literature"
HERE = os.path.dirname(os.path.abspath(__file__))
ELIGIBLE_PATH = os.path.join(HERE, "eligible_authors_v2.json")
CAPPED_PATH = os.path.join(HERE, "capped_bais_v2.json")
OUTPUT_PATH = os.path.join(HERE, "pairwise_fallback_counts.json")

MAX_WORKERS = 24
CHECKPOINT_EVERY = 500


def exact_cocitation_count(key, retries=4):
    bai_a, bai_b = key.split("|")
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
            return key, r.json()["hits"]["total"]
        except Exception:
            if attempt == retries - 1:
                return key, None
            time.sleep(1.5)


def main():
    with open(ELIGIBLE_PATH) as f:
        eligible = json.load(f)
    all_bais = [p["bai"] for p in eligible]

    with open(CAPPED_PATH) as f:
        capped = json.load(f)

    try:
        with open(OUTPUT_PATH) as f:
            results = json.load(f)
    except FileNotFoundError:
        results = {}

    needed_keys = set()
    for a in capped:
        for b in all_bais:
            if a == b:
                continue
            key = "|".join(sorted([a, b]))
            if key not in results:
                needed_keys.add(key)

    print(f"{len(capped)} capped people; {len(needed_keys)} unique pairs still needed "
          f"(of {len(capped)*(len(all_bais)-1)} raw capped x N, deduped for capped-capped overlap)", flush=True)

    lock = threading.Lock()
    done_count = [0]

    def checkpoint_if_due():
        with lock:
            if done_count[0] % CHECKPOINT_EVERY == 0:
                with open(OUTPUT_PATH, "w") as f:
                    json.dump(results, f)
                print(f"[{done_count[0]}/{len(needed_keys)}] checkpoint saved, "
                      f"{len(results)} pairs total cached", flush=True)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(exact_cocitation_count, key): key for key in needed_keys}
        for future in as_completed(futures):
            key, count = future.result()
            with lock:
                results[key] = count
                done_count[0] += 1
            checkpoint_if_due()

    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f)
    failed = sum(1 for v in results.values() if v is None)
    print(f"DONE: {len(results)} pairs cached ({failed} failed after retries)", flush=True)


if __name__ == "__main__":
    main()
