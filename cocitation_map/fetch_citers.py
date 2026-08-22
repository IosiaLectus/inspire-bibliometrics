#!/usr/bin/python3

################################################################################
# Fetches, for every person in eligible_authors.json, the full list of papers
# (by control_number) that cite them on INSPIRE (refersto:author:<BAI>).
#
# This is the "linear" data-collection method: O(N) requests dominated by
# pagination cost, rather than O(N^2) pairwise co-citation-count queries.
# Resumable -- safe to interrupt and rerun; already-fetched people are skipped.
################################################################################

import json
import os
import time

import requests

API = "https://inspirehep.net/api/literature"
HERE = os.path.dirname(os.path.abspath(__file__))
ELIGIBLE_PATH = os.path.join(HERE, "eligible_authors.json")
OUTPUT_PATH = os.path.join(HERE, "citing_papers.json")

PAGE_SIZE = 250
MAX_RESULT_WINDOW = 10000  # INSPIRE refuses to page past this for a single query


def get_citing_control_numbers(bai, retries=4):
    """Return the de-duplicated list of control_numbers of all papers citing `bai`.

    INSPIRE's default (relevance) sort has no stable tiebreaker, so paging
    through it reshuffles results between requests and the same record can
    reappear on multiple pages (observed ~30-40% duplication rate in testing).
    Pinning sort=mostrecent gives a stable, non-overlapping page sequence.
    """
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
                  f"pagination cap; truncating to the {MAX_RESULT_WINDOW} most "
                  f"recent citing papers.")
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

    for i, person in enumerate(eligible):
        bai = person["bai"]
        if bai in results:
            continue
        ids = get_citing_control_numbers(bai)
        results[bai] = ids
        if (i + 1) % 20 == 0 or i == total - 1:
            with open(OUTPUT_PATH, "w") as f:
                json.dump(results, f)
            print(f"[{i+1}/{total}] {person['name']:30s} {len(ids):5d} citing papers "
                  f"(checkpoint saved, {len(results)} total done)", flush=True)

    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f)
    print(f"DONE: {len(results)}/{total}", flush=True)


if __name__ == "__main__":
    main()
