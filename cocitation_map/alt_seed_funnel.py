#!/usr/bin/python3

################################################################################
# Diagnostic (not part of the production pipeline): re-runs just the
# inclusion funnel -- aggregate citations across a set of seed papers,
# threshold >=2 -- under an alternative seed-paper set, to compare against
# the existing eligible_authors.json population before committing to a
# rewrite. Does NOT apply the first-paper-year filter for people outside the
# existing eligible list (that would need one more fetch per new candidate;
# this script reports how many such candidates there are so that cost is
# visible before deciding whether to pay it).
################################################################################

import json
import os
import sys
import time
from collections import Counter

import requests

API = "https://inspirehep.net/api/literature"
HERE = os.path.dirname(os.path.abspath(__file__))
PAGE_SIZE = 250


def fetch_citer_bais(recid, retries=4):
    """Return a Counter of BAI -> number of distinct citing papers where that
    BAI appears as an author, for papers citing `recid`."""
    counts = Counter()
    page = 1
    while True:
        for attempt in range(retries):
            try:
                r = requests.get(
                    API,
                    params={
                        "q": f"refersto:recid:{recid}",
                        "size": PAGE_SIZE,
                        "page": page,
                        "sort": "mostrecent",
                        "fields": "control_number,authors.ids,authors.full_name",
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
        for h in hits:
            seen_this_paper = set()
            for a in h["metadata"].get("authors", []):
                bai = None
                for ident in a.get("ids", []):
                    if ident.get("schema") == "INSPIRE BAI":
                        bai = ident["value"]
                        break
                if bai and bai not in seen_this_paper:
                    counts[bai] += 1
                    seen_this_paper.add(bai)
        if len(hits) < PAGE_SIZE:
            break
        page += 1
        print(f"    recid {recid}: page {page-1} done, {len(counts)} distinct BAIs so far", flush=True)
    return counts


def main():
    seed_sets = {
        "original": {
            1215350: "Harlow-Hayden (firewalls)",
            759404: "Hayden-Preskill (BH-mirrors)",
            1409901: "BRSSZ (complexity=action)",
        },
        "alternative": {
            711505: "Ryu-Takayanagi (entanglement entropy)",
            1122534: "AMPS (firewalls)",
            1326012: "Susskind (entanglement is not enough)",
        },
    }

    results = {}
    for set_name, recids in seed_sets.items():
        print(f"\n=== fetching seed set: {set_name} ===", flush=True)
        total = Counter()
        for recid, label in recids.items():
            print(f"  {label} (recid {recid})...", flush=True)
            counts = fetch_citer_bais(recid)
            print(f"    {len(counts)} distinct citing BAIs", flush=True)
            total.update(counts)
        results[set_name] = total

    out = {name: dict(counter) for name, counter in results.items()}
    with open(os.path.join(HERE, "alt_seed_funnel_results.json"), "w") as f:
        json.dump(out, f)

    for set_name, counter in results.items():
        qualifying = {b for b, c in counter.items() if c >= 2}
        print(f"\n{set_name}: {len(counter)} total distinct citing BAIs, "
              f"{len(qualifying)} with aggregate count >= 2")

    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
