#!/usr/bin/python3

################################################################################
# Q3 standard-metrics baseline: fetches citation_count per paper for every
# eligible researcher, reusing the same author:{bai} query pattern as
# fetch_papers.py (just one extra field, no extra request cost). Used to
# build citation-count/h-index-style features at PhD graduation, to compare
# against the paper-embedding position as a predictor of faculty outcomes.
################################################################################

import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

API = "https://inspirehep.net/api/literature"
HERE = os.path.dirname(os.path.abspath(__file__))
ELIGIBLE_PATH = os.path.join(HERE, "eligible_researchers.json")
OUTPUT_PATH = os.path.join(HERE, "citations_by_researcher.json")

PAGE_SIZE = 250
MAX_RESULT_WINDOW = 10000
MAX_WORKERS = 8
CHECKPOINT_EVERY = 50
FIELDS = "control_number,earliest_date,citation_count"


def get_citation_counts(bai, retries=4):
    papers = []
    page = 1
    while True:
        for attempt in range(retries):
            try:
                r = requests.get(
                    API,
                    params={
                        "q": f"author:{bai}",
                        "size": PAGE_SIZE,
                        "page": page,
                        "sort": "mostrecent",
                        "fields": FIELDS,
                    },
                    timeout=30,
                )
                r.raise_for_status()
                data = r.json()
                break
            except Exception:
                if attempt == retries - 1:
                    raise
        hits = data["hits"]["hits"]
        if not hits:
            break
        for h in hits:
            m = h["metadata"]
            papers.append({
                "control_number": m["control_number"],
                "earliest_date": m.get("earliest_date"),
                "citation_count": m.get("citation_count", 0),
            })
        if len(hits) < PAGE_SIZE:
            break
        page += 1
        if page * PAGE_SIZE > MAX_RESULT_WINDOW:
            break
    return papers


def main():
    with open(ELIGIBLE_PATH) as f:
        all_bais = [p["bai"] for p in json.load(f)]

    try:
        with open(OUTPUT_PATH) as f:
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
                with open(OUTPUT_PATH, "w") as f:
                    json.dump(results, f)
                print(f"[{done[0]}/{len(needed)}] checkpoint saved", flush=True)

    def worker(bai):
        try:
            return bai, get_citation_counts(bai), None
        except Exception as e:
            return bai, None, str(e)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {ex.submit(worker, b): b for b in needed}
        for fut in as_completed(futs):
            bai, papers, err = fut.result()
            with lock:
                if err is not None:
                    print(f"  ERROR fetching {bai}: {err}", flush=True)
                    results[bai] = "ERROR"
                else:
                    results[bai] = papers
                done[0] += 1
            checkpoint()

    checkpoint(force=True)
    n_err = sum(1 for v in results.values() if v == "ERROR")
    print(f"DONE: {len(results)}/{len(all_bais)} ({n_err} errors)", flush=True)


if __name__ == "__main__":
    main()
