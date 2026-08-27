#!/usr/bin/python3

################################################################################
# For each researcher in eligible_researchers.json, fetch their authored
# papers' title, abstract, earliest release date, and total author count
# (the latter for coauthor-count discounting when the papers are later
# aggregated into a time-weighted per-researcher trajectory).
#
# Threaded + checkpointed like fetch_positions.py in the sibling
# cocitation_map/ pipeline. Output is keyed by BAI so it's resumable and so
# a paper coauthored by two eligible researchers appears under both.
################################################################################

import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

API = "https://inspirehep.net/api/literature"
HERE = os.path.dirname(os.path.abspath(__file__))
ELIGIBLE_PATH = os.path.join(HERE, "eligible_researchers.json")
OUTPUT_PATH = os.path.join(HERE, "papers_by_researcher.json")

PAGE_SIZE = 250
MAX_RESULT_WINDOW = 10000
MAX_WORKERS = 8
CHECKPOINT_EVERY = 20
FIELDS = "control_number,titles,abstracts,earliest_date,author_count"


def get_papers(bai, retries=4):
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
            titles = m.get("titles") or []
            abstracts = m.get("abstracts") or []
            papers.append({
                "control_number": m["control_number"],
                "title": titles[0]["title"] if titles else None,
                "abstract": abstracts[0]["value"] if abstracts else None,
                "earliest_date": m.get("earliest_date"),
                "author_count": m.get("author_count"),
            })
        if len(hits) < PAGE_SIZE:
            break
        page += 1
        if page * PAGE_SIZE > MAX_RESULT_WINDOW:
            print(f"  WARNING: {bai} exceeds {MAX_RESULT_WINDOW}-result pagination cap "
                  f"(unusual; truncating).", flush=True)
            break
    seen = set()
    deduped = []
    for p in papers:
        if p["control_number"] not in seen:
            deduped.append(p)
            seen.add(p["control_number"])
    return deduped


def main():
    with open(ELIGIBLE_PATH) as f:
        eligible = json.load(f)
    all_bais = [p["bai"] for p in eligible]

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
                print(f"[{done[0]}/{len(needed)}] checkpoint saved, {len(results)} total done", flush=True)

    def worker(bai):
        try:
            return bai, get_papers(bai), None
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
    n_papers = sum(len(v) for v in results.values() if isinstance(v, list))
    print(f"DONE: {len(results)}/{len(all_bais)} researchers ({n_err} errors), "
          f"{n_papers} researcher-paper edges", flush=True)


if __name__ == "__main__":
    main()
