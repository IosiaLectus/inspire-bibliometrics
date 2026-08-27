#!/usr/bin/python3

################################################################################
# v2 of fetch_own_papers.py: fetches each person's own authored papers, PLUS
# (same query, same page count, just two more fields -- no extra API cost)
# each paper's arXiv categories and this specific author's affiliation(s) on
# that paper. Feeds three things:
#   - own-paper control numbers -> coauthorship / directed-citation relations
#     (as before)
#   - arXiv categories -> validation-only layer (not embedded), primary
#     category weight 1.0, cross-listed weight 0.5
#   - per-paper affiliations -> institution bigraph relation (paper-count
#     weighted, keyed by INSPIRE's canonical institution record id so no
#     string-matching entity resolution is needed)
################################################################################

import json
import os
import time

import requests

API = "https://inspirehep.net/api/literature"
HERE = os.path.dirname(os.path.abspath(__file__))
ELIGIBLE_PATH = os.path.join(HERE, "eligible_authors_v2.json")
OUTPUT_PATH = os.path.join(HERE, "own_papers_v2.json")

PAGE_SIZE = 250
MAX_RESULT_WINDOW = 10000


def extract_person_record(hit_metadata, bai):
    """Given one paper's metadata, return this person's affiliation institution
    ids on that paper, plus the paper's arxiv categories (primary + cross-listed)."""
    affil_ids = []
    for a in hit_metadata.get("authors", []):
        ids = a.get("ids", [])
        if any(i.get("schema") == "INSPIRE BAI" and i.get("value") == bai for i in ids):
            for aff in a.get("affiliations", []):
                ref = aff.get("record", {}).get("$ref")
                if ref:
                    affil_ids.append(ref.rstrip("/").split("/")[-1])
            break
    categories = []
    for ep in (hit_metadata.get("arxiv_eprints") or []):
        cats = ep.get("categories") or []
        for j, c in enumerate(cats):
            categories.append({"category": c, "primary": j == 0})
    return affil_ids, categories


def get_own_papers(bai, retries=4):
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
                        "fields": "control_number,arxiv_eprints,authors.ids,authors.affiliations",
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
            m = h["metadata"]
            affil_ids, categories = extract_person_record(m, bai)
            papers.append({
                "control_number": m["control_number"],
                "affiliations": affil_ids,
                "categories": categories,
            })
        if len(hits) < PAGE_SIZE:
            break
        page += 1
        if page * PAGE_SIZE > MAX_RESULT_WINDOW:
            print(f"  WARNING: {bai} exceeds {MAX_RESULT_WINDOW}-result pagination cap on own papers "
                  f"(unusual; truncating).")
            break
    # de-dup by control_number (stable sort should already prevent this, but be defensive)
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
        papers = get_own_papers(bai)
        results[bai] = papers
        if (i + 1) % 20 == 0 or i == total - 1:
            with open(OUTPUT_PATH, "w") as f:
                json.dump(results, f)
            print(f"[{i+1}/{total}] {person['name']:30s} {len(papers):5d} own papers "
                  f"(checkpoint saved, {len(results)} total done)", flush=True)

    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f)
    print(f"DONE: {len(results)}/{total}", flush=True)


if __name__ == "__main__":
    main()
