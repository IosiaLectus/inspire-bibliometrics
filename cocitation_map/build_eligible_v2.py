#!/usr/bin/python3

################################################################################
# Builds the v2 eligible-author list under the three-part inclusion rule
# worked out via alt_seed_funnel.py-style diagnostics:
#
#   narrow set (5 papers): Harlow-Hayden, Hayden-Preskill, BRSSZ, AMPS,
#                           Susskind's "Entanglement is not enough"
#   broad set (narrow + 2): + Ryu-Takayanagi + HRT
#
#   INCLUDE iff:  narrow_count >= 2
#            AND  broad_count  >= 3
#            AND  distinct_broad_papers_cited >= 2
#
# No first-paper-year window is applied -- this is a deliberate, non-ad-hoc
# rule; whoever passes, passes. (See conversation record for why: the
# distinct-broad-papers condition kills single-paper mega-collaboration
# credit farming (e.g. LIGO/Virgo authors picking up incidental citations to
# AMPS or RT+HRT), and the narrow>=2 condition independently kills the
# RT+HRT-boilerplate-pair pathway, since RT/HRT are excluded from the narrow
# count.)
################################################################################

import json
import os
import sys
import time

import requests

API = "https://inspirehep.net/api/literature"
HERE = os.path.dirname(os.path.abspath(__file__))

NARROW_RECIDS = {
    1215350: "harlow_hayden",
    759404: "hayden_preskill",
    1409901: "brssz",
    1122534: "amps",
    1326012: "susskind",
}
BROAD_EXTRA_RECIDS = {
    711505: "rt",
    749637: "hrt",
}

NARROW_THRESHOLD = 2
BROAD_THRESHOLD = 3
DISTINCT_BROAD_THRESHOLD = 2


def fetch_citer_bais(recid, retries=4, page_size=250):
    from collections import Counter
    counts = Counter()
    page = 1
    while True:
        for attempt in range(retries):
            try:
                r = requests.get(
                    API,
                    params={
                        "q": f"refersto:recid:{recid}",
                        "size": page_size,
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
        if len(hits) < page_size:
            break
        page += 1
    return counts


def get_first_year(bai, retries=4):
    for attempt in range(retries):
        try:
            r = requests.get(API, params={"q": f"author:{bai}", "size": 1,
                                            "sort": "leastrecent", "fields": "earliest_date"}, timeout=30)
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
    return int(date[:4]) if date else None


def get_name(bai, retries=4):
    for attempt in range(retries):
        try:
            r = requests.get(API, params={"q": f"author:{bai}", "size": 1,
                                            "fields": "authors.full_name,authors.ids"}, timeout=30)
            r.raise_for_status()
            data = r.json()
            break
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(1.5)
    hits = data["hits"]["hits"]
    if not hits:
        return bai
    for a in hits[0]["metadata"].get("authors", []):
        for ident in a.get("ids", []):
            if ident.get("schema") == "INSPIRE BAI" and ident["value"] == bai:
                return a.get("full_name", bai)
    return bai


def main():
    cache_path = os.path.join(HERE, "seed_paper_counts_v2.json")
    try:
        with open(cache_path) as f:
            all_counts = json.load(f)
        print("using cached per-paper seed counts", flush=True)
    except FileNotFoundError:
        all_counts = {}
        for recid, label in {**NARROW_RECIDS, **BROAD_EXTRA_RECIDS}.items():
            print(f"fetching {label} (recid {recid})...", flush=True)
            counts = fetch_citer_bais(recid)
            all_counts[label] = dict(counts)
            print(f"  {len(counts)} distinct citing BAIs", flush=True)
        with open(cache_path, "w") as f:
            json.dump(all_counts, f)

    narrow_labels = list(NARROW_RECIDS.values())
    broad_labels = narrow_labels + list(BROAD_EXTRA_RECIDS.values())

    all_bais = set()
    for label in broad_labels:
        all_bais.update(all_counts[label].keys())
    print(f"total distinct BAIs across broad set: {len(all_bais)}", flush=True)

    def narrow_count(b):
        return sum(all_counts[l].get(b, 0) for l in narrow_labels)

    def broad_count(b):
        return sum(all_counts[l].get(b, 0) for l in broad_labels)

    def distinct_broad(b):
        return sum(1 for l in broad_labels if all_counts[l].get(b, 0) > 0)

    qualifying = sorted(
        b for b in all_bais
        if narrow_count(b) >= NARROW_THRESHOLD
        and broad_count(b) >= BROAD_THRESHOLD
        and distinct_broad(b) >= DISTINCT_BROAD_THRESHOLD
    )
    print(f"qualifying under three-part rule: {len(qualifying)}", flush=True)

    # Aaronson check -- report only, no exclusion.
    aaronson_bais = [b for b in all_bais if "aaronson" in b.lower()]
    for b in aaronson_bais:
        print(f"Aaronson candidate BAI found: {b}  narrow={narrow_count(b)} broad={broad_count(b)} "
              f"distinct_broad={distinct_broad(b)}  QUALIFIES={b in qualifying}", flush=True)
    if not aaronson_bais:
        print("No BAI matching 'aaronson' found in the broad citer set at all.", flush=True)

    # reuse cached names/first_years where available
    known_names = {}
    known_years = {}
    for path in ["/tmp/claude-0/-home-user-inspire-bibliometrics/6e1eca43-0b4b-59ec-815e-2276d63ad47b/scratchpad/candidate_data.json",
                 os.path.join(HERE, "new_candidate_first_years.json")]:
        try:
            with open(path) as f:
                d = json.load(f)
            for b, v in d.items():
                if isinstance(v, dict):
                    known_names.setdefault(b, v.get("name"))
                    known_years.setdefault(b, v.get("first_year"))
                else:
                    known_years.setdefault(b, v)
        except FileNotFoundError:
            pass
    try:
        with open(os.path.join(HERE, "eligible_authors.json")) as f:
            for p in json.load(f):
                known_names.setdefault(p["bai"], p["name"])
                known_years.setdefault(p["bai"], p["first_year"])
    except FileNotFoundError:
        pass

    out_path = os.path.join(HERE, "eligible_authors_v2.json")
    try:
        with open(out_path) as f:
            existing = {p["bai"]: p for p in json.load(f)}
    except FileNotFoundError:
        existing = {}

    result = []
    total = len(qualifying)
    for i, b in enumerate(qualifying):
        if b in existing and existing[b].get("first_year") is not None:
            result.append(existing[b])
            continue
        name = known_names.get(b) or get_name(b)
        fy = known_years.get(b)
        if fy is None:
            fy = get_first_year(b)
        result.append({"bai": b, "name": name, "first_year": fy})
        if (i + 1) % 50 == 0 or i == total - 1:
            with open(out_path, "w") as f:
                json.dump(result, f)
            print(f"[{i+1}/{total}] checkpoint saved", flush=True)

    with open(out_path, "w") as f:
        json.dump(result, f)
    print(f"DONE: {len(result)} people written to eligible_authors_v2.json", flush=True)


if __name__ == "__main__":
    main()
