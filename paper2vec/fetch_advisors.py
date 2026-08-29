#!/usr/bin/python3

################################################################################
# For each eligible researcher, fetches their PhD advisor (name + advisor's
# own INSPIRE author record id) and PhD graduation year (end_date of the
# PHD-rank entry in their own positions list), both from the INSPIRE
# AUTHORS API (not the literature API used elsewhere in this pipeline).
#
# The advisor sub-object on a student's record carries an INSPIRE ID, not a
# BAI, so a second pass resolves each distinct advisor's author-record id to
# their own BAI (fetch_advisor_bais) -- that BAI is what the rest of the
# pipeline (paper fetch, embeddings, trajectory) is keyed on.
################################################################################

import json
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

API = "https://inspirehep.net/api/authors"
HERE = os.path.dirname(os.path.abspath(__file__))
ELIGIBLE_PATH = os.path.join(HERE, "eligible_researchers.json")
STUDENT_OUT = os.path.join(HERE, "advisor_info.json")
ADVISOR_BAI_OUT = os.path.join(HERE, "advisor_bais.json")
MAX_WORKERS = 8
CHECKPOINT_EVERY = 50


def fetch_student_info(bai, retries=4):
    for attempt in range(retries):
        try:
            r = requests.get(
                API,
                params={"q": f"ids.value:{bai}", "size": 1, "fields": "advisors,positions"},
                timeout=30,
            )
            r.raise_for_status()
            hits = r.json().get("hits", {}).get("hits", [])
            break
        except Exception:
            if attempt == retries - 1:
                return bai, "ERROR"
    if not hits:
        return bai, None
    m = hits[0]["metadata"]

    phd_advisor = None
    for adv in (m.get("advisors") or []):
        if adv.get("degree_type") == "phd":
            ref = (adv.get("record") or {}).get("$ref")
            adv_control_number = None
            if ref:
                match = re.search(r"/authors/(\d+)", ref)
                if match:
                    adv_control_number = int(match.group(1))
            phd_advisor = {"name": adv.get("name"), "control_number": adv_control_number}
            break

    phd_end_year = None
    phd_start_year = None
    for pos in (m.get("positions") or []):
        if pos.get("rank") == "PHD":
            end = pos.get("end_date")
            start = pos.get("start_date")
            if end:
                phd_end_year = int(str(end)[:4])
            if start:
                phd_start_year = int(str(start)[:4])
            break

    return bai, {
        "phd_advisor": phd_advisor,
        "phd_start_year": phd_start_year,
        "phd_end_year": phd_end_year,
    }


def fetch_advisor_bai(control_number, retries=4):
    for attempt in range(retries):
        try:
            r = requests.get(f"{API}/{control_number}", params={"fields": "ids"}, timeout=30)
            r.raise_for_status()
            data = r.json()
            break
        except Exception:
            if attempt == retries - 1:
                return control_number, "ERROR"
    for ident in (data.get("metadata", {}).get("ids") or []):
        if ident.get("schema") == "INSPIRE BAI":
            return control_number, ident["value"]
    return control_number, None


def run_threaded(items, worker_fn, out_path, label):
    try:
        with open(out_path) as f:
            results = json.load(f)
    except FileNotFoundError:
        results = {}

    needed = [x for x in items if str(x) not in results]
    print(f"{label}: {len(needed)} of {len(items)} still needed", flush=True)

    lock = threading.Lock()
    done = [0]

    def checkpoint(force=False):
        with lock:
            if force or done[0] % CHECKPOINT_EVERY == 0:
                with open(out_path, "w") as f:
                    json.dump(results, f)
                print(f"  [{label}] [{done[0]}/{len(needed)}] checkpoint saved", flush=True)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {ex.submit(worker_fn, x): x for x in needed}
        for fut in as_completed(futs):
            key, value = fut.result()
            with lock:
                results[str(key)] = value
                done[0] += 1
            checkpoint()

    checkpoint(force=True)
    return results


def main():
    with open(ELIGIBLE_PATH) as f:
        eligible = json.load(f)
    all_bais = [p["bai"] for p in eligible]

    student_info = run_threaded(all_bais, fetch_student_info, STUDENT_OUT, "students")

    advisor_control_numbers = sorted({
        v["phd_advisor"]["control_number"]
        for v in student_info.values()
        if isinstance(v, dict) and v.get("phd_advisor") and v["phd_advisor"].get("control_number")
    })
    print(f"\n{len(advisor_control_numbers)} distinct PhD advisors to resolve to BAIs", flush=True)

    advisor_bais = run_threaded(advisor_control_numbers, fetch_advisor_bai, ADVISOR_BAI_OUT, "advisor_bais")

    n_with_advisor = sum(1 for v in student_info.values() if isinstance(v, dict) and v.get("phd_advisor"))
    n_with_year = sum(1 for v in student_info.values() if isinstance(v, dict) and v.get("phd_end_year"))
    n_resolved = sum(1 for v in advisor_bais.values() if v and v != "ERROR")
    print(f"\n{n_with_advisor}/{len(all_bais)} students have a listed PhD advisor", flush=True)
    print(f"{n_with_year}/{len(all_bais)} students have a PhD end year", flush=True)
    print(f"{n_resolved}/{len(advisor_control_numbers)} advisor control numbers resolved to a BAI", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
