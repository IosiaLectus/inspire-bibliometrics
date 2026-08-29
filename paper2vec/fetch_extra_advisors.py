#!/usr/bin/python3

################################################################################
# Extends papers_by_researcher.json with papers for PhD advisors who fall
# outside the original 1348-researcher eligible population but who are a
# listed advisor for someone inside it (per advisor_info.json /
# advisor_bais.json from fetch_advisors.py). Without this, ~half of the
# usable student-advisor pairs for Q1 would have no advisor trajectory to
# compare against.
#
# Reuses fetch_papers.get_papers() and just merges the new BAIs' entries
# into the same papers_by_researcher.json store other scripts already read,
# so compute_embeddings.py / compute_trajectories.py pick them up with no
# changes.
################################################################################

import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import fetch_papers

HERE = os.path.dirname(os.path.abspath(__file__))
ELIGIBLE_PATH = os.path.join(HERE, "eligible_researchers.json")
STUDENT_INFO_PATH = os.path.join(HERE, "advisor_info.json")
ADVISOR_BAI_PATH = os.path.join(HERE, "advisor_bais.json")
PAPERS_PATH = os.path.join(HERE, "papers_by_researcher.json")

MAX_WORKERS = 8
CHECKPOINT_EVERY = 10


def main():
    with open(ELIGIBLE_PATH) as f:
        eligible_bais = {p["bai"] for p in json.load(f)}
    with open(STUDENT_INFO_PATH) as f:
        student_info = json.load(f)
    with open(ADVISOR_BAI_PATH) as f:
        advisor_bais_map = json.load(f)

    advisor_bais = set()
    for info in student_info.values():
        if not isinstance(info, dict):
            continue
        adv = info.get("phd_advisor")
        if not adv or adv.get("control_number") is None:
            continue
        adv_bai = advisor_bais_map.get(str(adv["control_number"]))
        if adv_bai and adv_bai != "ERROR":
            advisor_bais.add(adv_bai)

    extra_bais = sorted(advisor_bais - eligible_bais)
    print(f"{len(extra_bais)} advisor BAIs outside the eligible population need fetching", flush=True)

    with open(PAPERS_PATH) as f:
        results = json.load(f)

    needed = [b for b in extra_bais if b not in results]
    print(f"{len(needed)} of {len(extra_bais)} still needed", flush=True)

    lock = threading.Lock()
    done = [0]

    def checkpoint(force=False):
        with lock:
            if force or done[0] % CHECKPOINT_EVERY == 0:
                with open(PAPERS_PATH, "w") as f:
                    json.dump(results, f)
                print(f"[{done[0]}/{len(needed)}] checkpoint saved, {len(results)} total in store", flush=True)

    def worker(bai):
        try:
            return bai, fetch_papers.get_papers(bai), None
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
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
