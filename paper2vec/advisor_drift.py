#!/usr/bin/python3

################################################################################
# Q1: does a researcher drift away from their PhD advisor after graduation,
# and how much?
#
# For each student with a resolvable PhD advisor (advisor_info.json /
# advisor_bais.json) and a usable graduation year, computes cosine distance
# between the STUDENT's position at graduation+k years and the ADVISOR's
# position AT THE GRADUATION YEAR (a fixed reference point -- deliberately
# not the advisor's own contemporaneous position, so this isolates the
# student's movement rather than mixing in the advisor's own drift too).
#
# Pools (years-since-graduation, distance) pairs across all matched
# student-advisor pairs to get an average drift-from-advisor curve, and
# compares it against the population self-drift curve from self_drift.py
# (position vs. one's own past) as the natural baseline: is drift-from-advisor
# faster, slower, or about the same as ordinary self-drift?
#
# CAVEAT: this is NOT an apples-to-apples comparison with self_drift.py's
# curve, and not for the reason it might first look like. The advisor's
# position is frozen, so it cannot "resist" the student moving away from it
# -- only the student's side is actually moving. What's asymmetric is
# sample size: even under the identical 5-year half-life decay, an
# advisor's graduation-year position is typically built from ~90
# effectively-weighted papers (an established PI's group keeps publishing),
# vs. ~7 for the student (a fresh PhD graduate, unsurprisingly). Averaging
# over more papers cancels more idiosyncratic per-paper variation and
# leaves a vector closer to whatever generic component the advisor's
# papers share -- and a more heavily-averaged reference point generically
# has HIGHER expected cosine similarity to any other point than a noisier,
# less-averaged one does, independent of who's moving. So part of why this
# curve sits flatter than self-drift is that self-drift compares two
# similarly-noisy points to each other, while this compares a noisy
# student point to a much-more-averaged advisor point -- not necessarily
# that departure from an advisor is smaller than departure from one's own
# past.
################################################################################

import json
import os
from collections import defaultdict

import numpy as np

import trajectory as traj

HERE = os.path.dirname(os.path.abspath(__file__))
STUDENT_INFO_PATH = os.path.join(HERE, "advisor_info.json")
ADVISOR_BAI_PATH = os.path.join(HERE, "advisor_bais.json")
OUT_PATH = os.path.join(HERE, "advisor_drift_by_lag.json")
PAIRS_OUT_PATH = os.path.join(HERE, "advisor_drift_pairs.json")

MIN_REAL_PAPERS_STUDENT = 3   # minimum distinct student papers to trust their post-PhD position
MAX_LAG = 30                  # years post-graduation to track
CURRENT_YEAR = 2026           # never query a trajectory position past today


def main():
    with open(STUDENT_INFO_PATH) as f:
        student_info = json.load(f)
    with open(ADVISOR_BAI_PATH) as f:
        advisor_bais_map = json.load(f)

    embeddings, manifest = traj.load_embeddings()
    researcher_rows = traj.load_researcher_paper_rows(embeddings, manifest)

    lag_cosines = defaultdict(list)
    pairs = []
    for student_bai, info in student_info.items():
        if not isinstance(info, dict):
            continue
        adv = info.get("phd_advisor")
        grad_year = info.get("phd_end_year")
        if not adv or not grad_year or adv.get("control_number") is None:
            continue
        advisor_bai = advisor_bais_map.get(str(adv["control_number"]))
        if not advisor_bai or advisor_bai == "ERROR" or advisor_bai == student_bai:
            continue

        student_rows = researcher_rows.get(student_bai, [])
        advisor_rows = researcher_rows.get(advisor_bai, [])
        if len(student_rows) < MIN_REAL_PAPERS_STUDENT or not advisor_rows:
            continue

        # advisor's position frozen at the graduation year (their own papers up to that year)
        advisor_at_grad = traj.researcher_trajectory(advisor_rows, embeddings, [grad_year])
        if not advisor_at_grad:
            continue
        advisor_vec = advisor_at_grad[0]["position"]

        last_year = min(grad_year + MAX_LAG, CURRENT_YEAR)
        if last_year < grad_year:
            continue
        student_points = traj.researcher_trajectory(
            student_rows, embeddings, range(grad_year, last_year + 1)
        )
        if not student_points:
            continue

        pairs.append({
            "student_bai": student_bai,
            "advisor_bai": advisor_bai,
            "advisor_name": adv.get("name"),
            "grad_year": grad_year,
            "n_student_years": len(student_points),
        })

        for p in student_points:
            lag = int(round(p["year"] - grad_year))
            cos = float(np.dot(p["position"], advisor_vec))
            lag_cosines[lag].append(cos)

    with open(PAIRS_OUT_PATH, "w") as f:
        json.dump(pairs, f, indent=2)

    by_lag = []
    for lag in sorted(lag_cosines):
        cos_vals = np.array(lag_cosines[lag])
        by_lag.append({
            "lag": lag,
            "mean_cosine": float(cos_vals.mean()),
            "median_cosine": float(np.median(cos_vals)),
            "p10_cosine": float(np.percentile(cos_vals, 10)),
            "p90_cosine": float(np.percentile(cos_vals, 90)),
            "n_students": int(len(cos_vals)),
        })
    with open(OUT_PATH, "w") as f:
        json.dump(by_lag, f, indent=2)

    print(f"{len(pairs)} usable student-advisor pairs "
          f"({len(set(p['advisor_bai'] for p in pairs))} distinct advisors)", flush=True)
    print("\ndrift from advisor's graduation-year position, by years since graduation:", flush=True)
    for row in by_lag:
        print(f"  +{row['lag']:2d}yr  mean_cosine={row['mean_cosine']:.4f}  "
              f"median={row['median_cosine']:.4f}  n={row['n_students']}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
