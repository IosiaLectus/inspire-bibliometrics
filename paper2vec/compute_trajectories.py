#!/usr/bin/python3

################################################################################
# Materializes an annual-grid researcher trajectory for every eligible
# researcher, using trajectory.py's default HALF_LIFE. Each grid point is one
# researcher-year: the coauthor-discounted, time-decayed, unit-normalized
# average of that researcher's papers up to (and including) that year.
#
# Grid runs from each researcher's own first paper year through the current
# year -- so a researcher's position keeps drifting (toward whatever recent
# work exists, decayed) even in years they didn't publish, which is exactly
# what a "did they go quiet / drift away" analysis needs.
################################################################################

import json
import os

import numpy as np

import trajectory as traj

HERE = os.path.dirname(os.path.abspath(__file__))
ELIGIBLE_PATH = os.path.join(HERE, "eligible_researchers.json")
OUT_EMBEDDINGS_PATH = os.path.join(HERE, "trajectory_embeddings.npy")
OUT_MANIFEST_PATH = os.path.join(HERE, "trajectory_manifest.json")

CURRENT_YEAR = 2026


def main():
    with open(ELIGIBLE_PATH) as f:
        eligible = json.load(f)
    names = {p["bai"]: p["name"] for p in eligible}

    print("Loading paper embeddings...", flush=True)
    embeddings, manifest = traj.load_embeddings()
    print(f"  {embeddings.shape[0]} unique papers", flush=True)

    print("Joining researcher paper lists to embedding rows...", flush=True)
    researcher_rows = traj.load_researcher_paper_rows(embeddings, manifest)

    all_positions = []
    out_manifest = []
    bais = sorted(researcher_rows.keys())
    for i, bai in enumerate(bais):
        rows = researcher_rows[bai]
        if not rows:
            continue
        start_year = int(min(r[1] for r in rows))
        years = range(start_year, CURRENT_YEAR + 1)
        points = traj.researcher_trajectory(rows, embeddings, years)
        for p in points:
            all_positions.append(p["position"])
            out_manifest.append({
                "bai": bai,
                "name": names.get(bai, bai),
                "year": p["year"],
                "n_papers": p["n_papers"],
                "effective_weight": p["effective_weight"],
            })
        if (i + 1) % 100 == 0 or i == len(bais) - 1:
            print(f"[{i+1}/{len(bais)}] researchers processed, "
                  f"{len(all_positions)} researcher-year points so far", flush=True)

    positions = np.stack(all_positions).astype(np.float32)
    np.save(OUT_EMBEDDINGS_PATH, positions)
    with open(OUT_MANIFEST_PATH, "w") as f:
        json.dump(out_manifest, f)

    print(f"Saved trajectory_embeddings.npy {positions.shape} and trajectory_manifest.json "
          f"({len(out_manifest)} researcher-year points across {len(bais)} researchers)", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
