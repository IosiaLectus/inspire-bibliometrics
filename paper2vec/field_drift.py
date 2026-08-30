#!/usr/bin/python3

################################################################################
# Does the field move? Tracks the mean (unit-renormalized) SPECTER embedding
# of papers in a rolling window over calendar time -- for the whole corpus,
# and separately for each of the 6 author communities from
# cluster_communities.py (a paper counts toward every cluster any of its
# in-population authors belongs to; a paper coauthored across cluster
# boundaries legitimately counts in more than one).
#
# METHODOLOGY NOTE, learned from self_drift.py's mechanical-overlap issue:
# a rolling window stepped by 1 year with a 5-year width shares 80% of its
# papers with the previous step, so year-over-year cosine similarity would
# be misleadingly high almost by construction, not primarily reflecting
# real movement. Instead this compares every window to one FIXED reference
# window (1995, chosen for adequate paper counts per cluster) -- overlap
# with that fixed point shrinks to zero once you're more than one window
# width past it, so the resulting curve is a real drift signal, not a
# window-overlap artifact. Same idea as advisor_drift.py's frozen-reference
# comparison, applied to a field/community aggregate instead of a person.
#
# Also computes mean pairwise cosine similarity AMONG the 6 cluster
# centroids over time -- are sub-communities converging or diverging from
# each other, independent of whether any of them is moving from its own
# past.
################################################################################

import json
import os
from collections import defaultdict

import numpy as np

import trajectory as traj

HERE = os.path.dirname(os.path.abspath(__file__))
CLUSTERS_PATH = os.path.join(HERE, "clusters.json")
OUT_PATH = os.path.join(HERE, "field_drift.json")

WINDOW_HALF = 2          # +/- years -> 5-year window
START_YEAR = 1990
END_YEAR = 2025
REFERENCE_YEAR = 1995
MIN_PAPERS_PER_WINDOW = 15

CLUSTER_LABELS = {
    0: "String Theory & Gauge/Gravity Duality",
    1: "Black Hole Physics & Gravity",
    2: "Gravitational Wave Astronomy",
    3: "Quantum Information Theory",
    4: "Holographic Entanglement & Complexity",
    5: "Collider & Detector Physics",
}


def main():
    embeddings, manifest = traj.load_embeddings()
    researcher_rows = traj.load_researcher_paper_rows(embeddings, manifest)

    with open(CLUSTERS_PATH) as f:
        cluster_data = json.load(f)
    bai_to_cluster = {p["bai"]: p["cluster"] for p in cluster_data["points"]}

    # control_number -> (embedding row index, year)
    paper_year = {}
    paper_idx = {}
    for i, m in enumerate(manifest):
        y = m.get("earliest_date")
        if y:
            paper_year[m["control_number"]] = int(y[:4])
            paper_idx[m["control_number"]] = i

    # papers per series: "field" = everything; cluster c = papers touched by
    # any researcher assigned to cluster c
    series_papers = defaultdict(set)
    for bai, rows in researcher_rows.items():
        cns = {manifest[r[0]]["control_number"] for r in rows}
        series_papers["field"].update(cns)
        c = bai_to_cluster.get(bai)
        if c is not None:
            series_papers[c].update(cns)

    def window_centroid(cns, center_year):
        idxs = [paper_idx[cn] for cn in cns
                if cn in paper_year and abs(paper_year[cn] - center_year) <= WINDOW_HALF]
        if len(idxs) < MIN_PAPERS_PER_WINDOW:
            return None, len(idxs)
        vecs = embeddings[idxs].astype(np.float64)
        centroid = vecs.mean(axis=0)
        norm = np.linalg.norm(centroid)
        return (centroid / norm if norm > 0 else None), len(idxs)

    series_keys = ["field"] + sorted(k for k in series_papers if isinstance(k, int))
    series_names = {"field": "Whole field (all clusters)", **CLUSTER_LABELS}

    print(f"Series: {[series_names[k] for k in series_keys]}", flush=True)
    for k in series_keys:
        print(f"  {series_names[k]}: {len(series_papers[k])} distinct papers", flush=True)

    centroids = {k: {} for k in series_keys}
    counts = {k: {} for k in series_keys}
    for k in series_keys:
        for year in range(START_YEAR, END_YEAR + 1):
            c, n = window_centroid(series_papers[k], year)
            centroids[k][year] = c
            counts[k][year] = n

    reference = {k: centroids[k].get(REFERENCE_YEAR) for k in series_keys}
    for k in series_keys:
        if reference[k] is None:
            print(f"  WARNING: no reference centroid for {series_names[k]} at {REFERENCE_YEAR} "
                  f"(only {counts[k][REFERENCE_YEAR]} papers)", flush=True)

    vs_reference = {k: [] for k in series_keys}
    for k in series_keys:
        for year in range(START_YEAR, END_YEAR + 1):
            c = centroids[k][year]
            if c is None or reference[k] is None:
                continue
            cos = float(np.dot(c, reference[k]))
            vs_reference[k].append({"year": year, "cosine_vs_1995": cos, "n_papers": counts[k][year]})

    # inter-cluster cohesion: mean pairwise cosine among the 6 cluster centroids, per year
    cluster_keys = [k for k in series_keys if isinstance(k, int)]
    cohesion = []
    for year in range(START_YEAR, END_YEAR + 1):
        vecs = [centroids[k][year] for k in cluster_keys if centroids[k][year] is not None]
        if len(vecs) < 2:
            continue
        pair_cos = []
        for i in range(len(vecs)):
            for j in range(i + 1, len(vecs)):
                pair_cos.append(float(np.dot(vecs[i], vecs[j])))
        cohesion.append({"year": year, "mean_pairwise_cosine": float(np.mean(pair_cos)), "n_clusters": len(vecs)})

    with open(OUT_PATH, "w") as f:
        json.dump({
            "series_names": series_names,
            "reference_year": REFERENCE_YEAR,
            "window_half_width": WINDOW_HALF,
            "vs_reference": vs_reference,
            "cohesion": cohesion,
        }, f, indent=2)

    print("\nCosine similarity to own 1995 window, selected years:", flush=True)
    for k in series_keys:
        rows = vs_reference[k]
        by_year = {r["year"]: r for r in rows}
        vals = " ".join(f"{y}:{by_year[y]['cosine_vs_1995']:.3f}" for y in [1995, 2000, 2010, 2020, 2025] if y in by_year)
        print(f"  {series_names[k]:36s} {vals}", flush=True)

    print("\nInter-cluster cohesion (mean pairwise cosine among cluster centroids):", flush=True)
    for row in cohesion:
        if row["year"] in (1990, 1995, 2000, 2005, 2010, 2015, 2020, 2025):
            print(f"  {row['year']}: {row['mean_pairwise_cosine']:.4f} (n_clusters={row['n_clusters']})", flush=True)

    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
