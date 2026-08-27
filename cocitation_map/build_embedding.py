#!/usr/bin/python3

################################################################################
# Builds a word2vec-style co-citation embedding from citing_papers.json.
#
# Method: for each pair of people (i, j), count papers that cite both (derived
# from the per-person citing-paper lists via a reverse index, not pairwise
# queries). Convert to a positive-PMI matrix, then take the full SVD. Rather
# than picking an embedding dimension by eye, scan cumulative explained
# variance and pick the smallest k that reaches a target (default 98%).
################################################################################

import json
import os
import math
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ELIGIBLE_PATH = os.path.join(HERE, "eligible_authors.json")
CITERS_PATH = os.path.join(HERE, "citing_papers.json")

VARIANCE_TARGET = 0.98


def build_cooccurrence(eligible, citing_papers):
    bais = [p["bai"] for p in eligible]
    n = len(bais)
    idx = {b: i for i, b in enumerate(bais)}

    # defensively de-duplicate each person's citing-paper list (fetch_citers.py
    # already does this, but a stale cache file might predate that fix)
    deduped = {b: set(citing_papers[b]) for b in bais}
    singles = np.array([len(deduped[b]) for b in bais], dtype=float)

    # reverse index: citing paper -> set of our people that it cites
    paper_to_people = defaultdict(set)
    for b in bais:
        i = idx[b]
        for cn in deduped[b]:
            paper_to_people[cn].add(i)

    count = np.zeros((n, n))
    for people in paper_to_people.values():
        if len(people) < 2:
            continue
        people = sorted(people)
        for a in range(len(people)):
            for c in range(a + 1, len(people)):
                i, j = people[a], people[c]
                count[i, j] += 1
                count[j, i] += 1

    return bais, singles, count


def build_ppmi(count, singles):
    n = count.shape[0]
    D = singles.sum()
    ppmi = np.zeros((n, n))
    nz = np.nonzero(count)
    for i, j in zip(*nz):
        if i == j:
            continue
        pmi = math.log((count[i, j] * D) / (singles[i] * singles[j]))
        ppmi[i, j] = max(0.0, pmi)
    return ppmi


def scan_svd_dimension(ppmi, target=VARIANCE_TARGET):
    U, S, Vt = np.linalg.svd(ppmi, full_matrices=False)
    cum = np.cumsum(S ** 2) / np.sum(S ** 2)
    k = int(np.searchsorted(cum, target) + 1)
    return U, S, Vt, cum, k


def main():
    with open(ELIGIBLE_PATH) as f:
        eligible = json.load(f)
    with open(CITERS_PATH) as f:
        citing_papers = json.load(f)

    missing = [p["bai"] for p in eligible if p["bai"] not in citing_papers]
    if missing:
        raise SystemExit(f"{len(missing)} people missing from {CITERS_PATH}; "
                          f"run fetch_citers.py to completion first.")

    print(f"Building co-occurrence matrix for {len(eligible)} people...", flush=True)
    bais, singles, count = build_cooccurrence(eligible, citing_papers)

    total_pairs = int((count > 0).sum() / 2)
    print(f"non-zero co-citation pairs: {total_pairs} / {len(bais) * (len(bais)-1)//2}", flush=True)

    print("Computing PPMI matrix...", flush=True)
    ppmi = build_ppmi(count, singles)

    print("Running full SVD and scanning for variance target...", flush=True)
    U, S, Vt, cum, k = scan_svd_dimension(ppmi, VARIANCE_TARGET)
    print(f"Singular value spectrum (first 15): {np.round(S[:15], 2)}", flush=True)
    checkpoints = [c for c in [10, 20, 30, 45, 60] if c <= len(cum)]
    summary = " / ".join(f"{cum[c-1]*100:.1f}%" for c in checkpoints)
    print(f"Cumulative variance at k={checkpoints}: {summary}", flush=True)
    print(f"==> smallest k reaching {VARIANCE_TARGET*100:.0f}% variance: k={k} "
          f"(captures {cum[k-1]*100:.2f}%)", flush=True)

    embedding = U[:, :k] * np.sqrt(S[:k])

    np.save(os.path.join(HERE, "cooccurrence_count.npy"), count)
    np.save(os.path.join(HERE, "ppmi.npy"), ppmi)
    np.save(os.path.join(HERE, "singular_values.npy"), S)
    np.save(os.path.join(HERE, "embedding.npy"), embedding)
    with open(os.path.join(HERE, "embedding_bais.json"), "w") as f:
        json.dump(bais, f)
    with open(os.path.join(HERE, "svd_scan.json"), "w") as f:
        json.dump({
            "variance_target": VARIANCE_TARGET,
            "chosen_k": k,
            "cumulative_variance": cum.tolist(),
        }, f)

    print("Saved: cooccurrence_count.npy, ppmi.npy, singular_values.npy, "
          "embedding.npy, embedding_bais.json, svd_scan.json", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
