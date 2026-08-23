#!/usr/bin/python3

################################################################################
# Combines three relation tables into one embedding per person ("Option 2"
# from the co-citation-atlas discussion):
#
#   - co-citation      (symmetric):  a third paper cites both i and j
#   - coauthorship     (symmetric):  i and j co-wrote a paper together
#   - directed citation (asymmetric): i's own paper cites j
#
# Recipe: factorize each relation separately (PPMI + SVD, same recipe as
# build_embedding.py), then concatenate each person's per-relation vectors
# into one combined embedding. The directed citation matrix is NOT
# symmetrized -- SVD of a rectangular/asymmetric matrix naturally yields two
# role vectors per person (a "citer" profile from U and a "citee" profile
# from V), exactly analogous to the target/context split in word2vec's own
# PMI matrix. Each block is scale-normalized (unit average row norm) before
# concatenation so no single relation dominates just by having larger counts.
################################################################################

import json
import os
import math
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ELIGIBLE_PATH = os.path.join(HERE, "eligible_authors.json")
CITERS_PATH = os.path.join(HERE, "citing_papers.json")
OWN_PATH = os.path.join(HERE, "own_papers.json")

VARIANCE_TARGET = 0.98


def build_symmetric_count(bais, sets_a, sets_b=None):
    """Co-occurrence-style symmetric count matrix from per-person paper sets.
    If sets_b is None, counts pairwise intersections within sets_a (co-citation
    style, via a reverse paper->people index). If sets_b is given, counts
    |sets_a[i] & sets_b[j]| directly for coauthorship (both are "own papers")."""
    n = len(bais)
    idx = {b: i for i, b in enumerate(bais)}
    count = np.zeros((n, n))

    if sets_b is None:
        paper_to_people = defaultdict(set)
        for b in bais:
            for cn in sets_a[b]:
                paper_to_people[cn].add(idx[b])
        for people in paper_to_people.values():
            if len(people) < 2:
                continue
            people = sorted(people)
            for a in range(len(people)):
                for c in range(a + 1, len(people)):
                    i, j = people[a], people[c]
                    count[i, j] += 1
                    count[j, i] += 1
    else:
        for i, bi in enumerate(bais):
            si = sets_a[bi]
            if not si:
                continue
            for j in range(i + 1, n):
                bj = bais[j]
                sj = sets_a[bj]
                if not sj:
                    continue
                c = len(si & sj)
                if c:
                    count[i, j] = c
                    count[j, i] = c

    return count


def build_directed_count(bais, own, citing):
    """directed[i, j] = |own[i] & citing[j]| = # of i's own papers that cite j."""
    n = len(bais)
    directed = np.zeros((n, n))
    for i, bi in enumerate(bais):
        oi = own[bi]
        if not oi:
            continue
        for j, bj in enumerate(bais):
            if i == j:
                continue
            cj = citing[bj]
            if not cj:
                continue
            c = len(oi & cj)
            if c:
                directed[i, j] = c
    return directed


def ppmi_symmetric(count, marg):
    """marg[i] must be each person's TRUE global frequency for this relation
    (e.g. total distinct citing papers, or total distinct own papers) --
    NOT derived from count.sum(axis=1), which would restrict the marginal to
    only the in-vocabulary submatrix and understate each person's true
    unigram frequency (the same mistake as computing P(w) in word2vec from
    only the contexts that happen to also be vocabulary words)."""
    n = count.shape[0]
    D = marg.sum()
    ppmi = np.zeros((n, n))
    nz = np.nonzero(count)
    for i, j in zip(*nz):
        if i == j or marg[i] == 0 or marg[j] == 0:
            continue
        pmi = math.log((count[i, j] * D) / (marg[i] * marg[j]))
        ppmi[i, j] = max(0.0, pmi)
    return ppmi


def ppmi_directed(count):
    """PMI for an asymmetric matrix, using row/column marginals derived from
    `count` itself. Unlike co-citation and coauthorship -- where each
    person's TRUE unrestricted frequency (total citations received / total
    papers written, from data we already have) is available and used as the
    marginal -- there's no unrestricted analogue here: we never fetched full
    reference lists, so "i's total citations made" restricted to this
    663-person cohort is the only citing-side quantity we have. Mixing that
    restricted row total with an unrestricted column total (e.g. i's true
    global citing rate) would put row and column probabilities on different
    scales and bias the result, so both sides are deliberately kept
    consistent within the same restricted, field-internal universe: this
    measures whether i cites j more than expected given i's and j's
    citing/cited propensity *within the field*, not the wider literature."""
    row_marg = count.sum(axis=1)
    col_marg = count.sum(axis=0)
    D = count.sum()
    n = count.shape[0]
    ppmi = np.zeros((n, n))
    nz = np.nonzero(count)
    for i, j in zip(*nz):
        if row_marg[i] == 0 or col_marg[j] == 0:
            continue
        pmi = math.log((count[i, j] * D) / (row_marg[i] * col_marg[j]))
        ppmi[i, j] = max(0.0, pmi)
    return ppmi


def scan_svd_dimension(M, target=VARIANCE_TARGET, symmetric=True):
    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    total = np.sum(S ** 2)
    if total == 0:
        return U, S, Vt, np.array([1.0]), 1
    cum = np.cumsum(S ** 2) / total
    k = int(np.searchsorted(cum, target) + 1)
    k = max(k, 1)
    return U, S, Vt, cum, k


def normalize_block(block):
    """Scale a block so its rows have unit average norm, so blocks with very
    different count magnitudes contribute comparably once concatenated."""
    norms = np.linalg.norm(block, axis=1)
    avg = norms[norms > 0].mean() if (norms > 0).any() else 1.0
    return block / avg if avg > 0 else block


def main():
    with open(ELIGIBLE_PATH) as f:
        eligible = json.load(f)
    with open(CITERS_PATH) as f:
        citing_papers_raw = json.load(f)
    with open(OWN_PATH) as f:
        own_papers_raw = json.load(f)

    bais = [p["bai"] for p in eligible]
    missing_citing = [b for b in bais if b not in citing_papers_raw]
    missing_own = [b for b in bais if b not in own_papers_raw]
    if missing_citing or missing_own:
        raise SystemExit(f"{len(missing_citing)} missing from citing_papers.json, "
                          f"{len(missing_own)} missing from own_papers.json; "
                          f"run fetch_citers.py / fetch_own_papers.py to completion first.")

    citing = {b: set(citing_papers_raw[b]) for b in bais}
    own = {b: set(own_papers_raw[b]) for b in bais}
    citing_singles = np.array([len(citing[b]) for b in bais], dtype=float)
    own_singles = np.array([len(own[b]) for b in bais], dtype=float)

    print(f"Building relation matrices for {len(bais)} people...", flush=True)

    print("  co-citation...", flush=True)
    cocitation_count = build_symmetric_count(bais, citing)
    print(f"    non-zero pairs: {int((cocitation_count > 0).sum() / 2)}", flush=True)

    print("  coauthorship...", flush=True)
    coauthor_count = build_symmetric_count(bais, own, sets_b=own)
    print(f"    non-zero pairs: {int((coauthor_count > 0).sum() / 2)}", flush=True)

    print("  directed citation (i's own paper cites j)...", flush=True)
    directed_count = build_directed_count(bais, own, citing)
    print(f"    non-zero directed edges: {int((directed_count > 0).sum())}", flush=True)

    blocks = {}
    dims = {}

    print("\nFactorizing co-citation...", flush=True)
    ppmi_cc = ppmi_symmetric(cocitation_count, citing_singles)
    U, S, Vt, cum, k = scan_svd_dimension(ppmi_cc)
    print(f"  k={k} for {VARIANCE_TARGET*100:.0f}% variance (captures {cum[k-1]*100:.2f}%)", flush=True)
    blocks["cocitation"] = normalize_block(U[:, :k] * np.sqrt(S[:k]))
    dims["cocitation"] = k

    print("Factorizing coauthorship...", flush=True)
    ppmi_ca = ppmi_symmetric(coauthor_count, own_singles)
    U, S, Vt, cum, k = scan_svd_dimension(ppmi_ca)
    print(f"  k={k} for {VARIANCE_TARGET*100:.0f}% variance (captures {cum[k-1]*100:.2f}%)", flush=True)
    blocks["coauthor"] = normalize_block(U[:, :k] * np.sqrt(S[:k]))
    dims["coauthor"] = k

    print("Factorizing directed citation...", flush=True)
    ppmi_dc = ppmi_directed(directed_count)
    U, S, Vt, cum, k = scan_svd_dimension(ppmi_dc, symmetric=False)
    print(f"  k={k} for {VARIANCE_TARGET*100:.0f}% variance (captures {cum[k-1]*100:.2f}%)", flush=True)
    citer = U[:, :k] * np.sqrt(S[:k])       # "who this person cites" profile
    citee = Vt[:k, :].T * np.sqrt(S[:k])    # "who cites this person" profile
    blocks["citer"] = normalize_block(citer)
    blocks["citee"] = normalize_block(citee)
    dims["citer"] = k
    dims["citee"] = k

    combined = np.concatenate([blocks["cocitation"], blocks["coauthor"],
                                blocks["citer"], blocks["citee"]], axis=1)
    print(f"\nCombined embedding shape: {combined.shape} "
          f"(cocitation={dims['cocitation']}, coauthor={dims['coauthor']}, "
          f"citer={dims['citer']}, citee={dims['citee']})", flush=True)

    np.save(os.path.join(HERE, "cocitation_count_v2.npy"), cocitation_count)
    np.save(os.path.join(HERE, "coauthor_count.npy"), coauthor_count)
    np.save(os.path.join(HERE, "directed_citation_count.npy"), directed_count)
    for name, block in blocks.items():
        np.save(os.path.join(HERE, f"embedding_{name}.npy"), block)
    np.save(os.path.join(HERE, "embedding_combined.npy"), combined)
    with open(os.path.join(HERE, "embedding_bais_v2.json"), "w") as f:
        json.dump(bais, f)
    with open(os.path.join(HERE, "multi_embedding_dims.json"), "w") as f:
        json.dump(dims, f)

    print("Saved: cocitation_count_v2.npy, coauthor_count.npy, "
          "directed_citation_count.npy, embedding_{cocitation,coauthor,citer,citee}.npy, "
          "embedding_combined.npy, embedding_bais_v2.json, multi_embedding_dims.json", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
