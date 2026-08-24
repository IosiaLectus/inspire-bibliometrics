#!/usr/bin/python3

################################################################################
# v2 of build_multi_embedding.py: adds a fourth relation (institution
# affiliation, paper-count weighted) to the three from v1 (co-citation,
# coauthorship, directed citation), and applies the pairwise-fallback exact
# counts in place of any truncated co-citation counts for capped people.
#
# Institution relation: for each person, a vector over institutions weighted
# by how many of their own papers list that institution (keyed by INSPIRE's
# canonical institution record id, so no string entity-resolution needed).
# The person-person relation is the bipartite projection P @ P.T -- the
# standard way to turn a weighted person-x-institution bigraph into a
# person-x-person co-occurrence matrix (analogous to how coauthorship's
# person-x-paper bigraph projects onto shared papers, except institution
# "sharing" is inherently fuzzy/weighted rather than a single discrete event).
#
# arXiv categories are NOT embedded here -- deliberately kept as a
# validation-only layer computed separately, since they're closer to a label
# for what the embedding should recover than a relation it should be trained
# on (see conversation record for why).
################################################################################

import json
import os
import math
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ELIGIBLE_PATH = os.path.join(HERE, "eligible_authors_v2.json")
CITERS_PATH = os.path.join(HERE, "citing_papers_v2.json")
OWN_PATH = os.path.join(HERE, "own_papers_v2.json")
CAPPED_PATH = os.path.join(HERE, "capped_bais_v2.json")
FALLBACK_PATH = os.path.join(HERE, "pairwise_fallback_counts.json")

VARIANCE_TARGET = 0.98
FINAL_VARIANCE_TARGET = 0.95


def build_symmetric_count(bais, sets_a, sets_b=None):
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


def apply_pairwise_fallback(count, bais, capped_set, fallback):
    """Overwrite count(i,j) with the exact pairwise value wherever available
    and at least one of i, j is a capped (truncated-citer-list) person.
    Entries where the exact query failed after retries (None) are left as
    the original approximate count rather than corrupting the matrix."""
    idx = {b: i for i, b in enumerate(bais)}
    n_overridden = 0
    n_failed = 0
    for key, exact in fallback.items():
        if exact is None:
            n_failed += 1
            continue
        a, b = key.split("|")
        if a not in idx or b not in idx:
            continue
        i, j = idx[a], idx[b]
        count[i, j] = exact
        count[j, i] = exact
        n_overridden += 1
    print(f"  pairwise fallback: overrode {n_overridden} pairs "
          f"({n_failed} skipped -- exact query failed, kept approximate count)", flush=True)
    return count


def build_directed_count(bais, own, citing):
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


def build_institution_projection(bais, own_papers_v2):
    """person x institution matrix, weighted by paper count, projected onto
    person x person via P @ P.T (zero diagonal)."""
    inst_ids = set()
    for b in bais:
        for p in own_papers_v2[b]:
            inst_ids.update(p.get("affiliations", []))
    inst_ids = sorted(inst_ids)
    inst_idx = {k: i for i, k in enumerate(inst_ids)}
    P = np.zeros((len(bais), len(inst_ids)))
    for i, b in enumerate(bais):
        for p in own_papers_v2[b]:
            for aff in p.get("affiliations", []):
                P[i, inst_idx[aff]] += 1
    projection = P @ P.T
    np.fill_diagonal(projection, 0)
    return projection, P.sum(axis=1)  # per-person total institution-weighted paper count


def ppmi_symmetric(count, marg):
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


def scan_svd_dimension(M, target=VARIANCE_TARGET):
    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    total = np.sum(S ** 2)
    if total == 0:
        return U, S, Vt, np.array([1.0]), 1
    cum = np.cumsum(S ** 2) / total
    k = int(np.searchsorted(cum, target) + 1)
    return U, S, Vt, cum, max(k, 1)


def normalize_block(block):
    norms = np.linalg.norm(block, axis=1)
    avg = norms[norms > 0].mean() if (norms > 0).any() else 1.0
    return block / avg if avg > 0 else block


def main():
    with open(ELIGIBLE_PATH) as f:
        eligible = json.load(f)
    with open(CITERS_PATH) as f:
        citing_papers_raw = json.load(f)
    with open(OWN_PATH) as f:
        own_papers_v2 = json.load(f)

    bais = [p["bai"] for p in eligible]
    missing_citing = [b for b in bais if b not in citing_papers_raw]
    missing_own = [b for b in bais if b not in own_papers_v2]
    if missing_citing or missing_own:
        raise SystemExit(f"{len(missing_citing)} missing from citing_papers_v2.json, "
                          f"{len(missing_own)} missing from own_papers_v2.json; "
                          f"run fetch_citers_v2.py / fetch_own_papers_v2.py to completion first.")

    citing = {b: set(citing_papers_raw[b]) for b in bais}
    own = {b: set(p["control_number"] for p in own_papers_v2[b]) for b in bais}
    citing_singles = np.array([len(citing[b]) for b in bais], dtype=float)
    own_singles = np.array([len(own[b]) for b in bais], dtype=float)

    print(f"Building relation matrices for {len(bais)} people...", flush=True)

    print("  co-citation...", flush=True)
    cocitation_count = build_symmetric_count(bais, citing)
    try:
        with open(CAPPED_PATH) as f:
            capped = set(json.load(f))
        with open(FALLBACK_PATH) as f:
            fallback = json.load(f)
        print(f"  applying pairwise fallback for {len(capped)} capped people...", flush=True)
        cocitation_count = apply_pairwise_fallback(cocitation_count, bais, capped, fallback)
    except FileNotFoundError:
        print("  WARNING: no pairwise fallback data found; using raw (possibly truncated) counts.", flush=True)
    print(f"    non-zero pairs: {int((cocitation_count > 0).sum() / 2)}", flush=True)

    print("  coauthorship...", flush=True)
    coauthor_count = build_symmetric_count(bais, own, sets_b=own)
    print(f"    non-zero pairs: {int((coauthor_count > 0).sum() / 2)}", flush=True)

    print("  directed citation (i's own paper cites j)...", flush=True)
    directed_count = build_directed_count(bais, own, citing)
    print(f"    non-zero directed edges: {int((directed_count > 0).sum())}", flush=True)

    print("  institution affiliation (paper-count weighted)...", flush=True)
    institution_count, institution_singles = build_institution_projection(bais, own_papers_v2)
    print(f"    non-zero pairs: {int((institution_count > 0).sum() / 2)}", flush=True)

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
    U, S, Vt, cum, k = scan_svd_dimension(ppmi_dc)
    print(f"  k={k} for {VARIANCE_TARGET*100:.0f}% variance (captures {cum[k-1]*100:.2f}%)", flush=True)
    citer = U[:, :k] * np.sqrt(S[:k])
    citee = Vt[:k, :].T * np.sqrt(S[:k])
    blocks["citer"] = normalize_block(citer)
    blocks["citee"] = normalize_block(citee)
    dims["citer"] = k
    dims["citee"] = k

    print("Factorizing institution affiliation...", flush=True)
    ppmi_inst = ppmi_symmetric(institution_count, institution_singles)
    U, S, Vt, cum, k = scan_svd_dimension(ppmi_inst)
    print(f"  k={k} for {VARIANCE_TARGET*100:.0f}% variance (captures {cum[k-1]*100:.2f}%)", flush=True)
    blocks["institution"] = normalize_block(U[:, :k] * np.sqrt(S[:k]))
    dims["institution"] = k

    combined = np.concatenate([blocks["cocitation"], blocks["coauthor"],
                                blocks["citer"], blocks["citee"], blocks["institution"]], axis=1)
    print(f"\nCombined (pre-compression) shape: {combined.shape} dims={dims}", flush=True)

    print("\nRunning final SVD pass on the concatenated embedding...", flush=True)
    Uc, Sc, Vtc, cum_c, k_c = scan_svd_dimension(combined, target=FINAL_VARIANCE_TARGET)
    print(f"  k={k_c} for {FINAL_VARIANCE_TARGET*100:.0f}% variance (captures {cum_c[k_c-1]*100:.2f}%)", flush=True)
    final_embedding = Uc[:, :k_c] * Sc[:k_c]
    dims["final"] = k_c
    print(f"Final embedding shape: {final_embedding.shape}", flush=True)

    np.save(os.path.join(HERE, "cocitation_count_v3.npy"), cocitation_count)
    np.save(os.path.join(HERE, "coauthor_count_v3.npy"), coauthor_count)
    np.save(os.path.join(HERE, "directed_citation_count_v3.npy"), directed_count)
    np.save(os.path.join(HERE, "institution_count_v3.npy"), institution_count)
    for name, block in blocks.items():
        np.save(os.path.join(HERE, f"embedding_v3_{name}.npy"), block)
    np.save(os.path.join(HERE, "embedding_v3_combined.npy"), combined)
    np.save(os.path.join(HERE, "embedding_v3_final.npy"), final_embedding)
    np.save(os.path.join(HERE, "final_singular_values_v3.npy"), Sc)
    with open(os.path.join(HERE, "embedding_bais_v3.json"), "w") as f:
        json.dump(bais, f)
    with open(os.path.join(HERE, "multi_embedding_dims_v3.json"), "w") as f:
        json.dump(dims, f)

    print("DONE", flush=True)


if __name__ == "__main__":
    main()
