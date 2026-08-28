#!/usr/bin/python3

################################################################################
# Turns the per-paper SPECTER embeddings (embeddings.npy / paper_manifest.json)
# plus each researcher's paper list (papers_by_researcher.json) into a
# per-researcher trajectory: a position vector in the fixed paper-embedding
# space at any given year t.
#
# position(bai, t) = L2-normalize( sum over papers with year <= t of
#     (1 / author_count) * 0.5 ** ((t - paper_year) / HALF_LIFE) * embedding )
#
# i.e. an exponentially time-decayed, coauthor-discounted weighted average of
# the researcher's own paper embeddings, L2-renormalized to unit norm so
# magnitude never confounds comparisons -- only direction is meant to be
# compared downstream (cosine distance/similarity). Only papers with
# year <= t are used, so a position at year t never looks ahead of it.
#
# HALF_LIFE is a free parameter, deliberately not hardcoded into a single
# precomputed dataset -- different questions (recent-drift vs. career-long
# stability) may want different decay rates, so this module also exposes
# researcher_trajectory() directly for ad hoc use with other half-lives.
################################################################################

import json
import os
from datetime import date

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EMBEDDINGS_PATH = os.path.join(HERE, "embeddings.npy")
MANIFEST_PATH = os.path.join(HERE, "paper_manifest.json")
PAPERS_PATH = os.path.join(HERE, "papers_by_researcher.json")

HALF_LIFE = 5.0  # years


def parse_year(date_str):
    """'YYYY', 'YYYY-MM', or 'YYYY-MM-DD' -> fractional year (e.g. 2020-08-19 -> ~2020.63)."""
    if not date_str:
        return None
    parts = date_str.split("-")
    year = int(parts[0])
    month = int(parts[1]) if len(parts) > 1 else 1
    day = int(parts[2]) if len(parts) > 2 else 1
    day_of_year = date(year, month, day).timetuple().tm_yday
    days_in_year = 366 if (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)) else 365
    return year + (day_of_year - 1) / days_in_year


def load_embeddings():
    embeddings = np.load(EMBEDDINGS_PATH)
    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)
    return embeddings, manifest


def load_researcher_paper_rows(embeddings=None, manifest=None):
    """bai -> list of (row_index, paper_year, author_count) for every paper of
    theirs with a resolvable year, keyed against the deduped embedding rows."""
    if manifest is None:
        _, manifest = load_embeddings()
    cn_to_row = {m["control_number"]: (i, parse_year(m["earliest_date"]), m.get("author_count") or 1)
                 for i, m in enumerate(manifest)}

    with open(PAPERS_PATH) as f:
        papers_by_researcher = json.load(f)

    result = {}
    for bai, papers in papers_by_researcher.items():
        if not isinstance(papers, list):
            continue
        rows = []
        for p in papers:
            entry = cn_to_row.get(p["control_number"])
            if entry is not None and entry[1] is not None:
                rows.append(entry)
        result[bai] = rows
    return result


def researcher_trajectory(rows, embeddings, years, half_life=HALF_LIFE):
    """rows: list of (row_index, paper_year, author_count) for one researcher.
    years: iterable of query years t. Returns a list of dicts, one per t that
    has at least one eligible (year <= t) paper, each with a unit-ish-norm
    'position' vector plus diagnostics (n_papers, effective_weight) that flag
    how much signal actually supports that point."""
    if not rows:
        return []
    idxs = np.array([r[0] for r in rows])
    paper_years = np.array([r[1] for r in rows], dtype=np.float64)
    base_weights = np.array([1.0 / r[2] for r in rows], dtype=np.float64)
    vecs = embeddings[idxs].astype(np.float64)

    out = []
    for t in years:
        mask = paper_years <= t
        if not mask.any():
            continue
        decay = 0.5 ** ((t - paper_years[mask]) / half_life)
        w = base_weights[mask] * decay
        pos = (vecs[mask] * w[:, None]).sum(axis=0)
        norm = np.linalg.norm(pos)
        if norm > 0:
            pos = pos / norm
        out.append({
            "year": float(t),
            "position": pos.astype(np.float32),
            "n_papers": int(mask.sum()),
            "effective_weight": float(w.sum()),
        })
    return out
