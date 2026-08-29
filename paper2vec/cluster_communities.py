#!/usr/bin/python3

################################################################################
# Clusters researchers' CURRENT (2026) trajectory positions with k-means
# (spherical, since positions are already unit-normalized) to see what
# community structure emerges from the embedding space itself, as opposed
# to the citation-graph-derived clustering in the sibling cocitation_map/
# pipeline.
#
# Only researchers with >=3 real papers are included (same floor as
# self_drift.py's pooling threshold -- fewer than that and the current
# position is too noisy to cluster meaningfully).
#
# k is chosen by silhouette score over a small scanned range rather than
# fixed by hand. Each cluster is characterized two ways: the researchers
# closest to its centroid (by cosine similarity), and the terms most
# distinctive of its members' paper titles+abstracts vs. every other
# cluster's (TF-IDF, one pooled "document" per cluster).
################################################################################

import json
import os

import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score

import trajectory as traj
from text_utils import strip_markup

HERE = os.path.dirname(os.path.abspath(__file__))
PAPERS_PATH = os.path.join(HERE, "papers_by_researcher.json")
CLUSTERS_OUT = os.path.join(HERE, "clusters.json")

CURRENT_YEAR = 2026
MIN_REAL_PAPERS = 3
K_RANGE = range(6, 26, 2)
TOP_TERMS = 10
TOP_REPS = 8


def main():
    embeddings, manifest = traj.load_embeddings()
    researcher_rows = traj.load_researcher_paper_rows(embeddings, manifest)
    with open(PAPERS_PATH) as f:
        papers_by_researcher = json.load(f)
    papers_lookup = {}
    for plist in papers_by_researcher.values():
        if isinstance(plist, list):
            for p in plist:
                papers_lookup.setdefault(p["control_number"], p)

    bais = sorted(b for b, rows in researcher_rows.items() if len(rows) >= MIN_REAL_PAPERS)
    print(f"{len(bais)} researchers with >= {MIN_REAL_PAPERS} real papers", flush=True)

    current_points = {}
    for bai in bais:
        pts = traj.researcher_trajectory(researcher_rows[bai], embeddings, [CURRENT_YEAR])
        if pts:
            current_points[bai] = pts[0]["position"]

    ordered_bais = sorted(current_points.keys())
    X = np.stack([current_points[b] for b in ordered_bais])
    print(f"{len(ordered_bais)} researchers with a {CURRENT_YEAR} position", flush=True)

    print("\nScanning k for silhouette score...", flush=True)
    best_k, best_score, best_labels, best_km = None, -1, None, None
    for k in K_RANGE:
        km = KMeans(n_clusters=k, n_init=10, random_state=0)
        labels = km.fit_predict(X)
        score = silhouette_score(X, labels, metric="cosine")
        print(f"  k={k:2d}  silhouette={score:.4f}", flush=True)
        if score > best_score:
            best_k, best_score, best_labels, best_km = k, score, labels, km

    print(f"\n==> chosen k={best_k} (silhouette={best_score:.4f})", flush=True)
    labels = best_labels
    centroids = best_km.cluster_centers_
    centroid_norms = centroids / np.linalg.norm(centroids, axis=1, keepdims=True)

    # per-cluster pooled text for TF-IDF term extraction
    cluster_docs = []
    for c in range(best_k):
        member_bais = [ordered_bais[i] for i in range(len(ordered_bais)) if labels[i] == c]
        cns = set()
        for b in member_bais:
            for row in researcher_rows[b]:
                cns.add(manifest[row[0]]["control_number"])
        texts = []
        for cn in cns:
            p = papers_lookup.get(cn)
            if p and p.get("title"):
                text = strip_markup(p["title"])
                if p.get("abstract"):
                    text += " " + strip_markup(p["abstract"])
                texts.append(text)
        cluster_docs.append(" ".join(texts))

    tfidf = TfidfVectorizer(max_features=20000, stop_words="english", ngram_range=(1, 2), min_df=1)
    tfidf_matrix = tfidf.fit_transform(cluster_docs)
    terms = np.array(tfidf.get_feature_names_out())

    clusters_summary = []
    for c in range(best_k):
        member_idxs = [i for i in range(len(ordered_bais)) if labels[i] == c]
        member_bais = [ordered_bais[i] for i in member_idxs]
        sims = [(float(np.dot(X[i], centroid_norms[c])), ordered_bais[i]) for i in member_idxs]
        sims.sort(reverse=True)

        row = tfidf_matrix[c].toarray().ravel()
        top_term_idx = np.argsort(row)[::-1][:TOP_TERMS]
        top_terms = [terms[i] for i in top_term_idx if row[i] > 0]

        clusters_summary.append({
            "cluster": c,
            "size": len(member_bais),
            "top_terms": top_terms,
            "representatives": [bai for _, bai in sims[:TOP_REPS]],
        })

    clusters_summary.sort(key=lambda c: -c["size"])
    with open(CLUSTERS_OUT, "w") as f:
        json.dump({
            "k": best_k,
            "silhouette": best_score,
            "clusters": clusters_summary,
            "assignments": {ordered_bais[i]: int(labels[i]) for i in range(len(ordered_bais))},
        }, f, indent=2)

    print(f"\n{best_k} clusters, sizes: {[c['size'] for c in clusters_summary]}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
