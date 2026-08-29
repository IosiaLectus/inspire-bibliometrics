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
# k is NOT chosen by maximizing silhouette score: over a scanned range
# (2-25) silhouette increases monotonically as k shrinks toward 2, which is
# the expected signature of a continuously-varying field with no crisp
# separated blobs, not evidence that 2 clusters is the informative answer.
# K_CHOSEN was instead picked by inspecting TF-IDF term interpretability
# across the scanned range: below it, real distinctions collapse together
# (e.g. gravitational-wave astronomy folded into black-hole GR); above it,
# clusters start duplicating each other (e.g. the collider-physics side
# splitting into near-identical fragments). The silhouette scan is still
# run and reported for transparency, just not used to pick k.
################################################################################

import json
import os

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score

import trajectory as traj
from text_utils import strip_markup

HERE = os.path.dirname(os.path.abspath(__file__))
PAPERS_PATH = os.path.join(HERE, "papers_by_researcher.json")
ELIGIBLE_PATH = os.path.join(HERE, "eligible_researchers.json")
ADVISOR_INFO_PATH = os.path.join(HERE, "advisor_info.json")
ADVISOR_BAI_PATH = os.path.join(HERE, "advisor_bais.json")
CLUSTERS_OUT = os.path.join(HERE, "clusters.json")

CURRENT_YEAR = 2026
MIN_REAL_PAPERS = 3
SILHOUETTE_SCAN_RANGE = range(2, 13)
K_CHOSEN = 6
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

    names = load_names()

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

    print("\nSilhouette scan (diagnostic only -- see module docstring):", flush=True)
    for k in SILHOUETTE_SCAN_RANGE:
        km = KMeans(n_clusters=k, n_init=10, random_state=0)
        labels = km.fit_predict(X)
        score = silhouette_score(X, labels, metric="cosine")
        marker = "  <-- chosen" if k == K_CHOSEN else ""
        print(f"  k={k:2d}  silhouette={score:.4f}{marker}", flush=True)

    km = KMeans(n_clusters=K_CHOSEN, n_init=10, random_state=0)
    labels = km.fit_predict(X)
    centroids = km.cluster_centers_
    centroid_norms = centroids / np.linalg.norm(centroids, axis=1, keepdims=True)

    print("\nComputing 2D PCA projection for visualization...", flush=True)
    pca = PCA(n_components=2, random_state=0)
    coords_2d = pca.fit_transform(X)
    print(f"  explained variance ratio: {pca.explained_variance_ratio_}", flush=True)

    # per-cluster pooled text for TF-IDF term extraction
    cluster_docs = []
    for c in range(K_CHOSEN):
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

    tfidf = TfidfVectorizer(max_features=20000, stop_words="english", ngram_range=(1, 2), min_df=2)
    tfidf_matrix = tfidf.fit_transform(cluster_docs)
    terms = np.array(tfidf.get_feature_names_out())

    clusters_summary = []
    for c in range(K_CHOSEN):
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
            "representatives": [
                {"bai": bai, "name": names.get(bai, bai)} for _, bai in sims[:TOP_REPS]
            ],
        })

    clusters_summary.sort(key=lambda c: -c["size"])
    with open(CLUSTERS_OUT, "w") as f:
        json.dump({
            "k": K_CHOSEN,
            "clusters": clusters_summary,
            "points": [
                {
                    "bai": ordered_bais[i],
                    "name": names.get(ordered_bais[i], ordered_bais[i]),
                    "cluster": int(labels[i]),
                    "x": float(coords_2d[i, 0]),
                    "y": float(coords_2d[i, 1]),
                }
                for i in range(len(ordered_bais))
            ],
        }, f)

    print(f"\n{K_CHOSEN} clusters, sizes: {[c['size'] for c in clusters_summary]}", flush=True)
    print("DONE", flush=True)


def load_names():
    """bai -> display name, for both the eligible population and the
    PhD-advisor BAIs that were pulled in from outside it (fetch_extra_advisors.py)."""
    names = {}
    with open(ELIGIBLE_PATH) as f:
        for p in json.load(f):
            names[p["bai"]] = p["name"]
    try:
        with open(ADVISOR_INFO_PATH) as f:
            student_info = json.load(f)
        with open(ADVISOR_BAI_PATH) as f:
            advisor_bais_map = json.load(f)
        for info in student_info.values():
            if not isinstance(info, dict):
                continue
            adv = info.get("phd_advisor")
            if not adv or adv.get("control_number") is None:
                continue
            adv_bai = advisor_bais_map.get(str(adv["control_number"]))
            if adv_bai and adv_bai != "ERROR" and adv_bai not in names and adv.get("name"):
                names[adv_bai] = adv["name"]
    except FileNotFoundError:
        pass
    return names


if __name__ == "__main__":
    main()
