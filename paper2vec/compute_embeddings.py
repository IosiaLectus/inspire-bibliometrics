#!/usr/bin/python3

################################################################################
# Computes a fixed, frozen SPECTER embedding (allenai-specter, via
# sentence-transformers) for every distinct paper appearing in
# papers_by_researcher.json. Papers are deduped by control_number first,
# since a paper coauthored by two eligible researchers appears twice in the
# per-researcher fetch -- embedding it once is both cheaper and the whole
# point of using a fixed embedding space (same paper, same vector, no matter
# whose trajectory it feeds into).
#
# Text fed to the model follows SPECTER's own convention:
# title + tokenizer.sep_token + abstract (title alone if no abstract).
#
# Output: embeddings.npy (N x dim, float32) plus paper_manifest.json, a list
# of {control_number, earliest_date, author_count} in the same row order as
# embeddings.npy.
################################################################################

import json
import os

import numpy as np
from sentence_transformers import SentenceTransformer

HERE = os.path.dirname(os.path.abspath(__file__))
PAPERS_PATH = os.path.join(HERE, "papers_by_researcher.json")
EMBEDDINGS_PATH = os.path.join(HERE, "embeddings.npy")
MANIFEST_PATH = os.path.join(HERE, "paper_manifest.json")

MODEL_NAME = "sentence-transformers/allenai-specter"
BATCH_SIZE = 32


def dedup_papers(papers_by_researcher):
    unique = {}
    for bai, papers in papers_by_researcher.items():
        if not isinstance(papers, list):
            continue
        for p in papers:
            cn = p["control_number"]
            if cn not in unique and p.get("title"):
                unique[cn] = p
    return unique


def build_text(paper, sep_token):
    title = paper["title"]
    abstract = paper.get("abstract")
    if abstract:
        return f"{title}{sep_token}{abstract}"
    return title


def main():
    with open(PAPERS_PATH) as f:
        papers_by_researcher = json.load(f)

    unique = dedup_papers(papers_by_researcher)
    control_numbers = sorted(unique.keys())
    print(f"{len(control_numbers)} distinct papers to embed "
          f"(from {sum(len(v) for v in papers_by_researcher.values() if isinstance(v, list))} "
          f"researcher-paper edges)", flush=True)

    print(f"Loading {MODEL_NAME}...", flush=True)
    model = SentenceTransformer(MODEL_NAME)
    sep_token = model.tokenizer.sep_token

    texts = [build_text(unique[cn], sep_token) for cn in control_numbers]

    print("Encoding...", flush=True)
    embeddings = model.encode(
        texts, batch_size=BATCH_SIZE, show_progress_bar=True, convert_to_numpy=True
    ).astype(np.float32)

    manifest = [
        {
            "control_number": cn,
            "earliest_date": unique[cn].get("earliest_date"),
            "author_count": unique[cn].get("author_count"),
        }
        for cn in control_numbers
    ]

    np.save(EMBEDDINGS_PATH, embeddings)
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f)

    print(f"Saved embeddings.npy {embeddings.shape} and paper_manifest.json", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
