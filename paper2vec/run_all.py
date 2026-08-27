#!/usr/bin/python3
"""Run the full pipeline: fetch every eligible researcher's papers, then
embed the deduped set of papers with a fixed SPECTER model."""

import fetch_papers
import compute_embeddings

if __name__ == "__main__":
    fetch_papers.main()
    compute_embeddings.main()
