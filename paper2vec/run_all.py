#!/usr/bin/python3
"""Run the full pipeline: fetch every eligible researcher's papers, embed the
deduped set of papers with a fixed SPECTER model, then aggregate into
per-researcher-year trajectories."""

import fetch_papers
import compute_embeddings
import compute_trajectories

if __name__ == "__main__":
    fetch_papers.main()
    compute_embeddings.main()
    compute_trajectories.main()
