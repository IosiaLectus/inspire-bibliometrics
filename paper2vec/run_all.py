#!/usr/bin/python3
"""Run the full pipeline: fetch every eligible researcher's papers, pull in
PhD-advisor papers for advisors outside the eligible population, embed the
deduped paper set with a fixed SPECTER model, aggregate into
per-researcher-year trajectories, then run the self-drift and
advisor-drift analyses."""

import fetch_papers
import fetch_advisors
import fetch_extra_advisors
import compute_embeddings
import compute_trajectories
import self_drift
import advisor_drift

if __name__ == "__main__":
    fetch_papers.main()
    fetch_advisors.main()
    fetch_extra_advisors.main()
    compute_embeddings.main()
    compute_trajectories.main()
    self_drift.main()
    advisor_drift.main()
