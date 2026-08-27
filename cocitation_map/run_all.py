#!/usr/bin/python3
"""Run the full pipeline: fetch citing papers for everyone in
eligible_authors.json, then build the PPMI/SVD co-citation embedding."""

import fetch_citers
import build_embedding

if __name__ == "__main__":
    fetch_citers.main()
    build_embedding.main()
