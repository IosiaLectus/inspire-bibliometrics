#!/usr/bin/python3

################################################################################
# Self-drift / autocorrelation analysis: how similar is a researcher's
# position to their own past, as a function of elapsed time?
#
# For every pair of years (t_i, t_j) within the same researcher's trajectory
# (the default half-life=5yr annual grid from compute_trajectories.py),
# computes cosine similarity between their positions at those two years.
# Pooling (lag, cosine) pairs across all researchers gives an empirical
# autocorrelation function of career position vs. elapsed years.
#
# Also fits, per researcher, a single "drift rate" -- the slope of cosine
# distance (1 - cosine) vs lag, forced through the origin (distance is
# trivially 0 at lag 0) -- for researchers with enough trajectory span to
# fit it meaningfully.
#
# CAVEAT baked into these numbers, not just this comment: positions are
# built with a 5-year half-life decay, so trajectory points close in time
# share overlapping paper windows by construction. That alone induces some
# short-lag correlation independent of any real scientific drift -- the
# informative part of the curve is how much it decays beyond what pure
# window overlap would predict, not the absolute short-lag correlation.
################################################################################

import json
import os
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TRAJ_EMB_PATH = os.path.join(HERE, "trajectory_embeddings.npy")
TRAJ_MANIFEST_PATH = os.path.join(HERE, "trajectory_manifest.json")
AUTOCORR_OUT = os.path.join(HERE, "autocorrelation_by_lag.json")
DRIFT_RATES_OUT = os.path.join(HERE, "researcher_drift_rates.json")

MIN_POINTS_FOR_RATE = 5   # minimum trajectory points to fit a per-researcher drift rate
MIN_SPAN_FOR_RATE = 4     # minimum years of span (in addition to point count)


def main():
    positions = np.load(TRAJ_EMB_PATH)
    with open(TRAJ_MANIFEST_PATH) as f:
        manifest = json.load(f)

    by_researcher = defaultdict(list)
    for i, m in enumerate(manifest):
        by_researcher[m["bai"]].append((m["year"], i, m["name"]))

    lag_cosines = defaultdict(list)  # lag (int years) -> list of cosine similarities
    drift_rates = []

    for bai, points in by_researcher.items():
        points.sort()
        n = len(points)
        if n < 2:
            continue
        years = np.array([p[0] for p in points])
        idxs = [p[1] for p in points]
        vecs = positions[idxs]
        name = points[0][2]

        pair_lags = []
        pair_cos = []
        for a in range(n):
            for b in range(a + 1, n):
                lag = int(round(years[b] - years[a]))
                cos = float(np.dot(vecs[a], vecs[b]))
                lag_cosines[lag].append(cos)
                pair_lags.append(lag)
                pair_cos.append(cos)

        span = float(years[-1] - years[0])
        if n >= MIN_POINTS_FOR_RATE and span >= MIN_SPAN_FOR_RATE:
            lags_arr = np.array(pair_lags, dtype=np.float64)
            dist_arr = 1.0 - np.array(pair_cos, dtype=np.float64)
            slope = float(np.sum(lags_arr * dist_arr) / np.sum(lags_arr ** 2))
            drift_rates.append({
                "bai": bai,
                "name": name,
                "drift_rate_per_year": slope,
                "span_years": span,
                "n_points": n,
            })

    autocorr = []
    for lag in sorted(lag_cosines):
        cos_vals = np.array(lag_cosines[lag])
        autocorr.append({
            "lag": lag,
            "mean_cosine": float(cos_vals.mean()),
            "std_cosine": float(cos_vals.std()),
            "median_cosine": float(np.median(cos_vals)),
            "p10_cosine": float(np.percentile(cos_vals, 10)),
            "p90_cosine": float(np.percentile(cos_vals, 90)),
            "n_pairs": int(len(cos_vals)),
        })

    with open(AUTOCORR_OUT, "w") as f:
        json.dump(autocorr, f, indent=2)
    with open(DRIFT_RATES_OUT, "w") as f:
        json.dump(drift_rates, f, indent=2)

    total_pairs = sum(len(v) for v in lag_cosines.values())
    print(f"{len(by_researcher)} researchers with a trajectory, {total_pairs} total (t_i,t_j) pairs", flush=True)
    print(f"lags observed: {min(lag_cosines)} to {max(lag_cosines)} years", flush=True)
    print(f"researchers with a fitted drift rate (>= {MIN_POINTS_FOR_RATE} pts, "
          f">= {MIN_SPAN_FOR_RATE}yr span): {len(drift_rates)}", flush=True)
    rates = np.array([d["drift_rate_per_year"] for d in drift_rates])
    print(f"drift rate (cosine distance / year): mean={rates.mean():.4f} "
          f"median={np.median(rates):.4f} std={rates.std():.4f}", flush=True)
    print("\nautocorrelation by lag:", flush=True)
    for row in autocorr:
        print(f"  lag={row['lag']:3d}yr  mean_cosine={row['mean_cosine']:.4f}  "
              f"median={row['median_cosine']:.4f}  n_pairs={row['n_pairs']}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
