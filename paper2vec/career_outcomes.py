#!/usr/bin/python3

################################################################################
# Q3: does paper-embedding position at PhD graduation predict time to a
# faculty appointment (JUNIOR/SENIOR/STAFF rank) and whether it happens at
# all, and how does that compare to standard bibliometric predictors
# (paper count, citation count at graduation)?
#
# Framed as a discrete-time ("person-period") hazard model, per Singer &
# Willett: each researcher contributes one row per year since graduation
# they were still un-tenured and observed, with outcome=1 in the year they
# first reach a faculty rank and 0 otherwise, dropping rows past their last
# observed year (right-censoring). This uses the partial information in
# researchers who graduated recently and haven't had time to reach faculty
# yet, rather than only using fully-resolved careers -- fitting on only
# resolved cases would bias toward long-ago graduates and bias the
# "did they make it" rate downward for no reason but recency.
#
# CAVEAT baked into the data, not just this comment: a handful of
# researchers show a faculty rank starting AT OR BEFORE their recorded PhD
# end date. Manually checked -- these are real career paths, not a data
# bug: several academic systems (observed here: China, Russia, Indonesia)
# have in-service faculty pursue a PhD while already holding a junior/
# senior appointment, so the clean "PhD -> postdoc -> faculty" pipeline
# this whole framing assumes is Western-academia-centric and doesn't
# universally apply. Those cases are floored to t=0 (treated as "already
# faculty by graduation") rather than dropped, to keep the sample size,
# but the framing itself should be read with that limitation in mind.
#
# Follow-up is capped at MAX_FOLLOWUP years post-PhD (typical
# tenure-track horizon) -- beyond that, year-since-PhD dummies get sparse
# for the handful of very old graduates in this population, which would
# just add noise to the baseline hazard estimate.
################################################################################

import json
import os

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

import trajectory as traj

HERE = os.path.dirname(os.path.abspath(__file__))
ADVISOR_INFO_PATH = os.path.join(HERE, "advisor_info.json")
POSITIONS_PATH = os.path.join(HERE, "career_positions.json")
CITATIONS_PATH = os.path.join(HERE, "citations_by_researcher.json")
OUT_PATH = os.path.join(HERE, "career_outcomes.json")

FACULTY_RANKS = {"SENIOR", "JUNIOR", "STAFF"}
CURRENT_YEAR = 2026
MAX_FOLLOWUP = 20
N_PCA_COMPONENTS = 15
N_FOLDS = 5


def year_of(date_str):
    return int(str(date_str)[:4]) if date_str else None


def build_labels():
    with open(ADVISOR_INFO_PATH) as f:
        advisor_info = json.load(f)
    with open(POSITIONS_PATH) as f:
        positions_data = json.load(f)

    records = []
    for bai, info in advisor_info.items():
        if not isinstance(info, dict):
            continue
        grad_year = info.get("phd_end_year")
        if not grad_year:
            continue
        positions = positions_data.get(bai) or []
        faculty_years = [year_of(p.get("start_date")) for p in positions if p.get("rank") in FACULTY_RANKS]
        faculty_years = [y for y in faculty_years if y is not None]

        if faculty_years:
            t_event = max(0, min(faculty_years) - grad_year)
            event = True
        else:
            t_event = None
            event = False

        t_censor = CURRENT_YEAR - grad_year
        if t_censor < 0:
            continue

        if event:
            t = min(t_event, MAX_FOLLOWUP)
            if t_event > MAX_FOLLOWUP:
                event = False  # censored at the horizon instead
        else:
            t = min(t_censor, MAX_FOLLOWUP)

        records.append({"bai": bai, "grad_year": grad_year, "event": event, "t": t})
    return records


def build_person_period(records, feature_fn):
    """One row per (researcher, year-since-PhD in [0, t]); outcome=1 only on
    the final row of an event researcher. feature_fn(bai, grad_year) -> dict
    of static covariates repeated on every row for that researcher."""
    rows = []
    for r in records:
        feats = feature_fn(r["bai"], r["grad_year"])
        if feats is None:
            continue
        for t in range(0, r["t"] + 1):
            outcome = 1 if (r["event"] and t == r["t"]) else 0
            row = {"bai": r["bai"], "t": t, "outcome": outcome}
            row.update(feats)
            rows.append(row)
            if outcome == 1:
                break
    return rows


def fit_and_evaluate(rows, feature_cols, label):
    X = np.array([[row[c] for c in feature_cols] for row in rows], dtype=np.float64)
    y = np.array([row["outcome"] for row in rows])
    groups = np.array([row["bai"] for row in rows])
    t = np.array([row["t"] for row in rows], dtype=np.float64)

    # baseline hazard trend: quadratic in years-since-PhD, plus the features
    X_full = np.column_stack([t, t ** 2, X])

    n_groups = len(set(groups))
    gkf = GroupKFold(n_splits=min(N_FOLDS, n_groups))
    aucs = []
    for train_idx, test_idx in gkf.split(X_full, y, groups):
        if y[test_idx].sum() == 0 or y[test_idx].sum() == len(test_idx):
            continue  # degenerate fold, skip
        scaler = StandardScaler().fit(X_full[train_idx])
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(scaler.transform(X_full[train_idx]), y[train_idx])
        pred = clf.predict_proba(scaler.transform(X_full[test_idx]))[:, 1]
        aucs.append(roc_auc_score(y[test_idx], pred))

    aucs = np.array(aucs)
    print(f"{label:28s}  n_rows={len(rows):5d}  n_researchers={n_groups:4d}  "
          f"CV AUC={aucs.mean():.4f} +/- {aucs.std():.4f}  ({len(aucs)} folds)", flush=True)
    return float(aucs.mean()), float(aucs.std())


def main():
    records = build_labels()
    n_event = sum(1 for r in records if r["event"])
    print(f"{len(records)} researchers with a usable grad year "
          f"({n_event} events, {len(records) - n_event} censored, "
          f"follow-up capped at {MAX_FOLLOWUP}y)", flush=True)

    embeddings, manifest = traj.load_embeddings()
    researcher_rows = traj.load_researcher_paper_rows(embeddings, manifest)

    with open(CITATIONS_PATH) as f:
        citations_data = json.load(f)

    # position at graduation, for the researchers we can compute it for
    grad_positions = {}
    for r in records:
        rows = researcher_rows.get(r["bai"], [])
        pts = traj.researcher_trajectory(rows, embeddings, [r["grad_year"]])
        if pts:
            grad_positions[r["bai"]] = pts[0]["position"]
    records = [r for r in records if r["bai"] in grad_positions]
    print(f"{len(records)} with a computable position at graduation", flush=True)

    pca_bais = sorted(grad_positions.keys())
    X_pos = np.stack([grad_positions[b] for b in pca_bais])
    pca = PCA(n_components=N_PCA_COMPONENTS, random_state=0)
    pos_pcs = pca.fit_transform(X_pos)
    print(f"PCA: {N_PCA_COMPONENTS} components explain "
          f"{pca.explained_variance_ratio_.sum():.3f} of variance", flush=True)
    pos_pc_lookup = {b: pos_pcs[i] for i, b in enumerate(pca_bais)}

    def standard_metrics(bai, grad_year):
        papers = citations_data.get(bai)
        if not isinstance(papers, list):
            return None
        by_grad = [p for p in papers if (year_of(p.get("earliest_date")) or 9999) <= grad_year]
        n_papers = len(by_grad)
        total_citations = sum(p.get("citation_count") or 0 for p in by_grad)
        return {"log_n_papers": np.log1p(n_papers), "log_citations": np.log1p(total_citations)}

    def position_features(bai, grad_year):
        pcs = pos_pc_lookup.get(bai)
        if pcs is None:
            return None
        return {f"pc{i}": float(pcs[i]) for i in range(N_PCA_COMPONENTS)}

    def combined_features(bai, grad_year):
        a = standard_metrics(bai, grad_year)
        b = position_features(bai, grad_year)
        if a is None or b is None:
            return None
        return {**a, **b}

    print("\nFitting discrete-time hazard models (5-fold grouped CV AUC):", flush=True)
    rows_std = build_person_period(records, standard_metrics)
    auc_std = fit_and_evaluate(rows_std, ["log_n_papers", "log_citations"], "standard metrics only")

    rows_pos = build_person_period(records, position_features)
    auc_pos = fit_and_evaluate(rows_pos, [f"pc{i}" for i in range(N_PCA_COMPONENTS)], "embedding position only")

    rows_combined = build_person_period(records, combined_features)
    auc_combined = fit_and_evaluate(
        rows_combined,
        ["log_n_papers", "log_citations"] + [f"pc{i}" for i in range(N_PCA_COMPONENTS)],
        "standard metrics + position")

    model_comparison = [
        {"model": "standard metrics only", "auc_mean": auc_std[0], "auc_std": auc_std[1]},
        {"model": "embedding position only", "auc_mean": auc_pos[0], "auc_std": auc_pos[1]},
        {"model": "standard metrics + position", "auc_mean": auc_combined[0], "auc_std": auc_combined[1]},
    ]

    # empirical (nonparametric) cumulative incidence, for the chart
    max_t = max(r["t"] for r in records)
    at_risk = np.zeros(max_t + 1)
    events = np.zeros(max_t + 1)
    for r in records:
        at_risk[: r["t"] + 1] += 1
        if r["event"]:
            events[r["t"]] += 1
    hazard = np.divide(events, at_risk, out=np.zeros_like(events), where=at_risk > 0)
    survival = np.cumprod(1 - hazard)
    cum_incidence = 1 - survival

    with open(OUT_PATH, "w") as f:
        json.dump({
            "n_researchers": len(records),
            "n_events": n_event,
            "pca_explained_variance": float(pca.explained_variance_ratio_.sum()),
            "model_comparison": model_comparison,
            "cumulative_incidence": [
                {"t": t, "cum_incidence": float(cum_incidence[t]), "at_risk": int(at_risk[t])}
                for t in range(max_t + 1)
            ],
        }, f, indent=2)

    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
