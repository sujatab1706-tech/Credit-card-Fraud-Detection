"""Build candidate pairs, train match-scoring models, choose thresholds, save the winner.

Time-based split by payment date (no peeking into the future):
  train  Jan - Aug 2025
  val    Sep - Oct 2025   (model selection + auto-match threshold)
  test   Nov 2025 onward  (reported once at the end)

Models compared:
  rules     typical ERP lockbox rule: one exact invoice number + exact amount, or one exact-amount
            invoice for a clearly named customer
  logreg    logistic regression on scaled features
  hgb       histogram gradient boosting (sklearn's LightGBM-style model)

Usage (from the project root):
  python src/train.py --data data --models models --results results
"""
import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from candidates import Ledger  # noqa: E402
from decide import decide, pick_auto_threshold, score_decisions, summarize  # noqa: E402
from features import FEATURES, build_pairs  # noqa: E402

VAL_START, TEST_START = pd.Timestamp("2025-09-01"), pd.Timestamp("2025-11-01")
REVIEW_THRESHOLD = 0.15
TARGET_PRECISION = 0.995


def load(data):
    d = Path(data)
    cust = pd.read_csv(d / "customers.csv")
    inv = pd.read_csv(d / "invoices.csv", dtype={"po_number": str})
    pay = pd.read_csv(d / "payments.csv", parse_dates=["payment_date"], keep_default_na=False,
                      na_values={"amount": [""]})
    m = pd.read_csv(d / "matches.csv")
    return cust, inv, pay, m


def split_of(dates):
    return np.where(dates < VAL_START, "train", np.where(dates < TEST_START, "val", "test"))


def rules_baseline(pairs: pd.DataFrame, payment_ids) -> pd.DataFrame:
    """Deterministic matching rules most ERPs ship with. Anything they can't decide goes to a person."""
    rows = []
    groups = dict(tuple(pairs.groupby("payment_id")))
    for pid in payment_ids:
        g = groups.get(pid)
        inv, dec = [], "REVIEW"
        if g is not None:
            r1 = g[(g.ref_exact == 1) & (g.amt_exact == 1)]
            r2 = g[(g.name_score >= 90) & (g.name_gap == 0) & (g.amt_exact == 1)]
            if (g.ref_exact == 1).sum() == 1 and len(r1) == 1:
                inv, dec = list(r1.invoice_number), "AUTO_MATCH"
            elif (g.ref_exact == 0).all() and len(r2) == 1:
                inv, dec = list(r2.invoice_number), "AUTO_MATCH"
        rows.append({"payment_id": pid, "decision": dec, "invoices": inv, "confidence": float(dec == "AUTO_MATCH"),
                     "suggestions": [{"invoice_number": i} for i in inv]})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--models", default="models")
    ap.add_argument("--results", default="results")
    ap.add_argument("--reuse-pairs", action="store_true", help="load results/pairs.csv.gz instead of rebuilding")
    args = ap.parse_args()
    models, results = Path(args.models), Path(args.results)
    models.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)

    cust, inv, pay, m = load(args.data)
    pay["split"] = split_of(pay.payment_date)
    if args.reuse_pairs and (results / "pairs.csv.gz").exists():
        pairs = pd.read_csv(results / "pairs.csv.gz")
    else:
        ledger = Ledger.from_history(cust, inv, pay, m)
        t0 = time.time()
        pairs = build_pairs(ledger, pay, m)
        print(f"{len(pairs):,} candidate pairs for {pairs.payment_id.nunique():,} payments in {time.time() - t0:.0f}s")
        pairs = pairs.merge(pay[["payment_id", "split", "payment_type"]], on="payment_id")
        pairs.to_csv(results / "pairs.csv.gz", index=False)

    # candidate recall: share of true links that survive step 1
    linked = m.merge(pay[["payment_id", "split"]], on="payment_id")
    found = pairs[pairs.is_match == 1].groupby("split").size()
    recall = (found / linked.groupby("split").size()).round(4).to_dict()
    print("candidate recall by split:", recall)

    tr, va, te = (pairs[pairs.split == s] for s in ("train", "val", "test"))
    ids = {s: pay.loc[pay.split == s, "payment_id"].tolist() for s in ("train", "val", "test")}
    print({s: len(v) for s, v in ids.items()}, "payments")

    candidates = {
        "logreg": make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)),
        "hgb": HistGradientBoostingClassifier(learning_rate=0.08, max_iter=400, max_leaf_nodes=31,
                                              min_samples_leaf=40, l2_regularization=1.0,
                                              early_stopping=True, validation_fraction=0.15, random_state=0),
    }
    rows, fitted = [], {}
    for name, model in candidates.items():
        model.fit(tr[FEATURES], tr.is_match)
        fitted[name] = model
        p = model.predict_proba(va[FEATURES])[:, 1]
        sv = va.assign(p_match=p)
        t, _ = pick_auto_threshold(sv, ids["val"], m, REVIEW_THRESHOLD, TARGET_PRECISION)
        s = summarize(score_decisions(decide(sv, ids["val"], t, REVIEW_THRESHOLD), m))
        rows.append({"model": name, "pair_pr_auc": average_precision_score(va.is_match, p),
                     "pair_roc_auc": roc_auc_score(va.is_match, p), "auto_threshold": t, **s})
    rb = summarize(score_decisions(rules_baseline(va, ids["val"]), m))
    rows.append({"model": "rules", "pair_pr_auc": np.nan, "pair_roc_auc": np.nan, "auto_threshold": np.nan, **rb})
    val_table = pd.DataFrame(rows)
    val_table.to_csv(results / "val_model_comparison.csv", index=False)
    print("\nValidation\n", val_table[["model", "pair_pr_auc", "auto_threshold", "auto_match_rate",
                                       "auto_match_precision", "review_rate"]].round(4).to_string(index=False))

    # pick the model that automates most at the precision target
    ml = val_table[val_table.model != "rules"]
    best_name = ml.sort_values(["auto_match_rate", "pair_pr_auc"], ascending=False).model.iat[0]
    best = fitted[best_name]
    sv = va.assign(p_match=best.predict_proba(va[FEATURES])[:, 1])
    auto_t, curve = pick_auto_threshold(sv, ids["val"], m, REVIEW_THRESHOLD, TARGET_PRECISION)
    curve.to_csv(results / "val_threshold_curve.csv", index=False)

    # final test report (touched once)
    st = te.assign(p_match=best.predict_proba(te[FEATURES])[:, 1])
    dec_test = score_decisions(decide(st, ids["test"], auto_t, REVIEW_THRESHOLD), m)
    test_ml = summarize(dec_test)
    test_rules = summarize(score_decisions(rules_baseline(te, ids["test"]), m))
    test_table = pd.DataFrame([{"model": f"{best_name} + decision layer", **test_ml}, {"model": "rules", **test_rules}])
    test_table.to_csv(results / "test_metrics.csv", index=False)
    st.to_csv(results / "test_pairs_scored.csv.gz", index=False)
    dec_test.merge(pay[["payment_id", "payment_type", "amount"]], on="payment_id").assign(
        invoices=lambda d: d.invoices.apply("|".join),
        true_invoices=lambda d: d.true_invoices.apply(lambda s: "|".join(sorted(s))),
        suggestions=lambda d: d.suggestions.apply(json.dumps)).to_csv(results / "test_decisions.csv", index=False)
    print("\nTest\n", test_table.T.to_string(header=False))

    joblib.dump(best, models / "match_model.joblib")
    meta = {"model": best_name, "features": FEATURES, "auto_threshold": auto_t, "review_threshold": REVIEW_THRESHOLD,
            "target_precision": TARGET_PRECISION, "candidate_recall": recall,
            "val_pair_pr_auc": round(float(ml.set_index("model").at[best_name, "pair_pr_auc"]), 4),
            "test_pair_pr_auc": round(float(average_precision_score(te.is_match, st.p_match)), 4),
            "test": test_ml, "test_rules": test_rules,
            "trained_on": f"payments before {VAL_START.date()}", "trained_at": pd.Timestamp.now("UTC").isoformat()}
    (models / "model_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nSaved {best_name} to {models / 'match_model.joblib'} (auto threshold {auto_t})")


if __name__ == "__main__":
    main()
