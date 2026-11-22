"""Compare live matching (the API logs) with the validation-period reference.

Cash application gets labels faster than most ML systems: every payment an analyst
resolves from the review queue, and every auto-match that is audited, tells us
whether the model was right. So we watch three things:

  1. DECISION MIX     auto-match / review / unapplied rates vs reference
                      (a falling auto-match rate means lost savings, a rising one may mean over-confidence)
  2. INPUT AND SCORE DRIFT, with the Population Stability Index (PSI):
        PSI < 0.10  no real change | 0.10-0.25 watch | > 0.25 drift, investigate
     on amount, confidence, candidates per payment, best name score, references found,
     plus the share of payments with no usable remittance text.
  3. LABELLED QUALITY from logs/feedback.csv
        audited auto-match precision   must stay above the target (default 99.5 %)
        review top-suggestion accuracy how often the first suggestion was the right one

Typical root causes: a bank changes its file format (remittance text truncated), a big
customer starts paying through a third-party portal (payer names stop matching), or a
billing change (new invoice-number format) breaks the reference parser.

Usage:
  python monitoring/monitor.py --reference monitoring/reference.csv --current logs/matches.csv \
      --feedback logs/feedback.csv
Exit code 1 if any alert fires (so it can gate a scheduled job).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

NUMERIC = ["amount", "confidence", "n_candidates", "top_name_score", "refs_found"]
RATES = ["text_empty"]


def psi(r, c, eps=1e-4):
    r, c = np.clip(r, eps, None), np.clip(c, eps, None)
    return float(np.sum((c - r) * np.log(c / r)))


def psi_numeric(ref, cur, bins=10):
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:   # (near-)constant column: compare value shares instead
        return psi_categorical(pd.Series(ref).round(2), pd.Series(cur).round(2))
    edges[0], edges[-1] = -np.inf, np.inf
    return psi(np.histogram(ref, edges)[0] / len(ref), np.histogram(cur, edges)[0] / len(cur))


def psi_categorical(ref, cur):
    cats = sorted(set(ref) | set(cur))
    return psi(ref.value_counts(normalize=True).reindex(cats, fill_value=0).values,
               cur.value_counts(normalize=True).reindex(cats, fill_value=0).values)


def drift_table(ref, cur):
    rows = []
    for col in NUMERIC:
        rows.append({"signal": col, "psi": round(psi_numeric(ref[col].values, cur[col].values), 3),
                     "ks_pvalue": round(float(ks_2samp(ref[col], cur[col]).pvalue), 4),
                     "reference": round(float(ref[col].mean()), 3), "current": round(float(cur[col].mean()), 3)})
    for col in RATES + ["decision"]:
        rows.append({"signal": col, "psi": round(psi_categorical(ref[col], cur[col]), 3), "ks_pvalue": np.nan,
                     "reference": round(float(ref[col].mean()), 3) if col != "decision" else np.nan,
                     "current": round(float(cur[col].mean()), 3) if col != "decision" else np.nan})
    df = pd.DataFrame(rows)
    df["status"] = pd.cut(df.psi, [-np.inf, 0.10, 0.25, np.inf], labels=["ok", "watch", "DRIFT"]).astype(str)
    return df


def decision_mix(ref, cur):
    order = ["AUTO_MATCH", "REVIEW", "UNAPPLIED"]
    return pd.DataFrame({"reference": ref.decision.value_counts(normalize=True).reindex(order, fill_value=0),
                         "current": cur.decision.value_counts(normalize=True).reindex(order, fill_value=0)}).round(4)


def feedback_quality(fb):
    audited = fb[fb.model_decision == "AUTO_MATCH"]
    reviewed = fb[fb.model_decision == "REVIEW"]
    return {"audited_auto_matches": len(audited),
            "audited_precision": float(audited.agreed.mean()) if len(audited) else np.nan,
            "reviewed": len(reviewed),
            "review_top_suggestion_correct": float(reviewed.top_suggestion_correct.mean()) if len(reviewed)
            else np.nan}


def run(ref, cur, fb=None, max_auto_drop=0.05, target_precision=0.995, min_audited=30):
    alerts = []
    mix = decision_mix(ref, cur)
    drop = mix.at["AUTO_MATCH", "reference"] - mix.at["AUTO_MATCH", "current"]
    if drop > max_auto_drop:
        alerts.append(f"auto-match rate fell from {mix.at['AUTO_MATCH', 'reference']:.1%} to "
                      f"{mix.at['AUTO_MATCH', 'current']:.1%}")
    drift = drift_table(ref, cur)
    alerts += [f"{r.signal} drifted (PSI {r.psi})" for r in drift[drift.status == "DRIFT"].itertuples()]
    quality = feedback_quality(fb) if fb is not None and len(fb) else {}
    if quality.get("audited_auto_matches", 0) >= min_audited and quality["audited_precision"] < target_precision:
        alerts.append(f"audited auto-match precision {quality['audited_precision']:.2%} is below "
                      f"the {target_precision:.1%} target")
    return {"mix": mix, "drift": drift, "quality": quality, "alerts": alerts}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", default="monitoring/reference.csv")
    ap.add_argument("--current", default="logs/matches.csv")
    ap.add_argument("--feedback", default="logs/feedback.csv")
    ap.add_argument("--min-rows", type=int, default=200)
    args = ap.parse_args()
    ref, cur = pd.read_csv(args.reference), pd.read_csv(args.current)
    if len(cur) < args.min_rows:
        sys.exit(f"Only {len(cur)} payments logged; need at least {args.min_rows} for a fair comparison.")
    fb = pd.read_csv(args.feedback) if Path(args.feedback).exists() else None
    r = run(ref, cur, fb)
    print("Decision mix\n", r["mix"].to_string())
    print("\nDrift (PSI)\n", r["drift"].to_string(index=False))
    if r["quality"]:
        print("\nLabelled quality from analyst feedback\n", pd.Series(r["quality"]).round(4).to_string())
    print("\nALERTS:\n  " + "\n  ".join(r["alerts"]) if r["alerts"] else "\nNo alerts.")
    return 1 if r["alerts"] else 0


if __name__ == "__main__":
    sys.exit(main())
