"""Compare live predictions (the API log) with the training-time reference.

Nobody knows yet whether today's customers will churn (labels arrive weeks
later), so we can't measure accuracy live. Instead we watch:

  1. FLAG RATE: share of customers flagged as likely churners.
  2. DATA DRIFT per input, with the Population Stability Index (PSI):
        PSI < 0.10  no real change
        0.10-0.25   moderate shift, keep an eye on it
        > 0.25      large shift, investigate / consider retraining
     plus a Kolmogorov-Smirnov test p-value for numeric columns.
  3. PREDICTION DRIFT: PSI on the churn probability itself.

When labels do arrive, run src/evaluate.py on them to check real performance.

Usage:
  python monitoring/monitor.py --reference monitoring/reference.csv --current logs/predictions.csv
"""
import argparse
import sys

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

NUMERIC = ["tenure", "MonthlyCharges", "TotalCharges", "churn_probability"]
CATEGORICAL = ["Contract", "InternetService", "PaymentMethod"]


def psi_numeric(ref, cur, bins=10):
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(ref, edges)[0] / len(ref)
    c = np.histogram(cur, edges)[0] / len(cur)
    return psi(r, c)


def psi_categorical(ref, cur):
    cats = sorted(set(ref) | set(cur))
    r = ref.value_counts(normalize=True).reindex(cats, fill_value=0).values
    c = cur.value_counts(normalize=True).reindex(cats, fill_value=0).values
    return psi(r, c)


def psi(r, c, eps=1e-4):
    r, c = np.clip(r, eps, None), np.clip(c, eps, None)
    return float(np.sum((c - r) * np.log(c / r)))


def drift_table(ref, cur):
    rows = []
    for col in NUMERIC:
        p = psi_numeric(ref[col].values, cur[col].values)
        rows.append({"column": col, "psi": round(p, 3), "ks_pvalue": round(ks_2samp(ref[col], cur[col]).pvalue, 4),
                     "ref_mean": round(ref[col].mean(), 3), "cur_mean": round(cur[col].mean(), 3)})
    for col in CATEGORICAL:
        rows.append({"column": col, "psi": round(psi_categorical(ref[col], cur[col]), 3), "ks_pvalue": np.nan,
                     "ref_mean": np.nan, "cur_mean": np.nan})
    df = pd.DataFrame(rows)
    df["status"] = pd.cut(df.psi, [-np.inf, 0.10, 0.25, np.inf], labels=["ok", "watch", "DRIFT"])
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", default="monitoring/reference.csv")
    ap.add_argument("--current", default="logs/predictions.csv")
    ap.add_argument("--min-rows", type=int, default=100)
    ap.add_argument("--max-flag-change", type=float, default=0.10)
    args = ap.parse_args()

    ref, cur = pd.read_csv(args.reference), pd.read_csv(args.current)
    if len(cur) < args.min_rows:
        sys.exit(f"Only {len(cur)} predictions logged; need at least {args.min_rows} for a fair comparison.")

    alerts = []
    r, c = ref.churn_flag.mean(), cur.churn_flag.mean()
    print(f"Flag rate: reference {r:.1%}, current {c:.1%}")
    if abs(c - r) > args.max_flag_change:
        alerts.append(f"flag rate moved from {r:.0%} to {c:.0%}")

    drift = drift_table(ref, cur)
    print("\nDrift (PSI)\n", drift.to_string(index=False))
    alerts += [f"{row.column} drifted (PSI {row.psi})" for row in drift[drift.status == "DRIFT"].itertuples()]
    print("\nALERTS:\n  " + "\n  ".join(alerts) if alerts else "\nNo alerts.")
    return 1 if alerts else 0


if __name__ == "__main__":
    sys.exit(main())
