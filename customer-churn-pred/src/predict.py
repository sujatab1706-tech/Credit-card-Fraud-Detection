"""Batch scoring: add churn probability and flag to a CSV of customers.

Usage:
  python src/predict.py --input data/telco_churn.csv --out results/scored.csv
"""
import argparse

import pandas as pd

from inference import Predictor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", default="results/scored.csv")
    ap.add_argument("--model", default="models/churn_logreg.joblib")
    ap.add_argument("--meta", default="models/model_meta.json")
    args = ap.parse_args()
    df = pd.read_csv(args.input)
    scored = Predictor(args.model, args.meta).predict(df)
    out = pd.concat([df.reset_index(drop=True), scored], axis=1).sort_values("churn_probability", ascending=False)
    out.to_csv(args.out, index=False)
    print(f"Scored {len(out):,} customers, {scored.churn_flag.mean():.1%} flagged -> {args.out}")


if __name__ == "__main__":
    main()
