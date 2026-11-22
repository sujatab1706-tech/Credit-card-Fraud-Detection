"""Training-time snapshot of what the matcher sees and decides, to compare live traffic against.

Uses the validation period (Sep-Oct 2025): the model never trained on it, so its decision mix is
what we expect in production. Columns match the API log (logs/matches.csv).

Usage:
  python monitoring/make_reference.py --data data --models models --results results
"""
import argparse
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from decide import decide  # noqa: E402
from train import load, split_of  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--models", default="models")
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="monitoring/reference.csv")
    args = ap.parse_args()
    meta = json.loads((Path(args.models) / "model_meta.json").read_text())
    model = joblib.load(Path(args.models) / "match_model.joblib")
    _, _, pay, _ = load(args.data)
    pay = pay[split_of(pay.payment_date) == "val"]
    pairs = pd.read_csv(Path(args.results) / "pairs.csv.gz")
    pairs = pairs[pairs.payment_id.isin(pay.payment_id)].copy()
    pairs["p_match"] = model.predict_proba(pairs[meta["features"]])[:, 1]

    dec = decide(pairs, pay.payment_id.tolist(), meta["auto_threshold"], meta["review_threshold"])
    sig = pairs.groupby("payment_id").agg(n_candidates=("invoice_number", "size"),
                                          top_name_score=("name_score", "max"),
                                          refs_found=("refs_found", "first"), text_empty=("text_empty", "first"))
    ref = (pay[["payment_id", "payment_date", "amount"]].merge(dec, on="payment_id")
           .merge(sig, left_on="payment_id", right_index=True, how="left")
           .fillna({"n_candidates": 0, "top_name_score": 0, "refs_found": 0, "text_empty": 1}))
    ref["n_invoices"] = ref.invoices.apply(len)
    cols = ["payment_id", "payment_date", "amount", "decision", "confidence", "n_invoices", "n_candidates",
            "top_name_score", "refs_found", "text_empty"]
    ref[cols].to_csv(args.out, index=False)
    print(f"{len(ref)} reference payments -> {args.out}")
    print(ref.decision.value_counts(normalize=True).round(3).to_string())


if __name__ == "__main__":
    main()
