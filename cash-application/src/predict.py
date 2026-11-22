"""Batch cash application: match a CSV of bank payments against the open-invoice ledger.

Input columns: payment_id, payment_date, amount, payer_name, remittance_text
Output: one row per payment with decision, invoices, confidence, reconciliation and the top suggestion.

Usage:
  python src/predict.py --payments bank_file.csv --out matched.csv
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from inference import Matcher  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--payments", required=True)
    ap.add_argument("--out", default="matched.csv")
    ap.add_argument("--model", default="models/match_model.joblib")
    ap.add_argument("--meta", default="models/model_meta.json")
    ap.add_argument("--customers", default="ledger/customers.csv")
    ap.add_argument("--invoices", default="ledger/open_invoices.csv")
    args = ap.parse_args()
    m = Matcher(args.model, args.meta, args.customers, args.invoices)
    pay = pd.read_csv(args.payments, keep_default_na=False).sort_values("payment_date")
    rows = []
    for p in pay.to_dict("records"):
        d = m.match(p)
        top = d["suggestions"][0] if d["suggestions"] else {}
        rows.append({"payment_id": p["payment_id"], "decision": d["decision"], "invoices": "|".join(d["invoices"]),
                     "confidence": d["confidence"], "reconciliation": d["reconciliation"],
                     "top_suggestion": top.get("invoice_number"), "top_suggestion_p": top.get("p_match")})
    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False)
    print(out.decision.value_counts().to_string())
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
