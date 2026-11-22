"""Freeze the AR ledger the API starts with: every invoice open on the cut-over date (with its
remaining balance after any part-payments) plus invoices issued after it. Matching posts cash
against this in memory, the way a cash application tool works through a day's bank file.

Usage:
  python src/make_ledger_snapshot.py --data data --out ledger --as-of 2025-11-01
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from candidates import Ledger  # noqa: E402
from train import load  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="ledger")
    ap.add_argument("--as-of", default="2025-11-01")
    args = ap.parse_args()
    cust, inv, pay, m = load(args.data)
    hist = Ledger.from_history(cust, inv, pay, m)
    as_of = pd.Timestamp(args.as_of)
    i = hist.inv
    keep = ((i.issue_date < as_of) & (i.closed_date.isna() | (i.closed_date >= as_of))) | (i.issue_date >= as_of)
    snap = i[keep].copy()
    snap["open_amount"] = hist.open_amounts(snap.index.values, as_of).round(2)
    snap = snap[["invoice_number", "customer_id", "po_number", "amount", "open_amount", "issue_date", "due_date"]]
    for c in ("issue_date", "due_date"):
        snap[c] = snap[c].dt.date
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cust.to_csv(out / "customers.csv", index=False)
    snap.to_csv(out / "open_invoices.csv", index=False)
    print(f"{len(snap)} open or upcoming invoices as of {as_of.date()} -> {out}/open_invoices.csv")


if __name__ == "__main__":
    main()
