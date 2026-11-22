"""Build the monitoring reference: what inputs and scores looked like at training time.

The API logs the same columns for every live prediction, so monitor.py can
compare live traffic against this file.

Usage:
  python monitoring/make_reference.py --data data/telco_churn.csv --out monitoring/reference.csv
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from features import clean  # noqa: E402
from inference import Predictor  # noqa: E402

COLUMNS = ["tenure", "MonthlyCharges", "TotalCharges", "Contract", "InternetService", "PaymentMethod"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/telco_churn.csv")
    ap.add_argument("--out", default="monitoring/reference.csv")
    ap.add_argument("--n", type=int, default=2000)
    args = ap.parse_args()
    df = clean(pd.read_csv(args.data)).sample(args.n, random_state=0).reset_index(drop=True)
    scored = Predictor().predict(df)
    ref = pd.concat([df[COLUMNS], scored], axis=1)
    ref["churn_flag"] = ref["churn_flag"].astype(int)
    ref.to_csv(args.out, index=False)
    print(f"reference: {len(ref)} rows, flag rate {ref.churn_flag.mean():.1%} -> {args.out}")


if __name__ == "__main__":
    main()
