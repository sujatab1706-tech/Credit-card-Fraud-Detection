"""Send fake customers to the running API so there is something to monitor.

--drift simulates a price rise plus a marketing push that brought in many new,
month-to-month, fibre customers: the kind of shift monitoring should catch.

Usage (API running on :8000):
  python monitoring/simulate_traffic.py --n 300            # normal traffic
  python monitoring/simulate_traffic.py --n 300 --drift    # shifted traffic
"""
import argparse
import sys
from pathlib import Path

import httpx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from features import RAW_INPUTS, clean  # noqa: E402
from generate_data import generate  # noqa: E402


def make_customers(n, drift=False, seed=1):
    df = clean(generate(n * 2, seed=seed)).drop(columns=["customerID", "Churn"]).head(n).copy()
    if drift:
        rng = np.random.default_rng(seed)
        new = rng.random(len(df)) < 0.5
        df.loc[new, ["Contract", "InternetService"]] = ["Month-to-month", "Fiber optic"]
        df.loc[new, "tenure"] = rng.integers(0, 6, new.sum())
        df["MonthlyCharges"] = (df["MonthlyCharges"] * 1.15).clip(upper=150).round(2)
        df["TotalCharges"] = (df["MonthlyCharges"] * df["tenure"]).round(2)
    return df[RAW_INPUTS].to_dict(orient="records")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--drift", action="store_true")
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    args = ap.parse_args()
    customers = make_customers(args.n, args.drift)
    r = httpx.post(f"{args.url}/predict_batch", json=customers, timeout=60)
    r.raise_for_status()
    flags = sum(p["churn_flag"] for p in r.json()["predictions"])
    print(f"sent {len(customers)} customers, {flags} flagged")


if __name__ == "__main__":
    main()
