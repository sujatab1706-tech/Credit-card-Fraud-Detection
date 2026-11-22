"""Generate a realistic, Telco-style customer churn dataset.

The columns copy the layout of the well-known IBM "Telco Customer Churn" sample
(customer demographics, services, contract, billing), so code written here also
works on the real file. The data is synthetic so the project is fully
reproducible and free of licence questions.

How churn is decided: every customer gets a churn probability from a hidden
logistic formula (month-to-month contracts, fibre internet, electronic cheque
payment and short tenure raise it; long contracts, tech support and online
security lower it). One interaction effect and some random noise are added on
purpose, so a plain logistic regression cannot be perfect.

The raw file is also a little messy, like real exports:
  * TotalCharges is text, with blanks for brand-new customers (tenure 0)
  * a few duplicated rows
  * inconsistent casing / spacing in PaymentMethod

Usage:
  python src/generate_data.py --n 7043 --out data/telco_churn.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def generate(n=7043, seed=42):
    rng = np.random.default_rng(seed)

    gender = rng.choice(["Male", "Female"], n)
    senior = rng.binomial(1, 0.16, n)
    partner = rng.choice(["Yes", "No"], n, p=[0.48, 0.52])
    dependents = np.where(partner == "Yes",
                          rng.choice(["Yes", "No"], n, p=[0.5, 0.5]),
                          rng.choice(["Yes", "No"], n, p=[0.1, 0.9]))

    contract = rng.choice(["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.21, 0.24])
    # Long contracts go with long tenure
    max_tenure = {"Month-to-month": 40, "One year": 60, "Two year": 72}
    tenure = np.array([int(min(72, rng.exponential(max_tenure[c] / 2.2))) if c == "Month-to-month"
                       else int(rng.integers(0, max_tenure[c] + 1) * 0.6 + rng.integers(0, 30))
                       for c in contract])
    tenure = np.clip(tenure, 0, 72)

    phone = rng.choice(["Yes", "No"], n, p=[0.9, 0.1])
    multiple = np.where(phone == "No", "No phone service", rng.choice(["Yes", "No"], n, p=[0.42, 0.58]))
    internet = rng.choice(["DSL", "Fiber optic", "No"], n, p=[0.34, 0.44, 0.22])

    def addon(p_yes):
        return np.where(internet == "No", "No internet service",
                        rng.choice(["Yes", "No"], n, p=[p_yes, 1 - p_yes]))

    security, backup, protection = addon(0.37), addon(0.44), addon(0.44)
    support, tv, movies = addon(0.37), addon(0.49), addon(0.50)

    paperless = rng.choice(["Yes", "No"], n, p=[0.59, 0.41])
    payment = rng.choice(["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
                         n, p=[0.34, 0.23, 0.22, 0.21])

    # Price follows the services a customer has
    monthly = (20 + (phone == "Yes") * 5 + (multiple == "Yes") * 5
               + np.select([internet == "DSL", internet == "Fiber optic"], [25, 50], 0)
               + sum((s == "Yes") * 5 for s in [security, backup, protection, support])
               + sum((s == "Yes") * 10 for s in [tv, movies])
               + rng.normal(0, 2.5, n))
    monthly = np.round(np.clip(monthly, 18.25, 118.75), 2)
    total = np.round(monthly * tenure * rng.uniform(0.95, 1.05, n), 2)

    # Hidden churn formula (log-odds)
    z = (-1.55
         + 1.25 * (contract == "Month-to-month") - 1.1 * (contract == "Two year")
         - 0.045 * tenure + 0.0007 * (tenure - 30) ** 2 / 10
         + 0.75 * (internet == "Fiber optic") - 0.55 * (internet == "No")
         + 0.55 * (payment == "Electronic check")
         + 0.35 * (paperless == "Yes")
         + 0.30 * senior
         - 0.45 * (security == "Yes") - 0.45 * (support == "Yes")
         - 0.15 * (backup == "Yes")
         - 0.20 * (dependents == "Yes")
         + 0.010 * (monthly - 65)
         + 0.45 * ((internet == "Fiber optic") & (support == "No") & (contract == "Month-to-month"))  # interaction
         + rng.normal(0, 0.6, n))  # things we never observe
    churn = rng.binomial(1, sigmoid(z))

    df = pd.DataFrame({
        "customerID": [f"{rng.integers(1000, 9999)}-{''.join(rng.choice(list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'), 5))}"
                       for _ in range(n)],
        "gender": gender, "SeniorCitizen": senior, "Partner": partner, "Dependents": dependents,
        "tenure": tenure, "PhoneService": phone, "MultipleLines": multiple, "InternetService": internet,
        "OnlineSecurity": security, "OnlineBackup": backup, "DeviceProtection": protection,
        "TechSupport": support, "StreamingTV": tv, "StreamingMovies": movies,
        "Contract": contract, "PaperlessBilling": paperless, "PaymentMethod": payment,
        "MonthlyCharges": monthly, "TotalCharges": total.astype(str),
        "Churn": np.where(churn == 1, "Yes", "No"),
    })

    # --- realistic mess ---
    df.loc[df.tenure == 0, "TotalCharges"] = " "               # blanks, like the IBM file
    idx = rng.choice(n, 40, replace=False)
    df.loc[idx, "PaymentMethod"] = df.loc[idx, "PaymentMethod"].str.lower() + " "
    df = pd.concat([df, df.sample(15, random_state=seed)], ignore_index=True)  # duplicates
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=7043)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data/telco_churn.csv")
    args = ap.parse_args()
    df = generate(args.n, args.seed)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df):,} rows to {args.out}  (churn rate {(df.Churn == 'Yes').mean():.1%})")


if __name__ == "__main__":
    main()
