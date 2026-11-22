"""Replay the test period (Nov 2025 onward) through the API, like daily bank files arriving.

  normal   payments exactly as generated
  shifted  a bank changes its file format: remittance text is cut to 8 characters and payer
           names to 12, the kind of silent upstream change that quietly kills match rates

Analysts are simulated too: every REVIEW / UNAPPLIED item is resolved with the true answer and
a random 10 % of auto-matches are audited, so monitoring has labels to work with.

Usage (in-process, no server needed):
  python monitoring/simulate_traffic.py --mode normal  --log-dir logs/normal
  python monitoring/simulate_traffic.py --mode shifted --log-dir logs/shifted
Or against a running server:  --url http://localhost:8000
"""
import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def make_client(url, log_dir):
    if url:
        def call(method, path, body=None):
            req = urllib.request.Request(url + path, method=method, data=json.dumps(body).encode() if body else None,
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req) as r:
                return json.loads(r.read())
        return call
    os.environ["LOG_DIR"] = str(log_dir)
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    from fastapi.testclient import TestClient
    import api.app as app_module
    client = TestClient(app_module.app)

    def call(method, path, body=None):
        r = client.request(method, path, json=body)
        r.raise_for_status()
        return r.json()
    return call


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["normal", "shifted"], default="normal")
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--n", type=int, default=800)
    ap.add_argument("--audit-rate", type=float, default=0.10)
    ap.add_argument("--url", default="")
    ap.add_argument("--log-dir", default=str(ROOT / "logs"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    pay = pd.read_csv(Path(args.data) / "payments.csv", keep_default_na=False)
    truth = pd.read_csv(Path(args.data) / "matches.csv").groupby("payment_id").invoice_number.apply(list)
    pay = pay[pay.payment_date >= "2025-11-01"].head(args.n).copy()
    if args.mode == "shifted":
        pay["remittance_text"] = pay.remittance_text.str[:8]
        pay["payer_name"] = pay.payer_name.str[:12]

    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    call = make_client(args.url, Path(args.log_dir).resolve())
    for day, g in pay.groupby("payment_date"):        # one bank file per day
        body = g[["payment_id", "payment_date", "amount", "payer_name", "remittance_text"]].to_dict("records")
        res = call("POST", "/match_batch", body)["results"]
        for r, p in zip(res, sorted(body, key=lambda b: b["payment_date"])):
            correct = truth.get(r["payment_id"], [])
            if r["decision"] != "AUTO_MATCH" or rng.random() < args.audit_rate:
                partial = pay.loc[pay.payment_id == r["payment_id"], "payment_type"].iat[0] == "partial"
                call("POST", "/feedback", {"payment_id": r["payment_id"], "invoices": correct, "partial": bool(partial),
                                           "analyst": "simulated"})
    print(f"{args.mode}: replayed {len(pay)} payments")
    print(json.dumps(call("GET", "/stats"), indent=2))


if __name__ == "__main__":
    main()
