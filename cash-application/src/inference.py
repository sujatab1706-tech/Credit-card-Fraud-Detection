"""Load the trained model and an open-invoice ledger, and match payments. Used by the API and batch scoring."""
import json
import threading
from pathlib import Path

import joblib
import pandas as pd

from candidates import Ledger
from decide import decide_one
from features import FEATURES, pair_features


class Matcher:
    def __init__(self, model_path="models/match_model.joblib", meta_path="models/model_meta.json",
                 customers_path="ledger/customers.csv", invoices_path="ledger/open_invoices.csv"):
        self.model = joblib.load(model_path)
        meta = json.loads(Path(meta_path).read_text())
        self.auto_threshold = float(meta["auto_threshold"])
        self.review_threshold = float(meta["review_threshold"])
        self.features = meta.get("features", FEATURES)
        customers = pd.read_csv(customers_path)
        invoices = pd.read_csv(invoices_path, dtype={"po_number": str})
        self.ledger = Ledger(customers, invoices)
        self._lock = threading.Lock()   # the ledger changes as cash is posted

    def match(self, payment: dict, post: bool = True) -> dict:
        """payment: payment_id, payment_date, amount, payer_name, remittance_text.
        Returns the decision plus the signals worth logging for monitoring."""
        with self._lock:
            cands = self.ledger.candidates(payment)
            if cands.empty:
                return {"decision": "UNAPPLIED", "invoices": [], "confidence": 0.0, "reconciliation": None,
                        "suggestions": [], "signals": {"n_candidates": 0, "top_name_score": 0.0, "refs_found": 0,
                                                       "text_empty": 1}}
            f = pair_features(payment, cands)
            g = pd.concat([cands, f[[c for c in f.columns if c not in cands.columns]]], axis=1)
            g["payment_amount"] = float(payment["amount"])
            g["p_match"] = self.model.predict_proba(f[self.features])[:, 1]
            d = decide_one(g, self.auto_threshold, self.review_threshold)
            names = dict(zip(cands.invoice_number, cands.customer_name))
            for s in d["suggestions"]:
                s["customer_name"] = names.get(s["invoice_number"])
            d["customer_name"] = names.get(d["invoices"][0]) if d["invoices"] else None
            d["signals"] = {"n_candidates": int(len(cands)), "top_name_score": round(float(f.name_score.max()), 1),
                            "refs_found": int(f.refs_found.iat[0]), "text_empty": int(f.text_empty.iat[0])}
            if post and d["decision"] == "AUTO_MATCH":
                self.ledger.apply_payment(d["invoices"], payment["amount"], payment["payment_date"],
                                          partial=d["reconciliation"] == "partial")
            return d

    def post(self, invoice_numbers, amount, date, partial=False):
        """Apply a payment an analyst resolved by hand."""
        with self._lock:
            self.ledger.apply_payment(invoice_numbers, amount, date, partial=partial)
