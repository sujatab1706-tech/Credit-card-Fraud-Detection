"""FastAPI cash application service: send a bank payment, get back the invoice(s) it pays.

  POST /match           one payment  -> AUTO_MATCH / REVIEW / UNAPPLIED with invoices and suggestions
  POST /match_batch     up to 1,000 payments (a day's bank file), processed in order
  GET  /review_queue    payments waiting for an analyst, with ranked suggestions
  POST /feedback        an analyst's resolution (or an audit of an auto-match); posts the cash
  GET  /open_invoices   the open AR ledger for one customer
  GET  /stats           decision mix, latency, audited precision
  GET  /health

Run locally:
  uvicorn api.app:app --reload        then open http://127.0.0.1:8000/docs
"""
import csv
import json
import os
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import List, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from inference import Matcher  # noqa: E402

MODEL_PATH = os.getenv("MODEL_PATH", "models/match_model.joblib")
META_PATH = os.getenv("META_PATH", "models/model_meta.json")
CUSTOMERS_PATH = os.getenv("CUSTOMERS_PATH", "ledger/customers.csv")
INVOICES_PATH = os.getenv("INVOICES_PATH", "ledger/open_invoices.csv")
LOG_DIR = Path(os.getenv("LOG_DIR", "logs"))
MODEL_VERSION = os.getenv("MODEL_VERSION", "hgb-v1")


class Payment(BaseModel):
    payment_id: str = Field(min_length=1, max_length=64)
    payment_date: date
    amount: float = Field(gt=0, le=10_000_000)
    payer_name: str = Field(default="", max_length=200)
    remittance_text: str = Field(default="", max_length=1000)

    model_config = {"json_schema_extra": {"example": {
        "payment_id": "BANK-2025-11-03-0042", "payment_date": "2025-11-03", "amount": 4862.17,
        "payer_name": "ACME IND SUPP", "remittance_text": "RMT INV-104512 / 4519"}}}


class Feedback(BaseModel):
    payment_id: str
    invoices: List[str] = Field(default_factory=list, description="the correct invoice(s); empty = no invoice")
    partial: bool = False
    analyst: Optional[str] = None


MATCH_FIELDS = ["timestamp", "model_version", "payment_id", "payment_date", "amount", "decision", "confidence",
                "n_invoices", "reconciliation", "n_candidates", "top_name_score", "refs_found", "text_empty",
                "invoices", "latency_ms"]
FEEDBACK_FIELDS = ["timestamp", "payment_id", "model_decision", "model_invoices", "correct_invoices", "agreed",
                   "top_suggestion_correct", "analyst"]

app = FastAPI(title="Cash Application Matcher",
              description="Matches incoming payments to open invoices: candidate generation, a gradient-boosted "
                          "match scorer and an auto-match / review / unapplied decision layer.")
matcher = Matcher(MODEL_PATH, META_PATH, CUSTOMERS_PATH, INVOICES_PATH)
decisions: dict = {}   # payment_id -> last decision (in-memory; a real system keeps this in a database)


def append_csv(name, fields, rows):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    path = LOG_DIR / name
    new = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new:
            w.writeheader()
        w.writerows(rows)


def run(payment: Payment) -> dict:
    start = time.perf_counter()
    p = payment.model_dump()
    p["payment_date"] = str(p["payment_date"])
    d = matcher.match(p)
    ms = (time.perf_counter() - start) * 1000
    sig = d.pop("signals")
    d = {"payment_id": payment.payment_id, **d}
    decisions[payment.payment_id] = {**d, "amount": payment.amount, "payment_date": p["payment_date"],
                                     "payer_name": payment.payer_name, "remittance_text": payment.remittance_text}
    append_csv("matches.csv", MATCH_FIELDS, [{
        "timestamp": datetime.now(timezone.utc).isoformat(), "model_version": MODEL_VERSION,
        "payment_id": payment.payment_id, "payment_date": p["payment_date"], "amount": payment.amount,
        "decision": d["decision"], "confidence": d["confidence"], "n_invoices": len(d["invoices"]),
        "reconciliation": d["reconciliation"] or "", **sig, "invoices": "|".join(d["invoices"]),
        "latency_ms": round(ms, 2)}])
    return d


@app.get("/health")
def health():
    return {"status": "ok", "model_version": MODEL_VERSION, "auto_threshold": matcher.auto_threshold,
            "open_invoices": int(matcher.ledger.open_mask(pd.Timestamp.now()).sum())}


@app.post("/match")
def match(payment: Payment):
    return run(payment)


@app.post("/match_batch")
def match_batch(payments: List[Payment]):
    if not 1 <= len(payments) <= 1000:
        raise HTTPException(status_code=422, detail="send between 1 and 1000 payments")
    results = [run(p) for p in sorted(payments, key=lambda p: p.payment_date)]
    mix = pd.Series([r["decision"] for r in results]).value_counts().to_dict()
    return {"results": results, "summary": mix}


@app.get("/review_queue")
def review_queue(limit: int = 50):
    q = [d for d in decisions.values() if d["decision"] in ("REVIEW", "UNAPPLIED") and not d.get("resolved")]
    q.sort(key=lambda d: -d["amount"])   # biggest cash first
    return {"waiting": len(q), "items": q[:limit]}


@app.post("/feedback")
def feedback(fb: Feedback):
    d = decisions.get(fb.payment_id)
    if d is None:
        raise HTTPException(status_code=404, detail="unknown payment_id; match it first")
    agreed = sorted(fb.invoices) == sorted(d["invoices"])
    top_ok = bool(d["suggestions"]) and d["suggestions"][0]["invoice_number"] in fb.invoices
    # post the cash unless the model already auto-posted exactly these invoices
    if fb.invoices and not (d["decision"] == "AUTO_MATCH" and agreed):
        matcher.post(fb.invoices, d["amount"], d["payment_date"], partial=fb.partial)
    d["resolved"] = True
    append_csv("feedback.csv", FEEDBACK_FIELDS, [{
        "timestamp": datetime.now(timezone.utc).isoformat(), "payment_id": fb.payment_id,
        "model_decision": d["decision"], "model_invoices": "|".join(d["invoices"]),
        "correct_invoices": "|".join(fb.invoices), "agreed": int(agreed),
        "top_suggestion_correct": int(top_ok), "analyst": fb.analyst or ""}])
    return {"payment_id": fb.payment_id, "agreed_with_model": agreed, "posted": fb.invoices}


@app.get("/open_invoices/{customer_id}")
def open_invoices(customer_id: str):
    inv = matcher.ledger.open_invoices()
    inv = inv[inv.customer_id == customer_id]
    return json.loads(inv.to_json(orient="records", date_format="iso"))


@app.get("/stats")
def stats():
    path = LOG_DIR / "matches.csv"
    if not path.exists():
        return {"payments": 0}
    log = pd.read_csv(path)
    out = {"payments": len(log),
           "decision_mix": log.decision.value_counts(normalize=True).round(4).to_dict(),
           "p95_latency_ms": round(float(log.latency_ms.quantile(0.95)), 2)}
    fb_path = LOG_DIR / "feedback.csv"
    if fb_path.exists():
        fb = pd.read_csv(fb_path)
        audited = fb[fb.model_decision == "AUTO_MATCH"]
        reviewed = fb[fb.model_decision == "REVIEW"]
        out["audited_auto_matches"] = len(audited)
        out["audited_auto_match_precision"] = round(float(audited.agreed.mean()), 4) if len(audited) else None
        out["review_resolved"] = len(reviewed)
        out["review_top_suggestion_correct"] = (round(float(reviewed.top_suggestion_correct.mean()), 4)
                                                if len(reviewed) else None)
    return out
