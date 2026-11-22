"""FastAPI app: send a customer's details, get a churn probability back.

Run locally:
  uvicorn api.app:app --reload
Then open http://127.0.0.1:8000/docs to try it in the browser.
"""
import csv
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Literal

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from inference import Predictor  # noqa: E402

MODEL_PATH = os.getenv("MODEL_PATH", "models/churn_logreg.joblib")
META_PATH = os.getenv("META_PATH", "models/model_meta.json")
LOG_PATH = Path(os.getenv("LOG_PATH", "logs/predictions.csv"))
MODEL_VERSION = os.getenv("MODEL_VERSION", "logreg-v1")

YesNo = Literal["Yes", "No"]
Addon = Literal["Yes", "No", "No internet service"]


class Customer(BaseModel):
    gender: Literal["Male", "Female"]
    SeniorCitizen: Literal[0, 1]
    Partner: YesNo
    Dependents: YesNo
    tenure: int = Field(ge=0, le=100, description="months with the company")
    PhoneService: YesNo
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Addon
    OnlineBackup: Addon
    DeviceProtection: Addon
    TechSupport: Addon
    StreamingTV: Addon
    StreamingMovies: Addon
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: YesNo
    PaymentMethod: Literal["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
    MonthlyCharges: float = Field(gt=0, le=500)
    TotalCharges: float = Field(ge=0)

    model_config = {"json_schema_extra": {"example": {
        "gender": "Female", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No", "tenure": 3,
        "PhoneService": "Yes", "MultipleLines": "No", "InternetService": "Fiber optic", "OnlineSecurity": "No",
        "OnlineBackup": "No", "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "Yes",
        "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check", "MonthlyCharges": 89.5, "TotalCharges": 268.5}}}


LOG_FIELDS = ["timestamp", "model_version", "tenure", "MonthlyCharges", "TotalCharges", "Contract", "InternetService",
              "PaymentMethod", "churn_probability", "churn_flag", "latency_ms"]

app = FastAPI(title="Customer Churn Predictor",
              description="Logistic regression that scores how likely a telecom customer is to leave.")
predictor = Predictor(MODEL_PATH, META_PATH)


def log_rows(rows):
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    new = not LOG_PATH.exists()
    with LOG_PATH.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        if new:
            w.writeheader()
        w.writerows(rows)


def score(customers: List[Customer], explain: bool):
    start = time.perf_counter()
    df = pd.DataFrame([c.model_dump() for c in customers])
    res = predictor.predict(df)
    latency = (time.perf_counter() - start) * 1000 / len(df)
    now = datetime.now(timezone.utc).isoformat()
    out, logs = [], []
    for i, r in res.iterrows():
        p = float(r.churn_probability)
        item = {"churn_probability": round(p, 4), "churn_flag": bool(r.churn_flag),
                "risk_band": "high" if p >= 0.5 else "medium" if p >= predictor.threshold else "low"}
        if explain:
            item["top_reasons"] = predictor.explain(df.iloc[[i]])
        out.append(item)
        logs.append({"timestamp": now, "model_version": MODEL_VERSION, **{k: df.at[i, k] for k in LOG_FIELDS[2:8]},
                     "churn_probability": round(p, 4), "churn_flag": int(r.churn_flag),
                     "latency_ms": round(latency, 2)})
    log_rows(logs)
    return out


@app.get("/health")
def health():
    return {"status": "ok", "model_version": MODEL_VERSION, "threshold": predictor.threshold}


@app.post("/predict")
def predict(customer: Customer, explain: bool = True):
    return {**score([customer], explain)[0], "threshold": predictor.threshold, "model_version": MODEL_VERSION}


@app.post("/predict_batch")
def predict_batch(customers: List[Customer]):
    if not 1 <= len(customers) <= 1000:
        raise HTTPException(status_code=422, detail="send between 1 and 1000 customers")
    return {"predictions": score(customers, explain=False), "threshold": predictor.threshold}


@app.get("/stats")
def stats():
    if not LOG_PATH.exists():
        return {"predictions": 0}
    log = pd.read_csv(LOG_PATH)
    return {"predictions": len(log), "flag_rate": round(float(log.churn_flag.mean()), 4),
            "mean_probability": round(float(log.churn_probability.mean()), 4),
            "p95_latency_ms": round(float(log.latency_ms.quantile(0.95)), 2)}
