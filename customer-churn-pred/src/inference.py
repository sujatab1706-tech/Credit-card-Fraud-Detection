"""Load the saved pipeline and score customers. Used by the API and batch scoring."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from features import RAW_INPUTS, prepare


class Predictor:
    def __init__(self, model_path="models/churn_logreg.joblib", meta_path="models/model_meta.json"):
        self.model = joblib.load(model_path)
        meta = json.loads(Path(meta_path).read_text())
        self.threshold = float(meta["threshold"])
        prep, clf = self.model.named_steps["prep"], self.model.named_steps["clf"]
        self.feature_names = prep.get_feature_names_out()
        self.coef = clf.coef_[0]

    def predict(self, customers: pd.DataFrame) -> pd.DataFrame:
        X, _ = prepare(customers[RAW_INPUTS], dedupe=False)
        prob = self.model.predict_proba(X)[:, 1]
        return pd.DataFrame({"churn_probability": prob, "churn_flag": prob >= self.threshold})

    def explain(self, customer: pd.DataFrame, top=3):
        """Top reasons for one customer: coefficient x transformed value (its push on the log-odds)."""
        X, _ = prepare(customer[RAW_INPUTS], dedupe=False)
        contrib = self.model.named_steps["prep"].transform(X)[0] * self.coef
        order = np.argsort(-np.abs(contrib))[:top]
        return [{"feature": str(self.feature_names[i]), "effect": round(float(contrib[i]), 3),
                 "direction": "raises churn risk" if contrib[i] > 0 else "lowers churn risk"} for i in order
                if contrib[i] != 0]
