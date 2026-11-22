"""Train baselines and a tuned logistic regression for churn.

Steps
  1. Clean the data and split it 60 / 20 / 20 into train / validation / test,
     stratified so each split keeps the same churn rate.
  2. Baselines: "nobody churns", "month-to-month customers churn", and a
     default logistic regression with no tuning.
  3. Tune logistic regression with 5-fold cross-validation on train:
       C (inverse regularisation strength), l1_ratio (0 = L2 ridge,
       0.5 = elastic net, 1 = L1 lasso, which can zero out features),
       class_weight None vs "balanced" (the imbalance handling).
  4. Pick a decision threshold on the validation set (not 0.5 by default).
  5. Score everything once on the untouched test set and save the model.

Usage:
  python src/train.py --data data/telco_churn.csv --out results --models models
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, f1_score, log_loss, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

from features import build_preprocessor, prepare

SEED = 42


def split(X, y):
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.4, stratify=y, random_state=SEED)
    X_val, X_te, y_val, y_te = train_test_split(X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=SEED)
    return X_tr, X_val, X_te, y_tr, y_val, y_te


def make_pipeline(**lr_kwargs):
    return Pipeline([("prep", build_preprocessor()),
                     ("clf", LogisticRegression(max_iter=5000, solver="saga", random_state=SEED, **lr_kwargs))])


def scores(y, prob, threshold=0.5):
    pred = (prob >= threshold).astype(int)
    out = {"roc_auc": roc_auc_score(y, prob), "pr_auc": average_precision_score(y, prob),
           "precision": precision_score(y, pred, zero_division=0), "recall": recall_score(y, pred),
           "f1": f1_score(y, pred), "flag_rate": pred.mean(), "threshold": threshold}
    if len(np.unique(prob)) > 2:  # probability metrics are meaningless for hard 0/1 rules
        out["log_loss"] = log_loss(y, np.clip(prob, 1e-6, 1 - 1e-6))
        out["brier"] = brier_score_loss(y, prob)
    return {k: round(float(v), 4) for k, v in out.items()}


def best_f1_threshold(y, prob):
    grid = np.arange(0.05, 0.95, 0.01)
    f1s = [f1_score(y, (prob >= t).astype(int)) for t in grid]
    return float(round(grid[int(np.argmax(f1s))], 2))


def cost_threshold(y, prob, offer_cost=50, churn_loss=500, save_rate=0.3):
    """Threshold that maximises expected profit of a retention campaign.

    Each flagged customer gets an offer costing `offer_cost`. A true churner who
    gets the offer stays with probability `save_rate`, saving `churn_loss`.
    """
    grid = np.arange(0.05, 0.95, 0.01)
    profit = [((prob >= t) & (y == 1)).sum() * save_rate * churn_loss - (prob >= t).sum() * offer_cost for t in grid]
    return float(round(grid[int(np.argmax(profit))], 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/telco_churn.csv")
    ap.add_argument("--out", default="results")
    ap.add_argument("--models", default="models")
    args = ap.parse_args()
    out, models = Path(args.out), Path(args.models)
    out.mkdir(parents=True, exist_ok=True)
    models.mkdir(parents=True, exist_ok=True)

    X, y = prepare(pd.read_csv(args.data))
    X_tr, X_val, X_te, y_tr, y_val, y_te = split(X, y)
    print(f"train {len(y_tr)}, val {len(y_val)}, test {len(y_te)}; churn rate {y.mean():.1%}")

    # ---- baselines ----
    majority = DummyClassifier(strategy="most_frequent").fit(X_tr, y_tr)
    default_lr = make_pipeline().fit(X_tr, y_tr)

    # ---- tuned logistic regression ----
    grid = GridSearchCV(
        make_pipeline(),
        param_grid={"clf__C": np.logspace(-3, 2, 11), "clf__l1_ratio": [0.0, 0.5, 1.0],
                    "clf__class_weight": [None, "balanced"]},
        scoring={"pr_auc": "average_precision", "roc_auc": "roc_auc", "neg_log_loss": "neg_log_loss"},
        refit="pr_auc", cv=StratifiedKFold(5, shuffle=True, random_state=SEED), n_jobs=-1)
    grid.fit(X_tr, y_tr)
    cv = pd.DataFrame(grid.cv_results_)
    cv_tidy = cv[["param_clf__C", "param_clf__l1_ratio", "param_clf__class_weight", "mean_test_pr_auc",
                  "std_test_pr_auc", "mean_test_roc_auc", "mean_test_neg_log_loss"]].copy()
    cv_tidy.columns = ["C", "l1_ratio", "class_weight", "cv_pr_auc", "cv_pr_auc_std", "cv_roc_auc", "cv_neg_log_loss"]
    cv_tidy["class_weight"] = cv_tidy["class_weight"].fillna("none")
    cv_tidy.sort_values("cv_pr_auc", ascending=False).to_csv(out / "cv_results.csv", index=False)
    print("best params:", grid.best_params_, f"CV PR-AUC {grid.best_score_:.4f}")

    # Same tuned settings but the other class_weight, to show what imbalance handling changes
    other_cw = None if grid.best_params_["clf__class_weight"] == "balanced" else "balanced"
    tuned = grid.best_estimator_
    contrast = make_pipeline(C=grid.best_params_["clf__C"], l1_ratio=grid.best_params_["clf__l1_ratio"],
                             class_weight=other_cw).fit(X_tr, y_tr)

    # ---- thresholds chosen on validation only ----
    p_val = tuned.predict_proba(X_val)[:, 1]
    thr = {"default": 0.5, "best_f1": best_f1_threshold(y_val, p_val), "max_profit": cost_threshold(y_val, p_val)}
    print("thresholds (val):", thr)

    # ---- final test scores ----
    month_rule = (X_te["Contract"] == "Month-to-month").astype(float).values
    probs = {
        "majority (nobody churns)": majority.predict_proba(X_te)[:, 1],
        "rule: month-to-month churns": month_rule,
        "logreg default": default_lr.predict_proba(X_te)[:, 1],
        f"logreg tuned, class_weight={other_cw or 'none'}": contrast.predict_proba(X_te)[:, 1],
        "logreg tuned (final)": tuned.predict_proba(X_te)[:, 1],
    }
    rows = []
    for name, p in probs.items():
        rows.append({"model": name, **scores(y_te, p)})
    p_final = probs["logreg tuned (final)"]
    rows.append({"model": "logreg tuned (final) @ best-F1 threshold", **scores(y_te, p_final, thr["best_f1"])})
    rows.append({"model": "logreg tuned (final) @ max-profit threshold", **scores(y_te, p_final, thr["max_profit"])})
    table = pd.DataFrame(rows)
    table.to_csv(out / "test_metrics.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(table.to_string(index=False))

    # ---- save ----
    preds = X_te.copy()
    preds["churn"] = y_te
    for name, p in probs.items():
        preds[f"p::{name}"] = p
    preds.to_csv(out / "test_predictions.csv", index=False)
    pd.DataFrame({"churn": y_val, "prob": p_val}).to_csv(out / "val_predictions.csv", index=False)

    joblib.dump(tuned, models / "churn_logreg.joblib")
    meta = {"threshold": thr["max_profit"], "thresholds": thr,
            "best_params": {k.replace("clf__", ""): (None if v is None else (float(v) if isinstance(v, float) else v))
                            for k, v in grid.best_params_.items()},
            "test": scores(y_te, p_final, thr["max_profit"]),
            "train_churn_rate": round(float(y_tr.mean()), 4)}
    (models / "model_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"saved {models/'churn_logreg.joblib'} and model_meta.json")


if __name__ == "__main__":
    main()
