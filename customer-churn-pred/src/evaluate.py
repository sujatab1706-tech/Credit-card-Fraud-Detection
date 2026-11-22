"""Evaluate the saved churn model in depth. Run after train.py.

Produces in results/:
  roc_pr_curves.png        ROC and precision-recall curves for every model
  calibration.png          are predicted probabilities honest? (reliability diagram)
  threshold_tradeoff.png   precision / recall / profit as the threshold moves
  confusion_matrix.png     at the chosen threshold
  coefficients.png/.csv    what drives churn, as odds ratios
  lift_by_decile.csv/png   how concentrated churners are in the top-scored customers
  segment_metrics.csv      performance per customer segment (gender, senior, contract)

Usage:
  python src/evaluate.py --results results --models models
"""
import argparse
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.metrics import (ConfusionMatrixDisplay, average_precision_score, confusion_matrix,  # noqa: E402
                             precision_recall_curve, precision_score, recall_score, roc_auc_score, roc_curve)

FINAL = "p::logreg tuned (final)"


def plot_curves(preds, out):
    y = preds["churn"].values
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5))
    for col in [c for c in preds if c.startswith("p::")]:
        name, p = col[3:], preds[col].values
        if len(np.unique(p)) <= 2 and "majority" in name:
            continue
        fpr, tpr, _ = roc_curve(y, p)
        a1.plot(fpr, tpr, label=f"{name} (AUC {roc_auc_score(y, p):.3f})", lw=2 if col == FINAL else 1)
        pr, rc, _ = precision_recall_curve(y, p)
        a2.plot(rc, pr, label=f"{name} (AP {average_precision_score(y, p):.3f})", lw=2 if col == FINAL else 1)
    a1.plot([0, 1], [0, 1], "k--", lw=0.8, label="random")
    a2.axhline(y.mean(), color="k", ls="--", lw=0.8, label=f"random (churn rate {y.mean():.2f})")
    a1.set(xlabel="False positive rate", ylabel="True positive rate (recall)", title="ROC curve")
    a2.set(xlabel="Recall", ylabel="Precision", title="Precision-recall curve")
    for a in (a1, a2):
        a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "roc_pr_curves.png", dpi=120)
    plt.close(fig)


def calibration(preds, out, bins=10):
    """Reliability diagram: of customers given ~30% churn risk, did ~30% churn?"""
    y = preds["churn"].values
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5))
    rows = []
    for col in [c for c in preds if c.startswith("p::logreg")]:
        p = preds[col].values
        frac, mean_p = calibration_curve(y, p, n_bins=bins, strategy="quantile")
        a1.plot(mean_p, frac, "o-", label=col[3:])
        a2.hist(p, bins=30, histtype="step", label=col[3:])
        rows.append({"model": col[3:], "mean_predicted": round(p.mean(), 4), "actual_rate": round(y.mean(), 4),
                     "ece": round(float(np.mean(np.abs(frac - mean_p))), 4)})
    a1.plot([0, 1], [0, 1], "k--", lw=0.8, label="perfect")
    a1.set(xlabel="Mean predicted probability", ylabel="Observed churn rate", title="Calibration (10 quantile bins)")
    a2.set(xlabel="Predicted probability", ylabel="Customers", title="Score distribution")
    a1.legend(fontsize=8)
    a2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "calibration.png", dpi=120)
    plt.close(fig)
    return pd.DataFrame(rows)


def threshold_tradeoff(preds, out, chosen, offer_cost=50, churn_loss=500, save_rate=0.3):
    y, p = preds["churn"].values, preds[FINAL].values
    grid = np.arange(0.05, 0.95, 0.01)
    prec = [precision_score(y, p >= t, zero_division=np.nan) for t in grid]  # undefined when nobody is flagged
    rec = [recall_score(y, p >= t) for t in grid]
    profit = [((p >= t) & (y == 1)).sum() * save_rate * churn_loss - (p >= t).sum() * offer_cost for t in grid]
    fig, a1 = plt.subplots(figsize=(8, 5))
    a1.plot(grid, prec, label="precision")
    a1.plot(grid, rec, label="recall")
    a1.axvline(chosen, color="grey", ls=":", label=f"chosen threshold {chosen}")
    a1.set(xlabel="Threshold", ylabel="Score")
    a2 = a1.twinx()
    a2.plot(grid, profit, color="tab:green", label="campaign profit (\\$)")
    a2.set_ylabel("Expected profit on test set (\\$)")
    h1, l1 = a1.get_legend_handles_labels()
    h2, l2 = a2.get_legend_handles_labels()
    a1.legend(h1 + h2, l1 + l2, loc="lower left", fontsize=8)
    a1.set_title(f"Threshold trade-off (offer \\${offer_cost}, churner worth \\${churn_loss}, {save_rate:.0%} saved)")
    fig.tight_layout()
    fig.savefig(out / "threshold_tradeoff.png", dpi=120)
    plt.close(fig)
    no_model = (y == 1).sum() * save_rate * churn_loss - len(y) * offer_cost  # offer to everybody
    best = max(profit)
    return {"profit_at_threshold": float(profit[int(round((chosen - 0.05) / 0.01))]), "best_profit": float(best),
            "profit_offer_everyone": float(no_model)}


def confusion(preds, out, thr):
    y, pred = preds["churn"].values, (preds[FINAL].values >= thr).astype(int)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay(confusion_matrix(y, pred), display_labels=["stays", "churns"]).plot(ax=ax, colorbar=False)
    ax.set_title(f"Test set, threshold {thr}")
    fig.tight_layout()
    fig.savefig(out / "confusion_matrix.png", dpi=120)
    plt.close(fig)


def coefficients(model, out):
    """Each coefficient is the change in log-odds of churn for +1 unit of the feature.

    Numbers are log-transformed then standardised, so +1 means +1 standard
    deviation of log(value). One-hot features compare with the dropped
    reference level. exp(coef) is the odds ratio: 2.0 = twice the odds of churning.
    """
    names = model.named_steps["prep"].get_feature_names_out()
    coef = model.named_steps["clf"].coef_[0]
    df = pd.DataFrame({"feature": names, "coef": coef, "odds_ratio": np.exp(coef)})
    df["abs"] = df.coef.abs()
    df = df.sort_values("abs", ascending=False).drop(columns="abs")
    df.round(4).to_csv(out / "coefficients.csv", index=False)
    plot = df[df.coef != 0].sort_values("coef")
    fig, ax = plt.subplots(figsize=(8, max(4, 0.28 * len(plot))))
    ax.barh(plot.feature, plot.coef, color=np.where(plot.coef > 0, "tab:red", "tab:blue"))
    ax.axvline(0, color="k", lw=0.8)
    ax.set(xlabel="Coefficient (log-odds of churn)  -> red raises churn, blue lowers it",
           title=f"Logistic regression coefficients ({(df.coef == 0).sum()} zeroed out by L1)")
    fig.tight_layout()
    fig.savefig(out / "coefficients.png", dpi=120)
    plt.close(fig)
    return df


def lift_table(preds, out):
    """Sort customers by risk, cut into 10 equal groups. Top decile = the 10% the model is most worried about."""
    df = preds[["churn", FINAL]].rename(columns={FINAL: "p"}).sort_values("p", ascending=False).reset_index(drop=True)
    df["decile"] = pd.qcut(df.index, 10, labels=range(1, 11))
    t = df.groupby("decile", observed=True).agg(customers=("churn", "size"), churners=("churn", "sum"),
                                                mean_score=("p", "mean"))
    t["churn_rate"] = t.churners / t.customers
    t["lift"] = t.churn_rate / df.churn.mean()
    t["cumulative_recall"] = t.churners.cumsum() / df.churn.sum()
    t = t.round(3)
    t.to_csv(out / "lift_by_decile.csv")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(t.index.astype(str), t.lift)
    ax.axhline(1, color="k", ls="--", lw=0.8)
    ax.set(xlabel="Risk decile (1 = highest scores)", ylabel="Lift over average", title="Lift by decile")
    fig.tight_layout()
    fig.savefig(out / "lift_by_decile.png", dpi=120)
    plt.close(fig)
    return t


def segments(preds, thr):
    """Does the model work equally well for every group? Big gaps are a warning sign."""
    rows = []
    for col in ["gender", "SeniorCitizen", "Contract", "InternetService"]:
        for val, g in preds.groupby(col):
            y, p = g["churn"].values, g[FINAL].values
            rows.append({"segment": f"{col}={val}", "customers": len(g), "churn_rate": y.mean(),
                         "roc_auc": roc_auc_score(y, p) if len(np.unique(y)) > 1 else np.nan,
                         "recall": recall_score(y, p >= thr, zero_division=0),
                         "precision": precision_score(y, p >= thr, zero_division=0)})
    return pd.DataFrame(rows).round(3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--models", default="models")
    args = ap.parse_args()
    out, models = Path(args.results), Path(args.models)
    preds = pd.read_csv(out / "test_predictions.csv")
    model = joblib.load(models / "churn_logreg.joblib")
    thr = json.loads((models / "model_meta.json").read_text())["threshold"]

    plot_curves(preds, out)
    print("Calibration\n", calibration(preds, out).to_string(index=False), "\n")
    print("Profit\n", threshold_tradeoff(preds, out, thr), "\n")
    confusion(preds, out, thr)
    print("Top coefficients\n", coefficients(model, out).head(12).round(3).to_string(index=False), "\n")
    print("Lift\n", lift_table(preds, out).to_string(), "\n")
    seg = segments(preds, thr)
    seg.to_csv(out / "segment_metrics.csv", index=False)
    print("Segments\n", seg.to_string(index=False))


if __name__ == "__main__":
    main()
