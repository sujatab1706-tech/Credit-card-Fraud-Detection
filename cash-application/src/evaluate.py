"""Evaluation report for the trained matcher (run after train.py).

Writes to results/:
  decision_mix_by_type.png / metrics_by_payment_type.csv   where automation works and where it doesn't
  threshold_tradeoff.png                                    auto-match rate vs precision vs cost (validation)
  pr_curve.png                                              pair-level precision-recall, model vs simple scores
  calibration.png                                           are match probabilities honest?
  feature_importance.png / feature_importance.csv           permutation importance on test pairs
  auto_match_errors.csv                                     every wrong auto-match on test, for error analysis
  business_impact.csv                                       analyst hours and cost, rules vs ML

Usage:
  python src/evaluate.py --data data --models models --results results
"""
import argparse
import json
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.metrics import average_precision_score, precision_recall_curve  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from decide import MANUAL_COST, WRONG_COST  # noqa: E402
from features import FEATURES  # noqa: E402

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
DECISION_COLORS = {"AUTO_MATCH": BLUE, "REVIEW": ORANGE, "UNAPPLIED": "#9a9893"}
PAYMENTS_PER_YEAR = 50_000      # scale the test results to a mid-size AR team
MINUTES_PER_MANUAL = 6          # research + apply one payment by hand

plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.titlecolor": INK,
                     "axes.titlesize": 11, "legend.frameon": False, "figure.dpi": 110})


def decision_mix(dec, out):
    order = ["exact", "wire_fee", "discount", "deduction", "partial", "combined", "combined_short", "unapplied"]
    mix = dec.groupby("payment_type").decision.value_counts(normalize=True).unstack(fill_value=0)
    mix = mix.reindex([o for o in order if o in mix.index])[[c for c in DECISION_COLORS if c in mix.columns]]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    left = np.zeros(len(mix))
    for col in mix.columns:
        ax.barh(mix.index, mix[col], left=left, color=DECISION_COLORS[col], edgecolor="white", linewidth=2,
                label=col.replace("_", " ").title(), height=0.65)
        for y, (l, v) in enumerate(zip(left, mix[col])):
            if v >= 0.08:
                ax.text(l + v / 2, y, f"{v:.0%}", ha="center", va="center", color="white", fontsize=8.5)
        left += mix[col].values
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
    ax.grid(axis="y", visible=False)
    ax.set_title("What happens to each kind of payment (test set)", loc="left")
    ax.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.25))
    fig.tight_layout()
    fig.savefig(out / "decision_mix_by_type.png", bbox_inches="tight")
    plt.close(fig)

    g = dec.groupby("payment_type")
    table = pd.DataFrame({
        "payments": g.size(),
        "auto_match_rate": g.apply(lambda d: (d.decision == "AUTO_MATCH").mean()),
        "auto_match_precision": g.apply(lambda d: d[d.decision == "AUTO_MATCH"].exact_set_correct.mean()),
        "review_rate": g.apply(lambda d: (d.decision == "REVIEW").mean()),
        "unapplied_rate": g.apply(lambda d: (d.decision == "UNAPPLIED").mean()),
    }).reindex([o for o in order if o in mix.index]).round(3)
    table.to_csv(out / "metrics_by_payment_type.csv")
    return table


def threshold_plot(curve, chosen, target, out):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    ax = axes[0]
    ax.plot(curve.auto_match_rate, curve.auto_match_precision, color=BLUE, lw=2)
    c = curve[curve.threshold == chosen].iloc[0]
    ax.scatter([c.auto_match_rate], [c.auto_match_precision], s=60, color=BLUE, edgecolor="white", lw=2, zorder=3)
    ax.annotate(f"chosen t={chosen:.2f}\n{c.auto_match_rate:.1%} auto, {c.auto_match_precision:.2%} precise",
                (c.auto_match_rate, c.auto_match_precision), xytext=(-150, -40), textcoords="offset points",
                color=INK, fontsize=8.5, arrowprops={"arrowstyle": "-", "color": MUTED})
    ax.axhline(target, color=MUTED, ls="--", lw=1)
    ax.text(curve.auto_match_rate.min(), target, f" target {target:.1%}", va="bottom", color=MUTED, fontsize=8)
    ax.set_xlabel("auto-match rate (share of payments posted with no human touch)")
    ax.set_ylabel("auto-match precision")
    ax.set_title("Coverage vs correctness as the threshold rises", loc="left")
    ax = axes[1]
    ax.plot(curve.threshold, curve.expected_cost, color=ORANGE, lw=2)
    ax.axvline(chosen, color=MUTED, ls="--", lw=1)
    ax.set_xlabel("auto-match threshold")
    ax.set_ylabel("expected cost on validation (USD)")
    ax.set_title(f"Cost: \\${MANUAL_COST:.0f} per manual payment, \\${WRONG_COST:.0f} per wrong match", loc="left")
    fig.tight_layout()
    fig.savefig(out / "threshold_tradeoff.png", bbox_inches="tight")
    plt.close(fig)


def pr_and_calibration(pairs, out):
    y, p = pairs.is_match, pairs.p_match
    simple = {"model (gradient boosting)": (p, BLUE),
              "name similarity only": (pairs.name_score, ORANGE),
              "invoice number in text only": (pairs.ref_exact + 0.5 * pairs.ref_fragment + 0.01 * pairs.amt_exact,
                                              AQUA)}
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    for label, (s, col) in simple.items():
        pr, rc, _ = precision_recall_curve(y, s)
        ax.plot(rc, pr, color=col, lw=2, label=f"{label}  AP={average_precision_score(y, s):.3f}")
    ax.set_xlabel("recall (true payment-invoice links found)")
    ax.set_ylabel("precision")
    ax.set_ylim(0, 1.02)
    ax.set_title("Pair-level precision-recall (test candidates)", loc="left")
    ax.legend(loc="lower left", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(out / "pr_curve.png", bbox_inches="tight")
    plt.close(fig)

    frac, mean = calibration_curve(y, p, n_bins=10, strategy="quantile")
    fig, ax = plt.subplots(figsize=(4.5, 4.2))
    ax.plot([0, 1], [0, 1], color=MUTED, ls="--", lw=1)
    ax.plot(mean, frac, color=BLUE, lw=2, marker="o", ms=6, mec="white", mew=1.5)
    ax.set_xlabel("predicted match probability")
    ax.set_ylabel("observed match rate")
    ax.set_title("Calibration (test candidates)", loc="left")
    fig.tight_layout()
    fig.savefig(out / "calibration.png", bbox_inches="tight")
    plt.close(fig)


def importance(model, pairs, out):
    sample = pairs.sample(min(20000, len(pairs)), random_state=0)
    r = permutation_importance(model, sample[FEATURES], sample.is_match, scoring="average_precision",
                               n_repeats=3, random_state=0, n_jobs=1)
    imp = pd.DataFrame({"feature": FEATURES, "importance": r.importances_mean}).sort_values("importance")
    imp.sort_values("importance", ascending=False).round(4).to_csv(out / "feature_importance.csv", index=False)
    top = imp.tail(12)
    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.barh(top.feature, top.importance, color=BLUE, height=0.6)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("drop in average precision when the feature is shuffled")
    ax.set_title("What the model relies on", loc="left")
    fig.tight_layout()
    fig.savefig(out / "feature_importance.png", bbox_inches="tight")
    plt.close(fig)
    return imp.sort_values("importance", ascending=False)


def business_impact(test_metrics, out):
    rows = []
    for r in test_metrics.itertuples():
        manual = r.review_rate + r.unapplied_rate
        rows.append({"approach": r.model, "auto_match_rate": r.auto_match_rate,
                     "auto_match_precision": r.auto_match_precision,
                     "manual_payments_per_year": round(PAYMENTS_PER_YEAR * manual),
                     "analyst_hours_per_year": round(PAYMENTS_PER_YEAR * manual * MINUTES_PER_MANUAL / 60),
                     "wrong_auto_matches_per_year": round(PAYMENTS_PER_YEAR * r.auto_match_rate
                                                          * (1 - r.auto_match_precision))})
    t = pd.DataFrame(rows)
    t["hours_saved_vs_rules"] = t.analyst_hours_per_year.iloc[-1] - t.analyst_hours_per_year
    t.to_csv(out / "business_impact.csv", index=False)
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--models", default="models")
    ap.add_argument("--results", default="results")
    args = ap.parse_args()
    out, models = Path(args.results), Path(args.models)
    meta = json.loads((models / "model_meta.json").read_text())
    model = joblib.load(models / "match_model.joblib")

    dec = pd.read_csv(out / "test_decisions.csv")
    pairs = pd.read_csv(out / "test_pairs_scored.csv.gz")
    curve = pd.read_csv(out / "val_threshold_curve.csv")
    test_metrics = pd.read_csv(out / "test_metrics.csv")

    by_type = decision_mix(dec, out)
    print("By payment type\n", by_type.to_string())
    threshold_plot(curve, meta["auto_threshold"], meta["target_precision"], out)
    pr_and_calibration(pairs, out)
    imp = importance(model, pairs, out)
    print("\nTop features\n", imp.head(8).round(4).to_string(index=False))

    pay = pd.read_csv(Path(args.data) / "payments.csv", keep_default_na=False)
    errors = dec[(dec.decision == "AUTO_MATCH") & (~dec.exact_set_correct)].merge(
        pay[["payment_id", "payer_name", "remittance_text"]], on="payment_id")
    errors[["payment_id", "payment_type", "amount", "payer_name", "remittance_text", "invoices", "true_invoices",
            "confidence", "reconciliation"]].to_csv(out / "auto_match_errors.csv", index=False)
    print(f"\n{len(errors)} wrong auto-matches on test, see results/auto_match_errors.csv")
    print("\nBusiness impact (scaled to", f"{PAYMENTS_PER_YEAR:,} payments a year)\n",
          business_impact(test_metrics, out).to_string(index=False))


if __name__ == "__main__":
    main()
