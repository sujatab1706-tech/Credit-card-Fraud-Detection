"""Step 3: turn pair scores into one decision per payment.

  AUTO_MATCH  post the cash against the chosen invoice(s) with no human touch
  REVIEW      send to the cash-app analyst queue with ranked suggestions
  UNAPPLIED   nothing plausible; park as unapplied cash / on-account and investigate

A payment is auto-matched only when the model is confident about every chosen
invoice AND the amounts reconcile (exact, a wire fee up to 50, a 2 % early-pay
discount, or a part-payment that names the invoice exactly). Everything else
goes to a person: precision matters more than coverage, because a wrong
auto-match is posted to the wrong customer and is expensive to unwind.
"""
import numpy as np
import pandas as pd

SELECT_P = 0.5


def amounts_reconcile(payment_amount, invoice_amounts, single_exact_ref=False):
    total = float(np.sum(invoice_amounts))
    diff = total - payment_amount
    if abs(diff) <= max(0.01, 0.0005 * payment_amount):
        return "exact"
    if 0 < diff <= 50:
        return "fee"
    if 0.975 <= payment_amount / total <= 0.985:
        return "discount"
    if single_exact_ref and len(invoice_amounts) == 1 and 0 < payment_amount < total:
        return "partial"
    return None


def assess_one(g: pd.DataFrame, top_k=3) -> dict:
    """Threshold-free summary of ONE payment's scored candidates (needs invoice_number, open_amount,
    payment_amount, ref_exact, p_match): what would be chosen, how confident, do the amounts reconcile."""
    g = g.sort_values("p_match", ascending=False)
    amount = float(g.payment_amount.iat[0])
    chosen = g[g.p_match >= SELECT_P]
    rec = None
    if len(chosen):
        rec = amounts_reconcile(amount, chosen.open_amount.values,
                                single_exact_ref=len(chosen) == 1 and bool(chosen.ref_exact.iat[0]))
    return {"invoices": list(chosen.invoice_number),
            "confidence": round(float(chosen.p_match.min() if len(chosen) else g.p_match.iat[0]), 4),
            "top_p": float(g.p_match.iat[0]), "reconciliation": rec,
            "suggestions": [{"invoice_number": r.invoice_number, "open_amount": round(float(r.open_amount), 2),
                             "p_match": round(float(r.p_match), 4)} for r in g.head(top_k).itertuples()]}


def assess(scored: pd.DataFrame, payment_ids) -> pd.DataFrame:
    groups = dict(tuple(scored.groupby("payment_id"))) if len(scored) else {}
    empty = {"invoices": [], "confidence": 0.0, "top_p": 0.0, "reconciliation": None, "suggestions": []}
    return pd.DataFrame([{"payment_id": pid, **(assess_one(groups[pid]) if pid in groups else empty)}
                         for pid in payment_ids])


def apply_thresholds(a: pd.DataFrame, auto_threshold: float, review_threshold: float) -> pd.DataFrame:
    d = a.copy()
    has = d.invoices.apply(len) > 0
    auto = has & (d.confidence >= auto_threshold) & d.reconciliation.notna()
    d["decision"] = np.where(auto, "AUTO_MATCH",
                             np.where(has | (d.top_p >= review_threshold), "REVIEW", "UNAPPLIED"))
    return d


def decide_one(g: pd.DataFrame, auto_threshold: float, review_threshold: float) -> dict:
    """Decision for one payment (used by the API)."""
    a = pd.DataFrame([assess_one(g)])
    return apply_thresholds(a, auto_threshold, review_threshold).drop(columns="top_p").iloc[0].to_dict()


def decide(scored: pd.DataFrame, payment_ids, auto_threshold: float, review_threshold: float) -> pd.DataFrame:
    """One row per payment id (payments with no candidates become UNAPPLIED)."""
    return apply_thresholds(assess(scored, payment_ids), auto_threshold, review_threshold)


def score_decisions(dec: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    """Add ground truth columns: is the predicted invoice set exactly right? Was the right invoice suggested?"""
    truth = matches.groupby("payment_id").invoice_number.apply(frozenset)
    d = dec.copy()
    d["true_invoices"] = d.payment_id.map(truth).apply(lambda s: s if isinstance(s, frozenset) else frozenset())
    d["exact_set_correct"] = [frozenset(a) == b for a, b in zip(d.invoices, d.true_invoices)]
    d["top1_correct"] = [bool(s) and s[0]["invoice_number"] in t for s, t in zip(d.suggestions, d.true_invoices)]
    d["has_invoice"] = d.true_invoices.apply(len) > 0
    return d


def summarize(d: pd.DataFrame) -> dict:
    """Business metrics from score_decisions output."""
    n = len(d)
    auto = d[d.decision == "AUTO_MATCH"]
    review = d[d.decision == "REVIEW"]
    unapplied = d[d.decision == "UNAPPLIED"]
    return {
        "payments": n,
        "auto_match_rate": round(len(auto) / n, 4),
        "auto_match_precision": round(float(auto.exact_set_correct.mean()) if len(auto) else 1.0, 4),
        "auto_matched_wrong": int((~auto.exact_set_correct).sum()),
        "review_rate": round(len(review) / n, 4),
        "review_top1_suggestion_correct": round(float(review[review.has_invoice].top1_correct.mean())
                                                if review.has_invoice.any() else np.nan, 4),
        "unapplied_rate": round(len(unapplied) / n, 4),
        "unapplied_truly_unmatched": round(float((~unapplied.has_invoice).mean()) if len(unapplied) else np.nan, 4),
        "matchable_payments_auto_matched": round(float(((d.decision == "AUTO_MATCH") & d.exact_set_correct
                                                        & d.has_invoice).sum() / d.has_invoice.sum()), 4),
    }


# rough unit costs used to pick the auto-match threshold (change to your own numbers)
MANUAL_COST = 5.0      # analyst minutes to research and apply one payment, in dollars
WRONG_COST = 150.0     # a wrong auto-match: reverse, re-apply, customer dispute, possible dunning error


def expected_cost(s: dict) -> float:
    manual = s["payments"] * (s["review_rate"] + s["unapplied_rate"])
    return manual * MANUAL_COST + s["auto_matched_wrong"] * WRONG_COST


def pick_auto_threshold(scored, payment_ids, matches, review_threshold, target_precision=0.995):
    """Auto-match threshold with the lowest expected cost, among those whose auto-match precision
    meets the target. Returns the threshold and the full curve for plotting."""
    grid = np.round(np.arange(0.50, 0.996, 0.01), 3)
    base = assess(scored, payment_ids)
    rows = []
    for t in grid:
        s = summarize(score_decisions(apply_thresholds(base, t, review_threshold), matches))
        rows.append({"threshold": t, **s, "expected_cost": expected_cost(s)})
    curve = pd.DataFrame(rows)
    ok = curve[curve.auto_match_precision >= target_precision]
    best = float(ok.sort_values(["expected_cost", "threshold"]).threshold.iat[0]) if len(ok) else float(grid[-1])
    return best, curve
