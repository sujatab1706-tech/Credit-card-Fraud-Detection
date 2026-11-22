"""Step 2 inputs: one feature row per (payment, candidate invoice) pair.

The same function is used for training, batch scoring and the API, so the
model always sees features computed the same way.
"""
from functools import lru_cache
from itertools import combinations

import numpy as np
import pandas as pd

from text import extract_refs, name_similarity

FEATURES = [
    # reference evidence
    "ref_exact", "ref_fragment", "ref_po", "refs_found", "refs_same_customer", "text_empty",
    # who paid
    "name_score", "name_gap",
    # how much
    "amt_ratio", "amt_abs_log_ratio", "amt_exact", "amt_fee_like", "amt_discount_like", "amt_partial_like",
    "subset_match", "subset_size", "amount_rank",
    # when
    "days_from_due", "days_since_issue", "age_rank",
    # context
    "n_candidates", "n_customer_open",
]

SUBSET_MAX_INVOICES = 14
SUBSET_MAX_SIZE = 5
SUBSET_TOL = 0.005


@lru_cache(maxsize=None)
def _combos(m, k):
    return np.array(list(combinations(range(m), k)), dtype=np.int16)


def subset_membership(amounts: np.ndarray, target: float):
    """For each invoice, the size of the smallest combination (2..5 invoices) containing it whose
    total matches the payment exactly or after a 2 % early-pay discount. 0 = in no such combination."""
    m = len(amounts)
    best = np.zeros(m, dtype=int)
    if m < 2 or m > SUBSET_MAX_INVOICES:
        return best
    for k in range(2, min(SUBSET_MAX_SIZE, m) + 1):
        c = _combos(m, k)
        sums = amounts[c].sum(axis=1)
        tol = max(1.0, SUBSET_TOL * target)
        hit = (np.abs(sums - target) <= tol) | (np.abs(sums * 0.98 - target) <= tol)
        for row in c[hit]:
            for i in row:
                if best[i] == 0:
                    best[i] = k
    return best


def pair_features(payment, cands: pd.DataFrame) -> pd.DataFrame:
    """Feature matrix for one payment and its candidate invoices (output of Ledger.candidates)."""
    if cands.empty:
        return pd.DataFrame(columns=FEATURES)
    amount = float(payment["amount"])
    date = pd.Timestamp(payment["payment_date"])
    text = payment.get("remittance_text", "")
    text = "" if text is None or text != text else str(text)
    refs = extract_refs(text)

    digits = cands.invoice_number.str.replace(r"\D", "", regex=True)
    f = pd.DataFrame(index=cands.index)
    f["ref_exact"] = digits.isin(refs["invoice_digits"]).astype(int)
    f["ref_fragment"] = digits.str[-4:].isin(refs["fragments"]).astype(int)
    f["ref_po"] = cands.po_number.astype(str).isin(refs["po_numbers"]).astype(int)
    hit = (f.ref_exact | f.ref_fragment | f.ref_po).astype(bool)
    f["refs_found"] = int(hit.sum())
    cust_hits = cands[hit].groupby("customer_id").size()
    f["refs_same_customer"] = cands.customer_id.map(cust_hits).fillna(0).astype(int) - hit.astype(int)
    f["text_empty"] = int(len(refs["invoice_digits"]) + len(refs["fragments"]) + len(refs["po_numbers"]) == 0)

    scores = {c: name_similarity(payment.get("payer_name", ""), n)
              for c, n in cands[["customer_id", "customer_name"]].drop_duplicates().itertuples(index=False)}
    f["name_score"] = cands.customer_id.map(scores).astype(float)
    f["name_gap"] = f.name_score - max(scores.values())

    open_amt = cands.open_amount.values.astype(float)
    ratio = amount / open_amt
    diff = open_amt - amount
    f["amt_ratio"] = np.clip(ratio, 0, 20)
    f["amt_abs_log_ratio"] = np.abs(np.log(np.clip(ratio, 1e-3, 1e3)))
    f["amt_exact"] = (np.abs(diff) <= 0.01).astype(int)
    f["amt_fee_like"] = ((diff > 0.01) & (diff <= 50)).astype(int)
    f["amt_discount_like"] = ((ratio >= 0.975) & (ratio <= 0.985)).astype(int)
    f["amt_partial_like"] = ((ratio > 0.2) & (ratio < 0.97)).astype(int)

    sub = np.zeros(len(cands), dtype=int)
    for _, ix in cands.groupby("customer_id").indices.items():
        sub[ix] = subset_membership(open_amt[ix], amount)
    f["subset_size"] = sub
    f["subset_match"] = (sub > 0).astype(int)
    f["amount_rank"] = pd.Series(np.abs(diff)).rank(method="min").values

    issue = pd.to_datetime(cands.issue_date)
    f["days_from_due"] = (date - pd.to_datetime(cands.due_date)).dt.days.astype(float)
    f["days_since_issue"] = (date - issue).dt.days.astype(float)
    f["age_rank"] = issue.groupby(cands.customer_id).rank(method="first").values  # 1 = oldest open invoice
    f["n_candidates"] = len(cands)
    f["n_customer_open"] = cands.customer_id.map(cands.customer_id.value_counts()).values
    return f[FEATURES]


def build_pairs(ledger, payments: pd.DataFrame, matches: pd.DataFrame | None = None) -> pd.DataFrame:
    """Candidates + features for many payments. With matches, adds the label `is_match`."""
    truth = set()
    if matches is not None:
        truth = set(zip(matches.payment_id, matches.invoice_number))
    out = []
    for p in payments.to_dict("records"):
        c = ledger.candidates(p)
        if c.empty:
            continue
        f = pair_features(p, c)
        block = pd.concat([c[["invoice_number", "customer_id", "open_amount", "source"]], f], axis=1)
        block.insert(0, "payment_id", p["payment_id"])
        block["payment_amount"] = float(p["amount"])
        out.append(block)
    pairs = pd.concat(out, ignore_index=True) if out else pd.DataFrame()
    if matches is not None and len(pairs):
        pairs["is_match"] = [int((a, b) in truth) for a, b in zip(pairs.payment_id, pairs.invoice_number)]
    return pairs
