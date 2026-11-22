import numpy as np
import pandas as pd

from candidates import Ledger
from decide import amounts_reconcile, decide_one
from features import FEATURES, build_pairs, subset_membership


def test_generated_data_is_consistent(small_data):
    cust, inv, pay, matches = small_data
    assert set(matches.invoice_number) <= set(inv.invoice_number)
    assert set(matches.payment_id) <= set(pay.payment_id)
    assert pay.payment_id.is_unique and inv.invoice_number.is_unique
    unapplied = pay[pay.payment_type == "unapplied"]
    assert not unapplied.payment_id.isin(matches.payment_id).any()


def test_candidates_recall_is_high(small_data):
    cust, inv, pay, matches = small_data
    ledger = Ledger.from_history(cust, inv, pay, matches)
    pairs = build_pairs(ledger, pay, matches)
    assert set(FEATURES) <= set(pairs.columns)
    assert pairs.is_match.sum() / len(matches) > 0.97


def test_subset_membership_finds_combined_payment():
    amounts = np.array([100.0, 250.0, 75.5, 990.0])
    sizes = subset_membership(amounts, 425.5)       # 100 + 250 + 75.5
    assert list(sizes) == [3, 3, 3, 0]
    assert list(subset_membership(amounts, 350.0 * 0.98)) == [2, 2, 0, 0]   # 2 % early-pay discount
    assert subset_membership(amounts, 5000.0).sum() == 0


def test_reconciliation_rules():
    assert amounts_reconcile(1000.0, [1000.0]) == "exact"
    assert amounts_reconcile(980.0, [1000.0]) == "fee"
    assert amounts_reconcile(9800.0, [5000.0, 5000.0]) == "discount"
    assert amounts_reconcile(400.0, [1000.0], single_exact_ref=True) == "partial"
    assert amounts_reconcile(400.0, [1000.0]) is None


def test_decision_layer():
    g = pd.DataFrame({"invoice_number": ["A", "B"], "open_amount": [500.0, 700.0], "payment_amount": 500.0,
                      "ref_exact": [1, 0], "p_match": [0.97, 0.02]})
    d = decide_one(g, auto_threshold=0.9, review_threshold=0.15)
    assert d["decision"] == "AUTO_MATCH" and d["invoices"] == ["A"]
    # confident, but the amount doesn't reconcile (a 30 % deduction) -> a person decides
    d = decide_one(g.assign(payment_amount=350.0, ref_exact=0), 0.9, 0.15)
    assert d["decision"] == "REVIEW"
    d = decide_one(g.assign(p_match=[0.05, 0.02]), 0.9, 0.15)
    assert d["decision"] == "UNAPPLIED"
