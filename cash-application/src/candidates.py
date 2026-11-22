"""Step 1 of matching: for each payment, shortlist the open invoices it could be paying.

Scoring every payment against every open invoice is wasteful and noisy, so we
keep an invoice as a candidate if ANY of three cheap, high-recall signals fires:

  1. REFERENCE  the remittance mentions its invoice number, last 4 digits or PO number
  2. NAME       its customer is one of the top fuzzy matches for the payer name
  3. AMOUNT     its open amount equals the payment amount (any customer)

The model in step 2 then scores each (payment, candidate invoice) pair.
"""
import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process

from text import extract_refs, invoice_digits, normalize_name

NAME_TOP_K = 3
NAME_MIN_SCORE = 60
AMOUNT_TOL = 0.002   # 0.2 % (and at least 1.00)
MAX_AGE_DAYS = 240   # ignore invoices older than this when shortlisting by name


class Ledger:
    """Accounts-receivable ledger that can tell which invoices were open on a given date.

    invoices needs: invoice_number, customer_id, po_number, amount, issue_date, due_date
    optional:       closed_date (NaT = still open), open_amount (live ledgers),
                    partials: DataFrame(invoice_number, payment_date, amount) of earlier part-payments
    """

    def __init__(self, customers: pd.DataFrame, invoices: pd.DataFrame, partials: pd.DataFrame | None = None):
        inv = invoices.copy()
        for c in ("issue_date", "due_date", "closed_date"):
            if c in inv:
                inv[c] = pd.to_datetime(inv[c])
        if "closed_date" not in inv:
            inv["closed_date"] = pd.NaT
        if "open_amount" not in inv:
            inv["open_amount"] = inv["amount"]
        inv["po_number"] = inv["po_number"].astype(str)
        inv["digits"] = inv.invoice_number.map(invoice_digits)
        inv = inv.merge(customers[["customer_id", "customer_name"]], on="customer_id", how="left")
        self.inv = inv.reset_index(drop=True)
        self.customers = customers.reset_index(drop=True)
        self.cust_norm = [normalize_name(n) for n in self.customers.customer_name]
        self._issue = self.inv.issue_date.values
        self._closed = self.inv.closed_date.fillna(pd.Timestamp("2262-01-01")).values.astype("datetime64[s]")
        self._amount = self.inv.open_amount.values.astype(float).copy()
        self._by_cust = {c: np.array(ix) for c, ix in self.inv.groupby("customer_id").indices.items()}
        self._by_digits = {d: i for i, d in enumerate(self.inv.digits)}
        self._by_po = self.inv.groupby("po_number").indices
        self.partials = None
        if partials is not None and len(partials):
            p = partials.copy()
            p["payment_date"] = pd.to_datetime(p.payment_date)
            self.partials = p.groupby("invoice_number")

    @classmethod
    def from_history(cls, customers, invoices, payments, matches):
        """Build a historical ledger: an invoice closes on the date of the last payment applied to it."""
        m = matches.merge(payments[["payment_id", "payment_date", "amount"]], on="payment_id")
        m["payment_date"] = pd.to_datetime(m.payment_date)
        closed = m.groupby("invoice_number").payment_date.max().rename("closed_date")
        inv = invoices.drop(columns=[c for c in ("closed_date",) if c in invoices]).merge(
            closed, left_on="invoice_number", right_index=True, how="left")
        # part-payments: payments that pay a single invoice which is paid by more than one payment
        per_pay = m.groupby("payment_id").invoice_number.transform("count")
        per_inv = m.groupby("invoice_number").payment_id.transform("count")
        partials = m[(per_pay == 1) & (per_inv > 1)][["invoice_number", "payment_date", "amount"]]
        return cls(customers, inv, partials)

    def open_mask(self, date) -> np.ndarray:
        d = np.datetime64(pd.Timestamp(date))
        return (self._issue <= d) & (self._closed >= d) & (self._amount > 0.005)

    def open_amounts(self, idx, date) -> np.ndarray:
        """Open balance of invoices idx on date (amount minus earlier part-payments)."""
        amt = self._amount[idx].copy()
        if self.partials is None:
            return amt
        d = pd.Timestamp(date)
        for k, i in enumerate(idx):
            num = self.inv.invoice_number.iat[i]
            if num in self.partials.groups:
                g = self.partials.get_group(num)
                amt[k] -= g.loc[g.payment_date < d, "amount"].sum()
        return amt

    def apply_payment(self, invoice_numbers, amount, date, partial=False):
        """Post cash: close the invoices (or reduce the balance for a part-payment) from `date` on,
        so later payments no longer see them as open. Used by the live API."""
        d = np.datetime64(pd.Timestamp(date))
        idx = [self._by_digits[invoice_digits(n)] for n in invoice_numbers if invoice_digits(n) in self._by_digits]
        if partial and len(idx) == 1:
            self._amount[idx[0]] = max(0.0, self._amount[idx[0]] - float(amount))
            if self._amount[idx[0]] > 0.005:
                return
        for i in idx:
            self._closed[i] = d - np.timedelta64(1, "s")

    def open_invoices(self, date=None) -> pd.DataFrame:
        date = date or pd.Timestamp.now()
        mask = self.open_mask(date)
        out = self.inv.loc[mask, ["invoice_number", "customer_id", "customer_name", "issue_date", "due_date"]].copy()
        out["open_amount"] = self._amount[mask]
        return out

    def name_matches(self, payer_name):
        """Top fuzzy-matched customers: list of (customer_id, score)."""
        q = normalize_name(payer_name)
        if not q:
            return []
        hits = process.extract(q, self.cust_norm, scorer=_name_scorer, limit=NAME_TOP_K, score_cutoff=NAME_MIN_SCORE)
        return [(self.customers.customer_id.iat[i], float(s)) for _, s, i in hits]

    def candidates(self, payment) -> pd.DataFrame:
        """Candidate invoices for one payment (a dict/Series with amount, payment_date, payer_name, remittance_text)."""
        date = pd.Timestamp(payment["payment_date"])
        amount = float(payment["amount"])
        open_ = self.open_mask(date)
        refs = extract_refs(payment.get("remittance_text", ""))
        src = {}

        def add(ix, how):
            for i in np.atleast_1d(ix):
                if open_[i]:
                    src.setdefault(int(i), set()).add(how)

        for d in refs["invoice_digits"]:
            if d in self._by_digits:
                add(self._by_digits[d], "ref")
        for po in refs["po_numbers"]:
            if po in self._by_po:
                add(self._by_po[po], "ref")
        if refs["fragments"]:
            frag_hits = np.flatnonzero(open_ & self.inv.digits.str[-4:].isin(refs["fragments"]).values)
            add(frag_hits, "ref")

        name_hits = self.name_matches(payment.get("payer_name", ""))
        min_issue = np.datetime64(date - pd.Timedelta(days=MAX_AGE_DAYS))
        for cid, _ in name_hits:
            ix = self._by_cust.get(cid, np.array([], int))
            add(ix[self._issue[ix] >= min_issue], "name")

        tol = max(1.0, AMOUNT_TOL * amount)
        add(np.flatnonzero(open_ & (np.abs(self._amount - amount) <= tol)), "amount")

        if not src:
            return pd.DataFrame()
        idx = np.array(sorted(src))
        out = self.inv.iloc[idx][["invoice_number", "customer_id", "customer_name", "po_number", "amount",
                                  "issue_date", "due_date"]].copy()
        out["open_amount"] = self.open_amounts(idx, date)
        out["issue_date"], out["due_date"] = out.issue_date.dt.date.astype(str), out.due_date.dt.date.astype(str)
        out["source"] = ["+".join(sorted(src[i])) for i in idx]
        out = out[out.open_amount > 0.005]
        return out.reset_index(drop=True)


def _name_scorer(a, b, **kw):
    return max(fuzz.token_set_ratio(a, b), fuzz.partial_ratio(a, b) * 0.95)
