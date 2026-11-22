"""Generate a realistic, messy accounts-receivable dataset for cash application.

Writes four CSVs to data/:
  customers.csv   customer master (id, legal name)
  invoices.csv    one row per invoice (number, customer, PO, amount, issue and due date)
  payments.csv    one row per incoming bank payment (amount, date, payer name, remittance text)
  matches.csv     ground truth: which invoice(s) each payment pays (empty for unapplied cash)

What makes it hard, like a real lockbox / bank statement feed:
  * payer names are bank-truncated, abbreviated, upper-cased, misspelt, or a third party pays
  * remittance text is free text: full invoice numbers, fragments, PO numbers, typos, or nothing
  * one payment can pay several invoices (combined), or one invoice can be paid in parts
  * amounts are short by early-pay discounts, deductions or wire fees
  * some cash has no invoice at all (prepayments, unknown payers)
  * look-alike customers ("Summit Foods Inc" vs "Summit Foods Distribution LLC")

Usage:
  python src/generate_data.py --out data
"""
import argparse
import string
from pathlib import Path

import numpy as np
import pandas as pd

PREFIXES = ["Acme", "Northwind", "Blue Ridge", "Summit", "Pioneer", "Evergreen", "Silverline", "Harbor", "Redwood",
            "Granite", "Lakeside", "Apex", "Crescent", "Golden Gate", "Ironwood", "Maple Leaf", "Oakridge", "Pacific",
            "Riverstone", "Sterling", "Trident", "Union", "Vertex", "Westfield", "Brightway", "Cascade", "Delta",
            "Eastgate", "Frontier", "Highland", "Keystone", "Liberty", "Meridian", "Nova", "Orion", "Prairie",
            "Quantum", "Sunrise", "Titan", "Valley"]
INDUSTRIES = ["Industrial Supplies", "Logistics", "Foods", "Medical Devices", "Electronics", "Packaging",
              "Construction", "Pharmaceuticals", "Auto Parts", "Hospitality Group", "Retail", "Chemicals",
              "Engineering", "Foods Distribution", "Office Solutions", "Energy Services", "Textiles", "Laboratories"]
SUFFIXES = ["Ltd", "LLC", "Inc", "Corp", "Co", "GmbH", "Pty Ltd", "Limited", "Corporation", "Holdings"]
ABBREV = {"Industrial": "IND", "Supplies": "SUPP", "Logistics": "LOGISTIC", "Medical": "MED", "Devices": "DEV",
          "Electronics": "ELEC", "Packaging": "PKG", "Construction": "CONSTR", "Pharmaceuticals": "PHARMA",
          "Distribution": "DIST", "Engineering": "ENG", "Hospitality": "HOSP", "Corporation": "CORP",
          "Limited": "LTD", "International": "INTL", "Laboratories": "LABS", "Services": "SVCS", "Holdings": "HLDGS"}
THIRD_PARTY = ["PAYPAL *{}", "BILL.COM {}", "AMEX SETTLEMENT", "WIRE TRANSFER", "ACH CREDIT", "{} PAYROLL ACCT"]

START = pd.Timestamp("2025-01-01")
DAYS = 365


def make_customers(rng, n):
    names = set()
    while len(names) < n:
        names.add(f"{rng.choice(PREFIXES)} {rng.choice(INDUSTRIES)} {rng.choice(SUFFIXES)}")
    names = sorted(names)
    rng.shuffle(names)
    cust = pd.DataFrame({"customer_id": [f"C{1000 + i}" for i in range(n)], "customer_name": names})
    # behaviour: average days late vs due date, chance to batch invoices into one payment, payment terms
    cust["avg_days_late"] = rng.gamma(2.0, 4.0, n).round(1) - 3
    cust["batch_prob"] = rng.choice([0.0, 0.0, 0.1, 0.5, 0.85], n)
    cust["terms_days"] = rng.choice([30, 30, 45, 60], n)
    cust["size"] = rng.lognormal(0, 0.8, n)  # drives invoice frequency and size
    return cust


def typo(rng, s):
    if len(s) < 5:
        return s
    i = rng.integers(1, len(s) - 2)
    kind = rng.integers(3)
    if kind == 0:   # swap neighbours
        return s[:i] + s[i + 1] + s[i] + s[i + 2:]
    if kind == 1:   # drop a char
        return s[:i] + s[i + 1:]
    return s[:i] + rng.choice(list(string.ascii_uppercase)) + s[i + 1:]


def payer_name(rng, name):
    """How the customer's name shows up on the bank statement."""
    words = name.split()
    r = rng.random()
    if r < 0.03:  # someone else pays, the name is useless
        return rng.choice(THIRD_PARTY).format(words[0].upper())
    if rng.random() < 0.5:  # drop legal suffix
        words = [w for w in words if w not in SUFFIXES and w not in ("Pty",)]
    if rng.random() < 0.4:
        words = [ABBREV.get(w, w) for w in words]
    s = " ".join(words)
    if rng.random() < 0.7:
        s = s.upper()
    if rng.random() < 0.15:
        s = typo(rng, s)
    if rng.random() < 0.35:  # bank field limit
        s = s[: rng.choice([16, 18, 22])].strip()
    return s


def ref_token(rng, inv_no, po):
    """One reference to an invoice the way a customer might type it."""
    digits = inv_no.split("-")[1]
    r = rng.random()
    if r < 0.30:
        return inv_no
    if r < 0.45:
        return f"INV{digits}"
    if r < 0.58:
        return digits
    if r < 0.66:
        return f"inv #{digits}"
    if r < 0.74:
        return f"PO {po}"
    if r < 0.84:  # fragment: last 4 digits
        return digits[-4:]
    # typo in the number (transposed digits)
    i = rng.integers(0, len(digits) - 1)
    d = digits[:i] + digits[i + 1] + digits[i] + digits[i + 2:]
    return f"INV {d}"


def remittance(rng, inv_rows):
    """Free-text remittance for a payment covering inv_rows."""
    if rng.random() < 0.18:  # nothing useful
        return rng.choice(["", "PAYMENT", "THANK YOU", "AP PAYMENT RUN", f"PMT {rng.integers(1, 13):02d}/2025",
                           "ACCOUNT SETTLEMENT", "SEE EMAIL"])
    refs = [ref_token(rng, r.invoice_number, r.po_number) for r in inv_rows]
    if len(refs) > 3 and rng.random() < 0.5:  # long lists get cut off by the bank
        refs = refs[:2] + ["..."]
    sep = rng.choice([" ", ", ", "/", "; "])
    body = sep.join(refs)
    prefix = rng.choice(["", "", "RMT ", "PAYMENT ", "REF ", "INVOICES ", "/RFB/", "Pmt for "])
    return (prefix + body)[: rng.choice([60, 80, 140])]


def generate(n_customers=320, seed=42):
    rng = np.random.default_rng(seed)
    cust = make_customers(rng, n_customers)

    # ---- invoices ---------------------------------------------------------
    inv = []
    for c in cust.itertuples():
        n = max(3, rng.poisson(28 * c.size))
        days = np.sort(rng.integers(0, DAYS - 20, n))
        for d in days:
            amt = round(float(rng.lognormal(np.log(2500 * c.size ** 0.5), 0.9)), 2)
            if rng.random() < 0.15:  # round, repeating amounts (subscriptions) make amount matching ambiguous
                amt = float(rng.choice([500, 1000, 1500, 2500, 5000]))
            inv.append({"customer_id": c.customer_id, "issue_date": START + pd.Timedelta(days=int(d)),
                        "terms": c.terms_days, "amount": max(amt, 25.0)})
    inv = pd.DataFrame(inv).sort_values("issue_date").reset_index(drop=True)
    inv["invoice_number"] = [f"INV-{100000 + i}" for i in range(len(inv))]
    inv["po_number"] = rng.integers(4_000_000, 4_999_999, len(inv)).astype(str)
    inv["due_date"] = inv.issue_date + pd.to_timedelta(inv.terms, unit="D")

    # ---- payments -----------------------------------------------------------
    cmap = cust.set_index("customer_id")
    pays, matches = [], []

    def add_payment(cid, rows, amount, date, kind, text=None, name=None):
        pid = f"P{len(pays) + 1:06d}"
        pays.append({"payment_id": pid, "payment_date": date, "amount": round(amount, 2),
                     "payer_name": name or payer_name(rng, cmap.at[cid, "customer_name"]),
                     "remittance_text": remittance(rng, rows) if text is None else text,
                     "payment_type": kind, "true_customer_id": cid})
        matches.extend({"payment_id": pid, "invoice_number": r.invoice_number} for r in rows)

    for cid, g in inv.groupby("customer_id"):
        c = cmap.loc[cid]
        rows = list(g.itertuples())
        i = 0
        while i < len(rows):
            r = rows[i]
            if rng.random() < 0.04:  # never paid (stays open, a distractor)
                i += 1
                continue
            lateness = int(round(rng.normal(c.avg_days_late, 6)))
            pay_date = r.due_date + pd.Timedelta(days=lateness)
            # batch: also pay later invoices issued within 30 days
            batch = [r]
            if rng.random() < c.batch_prob:
                k = int(rng.integers(2, 6))
                j = i + 1
                while j < len(rows) and len(batch) < k and (rows[j].issue_date - r.issue_date).days <= 30:
                    batch.append(rows[j])
                    j += 1
                pay_date = max(pay_date, max(b.issue_date for b in batch) + pd.Timedelta(days=3))
            i += len(batch)
            pay_date = max(pay_date, max(b.issue_date for b in batch) + pd.Timedelta(days=1))
            total = sum(b.amount for b in batch)

            u = rng.random()
            if len(batch) > 1:
                kind = "combined"
                if u < 0.15:
                    total *= 0.98   # early-pay discount on the lot
                    kind = "combined_short"
            elif u < 0.06:  # paid in two parts
                part = round(total * rng.uniform(0.3, 0.7), 2)
                add_payment(cid, batch, part, pay_date, "partial")
                add_payment(cid, batch, total - part, pay_date + pd.Timedelta(days=int(rng.integers(10, 40))),
                            "partial")
                continue
            elif u < 0.14:
                total -= float(rng.choice([15, 20, 25, 35, 45]))
                kind = "wire_fee"
            elif u < 0.20:
                total *= 0.98
                kind = "discount"
            elif u < 0.25:
                total *= 1 - rng.uniform(0.04, 0.2)
                kind = "deduction"
            else:
                kind = "exact"
            add_payment(cid, batch, total, pay_date, kind)

    # unapplied cash: prepayments / unknown payers with no invoice
    n_unapplied = int(0.04 * len(pays))
    for _ in range(n_unapplied):
        if rng.random() < 0.6:
            cid = rng.choice(cust.customer_id)
            name = payer_name(rng, cmap.at[cid, "customer_name"])
            text = rng.choice(["PREPAYMENT", "DEPOSIT", "ADVANCE PMT", "", "ORDER " + str(rng.integers(10000, 99999))])
        else:
            cid, name = "", f"{rng.choice(PREFIXES).upper()} {rng.choice(['TRADING', 'ENTERPRISES', 'PARTNERS'])}"
            text = rng.choice(["", "PAYMENT", "INV " + str(rng.integers(100000, 999999))])
        pid = f"P{len(pays) + 1:06d}"
        pays.append({"payment_id": pid, "payment_date": START + pd.Timedelta(days=int(rng.integers(20, DAYS + 60))),
                     "amount": round(float(rng.lognormal(np.log(3000), 1.0)), 2), "payer_name": name,
                     "remittance_text": text, "payment_type": "unapplied", "true_customer_id": cid})

    pays = pd.DataFrame(pays).sort_values(["payment_date", "payment_id"]).reset_index(drop=True)
    matches = pd.DataFrame(matches)
    # renumber so payment ids follow the bank statement order
    new_ids = {old: f"P{i + 1:06d}" for i, old in enumerate(pays.payment_id)}
    pays["payment_id"] = pays.payment_id.map(new_ids)
    matches["payment_id"] = matches.payment_id.map(new_ids)
    matches = matches.sort_values("payment_id").reset_index(drop=True)
    inv = inv[["invoice_number", "customer_id", "po_number", "amount", "issue_date", "due_date"]]
    return cust[["customer_id", "customer_name"]], inv, pays, matches


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--customers", type=int, default=320)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cust, inv, pays, matches = generate(args.customers, args.seed)
    cust.to_csv(out / "customers.csv", index=False)
    inv.to_csv(out / "invoices.csv", index=False)
    pays.to_csv(out / "payments.csv", index=False)
    matches.to_csv(out / "matches.csv", index=False)
    print(f"{len(cust)} customers, {len(inv)} invoices, {len(pays)} payments, {len(matches)} payment-invoice links")
    print(pays.payment_type.value_counts().to_string())


if __name__ == "__main__":
    main()
