"""Text clean-up shared by candidate generation, features, the API and the tests."""
import re

from rapidfuzz import fuzz

LEGAL_SUFFIXES = {"LTD", "LLC", "INC", "CORP", "CO", "GMBH", "PTY", "LIMITED", "CORPORATION", "HOLDINGS", "HLDGS",
                  "PLC", "SA", "AG", "BV"}
# bank / ERP abbreviations -> full word, so "ACME IND SUPP" and "Acme Industrial Supplies" line up
EXPAND = {"IND": "INDUSTRIAL", "SUPP": "SUPPLIES", "LOGISTIC": "LOGISTICS", "MED": "MEDICAL", "DEV": "DEVICES",
          "ELEC": "ELECTRONICS", "PKG": "PACKAGING", "CONSTR": "CONSTRUCTION", "PHARMA": "PHARMACEUTICALS",
          "DIST": "DISTRIBUTION", "ENG": "ENGINEERING", "HOSP": "HOSPITALITY", "INTL": "INTERNATIONAL",
          "LABS": "LABORATORIES", "SVCS": "SERVICES", "SVC": "SERVICES", "MFG": "MANUFACTURING"}
NOISE = {"PAYPAL", "BILL", "COM", "ACH", "CREDIT", "WIRE", "TRANSFER", "PAYMENT", "PMT", "ACCT", "PAYROLL"}

INV_RE = re.compile(r"(?<!\d)(1\d{5})(?!\d)")          # full invoice digits, e.g. 104512 / INV-104512
FRAG_RE = re.compile(r"(?<![\d#-])(\d{4})(?!\d)")      # a bare 4-digit fragment, e.g. "...4512"
PO_RE = re.compile(r"PO\W{0,3}(\d{7})", re.I)


def normalize_name(name: str) -> str:
    """Upper-case, strip punctuation, drop legal suffixes and bank noise, expand abbreviations."""
    words = re.sub(r"[^A-Z0-9 ]", " ", str(name).upper()).split()
    words = [EXPAND.get(w, w) for w in words if w not in LEGAL_SUFFIXES and w not in NOISE]
    return " ".join(words)


def name_similarity(a: str, b: str) -> float:
    """0-100. Takes the best of token-set (word order, extra words) and partial (bank truncation)."""
    a, b = normalize_name(a), normalize_name(b)
    if not a or not b:
        return 0.0
    return max(fuzz.token_set_ratio(a, b), fuzz.partial_ratio(a, b) * 0.95)


def extract_refs(text: str) -> dict:
    """Pull invoice numbers, 4-digit fragments and PO numbers out of free remittance text."""
    text = str(text) if text == text and text is not None else ""
    pos = set(PO_RE.findall(text))
    no_po = PO_RE.sub(" ", text)
    full = set(INV_RE.findall(no_po))
    frags = set(FRAG_RE.findall(INV_RE.sub(" ", no_po)))
    return {"invoice_digits": full, "fragments": frags, "po_numbers": pos}


def invoice_digits(invoice_number: str) -> str:
    return re.sub(r"\D", "", str(invoice_number))
