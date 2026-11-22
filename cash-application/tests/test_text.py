from text import extract_refs, name_similarity, normalize_name


def test_normalize_drops_suffix_and_expands_abbreviations():
    assert normalize_name("Acme Industrial Supplies Ltd") == "ACME INDUSTRIAL SUPPLIES"
    assert normalize_name("ACME IND SUPP") == "ACME INDUSTRIAL SUPPLIES"
    assert normalize_name("PAYPAL *ACME") == "ACME"


def test_name_similarity_handles_bank_truncation_and_case():
    assert name_similarity("BLUE RIDGE LOGISTI", "Blue Ridge Logistics LLC") > 90
    assert name_similarity("Blue Ridge Logistics", "Summit Foods Inc") < 60


def test_extract_refs_finds_numbers_fragments_and_po():
    r = extract_refs("RMT INV-104512, inv #104777 / 4519 PO 4123456")
    assert r["invoice_digits"] == {"104512", "104777"}
    assert r["fragments"] == {"4519"}
    assert r["po_numbers"] == {"4123456"}


def test_extract_refs_handles_empty_and_missing():
    for t in ["", None, float("nan"), "THANK YOU"]:
        r = extract_refs(t)
        assert not r["invoice_digits"] and not r["fragments"] and not r["po_numbers"]
