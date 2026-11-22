def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_exact_reference_payment_is_auto_matched_and_posted(client, ledger_invoice):
    inv = ledger_invoice
    pay = {"payment_id": "T1", "payment_date": "2025-11-05", "amount": inv["amount"],
           "payer_name": inv["customer_name"].upper(), "remittance_text": f"RMT {inv['invoice_number']}"}
    r = client.post("/match", json=pay).json()
    assert r["decision"] == "AUTO_MATCH"
    assert r["invoices"] == [inv["invoice_number"]]
    # the invoice is now closed, so paying it again must not auto-match it
    again = client.post("/match", json={**pay, "payment_id": "T2", "payment_date": "2025-11-06"}).json()
    assert inv["invoice_number"] not in again["invoices"]


def test_unknown_payer_goes_to_review_or_unapplied(client):
    pay = {"payment_id": "T3", "payment_date": "2025-11-05", "amount": 123456.78,
           "payer_name": "ZZQX HOLDINGS", "remittance_text": "DEPOSIT"}
    assert client.post("/match", json=pay).json()["decision"] in ("REVIEW", "UNAPPLIED")


def test_feedback_queue_and_stats(client):
    pay = {"payment_id": "T4", "payment_date": "2025-11-05", "amount": 77.0, "payer_name": "", "remittance_text": ""}
    client.post("/match", json=pay)
    assert client.get("/review_queue").json()["waiting"] >= 1
    r = client.post("/feedback", json={"payment_id": "T4", "invoices": []})
    assert r.status_code == 200
    s = client.get("/stats").json()
    assert s["payments"] == 1


def test_validation(client):
    assert client.post("/match", json={"payment_id": "x", "payment_date": "2025-11-05", "amount": -5}).status_code == 422
    assert client.post("/feedback", json={"payment_id": "never-seen", "invoices": []}).status_code == 404
