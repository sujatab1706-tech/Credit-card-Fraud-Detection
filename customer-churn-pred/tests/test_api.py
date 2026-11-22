def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_high_risk_customer(client, customer):
    body = client.post("/predict", json=customer).json()
    assert 0 <= body["churn_probability"] <= 1
    assert body["churn_flag"] is True           # new, month-to-month, fibre, e-cheque: classic churner
    assert len(body["top_reasons"]) >= 1


def test_loyal_customer_is_low_risk(client, customer):
    loyal = {**customer, "tenure": 70, "Contract": "Two year", "TechSupport": "Yes", "OnlineSecurity": "Yes",
             "PaymentMethod": "Credit card (automatic)", "InternetService": "DSL", "TotalCharges": 6000.0}
    body = client.post("/predict", json=loyal).json()
    assert body["churn_flag"] is False and body["risk_band"] == "low"


def test_invalid_input_rejected(client, customer):
    assert client.post("/predict", json={**customer, "Contract": "Weekly"}).status_code == 422
    assert client.post("/predict", json={**customer, "tenure": -1}).status_code == 422


def test_batch_and_logging(client, customer):
    r = client.post("/predict_batch", json=[customer] * 5)
    assert r.status_code == 200 and len(r.json()["predictions"]) == 5
    stats = client.get("/stats").json()
    assert stats["predictions"] == 5 and 0 <= stats["flag_rate"] <= 1
