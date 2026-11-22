import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


@pytest.fixture(scope="session")
def raw_df():
    from generate_data import generate
    return generate(n=600, seed=7)


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("LOG_PATH", str(tmp_path / "predictions.csv"))
    monkeypatch.chdir(ROOT)
    sys.modules.pop("api.app", None)
    sys.path.insert(0, str(ROOT))
    from fastapi.testclient import TestClient
    import api.app as app_module
    return TestClient(app_module.app)


@pytest.fixture()
def customer():
    return {"gender": "Female", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No", "tenure": 2,
            "PhoneService": "Yes", "MultipleLines": "No", "InternetService": "Fiber optic", "OnlineSecurity": "No",
            "OnlineBackup": "No", "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "Yes",
            "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check", "MonthlyCharges": 89.5, "TotalCharges": 179.0}
