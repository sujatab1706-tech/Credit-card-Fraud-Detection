import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


@pytest.fixture(scope="session")
def small_data():
    from generate_data import generate
    return generate(n_customers=40, seed=3)


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.chdir(ROOT)
    sys.modules.pop("api.app", None)
    sys.path.insert(0, str(ROOT))
    from fastapi.testclient import TestClient
    import api.app as app_module
    return TestClient(app_module.app)


@pytest.fixture(scope="session")
def ledger_invoice():
    """A real open invoice from the bundled ledger snapshot, with its customer's name."""
    import pandas as pd
    inv = pd.read_csv(ROOT / "ledger/open_invoices.csv", dtype={"po_number": str})
    cust = pd.read_csv(ROOT / "ledger/customers.csv")
    row = inv[(inv.issue_date < "2025-11-01") & (inv.open_amount == inv.amount)].iloc[0]
    name = cust.set_index("customer_id").at[row.customer_id, "customer_name"]
    return {**row.to_dict(), "customer_name": name}
