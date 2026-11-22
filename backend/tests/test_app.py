"""End-to-end API tests.  Run from backend/:  python -m pytest -q"""
import os
import tempfile

_tmp = tempfile.mkdtemp()
os.environ["DATABASE_URL"] = f"sqlite:///{_tmp}/test.db"
os.environ.setdefault("SECRET_KEY", "test-secret-key-for-pytest-only-0123456789")

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.auth import hash_password, verify_password  # noqa: E402
from app.database import SessionLocal  # noqa: E402
from app.main import app  # noqa: E402
from app.models import User  # noqa: E402
from app.services.risk import inventory_risk, supplier_risk  # noqa: E402

PW = "Secure123"


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def auth(client):
    r = client.post("/api/auth/register", json={"full_name": "Test User", "email": "Tester@Example.com",
                                                "password": PW, "confirm_password": PW})
    assert r.status_code == 201, r.text
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


# ---------------- Authentication ----------------
def test_register_validation(client):
    base = {"full_name": "A User", "email": "a@example.com", "password": PW, "confirm_password": PW}
    assert client.post("/api/auth/register", json={**base, "email": "not-an-email"}).status_code == 422
    r = client.post("/api/auth/register", json={**base, "confirm_password": "Other1234"})
    assert r.status_code == 422 and "do not match" in r.text
    assert client.post("/api/auth/register", json={**base, "password": "short1", "confirm_password": "short1"}).status_code == 422
    assert client.post("/api/auth/register", json={**base, "password": "lettersonly", "confirm_password": "lettersonly"}).status_code == 422


def test_duplicate_email_rejected(client, auth):
    r = client.post("/api/auth/register", json={"full_name": "Dup", "email": "tester@example.com",
                                                "password": PW, "confirm_password": PW})
    assert r.status_code == 409


def test_password_is_hashed(auth):
    with SessionLocal() as db:
        u = db.query(User).filter(User.email == "tester@example.com").one()
    assert u.password_hash != PW and PW not in u.password_hash
    assert u.password_hash.startswith("$2b$")
    assert verify_password(PW, u.password_hash) and not verify_password("Wrong123", u.password_hash)
    assert hash_password(PW) != hash_password(PW)  # unique salt per hash


def test_login_and_protected_routes(client, auth):
    assert client.post("/api/auth/login", json={"email": "tester@example.com", "password": "Wrong123"}).status_code == 401
    r = client.post("/api/auth/login", json={"email": "TESTER@example.com", "password": PW})
    assert r.status_code == 200 and r.json()["access_token"]
    assert client.get("/api/dashboard/summary").status_code == 401
    assert client.get("/api/dashboard/summary", headers={"Authorization": "Bearer invalid.token"}).status_code == 401
    me = client.get("/api/auth/me", headers=auth).json()
    assert me["email"] == "tester@example.com" and "password_hash" not in me
    assert client.post("/api/auth/logout", headers=auth).status_code == 200


# ---------------- Rule-based calculations ----------------
def test_supplier_risk_rules():
    good = supplier_risk(99, 0.3, 10, 0.5, 99.5, 5)
    bad = supplier_risk(65, 7.5, 20, 9, 78, 1)
    assert good["risk_level"] == "Low" and good["risk_score"] < 35
    assert bad["risk_level"] == "High" and bad["risk_score"] >= 60
    # exact formula check: delivery (100-90)*2.5=25, quality 2*12.5=25, lead 2/10*200=40, fulfil (100-95)*4=20, fin (5-3)/4*100=50
    mid = supplier_risk(90, 2, 10, 2, 95, 3)
    assert mid["risk_score"] == round(25 * .30 + 25 * .25 + 40 * .15 + 20 * .15 + 50 * .15, 1)


def test_inventory_rules():
    assert inventory_risk(1000, 100, 400, 80, 7, 20)["stockout_risk"] == "Low"
    med = inventory_risk(450, 100, 400, 80, 7, 20)          # net 350 <= reorder 400, cover 17.5 d > 7
    assert med["stockout_risk"] == "Medium" and med["recommended_order_qty"] == 400 + 20 * 30 - 350
    assert inventory_risk(100, 30, 400, 80, 7, 20)["stockout_risk"] == "High"   # net 70 <= safety 80
    assert inventory_risk(200, 0, 400, 50, 14, 20)["stockout_risk"] == "High"   # cover 10 d < lead 14 d


# ---------------- ML predictions ----------------
def test_machine_prediction(client, auth):
    normal = {"machine_type": "M", "air_temperature_k": 298.1, "process_temperature_k": 308.6,
              "rotational_speed_rpm": 1551, "torque_nm": 42.8, "tool_wear_min": 0}
    stressed = {"machine_type": "L", "air_temperature_k": 302.5, "process_temperature_k": 310.9,
                "rotational_speed_rpm": 1290, "torque_nm": 68.0, "tool_wear_min": 230}
    a = client.post("/api/machines/predict", json=normal, headers=auth).json()
    b = client.post("/api/machines/predict", json=stressed, headers=auth).json()
    assert 0 <= a["probability"] <= 1 and 0 <= b["probability"] <= 1
    assert b["probability"] > a["probability"]
    assert a["risk_level"] == "Low" and b["risk_level"] == "High" and b["prediction"] is True
    bad = client.post("/api/machines/predict", json={**normal, "machine_type": "X"}, headers=auth)
    assert bad.status_code == 422


def test_shipment_prediction(client, auth):
    order = {"order_date": "2017-06-10T10:00", "payment_type": "DEBIT", "shipping_mode": "First Class",
             "scheduled_shipping_days": 1, "market": "Europe", "order_region": "Western Europe",
             "category_name": "Cleats", "department_name": "Apparel", "customer_segment": "Consumer",
             "quantity": 1, "product_price": 59.99, "discount_rate": 0.05}
    r = client.post("/api/shipments/predict", json=order, headers=auth)
    assert r.status_code == 200, r.text
    first = r.json()
    std = client.post("/api/shipments/predict", json={**order, "shipping_mode": "Standard Class",
                                                      "scheduled_shipping_days": 4}, headers=auth).json()
    assert 0 <= first["probability"] <= 1 and first["risk_level"] in {"Low", "Medium", "High"}
    assert first["probability"] > std["probability"]  # First Class is late far more often in DataCo
    bad = client.post("/api/shipments/predict", json={**order, "market": "Mars"}, headers=auth)
    assert bad.status_code == 422


def test_history_records_predictions(client, auth):
    h = client.get("/api/history", headers=auth).json()
    assert h["total"] >= 4
    assert {i["model_name"] for i in h["items"]} == {"machine_failure", "late_delivery"}


# ---------------- CSV upload & CRUD ----------------
def test_supplier_csv_upload(client, auth):
    csv = ("supplier_code,name,country,category,on_time_delivery_rate,defect_rate,avg_lead_time_days,"
           "lead_time_variability_days,fulfillment_rate,financial_stability\n"
           "SUP-900,Good Co,Mexico,Apparel,98,0.5,10,1,99,5\n"
           "SUP-901,Bad Row,Mexico,Apparel,150,0.5,10,1,99,5\n")
    r = client.post("/api/suppliers/upload", files={"file": ("s.csv", csv, "text/csv")}, headers=auth).json()
    assert r["created"] == 1 and r["rejected"] == 1 and r["errors"][0]["row"] == 3
    missing = client.post("/api/suppliers/upload", files={"file": ("s.csv", "name\nx\n", "text/csv")}, headers=auth)
    assert missing.status_code == 400 and "Missing required column" in missing.json()["detail"]
    wrong_type = client.post("/api/suppliers/upload", files={"file": ("s.txt", "a", "text/plain")}, headers=auth)
    assert wrong_type.status_code == 400


def test_machine_csv_batch(client, auth):
    csv = ("machine_code,machine_type,air_temperature_k,process_temperature_k,rotational_speed_rpm,torque_nm,tool_wear_min\n"
           "T-1,M,298.1,308.6,1551,42.8,0\nT-2,L,302.5,310.9,1290,68,230\nT-3,Z,1,1,1,1,1\n")
    r = client.post("/api/machines/upload", files={"file": ("m.csv", csv, "text/csv")}, headers=auth).json()
    assert r["processed"] == 2 and r["rejected"] == 1


def test_supplier_crud_and_dashboard(client, auth):
    body = {"supplier_code": "SUP-950", "name": "Risky Ltd", "country": "X", "category": "Golf",
            "on_time_delivery_rate": 60, "defect_rate": 8, "avg_lead_time_days": 10,
            "lead_time_variability_days": 6, "fulfillment_rate": 70, "financial_stability": 1}
    before = client.get("/api/dashboard/summary", headers=auth).json()["cards"]
    created = client.post("/api/suppliers", json=body, headers=auth).json()
    assert created["risk_level"] == "High"
    assert client.post("/api/suppliers", json=body, headers=auth).status_code == 409
    after = client.get("/api/dashboard/summary", headers=auth).json()["cards"]
    assert after["total_suppliers"] == before["total_suppliers"] + 1
    assert after["high_risk_suppliers"] == before["high_risk_suppliers"] + 1
    assert client.delete(f"/api/suppliers/{created['id']}", headers=auth).status_code == 200


def test_inventory_crud(client, auth):
    pid = client.get("/api/products", headers=auth).json()[0]["id"]
    body = {"product_id": pid, "warehouse": "Test WH", "available_stock": 50, "reserved_stock": 10,
            "reorder_level": 300, "safety_stock": 60, "lead_time_days": 7, "avg_daily_demand": 30}
    r = client.post("/api/inventory", json=body, headers=auth).json()
    assert r["stockout_risk"] == "High" and r["net_stock"] == 40 and r["recommended_order_qty"] == 300 + 900 - 40
    assert client.post("/api/inventory", json={**body, "reserved_stock": 99}, headers=auth).status_code == 422
    assert client.delete(f"/api/inventory/{r['id']}", headers=auth).status_code == 200


# ---------------- Other modules ----------------
def test_demand_and_assistant(client, auth):
    o = client.get("/api/demand/overview", headers=auth).json()
    assert len(o["months"]) == 33 and o["trend"]["direction"] in {"Increasing", "Stable", "Decreasing"}
    live = client.post("/api/assistant/ask", json={"question": "Which suppliers are high risk?"}, headers=auth).json()
    assert live["type"] == "live_data" and "suppliers" in live["answer"]
    doc = client.post("/api/assistant/ask", json={"question": "How is supplier risk calculated?"}, headers=auth).json()
    assert doc["type"] == "documentation" and "Supplier risk calculation" in doc["sources"]
    m = client.post("/api/assistant/ask", json={"question": "What is the model accuracy?"}, headers=auth).json()
    assert "Random forest" in m["answer"] or "random forest" in m["answer"]
