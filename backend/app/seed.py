"""Seeds the SQLite database from data/operational/*.csv on first start (only empty tables are filled)."""
import logging

import pandas as pd
from sqlalchemy.orm import Session

from .config import OPERATIONAL_DIR
from .ml.predictor import ModelNotAvailable, predict_machines, predict_shipments
from .models import InventoryItem, Machine, Product, Shipment, Supplier
from .services.risk import supplier_risk

log = logging.getLogger("seed")


def _read(name):
    path = OPERATIONAL_DIR / name
    return pd.read_csv(path) if path.exists() else None


def seed_database(db: Session):
    if db.query(Supplier).count() == 0 and (df := _read("suppliers.csv")) is not None:
        for r in df.to_dict("records"):
            risk = supplier_risk(r["on_time_delivery_rate"], r["defect_rate"], r["avg_lead_time_days"],
                                 r["lead_time_variability_days"], r["fulfillment_rate"], int(r["financial_stability"]))
            db.add(Supplier(**r, risk_score=risk["risk_score"], risk_level=risk["risk_level"]))
        db.commit()
        log.info("Seeded %d suppliers", len(df))

    if db.query(Product).count() == 0 and (df := _read("products.csv")) is not None:
        suppliers_by_dept = {}
        for s in db.query(Supplier).all():
            suppliers_by_dept.setdefault(s.category, []).append(s.id)
        for i, r in enumerate(df.to_dict("records")):
            options = suppliers_by_dept.get(r["department"], [])
            db.add(Product(id=int(r["product_id"]), name=r["name"], category=r["category"], department=r["department"],
                           price=float(r["price"]), supplier_id=options[i % len(options)] if options else None))
        db.commit()
        log.info("Seeded %d products", len(df))

    if db.query(InventoryItem).count() == 0 and (df := _read("inventory.csv")) is not None:
        known = {p.id for p in db.query(Product.id).all()}
        for r in df.to_dict("records"):
            if int(r["product_id"]) in known:
                db.add(InventoryItem(product_id=int(r["product_id"]), warehouse=r["warehouse"],
                                     available_stock=int(r["available_stock"]), reserved_stock=int(r["reserved_stock"]),
                                     reorder_level=int(r["reorder_level"]), safety_stock=int(r["safety_stock"]),
                                     lead_time_days=int(r["lead_time_days"]), avg_daily_demand=float(r["avg_daily_demand"])))
        db.commit()
        log.info("Seeded inventory")

    try:
        if db.query(Machine).count() == 0 and (df := _read("machines.csv")) is not None:
            rows = df.to_dict("records")
            for r, p in zip(rows, predict_machines(rows)):
                db.add(Machine(machine_code=r["machine_code"], machine_type=r["machine_type"],
                               air_temperature_k=r["air_temperature_k"], process_temperature_k=r["process_temperature_k"],
                               rotational_speed_rpm=r["rotational_speed_rpm"], torque_nm=r["torque_nm"],
                               tool_wear_min=r["tool_wear_min"], failure_probability=p["probability"],
                               predicted_failure=p["prediction"], risk_level=p["risk_level"],
                               recommendation=p["recommendation"], actual_failure=bool(r["actual_failure"])))
            db.commit()
            log.info("Seeded %d machines with model predictions", len(rows))

        if db.query(Shipment).count() == 0 and (df := _read("shipments.csv")) is not None:
            known = {p.id for p in db.query(Product.id).all()}
            rows = df.to_dict("records")
            for r, p in zip(rows, predict_shipments(rows)):
                db.add(Shipment(order_id=str(r["order_id"]), order_date=pd.to_datetime(r["order_date"]).to_pydatetime(),
                                product_id=int(r["product_id"]) if int(r["product_id"]) in known else None,
                                category_name=r["category_name"], department_name=r["department_name"],
                                market=r["market"], order_region=r["order_region"], customer_segment=r["customer_segment"],
                                payment_type=r["payment_type"], shipping_mode=r["shipping_mode"],
                                scheduled_shipping_days=int(r["scheduled_shipping_days"]), quantity=int(r["quantity"]),
                                product_price=float(r["product_price"]), discount_rate=float(r["discount_rate"]),
                                sales=float(r["sales"]), late_probability=p["probability"], predicted_late=p["prediction"],
                                risk_level=p["risk_level"], recommendation=p["recommendation"],
                                actual_late=bool(r["actual_late"])))
            db.commit()
            log.info("Seeded %d shipments with model predictions", len(rows))
    except ModelNotAvailable as e:
        log.warning("Machines/shipments not seeded: %s", e)
