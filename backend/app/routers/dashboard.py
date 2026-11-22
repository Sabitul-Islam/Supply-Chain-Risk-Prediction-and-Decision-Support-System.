from fastapi import APIRouter, Depends
from sqlalchemy import func
from sqlalchemy.orm import Session

from ..auth import get_current_user
from ..database import get_db
from ..models import Machine, Product, Shipment, Supplier
from ..services import demand
from .inventory import all_items

router = APIRouter(prefix="/api/dashboard", tags=["dashboard"], dependencies=[Depends(get_current_user)])
LEVELS = ["Low", "Medium", "High"]


def _dist(db, col):
    counts = dict(db.query(col, func.count()).group_by(col).all())
    return [{"level": k, "count": counts.get(k, 0)} for k in LEVELS]


@router.get("/summary")
def summary(db: Session = Depends(get_db)):
    inventory = all_items(db)
    inv_alerts = [i for i in inventory if i["stockout_risk"] != "Low"]
    machine_alerts = db.query(Machine).filter(Machine.risk_level == "High").order_by(Machine.failure_probability.desc()).all()
    ship_alerts_q = db.query(Shipment).filter(Shipment.risk_level == "High")
    high_suppliers = db.query(Supplier).filter(Supplier.risk_level == "High").order_by(Supplier.risk_score.desc()).all()
    try:
        months = demand.overview("consistent")["months"][-12:]
    except demand.DemandDataMissing:
        months = []
    return {
        "cards": {
            "total_suppliers": db.query(Supplier).count(),
            "total_products": db.query(Product).count(),
            "total_shipments": db.query(Shipment).count(),
            "total_machines": db.query(Machine).count(),
            "high_risk_suppliers": len(high_suppliers),
            "machine_failure_alerts": len(machine_alerts),
            "shipment_delay_alerts": ship_alerts_q.count(),
            "inventory_alerts": len(inv_alerts),
        },
        "charts": {
            "supplier_risk": _dist(db, Supplier.risk_level),
            "machine_risk": _dist(db, Machine.risk_level),
            "shipment_risk": _dist(db, Shipment.risk_level),
            "inventory_risk": [{"level": k, "count": sum(1 for i in inventory if i["stockout_risk"] == k)} for k in LEVELS],
            "shipment_by_mode": [{"shipping_mode": m, "avg_late_probability": round(p or 0, 3), "shipments": n}
                                 for m, p, n in db.query(Shipment.shipping_mode, func.avg(Shipment.late_probability),
                                                         func.count()).group_by(Shipment.shipping_mode).all()],
            "monthly_demand": [{"month": r["month"], "quantity": r["quantity"]} for r in months],
        },
        "alerts": {
            "suppliers": [{"id": s.id, "name": s.name, "risk_score": s.risk_score} for s in high_suppliers[:5]],
            "machines": [{"id": m.id, "machine_code": m.machine_code, "failure_probability": m.failure_probability}
                         for m in machine_alerts[:5]],
            "shipments": [{"id": s.id, "order_id": s.order_id, "shipping_mode": s.shipping_mode,
                           "late_probability": s.late_probability}
                          for s in ship_alerts_q.order_by(Shipment.late_probability.desc()).limit(5).all()],
            "inventory": [{"id": i["id"], "product_name": i["product_name"], "stockout_risk": i["stockout_risk"],
                           "net_stock": i["net_stock"], "reorder_level": i["reorder_level"]} for i in inv_alerts[:5]],
        },
    }
