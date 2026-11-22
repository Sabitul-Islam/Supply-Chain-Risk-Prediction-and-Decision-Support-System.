from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import PlainTextResponse
from sqlalchemy.orm import Session

from ..auth import get_current_user
from ..database import get_db
from ..models import Supplier
from ..schemas import SupplierIn
from ..services.csv_utils import read_csv_upload, validate_rows
from ..services.risk import SUPPLIER_THRESHOLDS, SUPPLIER_WEIGHTS, supplier_risk

router = APIRouter(prefix="/api/suppliers", tags=["procurement"], dependencies=[Depends(get_current_user)])
COLUMNS = list(SupplierIn.model_fields)


def serialize(s: Supplier) -> dict:
    risk = supplier_risk(s.on_time_delivery_rate, s.defect_rate, s.avg_lead_time_days,
                         s.lead_time_variability_days, s.fulfillment_rate, s.financial_stability)
    return {"id": s.id, **{c: getattr(s, c) for c in COLUMNS}, **risk, "product_count": len(s.products)}


def apply(s: Supplier, data: SupplierIn):
    for k, v in data.model_dump().items():
        setattr(s, k, v)
    risk = supplier_risk(**{k: v for k, v in data.model_dump().items() if k in (
        "on_time_delivery_rate", "defect_rate", "avg_lead_time_days", "lead_time_variability_days",
        "fulfillment_rate", "financial_stability")})
    s.risk_score, s.risk_level = risk["risk_score"], risk["risk_level"]


@router.get("")
def list_suppliers(db: Session = Depends(get_db)):
    items = [serialize(s) for s in db.query(Supplier).all()]
    return sorted(items, key=lambda x: -x["risk_score"])


@router.get("/method")
def method():
    return {"weights": SUPPLIER_WEIGHTS, "thresholds": {"low_below": SUPPLIER_THRESHOLDS[0],
                                                        "high_from": SUPPLIER_THRESHOLDS[1]}}


@router.get("/template", response_class=PlainTextResponse)
def template():
    return ",".join(COLUMNS) + "\nSUP-101,Example Supplier,Mexico,Apparel,92.5,1.2,10,2,97,4\n"


@router.post("", status_code=201)
def create(data: SupplierIn, db: Session = Depends(get_db)):
    if db.query(Supplier).filter(Supplier.supplier_code == data.supplier_code).first():
        raise HTTPException(409, f"Supplier code {data.supplier_code} already exists.")
    s = Supplier()
    apply(s, data)
    db.add(s)
    db.commit()
    db.refresh(s)
    return serialize(s)


@router.put("/{supplier_id}")
def update(supplier_id: int, data: SupplierIn, db: Session = Depends(get_db)):
    s = db.get(Supplier, supplier_id)
    if not s:
        raise HTTPException(404, "Supplier not found.")
    clash = db.query(Supplier).filter(Supplier.supplier_code == data.supplier_code, Supplier.id != supplier_id).first()
    if clash:
        raise HTTPException(409, f"Supplier code {data.supplier_code} already exists.")
    apply(s, data)
    db.commit()
    return serialize(s)


@router.delete("/{supplier_id}")
def delete(supplier_id: int, db: Session = Depends(get_db)):
    s = db.get(Supplier, supplier_id)
    if not s:
        raise HTTPException(404, "Supplier not found.")
    db.delete(s)
    db.commit()
    return {"detail": "Supplier deleted"}


@router.post("/upload")
async def upload(file: UploadFile = File(...), db: Session = Depends(get_db)):
    df = await read_csv_upload(file, COLUMNS)
    valid, errors = validate_rows(df, SupplierIn, COLUMNS)
    created = updated = 0
    for data in valid:
        s = db.query(Supplier).filter(Supplier.supplier_code == data.supplier_code).first()
        if s:
            updated += 1
        else:
            s = Supplier()
            db.add(s)
            created += 1
        apply(s, data)
    db.commit()
    return {"created": created, "updated": updated, "rejected": len(errors), "errors": errors[:25]}
