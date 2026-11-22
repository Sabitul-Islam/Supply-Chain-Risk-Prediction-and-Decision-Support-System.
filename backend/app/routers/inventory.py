from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import PlainTextResponse
from sqlalchemy.orm import Session, joinedload

from ..auth import get_current_user
from ..database import get_db
from ..models import InventoryItem, Product
from ..schemas import InventoryIn
from ..services.csv_utils import read_csv_upload, validate_rows
from ..services.risk import inventory_risk

router = APIRouter(prefix="/api", tags=["inventory"], dependencies=[Depends(get_current_user)])
COLUMNS = list(InventoryIn.model_fields)


def serialize(i: InventoryItem) -> dict:
    return {"id": i.id, **{c: getattr(i, c) for c in COLUMNS},
            "product_name": i.product.name, "category": i.product.category,
            "supplier": i.product.supplier.name if i.product.supplier else None,
            **inventory_risk(i.available_stock, i.reserved_stock, i.reorder_level, i.safety_stock,
                             i.lead_time_days, i.avg_daily_demand)}


def all_items(db: Session) -> list[dict]:
    items = db.query(InventoryItem).options(joinedload(InventoryItem.product).joinedload(Product.supplier)).all()
    order = {"High": 0, "Medium": 1, "Low": 2}
    return sorted((serialize(i) for i in items), key=lambda x: (order[x["stockout_risk"]], x["days_of_cover"] or 0))


@router.get("/products")
def products(db: Session = Depends(get_db)):
    return [{"id": p.id, "name": p.name, "category": p.category, "department": p.department, "price": p.price,
             "supplier": p.supplier.name if p.supplier else None}
            for p in db.query(Product).options(joinedload(Product.supplier)).order_by(Product.name).all()]


@router.get("/inventory")
def list_inventory(db: Session = Depends(get_db)):
    return all_items(db)


@router.get("/inventory/template", response_class=PlainTextResponse)
def template(db: Session = Depends(get_db)):
    p = db.query(Product).first()
    pid = p.id if p else 365
    return ",".join(COLUMNS) + f"\n{pid},Central DC,500,40,420,60,7,45.5\n"


def _check_product(db, product_id):
    if not db.get(Product, product_id):
        raise HTTPException(400, f"Product {product_id} does not exist.")


@router.post("/inventory", status_code=201)
def create(data: InventoryIn, db: Session = Depends(get_db)):
    _check_product(db, data.product_id)
    item = InventoryItem(**data.model_dump())
    db.add(item)
    db.commit()
    db.refresh(item)
    return serialize(item)


@router.put("/inventory/{item_id}")
def update(item_id: int, data: InventoryIn, db: Session = Depends(get_db)):
    item = db.get(InventoryItem, item_id)
    if not item:
        raise HTTPException(404, "Inventory record not found.")
    _check_product(db, data.product_id)
    for k, v in data.model_dump().items():
        setattr(item, k, v)
    db.commit()
    db.refresh(item)
    return serialize(item)


@router.delete("/inventory/{item_id}")
def delete(item_id: int, db: Session = Depends(get_db)):
    item = db.get(InventoryItem, item_id)
    if not item:
        raise HTTPException(404, "Inventory record not found.")
    db.delete(item)
    db.commit()
    return {"detail": "Inventory record deleted"}


@router.post("/inventory/upload")
async def upload(file: UploadFile = File(...), db: Session = Depends(get_db)):
    df = await read_csv_upload(file, COLUMNS)
    valid, errors = validate_rows(df, InventoryIn, COLUMNS)
    known = {p.id for p in db.query(Product.id).all()}
    created = updated = 0
    for row_no, data in enumerate(valid):
        if data.product_id not in known:
            errors.append({"row": "-", "error": f"product_id {data.product_id} does not exist in products"})
            continue
        item = db.query(InventoryItem).filter(InventoryItem.product_id == data.product_id,
                                              InventoryItem.warehouse == data.warehouse).first()
        if item:
            updated += 1
        else:
            item = InventoryItem()
            db.add(item)
            created += 1
        for k, v in data.model_dump().items():
            setattr(item, k, v)
    db.commit()
    return {"created": created, "updated": updated, "rejected": len(errors), "errors": errors[:25]}
