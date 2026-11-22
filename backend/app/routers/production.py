from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import PlainTextResponse
from sqlalchemy.orm import Session

from ..auth import get_current_user
from ..database import get_db
from ..ml.predictor import ModelNotAvailable, predict_machines
from ..models import Machine, User
from ..schemas import MachineIn
from ..services.csv_utils import read_csv_upload, validate_rows
from ..services.history import record

router = APIRouter(prefix="/api/machines", tags=["production"])
FEATURES = ["machine_type", "air_temperature_k", "process_temperature_k", "rotational_speed_rpm", "torque_nm",
            "tool_wear_min"]


def serialize(m: Machine) -> dict:
    return {c.name: getattr(m, c.name) for c in Machine.__table__.columns}


def _predict(rows):
    try:
        return predict_machines(rows)
    except ModelNotAvailable as e:
        raise HTTPException(503, str(e))


def _save_machine(db, code, inputs, result):
    m = db.query(Machine).filter(Machine.machine_code == code).first() if code else None
    if m is None:
        m = Machine(machine_code=code or f"NEW-{db.query(Machine).count() + 1:04d}")
        db.add(m)
    for k in FEATURES:
        setattr(m, k, inputs[k])
    m.failure_probability, m.predicted_failure = result["probability"], result["prediction"]
    m.risk_level, m.recommendation, m.actual_failure = result["risk_level"], result["recommendation"], None
    return m


@router.get("")
def list_machines(db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    return [serialize(m) for m in db.query(Machine).order_by(Machine.failure_probability.desc()).all()]


@router.post("/predict")
def predict(data: MachineIn, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    inputs = data.model_dump(exclude={"save_to_records"})
    result = _predict([inputs])[0]
    record(db, user.id, "machine_failure", "manual", inputs, result)
    if data.save_to_records:
        _save_machine(db, data.machine_code, inputs, result)
    db.commit()
    return {"input": inputs, **result}


@router.post("/{machine_id}/predict")
def repredict(machine_id: int, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    m = db.get(Machine, machine_id)
    if not m:
        raise HTTPException(404, "Machine not found.")
    inputs = {k: getattr(m, k) for k in FEATURES}
    result = _predict([inputs])[0]
    m.failure_probability, m.predicted_failure = result["probability"], result["prediction"]
    m.risk_level, m.recommendation = result["risk_level"], result["recommendation"]
    record(db, user.id, "machine_failure", "record", {"machine_code": m.machine_code, **inputs}, result)
    db.commit()
    return serialize(m)


@router.delete("/{machine_id}")
def delete(machine_id: int, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    m = db.get(Machine, machine_id)
    if not m:
        raise HTTPException(404, "Machine not found.")
    db.delete(m)
    db.commit()
    return {"detail": "Machine deleted"}


@router.get("/template", response_class=PlainTextResponse)
def template(user: User = Depends(get_current_user)):
    return "machine_code," + ",".join(FEATURES) + "\nM-1001,M,298.1,308.6,1551,42.8,0\nL-2002,L,302.5,310.9,1290,65.2,215\n"


@router.post("/upload")
async def upload(file: UploadFile = File(...), db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    df = await read_csv_upload(file, FEATURES)
    cols = FEATURES + (["machine_code"] if "machine_code" in df.columns else [])
    valid, errors = validate_rows(df, MachineIn, cols)
    if not valid:
        return {"processed": 0, "rejected": len(errors), "errors": errors[:25], "results": []}
    rows = [v.model_dump(exclude={"save_to_records"}) for v in valid]
    results = _predict(rows)
    out = []
    for inputs, res in zip(rows, results):
        record(db, user.id, "machine_failure", "csv", inputs, res)
        m = _save_machine(db, inputs.get("machine_code"), inputs, res)
        out.append({"machine_code": m.machine_code, **res})
    db.commit()
    return {"processed": len(out), "rejected": len(errors), "errors": errors[:25], "results": out[:200]}
