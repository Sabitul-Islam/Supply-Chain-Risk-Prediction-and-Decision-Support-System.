import json

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from ..auth import get_current_user
from ..database import get_db
from ..models import PredictionHistory, User

router = APIRouter(prefix="/api/history", tags=["history"])


@router.get("")
def list_history(model: str | None = Query(None, pattern="^(machine_failure|late_delivery)$"),
                 page: int = Query(1, ge=1), page_size: int = Query(20, ge=1, le=100),
                 db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    q = db.query(PredictionHistory).filter(PredictionHistory.user_id == user.id)
    if model:
        q = q.filter(PredictionHistory.model_name == model)
    total = q.count()
    rows = q.order_by(PredictionHistory.created_at.desc(), PredictionHistory.id.desc()) \
        .offset((page - 1) * page_size).limit(page_size).all()
    return {"total": total, "page": page, "page_size": page_size,
            "items": [{"id": r.id, "model_name": r.model_name, "source": r.source, "input": json.loads(r.input_json),
                       "prediction": r.prediction, "probability": r.probability, "risk_level": r.risk_level,
                       "recommendation": r.recommendation, "created_at": r.created_at} for r in rows]}


@router.delete("/{item_id}")
def delete_item(item_id: int, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    r = db.get(PredictionHistory, item_id)
    if not r or r.user_id != user.id:
        raise HTTPException(404, "History entry not found.")
    db.delete(r)
    db.commit()
    return {"detail": "Deleted"}


@router.delete("")
def clear(db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    n = db.query(PredictionHistory).filter(PredictionHistory.user_id == user.id).delete()
    db.commit()
    return {"detail": f"Deleted {n} entries"}
