from fastapi import APIRouter, Depends, HTTPException

from ..auth import get_current_user
from ..ml.predictor import load_metrics, models_ready

router = APIRouter(prefix="/api/models", tags=["models"], dependencies=[Depends(get_current_user)])


@router.get("/metrics")
def metrics():
    m = load_metrics()
    if not m:
        raise HTTPException(503, "No trained models found. Run: python -m app.ml.train")
    return {**m, "ready": models_ready()}
