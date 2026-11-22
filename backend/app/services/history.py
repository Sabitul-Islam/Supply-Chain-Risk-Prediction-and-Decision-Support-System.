import json

from sqlalchemy.orm import Session

from ..models import PredictionHistory


def record(db: Session, user_id: int, model_name: str, source: str, inputs: dict, result: dict):
    db.add(PredictionHistory(user_id=user_id, model_name=model_name, source=source,
                             input_json=json.dumps(inputs, default=str), prediction=result["prediction"],
                             probability=result["probability"], risk_level=result["risk_level"],
                             recommendation=result["recommendation"]))
