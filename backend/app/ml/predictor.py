"""Loads the trained joblib models and produces real predictions."""
import json
import logging
from functools import lru_cache

import joblib
import pandas as pd
import sklearn

from ..config import MODELS_DIR
from .features import (MACHINE_THRESHOLDS, SHIPMENT_THRESHOLDS, machine_features, machine_recommendation,
                       risk_level, shipment_features, shipment_recommendation)

log = logging.getLogger(__name__)
MODEL_FILES = {"machine_failure": "machine_failure_rf.joblib", "late_delivery": "late_delivery_rf.joblib"}


class ModelNotAvailable(RuntimeError):
    pass


@lru_cache(maxsize=None)
def load_model(name: str):
    path = MODELS_DIR / MODEL_FILES[name]
    if not path.exists():
        raise ModelNotAvailable(f"Model file {path.name} not found. Run: python -m app.ml.train")
    return joblib.load(path)


@lru_cache(maxsize=1)
def load_metrics() -> dict:
    path = MODELS_DIR / "metrics.json"
    if not path.exists():
        return {}
    metrics = json.loads(path.read_text())
    if metrics.get("sklearn_version") != sklearn.__version__:
        log.warning("Models trained with scikit-learn %s but %s is installed - retrain with python -m app.ml.train",
                    metrics.get("sklearn_version"), sklearn.__version__)
    return metrics


@lru_cache(maxsize=1)
def shipment_options() -> dict:
    path = MODELS_DIR / "shipment_options.json"
    return json.loads(path.read_text()) if path.exists() else {}


def models_ready() -> dict:
    return {name: (MODELS_DIR / f).exists() for name, f in MODEL_FILES.items()}


def predict_machines(rows: list[dict]) -> list[dict]:
    model = load_model("machine_failure")
    X = machine_features(pd.DataFrame(rows))
    probs = model.predict_proba(X)[:, 1]
    out = []
    for row, p in zip(rows, probs):
        level = risk_level(float(p), MACHINE_THRESHOLDS)
        out.append({"probability": round(float(p), 4), "prediction": bool(p >= 0.5), "risk_level": level,
                    "recommendation": machine_recommendation(level, row)})
    return out


def predict_shipments(rows: list[dict]) -> list[dict]:
    model = load_model("late_delivery")
    X = shipment_features(pd.DataFrame(rows))
    probs = model.predict_proba(X)[:, 1]
    out = []
    for row, p in zip(rows, probs):
        level = risk_level(float(p), SHIPMENT_THRESHOLDS)
        out.append({"probability": round(float(p), 4), "prediction": bool(p >= 0.5), "risk_level": level,
                    "recommendation": shipment_recommendation(level, row)})
    return out
