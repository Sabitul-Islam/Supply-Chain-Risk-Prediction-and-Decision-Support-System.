"""Train and evaluate the two ML models.

    python -m app.ml.train

1. Machine failure (AI4I 2020)      - RandomForestClassifier vs LogisticRegression
2. Late delivery (DataCo)           - RandomForestClassifier vs LogisticRegression

Metrics are computed on a held-out test split and written to data/models/metrics.json.
The Random Forest pipelines are saved with joblib and used by the API. Hold-out rows (never
seen during training) are exported to data/operational/ to seed the Machines / Shipments tables.
"""
import json
import time
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ..config import MODELS_DIR, OPERATIONAL_DIR, RAW_DIR
from . import prepare_data
from .features import (AI4I_FILE, AI4I_IDS, AI4I_LEAKAGE, AI4I_RAW_TO_API, AI4I_TARGET, DATACO_LEAKAGE,
                       DATACO_RAW_TO_API, DATACO_TARGET, MACHINE_CATEGORICAL, MACHINE_NUMERIC,
                       SHIPMENT_CATEGORICAL, SHIPMENT_NUMERIC, machine_features, shipment_features)

RANDOM_STATE = 42


def preprocessor(categorical, numeric, scale: bool) -> ColumnTransformer:
    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("scale", StandardScaler()))
    return ColumnTransformer([
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
        ("num", Pipeline(num_steps), numeric),
    ])


def evaluate(model, X_test, y_test) -> dict:
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    cm = confusion_matrix(y_test, pred, labels=[0, 1])
    return {
        "accuracy": round(float(accuracy_score(y_test, pred)), 4),
        "precision": round(float(precision_score(y_test, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, proba)), 4),
        "confusion_matrix": {"tn": int(cm[0, 0]), "fp": int(cm[0, 1]), "fn": int(cm[1, 0]), "tp": int(cm[1, 1])},
    }


def feature_importance(rf_pipeline, top=12) -> list:
    names = rf_pipeline.named_steps["prep"].get_feature_names_out()
    imp = rf_pipeline.named_steps["model"].feature_importances_
    order = np.argsort(imp)[::-1][:top]
    return [{"feature": names[i].split("__", 1)[-1], "importance": round(float(imp[i]), 4)} for i in order]


def fit_both(name, X_train, y_train, X_test, y_test, categorical, numeric, rf_params):
    results, models = {}, {}
    candidates = {
        "random_forest": Pipeline([("prep", preprocessor(categorical, numeric, scale=False)),
                                   ("model", RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1, **rf_params))]),
        "logistic_regression": Pipeline([("prep", preprocessor(categorical, numeric, scale=True)),
                                         ("model", LogisticRegression(max_iter=2000, class_weight="balanced",
                                                                      random_state=RANDOM_STATE))]),
    }
    for key, pipe in candidates.items():
        t = time.time()
        pipe.fit(X_train, y_train)
        results[key] = evaluate(pipe, X_test, y_test)
        results[key]["train_seconds"] = round(time.time() - t, 1)
        models[key] = pipe
        print(f"[{name}] {key:20s} " + " ".join(f"{m}={results[key][m]}" for m in ["accuracy", "precision", "recall", "f1", "roc_auc"]))
    return results, models


def train_machine_model() -> dict:
    df = pd.read_csv(RAW_DIR / AI4I_FILE, encoding="utf-8-sig")
    missing = [c for c in list(AI4I_RAW_TO_API) + [AI4I_TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"ai4i2020.csv is missing columns: {missing}")
    n_raw = len(df)
    df = df.drop_duplicates(subset=[c for c in df.columns if c != "UDI"]).dropna(subset=[AI4I_TARGET])
    codes = df["Product ID"]
    y = df[AI4I_TARGET].astype(int)
    api = df.drop(columns=AI4I_LEAKAGE + AI4I_IDS + [AI4I_TARGET]).rename(columns=AI4I_RAW_TO_API)
    X = machine_features(api)

    X_train, X_test, y_train, y_test, _, codes_test = train_test_split(
        X, y, codes, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
    results, models = fit_both("machine", X_train, y_train, X_test, y_test, MACHINE_CATEGORICAL, MACHINE_NUMERIC,
                               dict(n_estimators=300, class_weight="balanced_subsample", min_samples_leaf=2))
    joblib.dump(models["random_forest"], MODELS_DIR / "machine_failure_rf.joblib", compress=3)
    joblib.dump(models["logistic_regression"], MODELS_DIR / "machine_failure_lr.joblib", compress=3)

    # Seed machines table with unseen test rows: all failures in test + a sample of normal rows
    test = api.loc[X_test.index].copy()
    test["machine_code"] = codes_test.values
    test["actual_failure"] = y_test.values
    seed = pd.concat([test[test.actual_failure == 1].head(15),
                      test[test.actual_failure == 0].sample(
                          min(45, len(test[test.actual_failure == 0])), random_state=RANDOM_STATE)])
    seed.sample(frac=1, random_state=RANDOM_STATE)[["machine_code"] + list(AI4I_RAW_TO_API.values()) + ["actual_failure"]] \
        .to_csv(OPERATIONAL_DIR / "machines.csv", index=False)

    return {
        "dataset": AI4I_FILE, "target": AI4I_TARGET, "rows_raw": n_raw, "rows_used": len(df),
        "positive_rate": round(float(y.mean()), 4), "train_rows": len(X_train), "test_rows": len(X_test),
        "split": "Stratified random 80/20 split (random_state=42)",
        "features": list(AI4I_RAW_TO_API.values()) + ["temp_diff_k (engineered)", "power_w (engineered)"],
        "excluded": {"failure_mode_columns (leakage)": AI4I_LEAKAGE, "identifiers": AI4I_IDS},
        "models": results, "selected_model": "random_forest",
        "feature_importance": feature_importance(models["random_forest"]),
    }


def train_shipment_model(df: pd.DataFrame) -> dict:
    missing = [c for c in list(DATACO_RAW_TO_API) + [DATACO_TARGET, "Order Id", "order_date"] if c not in df.columns]
    if missing:
        raise ValueError(f"DataCoSupplyChainDataset.csv is missing columns: {missing}")
    df = df.dropna(subset=[DATACO_TARGET]).drop(columns=[c for c in DATACO_LEAKAGE if c in df.columns])
    y = df[DATACO_TARGET].astype(int)
    api = df.rename(columns=DATACO_RAW_TO_API)
    X = shipment_features(api)

    # Items of the same order share shipping outcome -> split by Order Id to avoid leakage between splits.
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    train_idx, test_idx = next(splitter.split(X, y, groups=df["Order Id"]))
    X_train, X_test, y_train, y_test = X.iloc[train_idx], X.iloc[test_idx], y.iloc[train_idx], y.iloc[test_idx]
    results, models = fit_both("shipment", X_train, y_train, X_test, y_test, SHIPMENT_CATEGORICAL, SHIPMENT_NUMERIC,
                               dict(n_estimators=120, max_depth=16, min_samples_leaf=20))
    joblib.dump(models["random_forest"], MODELS_DIR / "late_delivery_rf.joblib", compress=3)
    joblib.dump(models["logistic_regression"], MODELS_DIR / "late_delivery_lr.joblib", compress=3)

    test = api.iloc[test_idx]
    seed = test.sample(min(800, len(test)), random_state=RANDOM_STATE).copy()
    seed["order_id"] = seed["Order Id"].astype(str)
    seed["product_id"] = seed["Product Card Id"]
    seed["actual_late"] = seed[DATACO_TARGET]
    seed["order_date"] = seed["order_date"].dt.strftime("%Y-%m-%d %H:%M")
    seed[["order_id", "order_date", "product_id"] + list(DATACO_RAW_TO_API.values()) + ["actual_late"]] \
        .to_csv(OPERATIONAL_DIR / "shipments.csv", index=False)

    options = {c: sorted(X[c].dropna().unique().tolist()) for c in SHIPMENT_CATEGORICAL}
    options["region_by_market"] = {m: sorted(g.unique().tolist()) for m, g in X.groupby("market")["order_region"]}
    options["category_by_department"] = {d: sorted(g.unique().tolist()) for d, g in X.groupby("department_name")["category_name"]}
    options["scheduled_days_by_mode"] = {m: sorted(int(v) for v in g.unique()) for m, g in X.groupby("shipping_mode")["scheduled_shipping_days"]}
    (MODELS_DIR / "shipment_options.json").write_text(json.dumps(options, indent=1))

    return {
        "dataset": "DataCoSupplyChainDataset.csv", "target": DATACO_TARGET, "rows_used": len(df),
        "positive_rate": round(float(y.mean()), 4), "train_rows": len(X_train), "test_rows": len(X_test),
        "split": "80/20 GroupShuffleSplit by Order Id (random_state=42)",
        "features": list(DATACO_RAW_TO_API.values()) + ["order_month", "order_weekday", "order_hour (from order date)"],
        "excluded": {"post-shipment / leakage": DATACO_LEAKAGE, "personal_information": prepare_data.DATACO_PII},
        "models": results, "selected_model": "random_forest",
        "feature_importance": feature_importance(models["random_forest"]),
    }


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OPERATIONAL_DIR.mkdir(parents=True, exist_ok=True)
    print("[train] Loading DataCo dataset (this can take ~20 s)...")
    dataco = prepare_data.load_dataco()
    prepare_data.main(dataco)
    metrics = {"trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "sklearn_version": sklearn.__version__,
               "machine_failure": train_machine_model(),
               "late_delivery": train_shipment_model(dataco)}
    (MODELS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"[train] Done. Models and metrics saved to {MODELS_DIR}")
    print("[train] Delete backend/supply_chain.db (if it exists) to re-seed the database with fresh predictions.")


if __name__ == "__main__":
    main()
