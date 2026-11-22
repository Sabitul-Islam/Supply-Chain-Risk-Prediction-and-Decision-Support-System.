"""FastAPI application entry point.  Run:  uvicorn app.main:app --reload"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import FRONTEND_ORIGIN
from .database import Base, SessionLocal, engine
from .ml.predictor import load_metrics, models_ready
from .routers import (assistant, auth, dashboard, demand, history, inventory, models_info, production, suppliers,
                      transportation)
from .seed import seed_database

logging.basicConfig(level=logging.INFO, format="%(levelname)s [%(name)s] %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)
    with SessionLocal() as db:
        seed_database(db)
    load_metrics()
    yield


app = FastAPI(title="Supply Chain Risk Prediction and Decision Support System", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=[FRONTEND_ORIGIN, "http://127.0.0.1:5173"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])

for r in [auth, dashboard, suppliers, inventory, production, transportation, demand, assistant, history, models_info]:
    app.include_router(r.router)


@app.get("/api/health")
def health():
    return {"status": "ok", "models": models_ready()}
