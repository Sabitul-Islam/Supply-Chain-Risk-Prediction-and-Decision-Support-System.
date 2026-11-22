"""CSV upload parsing and validation helpers."""
import io

import pandas as pd
from fastapi import HTTPException, UploadFile
from pydantic import BaseModel, ValidationError

from ..config import MAX_UPLOAD_BYTES


async def read_csv_upload(file: UploadFile, required: list[str]) -> pd.DataFrame:
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(400, "Please upload a .csv file.")
    content = await file.read()
    if len(content) == 0:
        raise HTTPException(400, "The uploaded file is empty.")
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(400, f"File is too large (max {MAX_UPLOAD_BYTES // (1024 * 1024)} MB).")
    try:
        try:
            df = pd.read_csv(io.BytesIO(content), encoding="utf-8-sig")
        except UnicodeDecodeError:
            df = pd.read_csv(io.BytesIO(content), encoding="latin-1")
    except Exception as e:  # pandas parser errors
        raise HTTPException(400, f"Could not read CSV: {e}")
    df.columns = [str(c).strip() for c in df.columns]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise HTTPException(400, f"Missing required column(s): {', '.join(missing)}. "
                                 f"Required columns: {', '.join(required)}")
    if df.empty:
        raise HTTPException(400, "The CSV has a header but no data rows.")
    if len(df) > 5000:
        raise HTTPException(400, "Too many rows (max 5000 per upload).")
    return df.drop_duplicates()


def validate_rows(df: pd.DataFrame, schema: type[BaseModel], columns: list[str]):
    """Returns (valid_models, errors). Row numbers are 1-based data rows (header = row 1 -> first data row = 2)."""
    valid, errors = [], []
    for i, rec in enumerate(df[columns].to_dict("records"), start=2):
        rec = {k: (None if pd.isna(v) else v) for k, v in rec.items()}
        try:
            valid.append(schema(**rec))
        except ValidationError as e:
            msg = "; ".join(f"{'.'.join(str(x) for x in err['loc']) or 'row'}: {err['msg']}" for err in e.errors())
            errors.append({"row": i, "error": msg})
    return valid, errors
