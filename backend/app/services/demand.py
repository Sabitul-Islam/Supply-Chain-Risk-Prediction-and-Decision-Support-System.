"""Demand analysis on real DataCo order lines (pre-aggregated by app.ml.prepare_data)."""
from functools import lru_cache

import numpy as np
import pandas as pd

from ..config import PROCESSED_DIR

# From Oct 2017 the DataCo dataset contains ~80 % fewer order lines and a different product mix.
# The "consistent" period excludes these months so that trends are not distorted by this data artefact.
CONSISTENT_END = "2017-09"


class DemandDataMissing(RuntimeError):
    pass


@lru_cache(maxsize=None)
def _load(name: str) -> pd.DataFrame:
    path = PROCESSED_DIR / name
    if not path.exists():
        raise DemandDataMissing("Demand data not found. Run: python -m app.ml.train")
    return pd.read_csv(path)


def clear_cache():
    _load.cache_clear()


def _filter(df: pd.DataFrame, period: str) -> pd.DataFrame:
    return df[df["month"] <= CONSISTENT_END] if period == "consistent" else df


def trend(series: pd.Series, months: int = 12) -> dict:
    """Least-squares linear trend over the last `months` points, expressed as % of the mean per month."""
    s = series.tail(months).astype(float)
    if len(s) < 3 or s.mean() == 0:
        return {"direction": "Insufficient data", "slope_per_month": 0.0, "pct_per_month": 0.0, "months": len(s)}
    slope = float(np.polyfit(np.arange(len(s)), s.values, 1)[0])
    pct = slope / float(s.mean()) * 100
    direction = "Increasing" if pct > 2 else ("Decreasing" if pct < -2 else "Stable")
    return {"direction": direction, "slope_per_month": round(slope, 2), "pct_per_month": round(pct, 2), "months": len(s)}


def _series(df: pd.DataFrame) -> list[dict]:
    df = df.sort_values("month").copy()
    df["moving_avg_3m"] = df["quantity"].rolling(3, min_periods=1).mean().round(1)
    return df.to_dict("records")


def overview(period: str) -> dict:
    total = _filter(_load("demand_monthly_total.csv"), period)
    stats = _load("product_demand_stats.csv")
    prod = _filter(_load("demand_monthly_product.csv"), period)
    cat = _filter(_load("demand_monthly_category.csv"), period)
    top = (prod.groupby("product_id").agg(quantity=("quantity", "sum"), sales=("sales", "sum")).reset_index()
              .merge(stats[["product_id", "name", "category"]], on="product_id")
              .sort_values("quantity", ascending=False).head(10))
    cats = cat.groupby("category").agg(quantity=("quantity", "sum"), sales=("sales", "sum")).reset_index() \
        .sort_values("sales", ascending=False).head(10)
    return {"period": period, "months": _series(total), "trend": trend(total.sort_values("month")["quantity"]),
            "totals": {"quantity": int(total["quantity"].sum()), "sales": round(float(total["sales"].sum()), 2),
                       "orders": int(total["orders"].sum())},
            "top_products": top.round(2).to_dict("records"), "categories": cats.round(2).to_dict("records"),
            "note": "From Oct 2017 the DataCo dataset contains far fewer order lines. Use the 'Consistent period' "
                    "(Jan 2015 - Sep 2017) for trend analysis."}


def products(period: str) -> list[dict]:
    stats = _load("product_demand_stats.csv")
    prod = _filter(_load("demand_monthly_product.csv"), period)
    out = []
    for pid, g in prod.groupby("product_id"):
        g = g.sort_values("month")
        out.append({"product_id": int(pid), "quantity": int(g["quantity"].sum()), "sales": round(float(g["sales"].sum()), 2),
                    "active_months": len(g), "last_month": g["month"].iloc[-1], "trend": trend(g["quantity"])["direction"]})
    df = pd.DataFrame(out).merge(stats[["product_id", "name", "category", "department", "avg_daily_demand"]], on="product_id")
    return df.sort_values("quantity", ascending=False).to_dict("records")


def product_detail(product_id: int, period: str) -> dict | None:
    stats = _load("product_demand_stats.csv")
    row = stats[stats["product_id"] == product_id]
    if row.empty:
        return None
    g = _filter(_load("demand_monthly_product.csv"), period)
    g = g[g["product_id"] == product_id]
    return {"product": row.iloc[0].to_dict(), "months": _series(g), "trend": trend(g.sort_values("month")["quantity"])}
