"""Builds the processed demand tables, the product catalogue and the operational CSV templates.

Run automatically by `python -m app.ml.train`, or standalone: `python -m app.ml.prepare_data`.

* products.csv  - real product catalogue extracted from DataCo.
* demand_*.csv  - real monthly demand aggregated from DataCo order lines.
* suppliers.csv / inventory.csv - NOT included in the downloaded datasets, so small SAMPLE templates are
  generated here (clearly marked as sample data). Average daily demand and its variability in the
  inventory template are computed from real DataCo demand; stock quantities are simulated.
"""
import math

import numpy as np
import pandas as pd

from ..config import OPERATIONAL_DIR, PROCESSED_DIR, RAW_DIR
from .features import DATACO_ENCODING, DATACO_FILE, DATACO_PII

RNG_SEED = 42


def load_dataco() -> pd.DataFrame:
    path = RAW_DIR / DATACO_FILE
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Copy DataCoSupplyChainDataset.csv into data/raw/.")
    df = pd.read_csv(path, encoding=DATACO_ENCODING)
    df = df.drop(columns=[c for c in DATACO_PII if c in df.columns])
    df = df.drop_duplicates()
    df["order_date"] = pd.to_datetime(df["order date (DateOrders)"], format="%m/%d/%Y %H:%M", errors="coerce")
    return df.dropna(subset=["order_date"])


def build_products_and_demand(df: pd.DataFrame) -> pd.DataFrame:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    OPERATIONAL_DIR.mkdir(parents=True, exist_ok=True)
    df = df[~df["Order Status"].isin(["CANCELED", "SUSPECTED_FRAUD"])].copy()  # not real demand
    df["month"] = df["order_date"].dt.to_period("M").astype(str)
    df["day"] = df["order_date"].dt.date

    products = (df.groupby("Product Card Id")
                  .agg(name=("Product Name", "first"), category=("Category Name", "first"),
                       department=("Department Name", "first"), price=("Product Price", "median"))
                  .reset_index().rename(columns={"Product Card Id": "product_id"}))
    products["name"] = products["name"].str.strip()

    monthly = (df.groupby(["Product Card Id", "month"])
                 .agg(quantity=("Order Item Quantity", "sum"), sales=("Sales", "sum"), orders=("Order Id", "nunique"))
                 .reset_index().rename(columns={"Product Card Id": "product_id"}))
    monthly["sales"] = monthly["sales"].round(2)
    monthly.to_csv(PROCESSED_DIR / "demand_monthly_product.csv", index=False)

    cat_monthly = (df.groupby(["Category Name", "month"])
                     .agg(quantity=("Order Item Quantity", "sum"), sales=("Sales", "sum"))
                     .reset_index().rename(columns={"Category Name": "category"}))
    cat_monthly["sales"] = cat_monthly["sales"].round(2)
    cat_monthly.to_csv(PROCESSED_DIR / "demand_monthly_category.csv", index=False)

    total = (df.groupby("month").agg(quantity=("Order Item Quantity", "sum"), sales=("Sales", "sum"),
                                     orders=("Order Id", "nunique")).reset_index())
    total["sales"] = total["sales"].round(2)
    total.to_csv(PROCESSED_DIR / "demand_monthly_total.csv", index=False)

    # daily demand statistics per product (zero-filled over the product's active period)
    daily = df.groupby(["Product Card Id", "day"])["Order Item Quantity"].sum()
    stats = []
    for pid, s in daily.groupby(level=0):
        s = s.droplevel(0)
        idx = pd.date_range(min(s.index), max(s.index), freq="D").date
        s = s.reindex(idx, fill_value=0)
        stats.append({"product_id": pid, "avg_daily_demand": round(float(s.mean()), 2),
                      "std_daily_demand": round(float(s.std(ddof=0)), 2), "total_quantity": int(s.sum())})
    products = products.merge(pd.DataFrame(stats), on="product_id", how="left")
    products.to_csv(PROCESSED_DIR / "product_demand_stats.csv", index=False)
    return products


SUPPLIER_NAMES = ["Alpha Industrial Parts", "Blue Ridge Components", "Cedar Line Manufacturing", "Delta Textile Works",
                  "Evergreen Sports Supply", "Falcon Electronics Ltd", "Granite Outdoor Goods", "Harbor Apparel Co",
                  "Ivory Footwear Group", "Juniper Fitness Gear", "Keystone Tech Assembly", "Lakeside Home Goods",
                  "Meridian Optics", "Northgate Toys & Games", "Orchard Kitchenware", "Pioneer Cycle Parts",
                  "Quarry Golf Equipment", "Riverbend Packaging", "Summit Camping Supply", "Trident Logistics Goods"]
COUNTRIES = ["China", "Vietnam", "India", "Mexico", "Germany", "United States", "Turkey", "Bangladesh",
             "Indonesia", "Brazil"]


def build_supplier_template(products: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED)
    departments = sorted(products["department"].unique())
    rows = []
    for i, name in enumerate(SUPPLIER_NAMES, start=1):
        quality = rng.uniform(0, 1)  # latent "quality" so that metrics are correlated, not random noise
        lead = round(float(rng.uniform(4, 30)), 1)
        rows.append({
            "supplier_code": f"SUP-{i:03d}",
            "name": name,
            "country": COUNTRIES[int(rng.integers(len(COUNTRIES)))],
            "category": departments[i % len(departments)],
            "on_time_delivery_rate": round(float(np.clip(70 + 28 * quality + rng.normal(0, 3), 55, 99.5)), 1),
            "defect_rate": round(float(np.clip(6 - 5.5 * quality + rng.normal(0, 0.7), 0.2, 9)), 2),
            "avg_lead_time_days": lead,
            "lead_time_variability_days": round(float(np.clip(lead * (0.45 - 0.35 * quality) + rng.normal(0, 0.8), 0.3, 15)), 1),
            "fulfillment_rate": round(float(np.clip(80 + 19 * quality + rng.normal(0, 2), 70, 100)), 1),
            "financial_stability": int(np.clip(round(1 + 4 * quality + rng.normal(0, 0.6)), 1, 5)),
        })
    sup = pd.DataFrame(rows)
    sup.to_csv(OPERATIONAL_DIR / "suppliers.csv", index=False)
    return sup


WAREHOUSES = ["Central DC", "East Warehouse", "West Warehouse"]


def build_inventory_template(products: pd.DataFrame) -> pd.DataFrame:
    """Reorder level and safety stock use the standard formulas with real DataCo demand:
    safety_stock = Z(95% = 1.65) * sigma_daily * sqrt(lead_time);  reorder_level = avg_daily * lead_time + safety_stock
    """
    rng = np.random.default_rng(RNG_SEED + 1)
    top = products.sort_values("total_quantity", ascending=False).head(60)
    rows = []
    for _, p in top.iterrows():
        lead = int(rng.integers(3, 15))
        avg, std = float(p["avg_daily_demand"]), float(p["std_daily_demand"])
        safety = math.ceil(1.65 * std * math.sqrt(lead))
        reorder = math.ceil(avg * lead + safety)
        # simulated stock position: between 0.3x and 3x the reorder level
        available = int(max(0, round(reorder * rng.uniform(0.3, 3.0))))
        reserved = int(round(available * rng.uniform(0.0, 0.35)))
        rows.append({"product_id": int(p["product_id"]), "product_name": p["name"], "category": p["category"],
                     "warehouse": WAREHOUSES[int(rng.integers(len(WAREHOUSES)))], "available_stock": available,
                     "reserved_stock": reserved, "reorder_level": reorder, "safety_stock": safety,
                     "lead_time_days": lead, "avg_daily_demand": round(avg, 2)})
    inv = pd.DataFrame(rows)
    inv.to_csv(OPERATIONAL_DIR / "inventory.csv", index=False)
    return inv


def main(df: pd.DataFrame | None = None):
    df = load_dataco() if df is None else df
    products = build_products_and_demand(df)
    products[["product_id", "name", "category", "department", "price"]].to_csv(OPERATIONAL_DIR / "products.csv", index=False)
    if not (OPERATIONAL_DIR / "suppliers.csv").exists():
        build_supplier_template(products)
        print("[prepare] suppliers.csv not found - created SAMPLE supplier template")
    if not (OPERATIONAL_DIR / "inventory.csv").exists():
        build_inventory_template(products)
        print("[prepare] inventory.csv not found - created SAMPLE inventory template")
    print(f"[prepare] {len(products)} products, demand tables written to {PROCESSED_DIR}")


if __name__ == "__main__":
    main()
