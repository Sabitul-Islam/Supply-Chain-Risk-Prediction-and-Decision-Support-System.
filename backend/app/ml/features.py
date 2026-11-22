"""Feature definitions shared by training and inference (single source of truth)."""
import numpy as np
import pandas as pd

# ---------------- AI4I 2020 (machine failure) ----------------
AI4I_FILE = "ai4i2020.csv"
AI4I_TARGET = "Machine failure"
# Failure-mode flags are *consequences* of a failure -> excluded (target leakage).
AI4I_LEAKAGE = ["TWF", "HDF", "PWF", "OSF", "RNF"]
AI4I_IDS = ["UDI", "Product ID"]
AI4I_RAW_TO_API = {
    "Type": "machine_type",
    "Air temperature [K]": "air_temperature_k",
    "Process temperature [K]": "process_temperature_k",
    "Rotational speed [rpm]": "rotational_speed_rpm",
    "Torque [Nm]": "torque_nm",
    "Tool wear [min]": "tool_wear_min",
}
MACHINE_CATEGORICAL = ["machine_type"]
MACHINE_NUMERIC = ["air_temperature_k", "process_temperature_k", "rotational_speed_rpm",
                   "torque_nm", "tool_wear_min", "temp_diff_k", "power_w"]


def machine_features(df: pd.DataFrame) -> pd.DataFrame:
    """df uses API column names. Adds physics-based engineered features (inputs only, no labels)."""
    out = df[list(AI4I_RAW_TO_API.values())].copy()
    out["machine_type"] = out["machine_type"].astype(str).str.upper().str.strip()
    for c in MACHINE_NUMERIC[:5]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out["temp_diff_k"] = out["process_temperature_k"] - out["air_temperature_k"]
    out["power_w"] = out["torque_nm"] * out["rotational_speed_rpm"] * 2 * np.pi / 60
    return out[MACHINE_CATEGORICAL + MACHINE_NUMERIC]


# ---------------- DataCo (late delivery) ----------------
DATACO_FILE = "DataCoSupplyChainDataset.csv"
DATACO_ENCODING = "latin-1"
DATACO_TARGET = "Late_delivery_risk"
# Known only after shipping/delivery -> excluded (target leakage)
DATACO_LEAKAGE = ["Days for shipping (real)", "Delivery Status", "shipping date (DateOrders)", "Order Status",
                  "Benefit per order", "Order Profit Per Order", "Order Item Profit Ratio"]
# Personal / identifying information -> removed
DATACO_PII = ["Customer Email", "Customer Fname", "Customer Lname", "Customer Password", "Customer Street",
              "Customer Zipcode", "Customer Id", "Order Customer Id", "Latitude", "Longitude", "Order Zipcode",
              "Product Image", "Product Description"]
DATACO_RAW_TO_API = {
    "Type": "payment_type",
    "Days for shipment (scheduled)": "scheduled_shipping_days",
    "Shipping Mode": "shipping_mode",
    "Market": "market",
    "Order Region": "order_region",
    "Category Name": "category_name",
    "Department Name": "department_name",
    "Customer Segment": "customer_segment",
    "Order Item Quantity": "quantity",
    "Order Item Product Price": "product_price",
    "Order Item Discount Rate": "discount_rate",
    "Sales": "sales",
}
SHIPMENT_CATEGORICAL = ["payment_type", "shipping_mode", "market", "order_region", "category_name",
                        "department_name", "customer_segment"]
SHIPMENT_NUMERIC = ["scheduled_shipping_days", "quantity", "product_price", "discount_rate", "sales",
                    "order_month", "order_weekday", "order_hour"]


def shipment_features(df: pd.DataFrame) -> pd.DataFrame:
    """df uses API column names plus 'order_date'. Only information available when the order is placed."""
    out = df[list(DATACO_RAW_TO_API.values())].copy()
    for c in SHIPMENT_CATEGORICAL:
        out[c] = out[c].astype(str).str.strip()
    for c in ["scheduled_shipping_days", "quantity", "product_price", "discount_rate", "sales"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    dates = pd.to_datetime(df["order_date"], errors="coerce")
    out["order_month"] = dates.dt.month
    out["order_weekday"] = dates.dt.day_of_week
    out["order_hour"] = dates.dt.hour
    return out[SHIPMENT_CATEGORICAL + SHIPMENT_NUMERIC]


# ---------------- Risk levels & recommendations ----------------
MACHINE_THRESHOLDS = (0.20, 0.50)   # < 0.20 Low, < 0.50 Medium, otherwise High
SHIPMENT_THRESHOLDS = (0.40, 0.60)


def risk_level(prob: float, thresholds) -> str:
    low, high = thresholds
    return "Low" if prob < low else ("Medium" if prob < high else "High")


def machine_recommendation(level: str, row: dict) -> str:
    notes = []
    if row.get("tool_wear_min", 0) >= 200:
        notes.append("tool wear is above 200 min - replace the tool")
    temp_diff = row.get("process_temperature_k", 0) - row.get("air_temperature_k", 0)
    if temp_diff < 8.6:
        notes.append("process/air temperature gap is below 8.6 K - check cooling / heat dissipation")
    power = row.get("torque_nm", 0) * row.get("rotational_speed_rpm", 0) * 2 * np.pi / 60
    if power < 3500 or power > 9000:
        notes.append("mechanical power is outside the 3.5-9 kW band - review load and speed settings")
    if row.get("torque_nm", 0) > 60:
        notes.append("torque is high - inspect for overstrain")
    base = {
        "High": "Schedule immediate maintenance and inspect the machine before the next run.",
        "Medium": "Plan preventive maintenance within the next shift and monitor closely.",
        "Low": "Machine is operating normally - continue routine monitoring.",
    }[level]
    return base + (" Checks: " + "; ".join(notes) + "." if notes else "")


def shipment_recommendation(level: str, row: dict) -> str:
    mode = row.get("shipping_mode", "")
    if level == "High":
        tip = "Upgrade the shipping mode or notify the customer of a possible delay; prioritise this order in the warehouse."
        if mode in ("First Class", "Second Class"):
            tip = (f"{mode} orders are frequently late in the historical data - confirm carrier capacity, "
                   "pick and pack immediately and proactively notify the customer.")
        return tip
    if level == "Medium":
        return "Monitor this shipment, confirm the carrier pickup time and keep a buffer on the promised date."
    return "Shipment is expected on time - no action needed."
