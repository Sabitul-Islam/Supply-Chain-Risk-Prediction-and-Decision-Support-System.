"""Rule-based risk calculations for suppliers (procurement) and inventory (stockout)."""
import math

# ---------------- Supplier risk ----------------
SUPPLIER_WEIGHTS = {
    "delivery": 0.30,      # on-time delivery rate
    "quality": 0.25,       # defect rate
    "lead_time": 0.15,     # lead-time variability relative to average lead time
    "fulfillment": 0.15,   # order fulfilment rate
    "financial": 0.15,     # financial stability rating
}
SUPPLIER_THRESHOLDS = (35, 60)  # score < 35 Low, < 60 Medium, otherwise High


def _clip(v: float) -> float:
    return max(0.0, min(100.0, v))


def supplier_risk(on_time_delivery_rate: float, defect_rate: float, avg_lead_time_days: float,
                  lead_time_variability_days: float, fulfillment_rate: float, financial_stability: int) -> dict:
    """Each factor is converted to a 0-100 sub-score (100 = worst), then weighted.

    delivery    = (100 - on_time%) * 2.5        -> 60 % on-time or worse = 100
    quality     = defect% * 12.5                -> 8 % defects or worse  = 100
    lead_time   = variability / avg_lead * 200  -> variability >= 50 % of lead time = 100
    fulfillment = (100 - fulfilment%) * 4       -> 75 % fulfilment or worse = 100
    financial   = (5 - rating) / 4 * 100        -> rating 1 = 100, rating 5 = 0
    """
    parts = {
        "delivery": _clip((100 - on_time_delivery_rate) * 2.5),
        "quality": _clip(defect_rate * 12.5),
        "lead_time": _clip((lead_time_variability_days / max(avg_lead_time_days, 0.1)) * 200),
        "fulfillment": _clip((100 - fulfillment_rate) * 4),
        "financial": _clip((5 - financial_stability) / 4 * 100),
    }
    score = round(sum(parts[k] * w for k, w in SUPPLIER_WEIGHTS.items()), 1)
    low, high = SUPPLIER_THRESHOLDS
    level = "Low" if score < low else ("Medium" if score < high else "High")
    return {"risk_score": score, "risk_level": level,
            "breakdown": {k: round(v, 1) for k, v in parts.items()},
            "recommendation": supplier_recommendation(level, parts)}


def supplier_recommendation(level: str, parts: dict) -> str:
    worst = max(parts, key=parts.get)
    focus = {
        "delivery": "late deliveries - agree delivery SLAs and track on-time performance weekly",
        "quality": "high defect rate - request a corrective action plan and increase incoming inspection",
        "lead_time": "unstable lead times - increase safety stock for this supplier's items",
        "fulfillment": "incomplete orders - confirm capacity and split orders where possible",
        "financial": "weak financial stability - qualify an alternative supplier",
    }[worst]
    if level == "High":
        return f"High risk: source a backup supplier. Main issue: {focus}."
    if level == "Medium":
        return f"Monitor closely. Main issue: {focus}."
    return "Low risk: maintain the relationship and review quarterly."


# ---------------- Inventory / stockout risk ----------------
REVIEW_PERIOD_DAYS = 30
Z_95 = 1.65


def suggested_safety_stock(std_daily_demand: float, lead_time_days: int) -> int:
    return math.ceil(Z_95 * std_daily_demand * math.sqrt(lead_time_days))


def suggested_reorder_level(avg_daily_demand: float, lead_time_days: int, safety_stock: int) -> int:
    return math.ceil(avg_daily_demand * lead_time_days + safety_stock)


def inventory_risk(available_stock: int, reserved_stock: int, reorder_level: int, safety_stock: int,
                   lead_time_days: int, avg_daily_demand: float) -> dict:
    """net = available - reserved
    High   : net <= safety stock, or the stock will run out before a new order can arrive (cover < lead time)
    Medium : net <= reorder level (time to reorder)
    Low    : otherwise
    Order quantity = reorder level + demand over the review period - net stock (if > 0)
    """
    net = available_stock - reserved_stock
    cover = round(net / avg_daily_demand, 1) if avg_daily_demand > 0 else None
    if net <= safety_stock or (cover is not None and cover < lead_time_days):
        level = "High"
    elif net <= reorder_level:
        level = "Medium"
    else:
        level = "Low"
    order_qty = 0
    if level != "Low":
        order_qty = max(0, math.ceil(reorder_level + avg_daily_demand * REVIEW_PERIOD_DAYS - net))
    if level == "High":
        rec = f"Urgent: place a purchase order for {order_qty} units now (expedite if possible)."
    elif level == "Medium":
        rec = f"Stock at or below reorder level - reorder {order_qty} units."
    else:
        rec = "Stock level is healthy - no reorder needed."
    return {"net_stock": net, "days_of_cover": cover, "stockout_risk": level,
            "recommended_order_qty": order_qty, "recommendation": rec}
