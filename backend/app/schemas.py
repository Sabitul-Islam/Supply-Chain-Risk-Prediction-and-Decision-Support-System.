"""Pydantic request/response schemas with validation."""
import re
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator, model_validator


# ---------------- Auth ----------------
class RegisterIn(BaseModel):
    full_name: str = Field(min_length=2, max_length=100)
    email: EmailStr
    password: str = Field(min_length=8, max_length=64)
    confirm_password: str

    @field_validator("full_name")
    @classmethod
    def strip_name(cls, v):
        v = v.strip()
        if len(v) < 2:
            raise ValueError("Full name must have at least 2 characters")
        return v

    @field_validator("password")
    @classmethod
    def strong_password(cls, v):
        if not re.search(r"[A-Za-z]", v) or not re.search(r"\d", v):
            raise ValueError("Password must contain at least one letter and one number")
        if len(v.encode("utf-8")) > 72:
            raise ValueError("Password is too long")
        return v

    @model_validator(mode="after")
    def passwords_match(self):
        if self.password != self.confirm_password:
            raise ValueError("Passwords do not match")
        return self


class LoginIn(BaseModel):
    email: EmailStr
    password: str = Field(min_length=1, max_length=128)


class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    full_name: str
    email: str
    created_at: datetime


class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserOut


# ---------------- Suppliers ----------------
class SupplierIn(BaseModel):
    supplier_code: str = Field(min_length=1, max_length=20)
    name: str = Field(min_length=1, max_length=150)
    country: str = Field(min_length=1, max_length=80)
    category: str = Field(min_length=1, max_length=80)
    on_time_delivery_rate: float = Field(ge=0, le=100)
    defect_rate: float = Field(ge=0, le=100)
    avg_lead_time_days: float = Field(gt=0, le=365)
    lead_time_variability_days: float = Field(ge=0, le=365)
    fulfillment_rate: float = Field(ge=0, le=100)
    financial_stability: int = Field(ge=1, le=5)


# ---------------- Inventory ----------------
class InventoryIn(BaseModel):
    product_id: int
    warehouse: str = Field(min_length=1, max_length=80)
    available_stock: int = Field(ge=0)
    reserved_stock: int = Field(ge=0)
    reorder_level: int = Field(ge=0)
    safety_stock: int = Field(ge=0)
    lead_time_days: int = Field(ge=1, le=365)
    avg_daily_demand: float = Field(ge=0)

    @model_validator(mode="after")
    def check(self):
        if self.reserved_stock > self.available_stock:
            raise ValueError("Reserved stock cannot exceed available stock")
        return self


# ---------------- ML inputs ----------------
class MachineIn(BaseModel):
    machine_code: str | None = Field(default=None, max_length=30)
    machine_type: Literal["L", "M", "H"]
    air_temperature_k: float = Field(ge=250, le=350)
    process_temperature_k: float = Field(ge=250, le=360)
    rotational_speed_rpm: float = Field(ge=500, le=4000)
    torque_nm: float = Field(ge=0, le=150)
    tool_wear_min: float = Field(ge=0, le=400)
    save_to_records: bool = False


class ShipmentIn(BaseModel):
    order_id: str | None = Field(default=None, max_length=30)
    order_date: datetime
    payment_type: str
    shipping_mode: str
    scheduled_shipping_days: int = Field(ge=0, le=30)
    market: str
    order_region: str
    category_name: str
    department_name: str
    customer_segment: str
    quantity: int = Field(ge=1, le=1000)
    product_price: float = Field(gt=0)
    discount_rate: float = Field(ge=0, le=1)
    sales: float | None = Field(default=None, ge=0)
    product_id: int | None = None
    save_to_records: bool = False


class AskIn(BaseModel):
    question: str = Field(min_length=2, max_length=500)
