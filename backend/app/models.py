"""Database tables."""
from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .database import Base


def utcnow():
    return datetime.now(timezone.utc)


class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    full_name: Mapped[str] = mapped_column(String(100))
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    predictions: Mapped[list["PredictionHistory"]] = relationship(back_populates="user", cascade="all, delete-orphan")


class Supplier(Base):
    __tablename__ = "suppliers"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    supplier_code: Mapped[str] = mapped_column(String(20), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(150))
    country: Mapped[str] = mapped_column(String(80))
    category: Mapped[str] = mapped_column(String(80))
    on_time_delivery_rate: Mapped[float] = mapped_column(Float)   # %
    defect_rate: Mapped[float] = mapped_column(Float)             # %
    avg_lead_time_days: Mapped[float] = mapped_column(Float)
    lead_time_variability_days: Mapped[float] = mapped_column(Float)
    fulfillment_rate: Mapped[float] = mapped_column(Float)        # %
    financial_stability: Mapped[int] = mapped_column(Integer)     # 1 (weak) .. 5 (strong)
    risk_score: Mapped[float] = mapped_column(Float, default=0)
    risk_level: Mapped[str] = mapped_column(String(10), default="Low")
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)

    products: Mapped[list["Product"]] = relationship(back_populates="supplier")


class Product(Base):
    __tablename__ = "products"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)  # DataCo "Product Card Id"
    name: Mapped[str] = mapped_column(String(200))
    category: Mapped[str] = mapped_column(String(100))
    department: Mapped[str] = mapped_column(String(100))
    price: Mapped[float] = mapped_column(Float)
    supplier_id: Mapped[int | None] = mapped_column(ForeignKey("suppliers.id", ondelete="SET NULL"), nullable=True)

    supplier: Mapped[Supplier | None] = relationship(back_populates="products")
    inventory_items: Mapped[list["InventoryItem"]] = relationship(back_populates="product", cascade="all, delete-orphan")
    shipments: Mapped[list["Shipment"]] = relationship(back_populates="product")


class InventoryItem(Base):
    __tablename__ = "inventory"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    product_id: Mapped[int] = mapped_column(ForeignKey("products.id", ondelete="CASCADE"), index=True)
    warehouse: Mapped[str] = mapped_column(String(80))
    available_stock: Mapped[int] = mapped_column(Integer)
    reserved_stock: Mapped[int] = mapped_column(Integer)
    reorder_level: Mapped[int] = mapped_column(Integer)
    safety_stock: Mapped[int] = mapped_column(Integer)
    lead_time_days: Mapped[int] = mapped_column(Integer)
    avg_daily_demand: Mapped[float] = mapped_column(Float)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)

    product: Mapped[Product] = relationship(back_populates="inventory_items")


class Machine(Base):
    __tablename__ = "machines"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    machine_code: Mapped[str] = mapped_column(String(30), index=True)
    machine_type: Mapped[str] = mapped_column(String(1))  # L / M / H quality variant
    air_temperature_k: Mapped[float] = mapped_column(Float)
    process_temperature_k: Mapped[float] = mapped_column(Float)
    rotational_speed_rpm: Mapped[float] = mapped_column(Float)
    torque_nm: Mapped[float] = mapped_column(Float)
    tool_wear_min: Mapped[float] = mapped_column(Float)
    failure_probability: Mapped[float | None] = mapped_column(Float, nullable=True)
    predicted_failure: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    risk_level: Mapped[str | None] = mapped_column(String(10), nullable=True)
    recommendation: Mapped[str | None] = mapped_column(Text, nullable=True)
    actual_failure: Mapped[bool | None] = mapped_column(Boolean, nullable=True)  # label from dataset, if known
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)


class Shipment(Base):
    __tablename__ = "shipments"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    order_id: Mapped[str] = mapped_column(String(30), index=True)
    order_date: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    product_id: Mapped[int | None] = mapped_column(ForeignKey("products.id", ondelete="SET NULL"), nullable=True)
    category_name: Mapped[str] = mapped_column(String(100))
    department_name: Mapped[str] = mapped_column(String(100))
    market: Mapped[str] = mapped_column(String(50))
    order_region: Mapped[str] = mapped_column(String(80))
    customer_segment: Mapped[str] = mapped_column(String(50))
    payment_type: Mapped[str] = mapped_column(String(30))
    shipping_mode: Mapped[str] = mapped_column(String(30))
    scheduled_shipping_days: Mapped[int] = mapped_column(Integer)
    quantity: Mapped[int] = mapped_column(Integer)
    product_price: Mapped[float] = mapped_column(Float)
    discount_rate: Mapped[float] = mapped_column(Float)
    sales: Mapped[float] = mapped_column(Float)
    late_probability: Mapped[float | None] = mapped_column(Float, nullable=True)
    predicted_late: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    risk_level: Mapped[str | None] = mapped_column(String(10), nullable=True)
    recommendation: Mapped[str | None] = mapped_column(Text, nullable=True)
    actual_late: Mapped[bool | None] = mapped_column(Boolean, nullable=True)  # label from dataset, if known

    product: Mapped[Product | None] = relationship(back_populates="shipments")


class PredictionHistory(Base):
    __tablename__ = "prediction_history"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    model_name: Mapped[str] = mapped_column(String(40))  # machine_failure | late_delivery
    source: Mapped[str] = mapped_column(String(20))      # manual | csv
    input_json: Mapped[str] = mapped_column(Text)
    prediction: Mapped[bool] = mapped_column(Boolean)
    probability: Mapped[float] = mapped_column(Float)
    risk_level: Mapped[str] = mapped_column(String(10))
    recommendation: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)

    user: Mapped[User] = relationship(back_populates="predictions")
