from fastapi import APIRouter, Depends, HTTPException, Query

from ..auth import get_current_user
from ..services import demand

router = APIRouter(prefix="/api/demand", tags=["demand"], dependencies=[Depends(get_current_user)])
Period = Query("consistent", pattern="^(consistent|all)$")


def _guard(fn, *args):
    try:
        return fn(*args)
    except demand.DemandDataMissing as e:
        raise HTTPException(503, str(e))


@router.get("/overview")
def overview(period: str = Period):
    return _guard(demand.overview, period)


@router.get("/products")
def products(period: str = Period):
    return _guard(demand.products, period)


@router.get("/products/{product_id}")
def product(product_id: int, period: str = Period):
    res = _guard(demand.product_detail, product_id, period)
    if res is None:
        raise HTTPException(404, "Product not found in demand data.")
    return res
