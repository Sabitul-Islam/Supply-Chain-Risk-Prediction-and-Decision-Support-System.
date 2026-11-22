from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from ..auth import get_current_user
from ..database import get_db
from ..models import User
from ..schemas import AskIn
from ..services.assistant import SUGGESTIONS, answer

router = APIRouter(prefix="/api/assistant", tags=["assistant"])


@router.post("/ask")
def ask(data: AskIn, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    return answer(data.question, db, user)


@router.get("/suggestions")
def suggestions(user: User = Depends(get_current_user)):
    return SUGGESTIONS
