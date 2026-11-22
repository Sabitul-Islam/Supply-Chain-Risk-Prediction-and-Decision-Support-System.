"""Application configuration loaded from environment variables / backend/.env."""
import os
import secrets
from pathlib import Path

from dotenv import load_dotenv

BACKEND_DIR = Path(__file__).resolve().parent.parent
PROJECT_DIR = BACKEND_DIR.parent
DATA_DIR = PROJECT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
OPERATIONAL_DIR = DATA_DIR / "operational"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = DATA_DIR / "models"

ENV_FILE = BACKEND_DIR / ".env"
load_dotenv(ENV_FILE)


def _ensure_secret_key() -> str:
    """Use SECRET_KEY from the environment; otherwise generate one once and persist it."""
    key = os.getenv("SECRET_KEY", "").strip()
    if key:
        return key
    key = secrets.token_hex(32)
    lines = []
    if ENV_FILE.exists():
        lines = [l for l in ENV_FILE.read_text(encoding="utf-8").splitlines() if not l.startswith("SECRET_KEY=")]
    lines.append(f"SECRET_KEY={key}")
    ENV_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.environ["SECRET_KEY"] = key
    print("[config] No SECRET_KEY found - generated a random key and saved it to backend/.env")
    return key


SECRET_KEY = _ensure_secret_key()
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "120"))
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{(BACKEND_DIR / 'supply_chain.db').as_posix()}")
if DATABASE_URL.startswith("sqlite:///./"):
    DATABASE_URL = f"sqlite:///{(BACKEND_DIR / DATABASE_URL[len('sqlite:///./'):]).as_posix()}"
FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "http://localhost:5173")
MAX_UPLOAD_BYTES = 5 * 1024 * 1024
