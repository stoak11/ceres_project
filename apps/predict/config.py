"""Re-export API settings for Streamlit (added later)."""
from ceres_package.interface.config import YIELD_UNIT, load_ml_winner_meta
from ceres_package.params import PROJECT_ROOT
import os

API_HOST = os.environ.get("CERES_API_HOST", "127.0.0.1")
API_PORT = int(os.environ.get("CERES_API_PORT", "8000"))
API_BASE = os.environ.get("CERES_API_BASE", f"http://{API_HOST}:{API_PORT}")

__all__ = ["API_BASE", "API_HOST", "API_PORT", "PROJECT_ROOT", "YIELD_UNIT", "load_ml_winner_meta"]
