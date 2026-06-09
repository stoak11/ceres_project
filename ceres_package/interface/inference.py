"""Public predict() — used by FastAPI and tests."""
from ceres_package.interface.predictor import get_predictor


def predict(dept_id: str, year: int) -> dict:
    return get_predictor().predict(dept_id, year)
