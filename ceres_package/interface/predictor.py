"""Load GCS model + feature mart, run model.predict() for one dept × year."""
from __future__ import annotations

from ceres_package.interface.config import GCS_MODEL_BLOB, RANDOM_STATE, TEST_SIZE, YIELD_UNIT
from ceres_package.interface.gcs import load_model_from_gcs
from ceres_package.interface.mart import build_features_and_scaler


class Predictor:
    def __init__(self):
        self.model = load_model_from_gcs()
        self.meta, self.X, self.scaler = build_features_and_scaler()

    def predict(self, dept_id: str, year: int) -> dict:
        dept = str(dept_id).strip().zfill(2)
        year = int(year)

        row = self.meta[(self.meta["DEPT_ID"] == dept) & (self.meta["harvest_year"] == year)]
        if row.empty:
            raise ValueError(f"No data for dept={dept} year={year}")

        i = int(row.index[0])
        X_row = self.scaler.transform(self.X.iloc[[i]])
        pred_qha = float(self.model.predict(X_row)[0])
        actual_qha = float(row["RENDEMENT"].iloc[0])

        return {
            "dept_id": dept,
            "year": year,
            "yield_unit": YIELD_UNIT,
            "predict_qha": round(pred_qha, 3),
            "actual_qha": round(actual_qha, 3),
            "model_source": "gcs",
            "model_blob": GCS_MODEL_BLOB,
            "split": {"test_size": TEST_SIZE, "random_state": RANDOM_STATE},
        }


_instance: Predictor | None = None


def get_predictor() -> Predictor:
    global _instance
    if _instance is None:
        _instance = Predictor()
    return _instance
