"""Department id normalization."""

from __future__ import annotations

import pandas as pd

from ceres_package.ndvi.constants import DEPT_ID


def normalize_dept_id(values: pd.Series) -> pd.Series:
    """Zero-pad numeric codes; preserve ``2A`` / ``2B``."""
    s = values.astype(str).str.replace(r"\.0$", "", regex=True).str.strip()

    def _one(code: str) -> str:
        if code.isdigit():
            return code.zfill(2)
        return code.upper()

    return s.map(_one)


def ensure_dept_id_column(df: pd.DataFrame) -> pd.DataFrame:
    """Rename ``department_code`` → ``DEPT_ID`` and normalize."""
    out = df.copy()
    if DEPT_ID in out.columns:
        out[DEPT_ID] = normalize_dept_id(out[DEPT_ID])
        return out
    if "department_code" in out.columns:
        out = out.rename(columns={"department_code": DEPT_ID})
        out[DEPT_ID] = normalize_dept_id(out[DEPT_ID])
    return out
