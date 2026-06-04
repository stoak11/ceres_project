"""
Score rows from a registered run (production alias or specific run_id).
"""
from __future__ import annotations

import pandas as pd

from ceres_package.ml_logic.model_specs import get_model_spec, predict_from_spec
from ceres_package.ml_logic.registry import LoadedRun, load_production, load_run


def predict_frame(loaded: LoadedRun, df: pd.DataFrame) -> pd.Series:
    """Return predictions aligned to df.index (uses manifest feature_cols)."""
    m = loaded.manifest
    spec = get_model_spec(m.model_id)
    return predict_from_spec(
        spec,
        df,
        m.feature_cols,
        pipeline=loaded.pipeline,
        dept_col=m.dept_col,
        year_col=m.year_col,
        target_col=m.target_col,
    )


def predict(
    df: pd.DataFrame,
    *,
    run_id: str | None = None,
    model_id: str | None = None,
) -> pd.DataFrame:
    """
    Add yield_pred, model_run_id, model_id. Use run_id or production model_id.
    """
    if run_id:
        loaded = load_run(run_id)
    elif model_id:
        loaded = load_production(model_id)
    else:
        raise ValueError("Provide run_id or model_id")

    out = df.copy()
    out["yield_pred"] = predict_frame(loaded, df)
    out["model_run_id"] = loaded.manifest.run_id
    out["model_id"] = loaded.manifest.model_id
    return out


def predict_production(df: pd.DataFrame, model_id: str) -> pd.DataFrame:
    """Score with the production alias for model_id."""
    return predict(df, model_id=model_id)
