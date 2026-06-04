"""End-to-end registry workflow smoke test (no GCS)."""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pandas as pd
import pytest

from ceres_package.ml_logic.predict import predict, predict_production
from ceres_package.ml_logic.registry import (
    list_runs,
    load_production,
    load_run,
    promote_to_production,
    registry_root,
)
from ceres_package.ml_logic.train import TemporalSplit, train_and_register


@pytest.fixture
def registry_sandbox(tmp_path, monkeypatch):
    root = tmp_path / "registry"
    monkeypatch.setenv("LOCAL_REGISTRY_PATH", str(root))
    yield root
    if root.exists():
        shutil.rmtree(root)


def _synthetic_mart() -> pd.DataFrame:
    rows = []
    for dept in ("59", "62"):
        for year in range(2014, 2025):
            rows.append(
                {
                    "DEPT_ID": dept,
                    "ANNEE": year,
                    "RENDEMENT": 7.0 + 0.1 * (year - 2014) + (0.5 if dept == "59" else 0),
                    "feat_a": float(year - 2010),
                    "feat_b": 100.0 + year,
                }
            )
    return pd.DataFrame(rows)


def test_full_registry_workflow_sklearn(registry_sandbox):
    mart = _synthetic_mart()
    split = TemporalSplit(train_max_year=2019, test_min_year=2020)

    run_id = train_and_register(
        mart,
        model_id="ridge_default",
        feature_cols=["feat_a", "feat_b"],
        split=split,
        extra_params={"test_tag": "workflow"},
    )

    run_dir = registry_sandbox / run_id
    assert (run_dir / "manifest.json").exists()
    assert (run_dir / "pipeline.joblib").exists()

    loaded = load_run(run_id)
    assert loaded.manifest.metrics["n_test"] > 0
    assert loaded.manifest.metrics["mae"] == loaded.manifest.metrics["mae"]
    assert loaded.pipeline is not None

    promote_to_production(run_id)
    ptr = json.loads((registry_sandbox / "production.json").read_text())
    assert ptr["ridge_default"]["run_id"] == run_id

    prod = load_production("ridge_default")
    assert prod.manifest.run_id == run_id

    scored = predict_production(mart, model_id="ridge_default")
    assert "yield_pred" in scored.columns
    assert (scored["model_run_id"] == run_id).all()

    by_run = predict(mart, run_id=run_id)
    pd.testing.assert_series_equal(
        scored["yield_pred"], by_run["yield_pred"], check_names=False
    )

    runs = list_runs(model_id="ridge_default")
    assert any(m.run_id == run_id for m in runs)


def test_lag1_rule_no_pipeline(registry_sandbox):
    mart = _synthetic_mart()
    split = TemporalSplit(train_max_year=2019, test_min_year=2020)
    run_id = train_and_register(
        mart,
        model_id="lag1",
        feature_cols=[],
        split=split,
    )
    loaded = load_run(run_id)
    assert loaded.pipeline is None
    assert not (registry_sandbox / run_id / "pipeline.joblib").exists()
    scored = predict(mart, run_id=run_id)
    assert scored["yield_pred"].notna().sum() > 0


def test_temporal_split_rejects_overlap(registry_sandbox):
    with pytest.raises(ValueError, match="test_min_year"):
        TemporalSplit(train_max_year=2020, test_min_year=2020)
