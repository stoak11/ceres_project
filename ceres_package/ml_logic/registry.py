"""
Versioned model storage: save_run, load_run, list_runs, production pointer.

Scoring lives in predict.py. GCS upload is explicit (upload_run_to_gcs).
"""
from __future__ import annotations

import json
import os
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import joblib
from sklearn.pipeline import Pipeline

from ceres_package.ml_logic.model_specs import get_model_spec
from ceres_package.params import BUCKET_NAME, GCP_PROJECT, GCS_MODEL_PREFIX, PROJECT_ROOT

MANIFEST_NAME = "manifest.json"
PIPELINE_NAME = "pipeline.joblib"
PRODUCTION_POINTER = "production.json"
SCHEMA_VERSION = "1"


@dataclass
class RunManifest:
    run_id: str
    model_id: str
    created_at: str
    schema_version: str = SCHEMA_VERSION
    feature_cols: list[str] = field(default_factory=list)
    target_col: str = "RENDEMENT"
    dept_col: str = "DEPT_ID"
    year_col: str = "ANNEE"
    train_years: list[int] | None = None
    test_years: list[int] | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunManifest:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class LoadedRun:
    manifest: RunManifest
    pipeline: Pipeline | None
    run_dir: Path | None = None


def registry_root() -> Path:
    """Resolve registry dir from LOCAL_REGISTRY_PATH env (read each call, not at import)."""
    raw = os.environ.get("LOCAL_REGISTRY_PATH")
    root = Path(raw) if raw else Path(PROJECT_ROOT) / "models" / "registry"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _run_dir(run_id: str) -> Path:
    return registry_root() / run_id


def _new_run_id(model_id: str) -> str:
    ts = time.strftime("%Y%m%d-%H%M%S")
    return f"{model_id}_{ts}"


def save_run(
    *,
    model_id: str,
    pipeline: Pipeline | None,
    feature_cols: list[str],
    metrics: dict[str, float] | None = None,
    params: dict[str, Any] | None = None,
    target_col: str = "RENDEMENT",
    dept_col: str = "DEPT_ID",
    year_col: str = "ANNEE",
    train_years: list[int] | None = None,
    test_years: list[int] | None = None,
    run_id: str | None = None,
) -> RunManifest:
    """Write manifest (+ joblib for sklearn) under LOCAL_REGISTRY_PATH. Disk only."""
    spec = get_model_spec(model_id)
    if spec.kind == "sklearn" and pipeline is None:
        raise ValueError(f"model_id={model_id} requires a fitted sklearn Pipeline")
    if spec.kind == "rule" and pipeline is not None:
        raise ValueError(f"model_id={model_id} is rule-based; do not pass pipeline")

    rid = run_id or _new_run_id(model_id)
    out = _run_dir(rid)
    out.mkdir(parents=True, exist_ok=True)

    manifest = RunManifest(
        run_id=rid,
        model_id=model_id,
        created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        feature_cols=list(feature_cols),
        target_col=target_col,
        dept_col=dept_col,
        year_col=year_col,
        train_years=train_years,
        test_years=test_years,
        metrics=dict(metrics or {}),
        params=dict(params or {}),
    )

    (out / MANIFEST_NAME).write_text(
        json.dumps(manifest.to_dict(), indent=2), encoding="utf-8"
    )
    if pipeline is not None:
        joblib.dump(pipeline, out / PIPELINE_NAME)

    print(f"Saved run {rid} -> {out}")
    return manifest


def load_run(run_id: str, *, from_gcs: bool = False) -> LoadedRun:
    """Load manifest + pipeline for a run_id."""
    if from_gcs and not _run_dir(run_id).exists():
        download_run_from_gcs(run_id)

    root = _run_dir(run_id)
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.exists():
        raise FileNotFoundError(f"No run at {root}")

    manifest = RunManifest.from_dict(json.loads(manifest_path.read_text(encoding="utf-8")))
    pipe_path = root / PIPELINE_NAME
    pipeline = joblib.load(pipe_path) if pipe_path.exists() else None
    return LoadedRun(manifest=manifest, pipeline=pipeline, run_dir=root)


def promote_to_production(run_id: str, *, alias: str | None = None) -> None:
    """Set production.json alias -> run_id (explicit step after training)."""
    loaded = load_run(run_id)
    alias = alias or loaded.manifest.model_id
    pointer = {
        "run_id": run_id,
        "model_id": loaded.manifest.model_id,
        "promoted_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    ptr_path = registry_root() / PRODUCTION_POINTER
    all_ptr: dict[str, Any] = {}
    if ptr_path.exists():
        all_ptr = json.loads(ptr_path.read_text(encoding="utf-8"))
    all_ptr[alias] = pointer
    ptr_path.write_text(json.dumps(all_ptr, indent=2), encoding="utf-8")
    print(f"Production alias '{alias}' -> {run_id}")


def load_production(model_id: str | None = None) -> LoadedRun:
    """Load the run pointed to by production.json."""
    ptr_path = registry_root() / PRODUCTION_POINTER
    if not ptr_path.exists():
        raise FileNotFoundError(
            f"No {PRODUCTION_POINTER}. Train, then promote_to_production(run_id)."
        )
    ptr = json.loads(ptr_path.read_text(encoding="utf-8"))
    if model_id is None:
        if len(ptr) != 1:
            raise ValueError(f"Specify model_id; production.json has: {list(ptr)}")
        model_id = next(iter(ptr))
    if model_id not in ptr:
        raise KeyError(f"No production alias for {model_id}. Available: {list(ptr)}")
    return load_run(ptr[model_id]["run_id"])


def list_runs(model_id: str | None = None) -> list[RunManifest]:
    """List manifests sorted by run_id."""
    root = registry_root()
    manifests: list[RunManifest] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        mp = child / MANIFEST_NAME
        if not mp.exists():
            continue
        m = RunManifest.from_dict(json.loads(mp.read_text(encoding="utf-8")))
        if model_id is None or m.model_id == model_id:
            manifests.append(m)
    return manifests


def upload_run_to_gcs(run_id: str) -> None:
    """Upload an existing local run folder to GCS (optional, explicit)."""
    local = _run_dir(run_id)
    if not local.exists():
        raise FileNotFoundError(f"No local run at {local}")
    if not BUCKET_NAME:
        raise RuntimeError("BUCKET_NAME unset; cannot upload")
    from google.cloud import storage

    client = storage.Client(project=GCP_PROJECT)
    bucket = client.bucket(BUCKET_NAME)
    prefix = f"{GCS_MODEL_PREFIX}/{run_id}"
    for path in local.rglob("*"):
        if path.is_file():
            blob_name = f"{prefix}/{path.relative_to(local).as_posix()}"
            bucket.blob(blob_name).upload_from_filename(str(path))
    print(f"Uploaded run to gs://{BUCKET_NAME}/{prefix}")


def download_run_from_gcs(run_id: str) -> None:
    """Download a run from GCS into LOCAL_REGISTRY_PATH."""
    if not BUCKET_NAME:
        raise RuntimeError("BUCKET_NAME required for GCS download")
    from google.cloud import storage

    client = storage.Client(project=GCP_PROJECT)
    prefix = f"{GCS_MODEL_PREFIX}/{run_id}/"
    local = _run_dir(run_id)
    if local.exists():
        shutil.rmtree(local)
    local.mkdir(parents=True, exist_ok=True)
    for blob in client.list_blobs(BUCKET_NAME, prefix=prefix):
        if blob.name.endswith("/"):
            continue
        rel = blob.name[len(prefix) :]
        dest = local / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(str(dest))
    print(f"Downloaded gs://{BUCKET_NAME}/{prefix} -> {local}")
