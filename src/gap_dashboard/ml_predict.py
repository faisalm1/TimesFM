"""Load trained LightGBM model (realized-gap labels) and score the latest bar only."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from gap_dashboard.config import ROOT
from gap_dashboard.ml_features import FEATURE_ORDER, MIN_HISTORY_ROWS, feature_vector

_MODEL: Any = None
_FEATURES_PATH: Path | None = None
_MODEL_PATH: Path | None = None


def ml_artifact_dir() -> Path:
    d = ROOT / "data" / "ml"
    d.mkdir(parents=True, exist_ok=True)
    return d


def model_path() -> Path:
    return ml_artifact_dir() / "lgbm_gap.pkl"


def features_manifest_path() -> Path:
    return ml_artifact_dir() / "features.json"


def load_ml_model():
    """Return (sklearn LGBMClassifier or None, reason_if_none)."""
    global _MODEL, _FEATURES_PATH, _MODEL_PATH
    mp = model_path()
    fp = features_manifest_path()
    if _MODEL is not None and _MODEL_PATH == mp and _FEATURES_PATH == fp and mp.exists():
        return _MODEL, None
    if not mp.exists() or not fp.exists():
        return None, "no_model_files"
    order = json.loads(fp.read_text(encoding="utf-8"))
    if order != FEATURE_ORDER:
        return None, "feature_mismatch"
    try:
        import joblib
    except ImportError:
        return None, "joblib_missing"
    _MODEL = joblib.load(mp)
    _MODEL_PATH = mp
    _FEATURES_PATH = fp
    return _MODEL, None


def ml_probability_last_bar(df: pd.DataFrame) -> tuple[float | None, str | None]:
    """
    P(label | features at last row). Features use only history through last close.
    Returns (prob_positive, skip_reason).
    """
    model, reason = load_ml_model()
    if model is None:
        return None, reason
    if len(df) <= MIN_HISTORY_ROWS:
        return None, "short_history"
    idx = len(df) - 1
    x = feature_vector(df, idx).reshape(1, -1)
    try:
        proba = model.predict_proba(x)[0]
        # binary: class 1 = event
        p = float(proba[1]) if proba.shape[0] > 1 else float(proba[0])
    except Exception:
        return None, "predict_failed"
    return p, None
