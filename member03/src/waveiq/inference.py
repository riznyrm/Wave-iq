"""Validated raw-input prediction from the predictor frozen by development CV."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

from .data import sha256_file
from .features import BASE_NUM, RAW_FEATURES


def validate_inputs(data: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(set(RAW_FEATURES) - set(data.columns))
    if missing:
        raise ValueError(f"Missing input columns: {missing}")
    result = data[RAW_FEATURES].copy()
    if result.empty:
        raise ValueError("At least one input row is required.")
    for column in BASE_NUM:
        numeric = pd.to_numeric(result[column], errors="coerce")
        invalid_text = result[column].notna() & numeric.isna()
        if invalid_text.any() or np.isinf(numeric.to_numpy(dtype=float)).any():
            raise ValueError(f"{column} contains nonnumeric or infinite input values.")
        if numeric.lt(0).any():
            raise ValueError(f"{column} must not be negative.")
        result[column] = numeric
    return result


def load_predictor(project_root: str | Path, model_name: str | None = None):
    root = Path(project_root).resolve()
    environment = json.loads((root / "artifacts/environment.json").read_text(encoding="utf-8"))
    if sklearn.__version__ != environment["scikit-learn"]:
        raise RuntimeError("Install the pinned requirements before loading these models, or retrain in your environment.")
    selection = json.loads((root / "artifacts/selection.json").read_text(encoding="utf-8"))
    name = model_name or selection["selected_predictor"]
    if name not in selection["model_sha256"]:
        raise ValueError(f"Unknown saved model: {name!r}")
    model_path = root / "artifacts" / f"{name}.joblib"
    if sha256_file(model_path) != selection["model_sha256"][name]:
        raise ValueError(f"Saved model hash mismatch: {name}")
    return joblib.load(model_path), name


def predict_frame(data: pd.DataFrame, project_root: str | Path, model_name: str | None = None) -> pd.DataFrame:
    clean = validate_inputs(data)
    model, name = load_predictor(project_root, model_name)
    prediction = model.predict(clean)
    if not np.isfinite(prediction).all() or np.any(prediction < 0):
        raise RuntimeError("Model returned an invalid prediction.")
    result = data.copy()
    result["Predicted_Rejection"] = prediction
    result["Model"] = name
    result["Prediction_Unit"] = "unverified_recorded_amount"
    return result
