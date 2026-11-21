"""Retraining logic for the garment-productivity API.

This module is deliberately free of any web code: ``prediction.py`` (the FastAPI app) calls
``parse_and_validate_csv`` to check an uploaded file and ``retrain`` to build the updated model.

Strategy (chosen from the type of the best model that was saved by the notebook):
* ``SGDRegressor``      -> keep the fitted encoder/scaler and continue training with ``partial_fit`` on the new rows.
* any other model       -> refit the whole pipeline on the original training data + the new rows.
"""
from __future__ import annotations

import copy
import io
import os
from pathlib import Path
from typing import Type

import joblib
import numpy as np
import pandas as pd
from pydantic import BaseModel, ValidationError
from sklearn.base import clone
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error

# Column order must match the notebook (multivariate.ipynb, section 4).
FEATURES = ["department", "day", "team", "week_of_month", "targeted_productivity", "smv", "wip",
            "over_time", "incentive", "idle_men", "no_of_style_change", "no_of_workers"]
TARGET = "actual_productivity"

PARTIAL_FIT_EPOCHS = 20       # passes over the new rows when the model is an SGDRegressor
MAX_ROWS_PER_UPLOAD = 20_000  # protects the free Render instance (512 MB RAM)


class CSVError(ValueError):
    """The uploaded CSV cannot be used (message is safe to show to the user)."""


def parse_and_validate_csv(raw: bytes, row_schema: Type[BaseModel], max_errors: int = 5) -> pd.DataFrame:
    """Read the uploaded bytes and return a clean DataFrame with FEATURES + TARGET, or raise CSVError."""
    try:
        df = pd.read_csv(io.BytesIO(raw), encoding="utf-8-sig", skipinitialspace=True)
    except Exception as exc:  # empty file, binary file, broken quoting ...
        raise CSVError(f"The file is not a readable CSV: {exc}") from exc

    df.columns = [str(c).strip() for c in df.columns]
    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise CSVError(f"The CSV is missing required column(s): {', '.join(missing)}. "
                       f"Expected columns: {', '.join(FEATURES + [TARGET])}.")
    df = df[FEATURES + [TARGET]]
    if df.empty:
        raise CSVError("The CSV has a header but no rows.")
    if len(df) > MAX_ROWS_PER_UPLOAD:
        raise CSVError(f"Too many rows ({len(df)}). The limit is {MAX_ROWS_PER_UPLOAD} rows per upload.")
    if df.isna().any().any():
        bad = df.isna().sum()
        raise CSVError("The CSV contains empty cells in: " + ", ".join(f"{c} ({n})" for c, n in bad[bad > 0].items()))

    clean, errors = [], []
    for position, record in enumerate(df.to_dict(orient="records")):
        try:
            clean.append(row_schema.model_validate(record).model_dump())
        except ValidationError as exc:
            first = exc.errors()[0]
            errors.append(f"row {position + 2}: {'.'.join(str(p) for p in first['loc'])} - {first['msg']}")
            if len(errors) >= max_errors:
                break
    if errors:
        raise CSVError("Invalid values in the CSV (showing up to "
                       f"{max_errors}): " + "; ".join(errors))
    return pd.DataFrame(clean)[FEATURES + [TARGET]]


def retrain(pipeline, new_rows: pd.DataFrame, training_data: pd.DataFrame):
    """Return (updated_pipeline, updated_training_data, report). The input objects are not modified."""
    model = pipeline.named_steps["model"]
    X_new, y_new = new_rows[FEATURES], new_rows[TARGET]
    mse_before = float(mean_squared_error(y_new, pipeline.predict(X_new)))
    combined = pd.concat([training_data[FEATURES + [TARGET]], new_rows], ignore_index=True)

    if isinstance(model, SGDRegressor):
        updated = copy.deepcopy(pipeline)
        X_new_t = updated.named_steps["prep"].transform(X_new)   # encoder + scaler stay as fitted on the original data
        for _ in range(PARTIAL_FIT_EPOCHS):
            updated.named_steps["model"].partial_fit(X_new_t, y_new)
        strategy = f"SGDRegressor continued with partial_fit ({PARTIAL_FIT_EPOCHS} epochs over the new rows)"
    else:
        updated = clone(pipeline).fit(combined[FEATURES], combined[TARGET])
        strategy = f"{type(model).__name__} refitted on the original training data + the new rows"

    report = {
        "strategy": strategy,
        "rows_received": int(len(new_rows)),
        "training_rows_after": int(len(combined)),
        "mse_on_new_rows_before": round(mse_before, 5),
        "mse_on_new_rows_after": round(float(mean_squared_error(y_new, updated.predict(X_new))), 5),
    }
    return updated, combined, report


def save_atomically(pipeline, training_data: pd.DataFrame, model_path: Path, training_path: Path) -> bool:
    """Write the new model and data to disk (temp file + replace, so a crash never leaves a half-written file).

    Returns False when the disk is read-only; the in-memory model is still used in that case.
    """
    try:
        tmp_model = model_path.with_name(model_path.name + ".tmp")
        joblib.dump(pipeline, tmp_model, compress=3)
        os.replace(tmp_model, model_path)
        tmp_data = training_path.with_name(training_path.name + ".tmp")
        training_data.to_csv(tmp_data, index=False)
        os.replace(tmp_data, training_path)
        return True
    except OSError:
        return False
