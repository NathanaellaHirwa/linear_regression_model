"""FastAPI service: predict a garment team's actual productivity for a working day and retrain the model from an uploaded CSV.

Run locally (from this folder):   uv run uvicorn prediction:app --reload
On Render (root directory API):    uvicorn prediction:app --host 0.0.0.0 --port $PORT
Swagger UI:                        /docs
"""
from __future__ import annotations

import logging
import os
import threading
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field

import retrain as retrain_module
from retrain import FEATURES, TARGET, CSVError

logger = logging.getLogger("garment-productivity-api")

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "best_model.pkl"          # loaded from the repository at start-up
TRAINING_DATA_PATH = BASE_DIR / "training_data.csv"
MAX_UPLOAD_BYTES = 5 * 1024 * 1024
MAX_PRODUCTIVITY = 1.2                           # productivity in the data never exceeds 1.12


# --------------------------------------------------------------------------- request / response models
class TeamDayInput(BaseModel):
    """One team-day in the factory. Every field has an enforced type and a realistic range (taken from the training data)."""

    # strict=True: "34" (a string) is NOT silently converted to 34 -> wrong types return a validation error.
    # extra="forbid": unknown fields are rejected instead of being ignored.
    model_config = ConfigDict(
        strict=True,
        extra="forbid",
        json_schema_extra={"examples": [{
            "department": "sewing", "day": "Tuesday", "team": 1, "week_of_month": 2, "targeted_productivity": 0.75,
            "smv": 22.5, "wip": 900, "over_time": 5000, "incentive": 0, "idle_men": 0,
            "no_of_style_change": 0, "no_of_workers": 40}]},
    )

    department: Literal["sewing", "finishing"] = Field(..., description="Production department")
    day: Literal["Saturday", "Sunday", "Monday", "Tuesday", "Wednesday", "Thursday"] = Field(
        ..., description="Working day of the week (the factory does not work on Fridays)")
    team: int = Field(..., ge=1, le=12, description="Team number (1-12)")
    week_of_month: int = Field(..., ge=1, le=5, description="Week of the month (1-5)")
    targeted_productivity: float = Field(..., ge=0.05, le=1.0, description="Productivity target set by management (0.05-1.0)")
    smv: float = Field(..., ge=1, le=100, description="Standard minute value: allocated minutes per garment (task difficulty), 1-100")
    wip: int = Field(..., ge=0, le=30000, description="Work in progress, unfinished garments (0 for the finishing department), 0-30000")
    over_time: int = Field(..., ge=0, le=30000, description="Overtime of the team in minutes (0-30000)")
    incentive: int = Field(..., ge=0, le=5000, description="Financial incentive for the team (0-5000)")
    idle_men: int = Field(..., ge=0, le=50, description="Number of idle workers because of production interruptions (0-50)")
    no_of_style_change: int = Field(..., ge=0, le=5, description="Number of style changes in the product (0-5)")
    no_of_workers: float = Field(..., ge=1, le=100, description="Number of workers in the team (1-100)")


class TrainingRow(TeamDayInput):
    """One row of a retraining CSV: the 12 inputs plus the real (observed) productivity.

    CSV cells are text, so this schema is NOT strict (a cell "34" is read as the number 34), but the same ranges apply.
    """
    model_config = ConfigDict(strict=False, extra="ignore")
    actual_productivity: float = Field(..., ge=0, le=1.5, description="Observed actual productivity (0-1.5)")


class PredictionResponse(BaseModel):
    predicted_productivity: float = Field(..., description="Predicted actual productivity (1.0 = 100% of the planned output)")
    target_productivity: float
    gap: float = Field(..., description="predicted minus target (negative = expected shortfall)")
    status: Literal["meets_target", "below_target"]
    message: str
    model_name: str
    model_version: int = Field(..., description="0 = model from the repository, +1 after every successful /retrain")


class RetrainResponse(BaseModel):
    message: str
    strategy: str
    rows_received: int
    training_rows_after: int
    mse_on_new_rows_before: float
    mse_on_new_rows_after: float
    model_name: str
    model_version: int
    saved_to_disk: bool


# --------------------------------------------------------------------------- model state
@dataclass
class ModelState:
    pipeline: object
    training_data: pd.DataFrame
    version: int = 0

    @property
    def name(self) -> str:
        return type(self.pipeline.named_steps["model"]).__name__


STATE: ModelState | None = None
_retrain_lock = threading.Lock()


def predict_one(pipeline, row: dict) -> float:
    """Same logic as predict_row() in the notebook: ONE team-day in, predicted productivity out (clipped to 0-1.2)."""
    prediction = float(pipeline.predict(pd.DataFrame([row])[FEATURES])[0])
    return float(np.clip(prediction, 0, MAX_PRODUCTIVITY))


@asynccontextmanager
async def lifespan(_: FastAPI):
    global STATE
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model file not found: {MODEL_PATH}. Run the notebook (it saves best_model.pkl) or restore the file.")
    STATE = ModelState(pipeline=joblib.load(MODEL_PATH), training_data=pd.read_csv(TRAINING_DATA_PATH))
    logger.info("Loaded %s from %s", STATE.name, MODEL_PATH)
    yield


app = FastAPI(
    title="Garment team productivity prediction API",
    description=("Predicts a garment team's actual productivity (1.0 = 100% of planned output) from 12 inputs known before the shift, so supervisors "
                 "can act on teams likely to miss their target. `/retrain` accepts a CSV of new labelled rows and updates the model without a redeploy."),
    version="1.0.0",
    lifespan=lifespan,
)

# --------------------------------------------------------------------------- CORS
# WHAT IS ALLOWED: only the browser origins listed below may call this API from a web page - the deployed Render URL
#   (the Swagger UI lives there), local uvicorn/Swagger, and the local Flutter *web* debug server (port 5000). Extra origins can be
#   added with the ALLOWED_ORIGINS environment variable (comma separated). Only the methods the API really uses (GET, POST, OPTIONS
#   for the browser's pre-flight check) and only the two headers it needs (Content-Type for JSON / file upload, Accept) are allowed.
# WHAT IS RESTRICTED: no wildcard "*" anywhere, so an arbitrary website cannot make a visitor's browser call the API; PUT/DELETE/PATCH
#   are refused; allow_credentials=False because the API has no cookies or sessions, so browsers never send credentials and a stolen
#   cookie can never be replayed against it.
# WHY: the least privilege that still lets our own front-ends work. The Flutter *mobile* app is not a browser, so CORS does not apply to it.
ALLOWED_ORIGINS = [
    "https://garment-productivity-api.onrender.com",  # <- replace with your own Render URL if the service has another name
    "http://localhost:8000", "http://127.0.0.1:8000",  # local uvicorn + local Swagger UI
    "http://localhost:5000", "http://127.0.0.1:5000",  # Flutter web debug: flutter run -d chrome --web-port 5000
]
if os.getenv("RENDER_EXTERNAL_URL"):           # Render sets this automatically to the service's public URL
    ALLOWED_ORIGINS.append(os.environ["RENDER_EXTERNAL_URL"].rstrip("/"))
ALLOWED_ORIGINS += [o.strip().rstrip("/") for o in os.getenv("ALLOWED_ORIGINS", "").split(",") if o.strip()]
ALLOWED_ORIGINS = sorted({o for o in ALLOWED_ORIGINS if o != "*"})

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Accept"],
    allow_credentials=False,
)


# --------------------------------------------------------------------------- routes
@app.get("/", tags=["health"], summary="Service status")
def root():
    return {"status": "ok", "docs": "/docs", "model_name": STATE.name, "model_version": STATE.version,
            "training_rows": int(len(STATE.training_data))}


@app.post("/predict", response_model=PredictionResponse, tags=["prediction"], summary="Predict a team's actual productivity for a day")
def predict(team_day: TeamDayInput):
    state = STATE  # read once so a concurrent /retrain cannot change the model mid-request
    row = team_day.model_dump()
    predicted = round(predict_one(state.pipeline, row), 3)
    target = row["targeted_productivity"]
    gap = round(predicted - target, 3)
    meets = gap >= 0
    advice = "expected to meet its target" if meets else \
        f"expected shortfall of {abs(gap):.2f} - consider more workers, overtime or an incentive before the shift"
    return PredictionResponse(
        predicted_productivity=predicted,
        target_productivity=target,
        gap=gap,
        status="meets_target" if meets else "below_target",
        message=f"Predicted productivity {predicted:.2f} vs target {target:.2f}: {advice}.",
        model_name=state.name,
        model_version=state.version,
    )


@app.post("/retrain", response_model=RetrainResponse, tags=["retraining"],
          summary="Upload a CSV of new labelled rows; the model is retrained and used immediately")
def retrain_endpoint(file: UploadFile = File(..., description=f"CSV with columns: {', '.join(FEATURES + [TARGET])}")):
    global STATE
    raw = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"File too large (limit {MAX_UPLOAD_BYTES // (1024 * 1024)} MB).")
    try:
        new_rows = retrain_module.parse_and_validate_csv(raw, TrainingRow)
    except CSVError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    if not _retrain_lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="A retraining job is already running. Try again in a few seconds.")
    try:
        current = STATE
        updated, combined, report = retrain_module.retrain(current.pipeline, new_rows, current.training_data)
        saved = retrain_module.save_atomically(updated, combined, MODEL_PATH, TRAINING_DATA_PATH)
        # Atomic swap: from the next request on, /predict uses the updated model - no redeploy or restart needed.
        STATE = ModelState(pipeline=updated, training_data=combined, version=current.version + 1)
    finally:
        _retrain_lock.release()

    return RetrainResponse(
        message="Model retrained and now used by /predict." + ("" if saved else " (Disk is read-only: the update lives in memory only.)"),
        model_name=STATE.name, model_version=STATE.version, saved_to_disk=saved, **report,
    )


if __name__ == "__main__":  # convenient on Windows:  uv run python prediction.py
    import uvicorn
    uvicorn.run("prediction:app", host="127.0.0.1", port=int(os.getenv("PORT", "8000")), reload=False)
