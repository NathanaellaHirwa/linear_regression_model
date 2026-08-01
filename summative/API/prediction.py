"""
Crop Yield Prediction API
Task 2 - Summative Assignment

Serves the best-performing model (RandomForest, saved from Task 1) behind a
FastAPI POST /predict endpoint, with Pydantic-enforced data types + range
constraints, explicit CORS configuration, and a /retrain endpoint that lets
the model be updated when new labelled data becomes available.

Run locally:
    uvicorn prediction:app --reload

Swagger UI (interactive docs): http://127.0.0.1:8000/docs
"""

import io
import json
import os
from enum import Enum
from typing import List

import joblib
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best_model.joblib")
METADATA_PATH = os.path.join(BASE_DIR, "model_metadata.json")

with open(METADATA_PATH) as f:
    METADATA = json.load(f)

model = joblib.load(MODEL_PATH)

AREA_OPTIONS = METADATA["categorical_options"]["Area"]
ITEM_OPTIONS = METADATA["categorical_options"]["Item"]
RANGES = METADATA["numeric_ranges"]

# Build Enums dynamically from the exact categories the model was trained on,
# so the API can never receive an Area/Item the model hasn't seen.
AreaEnum = Enum("AreaEnum", {a.replace(" ", "_").replace("'", "").replace(",", ""): a for a in AREA_OPTIONS})
ItemEnum = Enum("ItemEnum", {i.replace(" ", "_").replace("'", "").replace(",", ""): i for i in ITEM_OPTIONS})

# ---------------------------------------------------------------------------
# FastAPI app + CORS
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Crop Yield Prediction API",
    description="Predicts crop yield (hg/ha) from country, crop, year, rainfall, "
                "pesticide use, and average temperature. Built for the ALU ML "
                "summative assignment.",
    version="1.0.0",
)

# CORS configuration reasoning:
# - allow_origins: kept to a small explicit allow-list (Flutter app running on
#   localhost during dev, plus "*" fallback removed on purpose) instead of the
#   wildcard "*", because this API accepts POST requests that could be abused
#   for scraping/spamming predictions if left fully open. In production this
#   list should be the deployed Flutter web build's origin / mobile app has no
#   origin restriction since it isn't a browser context.
# - allow_methods: only GET and POST are needed (read docs, submit predictions/
#   retrain) - no PUT/DELETE/PATCH since the API has no mutable resource state
#   beyond the model file itself, which is only touched by /retrain (POST).
# - allow_headers: Content-Type (JSON/multipart bodies) and Authorization
#   (future auth) are the only headers the API actually reads.
# - allow_credentials: False - the API does not use cookies/session auth, so
#   there is no reason to allow credentialed cross-origin requests.
ALLOWED_ORIGINS = [
    "http://localhost",
    "http://localhost:3000",
    "http://localhost:8080",
    "http://127.0.0.1",
    "http://127.0.0.1:8080",
    "https://sorataxai.com",
    "https://aaatrustees.rw",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)


# ---------------------------------------------------------------------------
# Pydantic request/response schemas — typed + range-constrained
# ---------------------------------------------------------------------------
class YieldPredictionRequest(BaseModel):
    area: AreaEnum = Field(..., description="Country where the crop is grown")
    item: ItemEnum = Field(..., description="Crop type")
    year: int = Field(
        ..., ge=RANGES["Year"][0], le=RANGES["Year"][1] + 10,
        description=f"Harvest year (dataset range {RANGES['Year'][0]}-{RANGES['Year'][1]}, "
                     f"extended slightly for near-future forecasts)"
    )
    average_rain_fall_mm_per_year: float = Field(
        ..., ge=0, le=RANGES["average_rain_fall_mm_per_year"][1] * 1.2,
        description="Average annual rainfall in mm"
    )
    pesticides_tonnes: float = Field(
        ..., ge=0, le=RANGES["pesticides_tonnes"][1] * 1.2,
        description="Pesticide use in tonnes"
    )
    avg_temp: float = Field(
        ..., ge=-10, le=RANGES["avg_temp"][1] + 10,
        description="Average temperature in degrees Celsius"
    )

    class Config:
        use_enum_values = True
        json_schema_extra = {
            "example": {
                "area": "Rwanda",
                "item": "Cassava",
                "year": 2024,
                "average_rain_fall_mm_per_year": 1212.0,
                "pesticides_tonnes": 157.0,
                "avg_temp": 19.39,
            }
        }


class YieldPredictionResponse(BaseModel):
    predicted_yield_hg_per_ha: float
    predicted_yield_tonnes_per_ha: float
    model_used: str


class RetrainResponse(BaseModel):
    message: str
    rows_used: int
    previous_test_r2: float
    new_test_r2: float
    new_test_mse: float


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/")
def root():
    return {
        "message": "Crop Yield Prediction API is running.",
        "docs": "/docs",
        "predict_endpoint": "/predict",
        "retrain_endpoint": "/retrain",
    }


@app.get("/health")
def health():
    return {"status": "ok", "model": METADATA["best_model_name"]}


@app.post("/predict", response_model=YieldPredictionResponse)
def predict(payload: YieldPredictionRequest):
    row = pd.DataFrame([{
        "Area": payload.area,
        "Item": payload.item,
        "Year": payload.year,
        "average_rain_fall_mm_per_year": payload.average_rain_fall_mm_per_year,
        "pesticides_tonnes": payload.pesticides_tonnes,
        "avg_temp": payload.avg_temp,
    }])

    try:
        prediction = float(model.predict(row)[0])
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {exc}")

    return YieldPredictionResponse(
        predicted_yield_hg_per_ha=round(prediction, 2),
        predicted_yield_tonnes_per_ha=round(prediction / 10000, 4),
        model_used=METADATA["best_model_name"],
    )


@app.post("/retrain", response_model=RetrainResponse)
async def retrain(file: UploadFile = File(...)):
    """
    Upload a CSV with the same schema as the training data
    (Area, Item, Year, hg/ha_yield, average_rain_fall_mm_per_year,
    pesticides_tonnes, avg_temp) to retrain the model on new/expanded data.
    The newly trained model replaces best_model.joblib on disk.
    """
    global model

    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a .csv file")

    contents = await file.read()
    try:
        new_df = pd.read_csv(io.BytesIO(contents))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {exc}")

    required_cols = {"Area", "Item", "Year", "hg/ha_yield",
                      "average_rain_fall_mm_per_year", "pesticides_tonnes", "avg_temp"}
    missing = required_cols - set(new_df.columns)
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing columns: {missing}")

    new_df = new_df.dropna(subset=list(required_cols))
    if len(new_df) < 20:
        raise HTTPException(status_code=400, detail="Need at least 20 valid rows to retrain")

    features = ["Area", "Item", "Year", "average_rain_fall_mm_per_year", "pesticides_tonnes", "avg_temp"]
    X = new_df[features]
    y = new_df["hg/ha_yield"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), ["Year", "average_rain_fall_mm_per_year", "pesticides_tonnes", "avg_temp"]),
            ("cat", OneHotEncoder(handle_unknown="ignore"), ["Area", "Item"]),
        ]
    )
    new_pipeline = Pipeline([
        ("prep", preprocessor),
        ("model", RandomForestRegressor(n_estimators=80, max_depth=12, min_samples_leaf=2,
                                         random_state=42, n_jobs=-1)),
    ])
    new_pipeline.fit(X_train, y_train)

    new_pred = new_pipeline.predict(X_test)
    new_r2 = r2_score(y_test, new_pred)
    new_mse = mean_squared_error(y_test, new_pred)
    previous_r2 = METADATA.get("test_r2", 0.0)

    # Persist the retrained model + update metadata in place
    joblib.dump(new_pipeline, MODEL_PATH)
    METADATA["test_r2"] = float(new_r2)
    METADATA["test_mse"] = float(new_mse)
    with open(METADATA_PATH, "w") as f:
        json.dump(METADATA, f, indent=2)

    model = new_pipeline

    return RetrainResponse(
        message="Model retrained and saved successfully.",
        rows_used=len(new_df),
        previous_test_r2=previous_r2,
        new_test_r2=float(new_r2),
        new_test_mse=float(new_mse),
    )
