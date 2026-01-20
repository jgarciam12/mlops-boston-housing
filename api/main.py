from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd
from pathlib import Path
import logging

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
from src.data.preprocess import FEATURES

# -------------------------------------------------
# Logging
# -------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api")

# -------------------------------------------------
# Model version
# -------------------------------------------------
MODEL_VERSION = "1.0.0"

# -------------------------------------------------
# App
# -------------------------------------------------
app = FastAPI(
    title="Boston Housing Prediction API",
    description="Predict house using a trained ML model",
    version=MODEL_VERSION,
)

# -------------------------------------------------
# Load model
# -------------------------------------------------
MODEL_PATH = Path("models/model.pkl")

try:
    model = joblib.load(MODEL_PATH)
    logger.info("Model loaded")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    model = None

# -------------------------------------------------
# Request schema
# -------------------------------------------------
class HouseFeatures(BaseModel):
    CRIM: float = Field(..., ge=0, description="Per capita crime rate")
    ZN: float = Field(..., ge=0, le=100, description="Residential land zoned (%)")
    INDUS: float = Field(..., ge=0, description="Proportion of non-retail business acres")
    CHAS: int = Field(..., ge=0, le=1, description="1 if tract bounds Charles River, else 0")
    NOX: float = Field(..., ge=0, le=1, description="Nitric oxides concentration")
    RM: float = Field(..., ge=0, description="Average number of rooms")
    AGE: float = Field(..., ge=0, le=100, description="Owner-occupied units built prior to 1940 (%)")
    DIS: float = Field(..., ge=0, description="Distance to employment centers")
    RAD: float = Field(..., ge=0, description="Accessibility to highways")
    TAX: float = Field(..., ge=0, description="Property tax rate")
    PTRATIO: float = Field(..., ge=0, description="Pupil-teacher ratio")
    B: float = Field(..., ge=0, description="1000(Bk - 0.63)^2")
    LSTAT: float = Field(..., ge=0, description="% lower status of population")

class PredictionResponse(BaseModel):
    prediction: float

# -------------------------------------------------
# Endpoints
# -------------------------------------------------
@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_version": MODEL_VERSION,
    }

@app.post("/predict", response_model=PredictionResponse)
def predict(features: HouseFeatures):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not found")

    try:
        # Convert input to DataFrame
        data = pd.DataFrame([features.model_dump()])
        data = data[FEATURES] # enforce column order

        pred = model.predict(data)[0]

        return PredictionResponse(
            prediction=float(pred)
        )

    except Exception as e:
        logger.error(f"Error predicting: {e}")
        raise HTTPException(status_code=400, detail=str(e))

