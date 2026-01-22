"""
Código para servir una API y usar el modelo de predicción entrenado
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd
from pathlib import Path
import logging

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
from src.inference.predict import Predictor


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
# Load predictor
# -------------------------------------------------
try:
    predictor = Predictor()
    logger.info("Predictor initialized successfully")
except Exception as e:
    logger.error(f"Error initializing predictor: {e}")
    predictor = None

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
        "model_loaded": predictor is not None,
        "model_version": MODEL_VERSION,
    }

@app.post("/predict", response_model=PredictionResponse)
def predict(features: HouseFeatures):
    if predictor is None:
        raise HTTPException(status_code=500, detail="Model not found")

    try:
        pred = predictor.predict(features.model_dump())

        return PredictionResponse(
            prediction=float(pred)
        )

    except Exception as e:
        logger.error(f"Error predicting: {e}")
        raise HTTPException(status_code=400, detail=str(e))

