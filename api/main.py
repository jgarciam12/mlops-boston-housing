from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
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
# App
# -------------------------------------------------
app = FastAPI(
    title="Boston Housing Prediction API",
    description="Predict house using a trained ML model",
    version="1.0.0",
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
    CRIM: float
    ZN: float
    INDUS: float
    CHAS: int
    NOX: float
    RM: float
    AGE: float
    DIS: float
    RAD: float
    TAX: float
    PTRATIO: float
    B: float
    LSTAT: float

# -------------------------------------------------
# Endpoints
# -------------------------------------------------
@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/predict")
def predict(features: HouseFeatures):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not found")

    try:
        # Convert input to DataFrame
        data = pd.DataFrame([features.model_dump()])
        data = data[FEATURES] # enforce column order

        prediction = model.predict(data)[0]

        return {
            "prediction": float(prediction)
        }

    except Exception as e:
        logger.error(f"Error predicting: {e}")
        raise HTTPException(status_code=400, detail=str(e))

