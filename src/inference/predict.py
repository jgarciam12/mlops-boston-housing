"""
Código exclusivo para generar inferencia de los datos nuevos
"""

import joblib
import pandas as pd
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
from src.data.preprocess import FEATURES

MODEL_PATH = Path("models/model.pkl")

class Predictor:
    def __init__(self, model_path: Path = MODEL_PATH):
        self.model = joblib.load(model_path)

    def predict(self, features: dict) -> float:
        """
        Ejecuta una predicción a partir de un diccionario de features
        """
        df = pd.DataFrame([features])
        df = df[FEATURES]
        prediction = self.model.predict(df)
        return float(prediction[0])
