"""
Este código está diseñado para evaluar varios modelos de ML en la
predicción de la variable MEDV y escoger el mejor y guardarlo
"""

import pandas as pd
import joblib
import logging
from pathlib import Path

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

import mlflow
import mlflow.sklearn

# MLflow configuration
mlflow.set_tracking_uri("file:./mlruns")

import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))
from src.data.preprocess import build_preprocessing_pipeline, FEATURES, TARGET

# -------------------------------------------------
# Logging
# -------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("training")

# -------------------------------------------------
# Paths
# -------------------------------------------------
DATA_PATH = Path("data/raw/HousingData.csv")
MODEL_PATH = Path("models/model.pkl")

# -------------------------------------------------
# Training function
# -------------------------------------------------
def train():
    logger.info(f"Loading data: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)

    X = df[FEATURES]
    y = df[TARGET]

    logger.info(f"Splitting data")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    preprocessor = build_preprocessing_pipeline()

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", LinearRegression()),
        ]
    )

    param_grid = [
        {
            "model": [LinearRegression()]
        },
        {
            "model": [Ridge()],
            "model__alpha": [0.1, 1.0, 10.0]
        },
        {
            "model": [RandomForestRegressor(random_state=42)],
            "model__n_estimators": [100, 200, 300],
            "model__min_samples_split": [2, 5, 10, 20],
        },
        {
            "model": [GradientBoostingRegressor(random_state=42)],
            'model__n_estimators': [100, 250],
            'model__learning_rate': [0.1, 0.05],
            'model__max_depth': [3, 10]
        },
        {
            "model": [XGBRegressor(objective='reg:squarederror')],
            'model__n_estimators': [100, 150],
            'model__learning_rate': [0.1, 0.05],
            'model__max_depth': [3, 10]
        }
    ]

    logger.info("Running grid search")
    grid_search = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        scoring="neg_mean_absolute_error",
        cv=5,
        n_jobs=-1,
        verbose=1
    )
    mlflow.set_experiment("boston_housing_regression")
    with mlflow.start_run():
        grid_search.fit(X_train, y_train)
        mlflow.set_tag(
            "best_model",
            type(grid_search.best_estimator_.named_steps["model"]).__name__
        )
        logger.info("Best model found by GridSearch")
        logger.info(f"Best estimator: {type(grid_search.best_estimator_.named_steps['model']).__name__}")
        logger.info(f"Best params: {grid_search.best_params_}")
        logger.info(f"Best CV MAE: {-grid_search.best_score_:.4f}")

        best_model = grid_search.best_estimator_
        logger.info(f"Best model: {best_model.__class__.__name__}")

        logger.info('Evaluating best model')
        y_pred = best_model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        mlflow.log_params(grid_search.best_params_)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)

        logger.info(f"MAE: {mae:.3f}")
        logger.info(f"R2: {r2:.3f}")

        mlflow.sklearn.log_model(
            best_model,
            artifact_path="model"
        )

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)
    logger.info(f"Saved model to {MODEL_PATH}")

if __name__ == "__main__":
    train()



