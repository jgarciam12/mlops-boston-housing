# MLOps – Boston Housing Price Prediction
## Descripción general

Este proyecto implementa un pipeline completo de Machine Learning con enfoque MLOps, desde el análisis exploratorio y entrenamiento del modelo hasta su despliegue como una API REST utilizando FastAPI y Docker.

El objetivo es demostrar buenas prácticas de MLOps como:

reproducibilidad

modularidad

automatización

versionado

diseño preparado para producción

## Objetivos del proyecto

Construir un pipeline de entrenamiento de ML reproducible

Persistir y versionar artefactos del modelo

Evaluar múltiples modelos y seleccionar el mejor

Servir el modelo mediante una API REST

Empaquetar la solución usando Docker

Aplicar buenas prácticas de MLOps (modularidad, automatización, escalabilidad)

## Estructura del proyecto
mlops-boston-housing/
│
├── data/
│   ├── raw/                # Dataset original
│   └── processed/          # Datos transformados
│
├── notebooks/
│   └── exploring_data_analysis.ipynb  # Análisis exploratorio
│
├── src/
│   ├── data/
│   │   └── preprocess.py   # Limpieza y feature engineering
│   │
│   ├── training/
│   │   └── train.py        # Entrenamiento y evaluación
│   │
│   ├── inference/
│   │   └── predict.py      # Lógica de inferencia
│   │
│   └── utils/
│       └── io.py           # Punto de extensión para IO y monitoreo
│
├── models/
│   └── model.pkl           # Modelo entrenado
│
├── api/
│   └── main.py             # API FastAPI
│
├── requirements.txt
├── Dockerfile
├── README.md
└── .gitignore

## Dataset

El proyecto utiliza el dataset Boston Housing.

El archivo debe ubicarse en:

data/raw/HousingData.csv


Con la siguiente estructura:

CRIM,ZN,INDUS,CHAS,NOX,RM,AGE,DIS,RAD,TAX,PTRATIO,B,LSTAT,MEDV
0.00632,18,2.31,0,0.538,6.575,65.2,4.09,1,296,15.3,396.9,4.98,24


Variable objetivo: MEDV

CHAS es tratada como variable binaria (0/1)

## Ejecución del proyecto (local)
### 1. Crear entorno virtual
python -m venv .venv
source .venv/bin/activate   # Linux / Mac
.venv\Scripts\activate      # Windows

### 2. Instalar dependencias
pip install -r requirements.txt

#### 3. Entrenamiento del modelo
python -m src.training.train


Este paso:

ejecuta el pipeline de preprocesamiento

evalúa múltiples modelos con GridSearch

registra métricas con MLflow

guarda el mejor modelo en models/model.pkl

### 4. Visualizar métricas con MLflow
mlflow ui


Luego abrir:

http://localhost:5000

### 5. Despliegue de la API
Ejecutar la API localmente
uvicorn api.main:app --reload


Swagger UI:

http://localhost:8000/docs

Ejemplo de request /predict
{
  "CRIM": 0.00632,
  "ZN": 18.0,
  "INDUS": 2.31,
  "CHAS": 0,
  "NOX": 0.538,
  "RM": 6.575,
  "AGE": 65.2,
  "DIS": 4.09,
  "RAD": 1,
  "TAX": 296,
  "PTRATIO": 15.3,
  "B": 396.9,
  "LSTAT": 4.98
}

Endpoint de salud
GET /health


Permite verificar:

estado del servicio

carga del modelo

versión del modelo

## Despliegue con Docker
Construir la imagen
docker build -t boston-housing-api .

Ejecutar el contenedor
docker run -p 8000:8000 boston-housing-api


Acceso:

http://localhost:8000/docs

http://localhost:8000/health

## Monitoreo y reentrenamiento
### Monitoreo de entrenamiento

Durante el entrenamiento se utiliza MLflow para registrar:

- métricas (MAE, R²)

- hiperparámetros

- artefactos del modelo

Esto permite comparar ejecuciones y soportar el reentrenamiento.

### Monitoreo de inferencias (diseño)

Aunque el sistema no está en producción, el diseño contempla:

- logging estructurado de inputs y predicciones

- versionado explícito del modelo servido

- análisis de distribución de predicciones

- detección de data drift

Esto permitiría integrar herramientas como:

- Prometheus / Grafana

- ELK

- análisis batch offline

### Reentrenamiento

El pipeline es reproducible y automatizable, permitiendo reentrenar el modelo cuando:

- cambie la distribución de los datos

- se degrade el desempeño

## Posibles mejoras

- Pruebas unitarias e integración (pytest)

- Análisis automático de calidad de código (ruff, mypy, black)

- Monitoreo real de inferencias y data drift

- Versionado de datos (DVC)

- Orquestación del pipeline (Airflow, Prefect, Dagster)

- Seguridad del API (auth, rate limiting)

- Optimización del entrenamiento (Optuna, early stopping)