# MLOps Boston Housing – ML Pipeline & API
Este proyecto implementa un end-to-end de un pipeline de MLOps para entrenar, desplegar y servir un modelo de ML para predecir el precio de viviendas usando el dataset de Boston Housing.

## 🚀 Objetivos del proyecto
- Construir un pipeline de entrenamienrode ML reproducible
- Artefactos del modelo de persistencia y versión
- Servir un modelo de predicción a través de una API REST
- Empaquetamiento del modelo usando Docker
- Usar buenas prácticas de MLOps (modularidad, automatización, escabilidad)

## 📁 Estructura del proyecto
mlops-boston-housing/
│
├── data/
│   ├── raw/                # Dataset original (sin modificar)
│   └── processed/          # Dataset limpio / transformado
├── notebooks/              # EDA para explorar los datos
│   └── exploring_data_analysis.ipynb 
├── src/
│   ├── data/
│   │   └── preprocess.py   # Limpieza y feature engineering
│   │
│   ├── training/
│   │   └── train.py        # Entrenamiento y evaluación
│   │
│   ├── inference/
│   │   └── predict.py      # Lógica de inferencia (usada por la API)
│   │
│   └── utils/
│       └── io.py           # Carga/guardado de modelos y datos
│
├── models/
│   └── model.pkl           # Modelo entrenado (artefacto)
│
├── api/
│   └── main.py             # FastAPI /predict
│
├── scripts/
│   ├── train.sh            # Script de entrenamiento
│   └── serve.sh            # Script para levantar la API
│
├── requirements.txt
├── Dockerfile
├── README.md
└── .gitignore