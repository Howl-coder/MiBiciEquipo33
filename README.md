@'
# MiBici Mobility Analytics & MLOps Pipeline (Equipo 33)

Pipeline reproducible y modular de Machine Learning para el análisis, pronóstico de demanda y balanceo de estaciones del sistema de bicicletas públicas MiBici en el Área Metropolitana de Guadalajara (AMG).

---

## Objetivos del Proyecto

* **P1 (Regresión):** Predicción de la demanda horaria de salidas por cicloestación.
* **P2 (Clasificación):** Detección oportuna del riesgo de desbalance crítico (saturación o desabasto).
* **P3 (Clustering):** Segmentación funcional y espacial de cicloestaciones para planeación urbana.

---

## Estructura del Repositorio (CCDS)

```text
├── config.py              # Parámetros globales y rutas centralizadas
├── requirements.txt       # Dependencias fijas del entorno
├── .gitignore             # Exclusión estricta de datos crudos y entornos
├── README.md              # Documentación y reproducción del proyecto
├── data/
│   ├── raw/               # CSVs originales de MiBici (en .gitignore)
│   ├── interim/           # Dataset limpio y delimitado al AMG (Parquet)
│   └── processed/         # Malla continua y variables para modelos (Parquet)
├── notebooks/
│   └── 1.0-lcp-exploratorio.ipynb  # Análisis exploratorio de datos (EDA)
├── reports/
│   └── figures/           # Gráficas generadas durante el EDA
└── src/
    ├── __init__.py
    ├── dataset.py         # Ingesta, saneamiento y filtrado espacial AMG
    └── features.py        # Malla continua, variables cíclicas y rezagos (lags)

    git clone [https://github.com/Howl-coder/MiBiciEquipo33.git](https://github.com/Howl-coder/MiBiciEquipo33.git)
cd MiBiciEquipo33

# Crear y activar entorno virtual en PowerShell
python -m venv .venv311
.\.venv311\Scripts\Activate.ps1

# Instalar dependencias exactas
pip install -r requirements.txt

# 1. Ingesta y limpieza espacial hacia data/interim/
python -m src.dataset

# 2. Generación de features y malla regular hacia data/processed/
python -m src.features
