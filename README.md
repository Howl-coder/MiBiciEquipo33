# MiBici Mobility Analytics & MLOps Pipeline (Equipo 33)

Pipeline reproducible de Machine Learning para pronosticar la demanda, detectar el riesgo de desbalance y segmentar las cicloestaciones del sistema de bicicletas públicas MiBici en el Área Metropolitana de Guadalajara (AMG).

## Equipo 33

- Leonardo Contreras Palomino
- Sofía Donlucas Bañuelos
- Gerardo Duarte Ceballos
- Alejandro Gutiérrez Saldaña

## Flujo de trabajo con Git y GitHub

Usamos GitHub Flow: los cambios se desarrollan en ramas y se integran a `main` mediante Pull Requests (PR). No se hacen cambios directamente en `main`.

- **Ramas:** `feat/<descripcion>`, `fix/<descripcion>` o `docs/<descripcion>`.
- **Commits:** usamos Conventional Commits, por ejemplo `docs: documentar flujo de trabajo`.
- **Revisión:** cada PR debe ser revisado por al menos otro integrante antes de integrarse.
- **Protección de `main`:** los cambios deben entrar mediante PR. La protección de la rama se configura en GitHub.

## Preguntas del proyecto

| | Tipo | Pregunta | Target |
|---|---|---|---|
| **P1** | Regresión | ¿Cuántos viajes iniciarán en una estación en la próxima hora? | `demanda_salidas` |
| **P2** | Clasificación | ¿Una estación estará en riesgo de quedarse sin bicis o sin anclajes en las próximas horas? | `en_riesgo_desbalance` |
| **P3** | Clustering | ¿Qué grupos de estaciones comparten patrones de uso y dónde se ubican? | `cluster_estacion` |

El planteamiento completo está en el [ML Canvas](references/ml_canvas_v1.2.docx) y en la [definición de datos y pipeline](references/definicion_datos_y_pipeline.docx). El flujo de punta a punta está en el [diagrama del pipeline](references/diagrama_pipeline.md).

## Datos

- **Fuente:** [Datos abiertos de MiBici](https://www.mibici.net/es/datos-abiertos/), viajes mensuales y nomenclatura de estaciones.
- **Rango:** 2026-04-01 a 2026-09-30 (6 meses).
- **Partición temporal:** entrenamiento de abril a agosto y prueba en septiembre. No se usa partición aleatoria, para no filtrar información del futuro.
- **Archivos esperados en `data/raw/`:** `datos_abiertos_2026_04.csv` … `datos_abiertos_2026_09.csv` y `nomenclatura_2026_09.csv`. Se descargan del portal y **no se versionan en Git**.

### Versionado de datos

[references/manifiesto_datos.json](references/manifiesto_datos.json), que sí está en Git, guarda el tamaño y el SHA-256 de cada archivo crudo y un hash global de la versión (`data_version` = `mibici-datos-abiertos-2026_04-2026_09`).

- `python -m src.dataset` compara los CSV locales contra el manifiesto y **se detiene si no coinciden**, indicando qué archivo cambió.
- Cada run de MLflow verifica los datos otra vez y registra `data_version` y `data_sha256`.
- Para registrar una versión nueva a propósito, por ejemplo al agregar un mes, ejecuta `python -m src.dataset --actualizar-manifiesto` y haz commit del manifiesto.

## Estructura del repositorio

Generada con [Cookiecutter Data Science](https://cookiecutter-data-science.drivendata.org/) (`ccds`, `module_name = src`).

```text
├── Makefile               # make requirements | lint | format | clean
├── README.md
├── config.py              # Rutas, fechas, filtros AMG y configuración de MLflow
├── pyproject.toml         # Metadatos del proyecto y configuración de ruff
├── requirements.txt       # Dependencias con versión fija (Python 3.12)
├── data/                  # No versionado
│   ├── raw/               # CSV originales de MiBici
│   ├── interim/           # viajes_amg_limpios.parquet
│   └── processed/         # features_demanda_p1.parquet
├── docs/
├── models/                # Modelos locales (no versionados; los oficiales viven en MLflow)
├── notebooks/
│   └── 1.0-lcp-exploratorio.ipynb   # Análisis exploratorio (EDA)
├── references/
│   ├── ml_canvas_v1.2.docx
│   ├── definicion_datos_y_pipeline.docx
│   ├── diagrama_pipeline.md
│   └── manifiesto_datos.json        # Versión de los datos crudos (SHA-256)
├── reports/
│   └── figures/           # Figuras generadas (también se guardan como artefactos en MLflow)
└── src/
    ├── dataset.py         # Verificación de versión, limpieza y filtrado AMG
    ├── features.py        # Malla horaria, variables cíclicas, rezagos, targets P2 y perfiles P3
    ├── tracking.py        # Utilidades comunes de MLflow (tags, parámetros, verificación de datos)
    ├── train_regresion.py       # Baselines P1
    ├── train_clasificacion.py   # Baselines P2
    └── train_clustering.py      # Baseline P3
```

## Reproducción

Requiere Python 3.12. Los comandos son para PowerShell; en Git Bash activa el entorno con `source .venv/Scripts/activate`.

```powershell
git clone https://github.com/Howl-coder/MiBiciEquipo33.git
cd MiBiciEquipo33

python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Copiar los 7 CSV del portal de datos abiertos a data/raw/

python -m src.dataset              # 1. Verifica la versión de los datos y limpia -> data/interim/
python -m src.features             # 2. Malla horaria y variables -> data/processed/
python -m src.train_regresion      # 3. P1
python -m src.train_clasificacion  #    P2
python -m src.train_clustering     #    P3

mlflow ui --backend-store-uri sqlite:///mlflow.db   # 4. Ver resultados en http://127.0.0.1:5000
```

> Para que los runs queden ligados a un commit exacto, ejecuta los scripts con el árbol de trabajo limpio (`git status` sin cambios). Si hay cambios sin commit, el run se etiqueta con `git_dirty=True`.

## Experimento en MLflow

- **Experimento:** `mibici-equipo33`, con backend local `mlflow.db` (no versionado).
- **Tags de cada run:** `equipo`, `pregunta` (`P1`/`P2`/`P3`), `git_commit`, `git_dirty`, `tipo_modelo`, `etapa=baseline`.
- **Parámetros de cada run:** `fecha_inicio`, `fecha_fin`, `fecha_fin_train`, `fecha_inicio_test`, `data_version`, `data_sha256`, archivos usados, `random_state` e hiperparámetros del modelo.
- **Artefactos:** figura de evaluación de cada run y el modelo serializado. P3 agrega además el mapa de clusters, los perfiles horarios, la curva de selección de k y la tabla de asignación de estaciones.

| Run | Pregunta | Modelo |
|---|---|---|
| `P1-baseline-naive-lag1h` | P1 | Persistencia: misma demanda que la hora anterior |
| `P1-baseline-naive-estacional-168h` | P1 | Naive estacional: misma hora de la semana anterior |
| `P1-baseline-histgb-regressor` | P1 | `HistGradientBoostingRegressor` |
| `P2-baseline-logistic-regression` | P2 | `LogisticRegression` (`class_weight=balanced`) |
| `P2-baseline-histgb-classifier` | P2 | `HistGradientBoostingClassifier` (`class_weight=balanced`) |
| `P3-baseline-kmeans-perfiles-estacion` | P3 | `StandardScaler` + `KMeans`, k ∈ [3, 10] elegido por silhouette |

### Resultados baseline (test: septiembre 2026)

| Pregunta | Mejor baseline | Métricas |
|---|---|---|
| P1 | HistGradientBoosting | RMSE 1.48 · MAE 0.84 · R² 0.69 (naive 168 h: RMSE 2.00 · R² 0.44) |
| P2 | HistGradientBoosting | ROC-AUC 0.86 · PR-AUC 0.34 · Recall 0.88 · F1 0.32 (positivos ≈ 6.6 %) |
| P3 | K-Means, k = 3 | Silhouette 0.09 · Davies-Bouldin 2.61 |

### Decisiones de diseño

- **Target de P2:** los datos abiertos no incluyen la ocupación real de las estaciones. Por eso se aproxima como `|balance neto acumulado en t+1..t+3| ≥ 30 % de dpcapacity`. Los dos valores son constantes en `src/train_clasificacion.py`.
- **P3:** el perfil de cada estación se calcula solo con el periodo de entrenamiento. La latitud y la longitud se usan para el mapa, no para agrupar.
- **Calidad de datos:** se eliminan los `Viaje_Id` duplicados, los viajes de menos de 1 minuto o más de 24 horas y las estaciones fuera del AMG o sin registro en la nomenclatura.
- **Privacidad:** `Usuario_Id`, `Genero` y `Anio_de_nacimiento` no se usan como variables de los modelos.
