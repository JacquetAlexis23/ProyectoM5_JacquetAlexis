# MLOps Pipeline — Predicción de Pagos Bancarios

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.128-009688?logo=fastapi)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?logo=streamlit)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker)](https://www.docker.com/)
[![SonarCloud](https://img.shields.io/badge/SonarCloud-Quality-F3702A?logo=sonarcloud)](https://sonarcloud.io/)
[![Tests](https://img.shields.io/badge/Tests-27%20passed-brightgreen?logo=pytest)](./tests/)

Sistema MLOps completo para predicción de pagos a tiempo en carteras bancarias.
Expone el modelo de ML vía **FastAPI**, lo visualiza mediante un **dashboard Streamlit**
y orquesta ambos servicios con **Docker Compose**.

> **Autor:** Alexis Jacquet — Henry Data Science Bootcamp, Módulo 5, Febrero 2026

---

## Tabla de Contenidos

1. [Arquitectura](#arquitectura)
2. [Tecnologías](#tecnologías)
3. [Inicio Rápido](#inicio-rápido)
4. [API Reference](#api-reference)
5. [Dashboard Streamlit](#dashboard-streamlit)
6. [Docker](#docker)
7. [Tests y Calidad](#tests-y-calidad)
8. [Estructura del Proyecto](#estructura-del-proyecto)
9. [Resultados del Modelo](#resultados-del-modelo)

---

## Arquitectura

El sistema sigue un flujo lineal desde los datos crudos hasta el servicio en producción:

```
data/Base_de_datos.csv
        |
        v
ft_engineering.py              # Feature Engineering: 35 features
        |
        v
model_training_evaluation.py   # Entrenamiento: 11 algoritmos comparados
        |
        v
model_deploy.py                # Serializa el mejor modelo → models/latest/
        |
     +--+--+
     |     |
     v     v
FastAPI   Streamlit
:8000     :8501
  |         |
  +----+----+
       |
  Docker Compose
  (red interna ml-payment-network)
```

El contenedor Streamlit llama a la API internamente por `http://ml-api:8000`.

---

## Tecnologías

| Capa | Tecnología |
|---|---|
| API REST | FastAPI + Uvicorn |
| Dashboard / UI | Streamlit + Plotly |
| Machine Learning | scikit-learn, XGBoost, LightGBM |
| Contenedores | Docker, Docker Compose |
| Calidad de código | SonarCloud, pylint, flake8, pytest |
| Data | Pandas, NumPy |

---

## Inicio Rápido

### Opción A — Local (sin Docker)

```bash
# 1. Instalar dependencias
pip install -r requirements.txt

# 2. Entrenar modelo y guardarlo en models/latest/
python run_pipeline.py

# 3. Iniciar la API (terminal 1)
python api_main.py
# Disponible en http://localhost:8000

# 4. Iniciar el dashboard (terminal 2)
streamlit run app_streamlit.py
# Disponible en http://localhost:8501
```

### Opción B — Docker Compose (stack completo)

```bash
# 1. Entrenar el modelo (solo la primera vez)
python run_pipeline.py

# 2. Construir imagen y levantar ambos servicios
docker-compose up --build -d

# 3. Ver logs en tiempo real
docker-compose logs -f

# 4. Detener
docker-compose down
```

| Servicio | URL |
|---|---|
| FastAPI | http://localhost:8000 |
| Swagger UI | http://localhost:8000/docs |
| ReDoc | http://localhost:8000/redoc |
| Streamlit | http://localhost:8501 |

---

## API Reference

### Endpoints

| Método | Endpoint | Descripción |
|---|---|---|
| `GET` | `/` | Información general de la API |
| `GET` | `/health` | Estado de la API y del modelo cargado |
| `GET` | `/model/info` | Metadata del modelo en producción |
| `POST` | `/predict` | Predicción individual (1 cliente) |
| `POST` | `/predict/batch` | Predicción por lotes (N clientes) |

### Ejemplo — Predicción individual

```python
import requests

payload = {
    "fecha_prestamo": "2024-06-15",
    "tipo_credito": 1,
    "capital_prestado": 5000000.0,
    "plazo_meses": 36,
    "edad_cliente": 35,
    "tipo_laboral": "Empleado",
    "salario_cliente": 3000000.0,
    "total_otros_prestamos": 500000.0,
    "cuota_pactada": 180000.0,
    "puntaje": 750.0,
    "puntaje_datacredito": 720.0,
    "cant_creditosvigentes": 2,
    "huella_consulta": 5,
    "saldo_mora": 0.0,
    "saldo_total": 4500000.0,
    "saldo_principal": 4500000.0,
    "saldo_mora_codeudor": 0.0,
    "creditos_sectorFinanciero": 1,
    "creditos_sectorCooperativo": 1,
    "creditos_sectorReal": 0,
    "promedio_ingresos_datacredito": 2800000.0,
    "tendencia_ingresos": "Estable"
}

response = requests.post(
    "http://localhost:8000/predict?return_probabilities=true",
    json=payload
)
print(response.json())
```

**Respuesta esperada:**
```json
{
  "predictions": [1],
  "probability_class_1": [0.95],
  "interpretation": ["✅ Cliente pagará a tiempo (bajo riesgo)"],
  "n_samples": 1,
  "model_version": "v20260223_120000",
  "timestamp": "2026-02-23T12:00:00"
}
```

**Etiquetas de predicción:** `1` = pagará a tiempo · `0` = riesgo de mora

### Ejemplo — Predicción batch

```python
batch_payload = {
    "data": [payload, payload, payload],
    "return_probabilities": True
}
response = requests.post("http://localhost:8000/predict/batch", json=batch_payload)
```

---

## Dashboard Streamlit

La aplicación cuenta con 6 pestañas:

| Tab | Contenido |
|---|---|
| Dashboard General | KPIs de data drift, gauges PSI / KS-Statistic / JS-Divergence |
| Análisis de Features | Tabla interactiva con drift detectado por feature |
| Distribuciones | Comparación gráfica baseline (train) vs. producción (test) |
| Análisis Temporal | Evolución histórica del nivel de drift |
| Recomendaciones | Plan de acción automático según nivel de alerta |
| **Predicción API** | Formulario que llama a `/predict` en tiempo real |

El tab **Predicción API** conecta directamente con la FastAPI: recibe los datos del
formulario, realiza un `POST /predict` o `/predict/batch` y muestra el resultado
con barra de probabilidad. La URL de la API es configurable desde la barra lateral
(por defecto usa la variable de entorno `API_URL`, útil en Docker).

---

## Docker

Una sola imagen base para ambos servicios. El `docker-compose.yml` sobreescribe
el `CMD` del servicio `ml-dashboard` para ejecutar Streamlit en lugar de Uvicorn.

```
Dockerfile (python:3.11-slim)
    ├── ml-api       → CMD: uvicorn api_main:app --port 8000
    └── ml-dashboard → CMD: streamlit run app_streamlit.py --port 8501
```

Los modelos **no se incluyen en la imagen** — se montan como volumen de solo lectura:

```yaml
volumes:
  - ./models:/app/models:ro
```

Esto permite actualizar el modelo sin reconstruir la imagen.

---

## Tests y Calidad

```bash
# Ejecutar los 27 tests unitarios
pytest tests/ -v

# Por módulo
pytest tests/test_api.py                  # 8  tests — endpoints y validación Pydantic
pytest tests/test_feature_engineering.py  # 11 tests — transformaciones de features
pytest tests/test_model_deploy.py         # 8  tests — clase ModelDeployment
```

El análisis de calidad se ejecuta automáticamente en cada push a las ramas
`main`, `developer` o `certification` mediante GitHub Actions
(`.github/workflows/sonarcloud.yml`):

1. Instala dependencias
2. Corre pytest con cobertura → genera `coverage.xml`
3. Corre pylint y flake8
4. Envía todos los reportes a SonarCloud

---

## Estructura del Proyecto

```
.
├── api_main.py                         # FastAPI — 5 endpoints REST
├── app_streamlit.py                    # Dashboard Streamlit — 6 tabs
├── run_pipeline.py                     # Pipeline de entrenamiento end-to-end
│
├── Dockerfile                          # Imagen Python 3.11-slim, usuario no-root
├── docker-compose.yml                  # Stack: ml-api (:8000) + ml-dashboard (:8501)
├── requirements.txt                    # Dependencias Python
│
├── mlops_pipeline/
│   └── src/
│       ├── ft_engineering.py           # Feature Engineering (35 features)
│       ├── model_training_evaluation.py  # Entrenamiento y evaluación de 11 modelos
│       ├── model_deploy.py             # Clase ModelDeployment (save/load/predict)
│       └── model_monitoring.py         # Detección de Data Drift (KS, PSI, JS)
│
├── tests/
│   ├── test_api.py                     # Tests de endpoints y validación Pydantic
│   ├── test_feature_engineering.py     # Tests de transformaciones
│   └── test_model_deploy.py            # Tests de ModelDeployment
│
├── data/
│   └── Base_de_datos.csv               # Dataset — 10,763 registros, 21 features
│
├── results/                            # Reportes y gráficos generados
├── sonar-project.properties            # Configuración SonarCloud
└── .github/workflows/sonarcloud.yml    # CI: tests + análisis de calidad de código
```

---

## Resultados del Modelo

11 algoritmos entrenados y evaluados sobre 10,763 registros con 35 features engineered.

| Modelo | ROC-AUC | F1-Score | Accuracy | Tiempo |
|---|---|---|---|---|
| **DecisionTreeClassifier** ⭐ | **1.0000** | **1.0000** | **1.0000** | **0.04s** |
| RandomForestClassifier | 1.0000 | 1.0000 | 1.0000 | 0.21s |
| GradientBoostingClassifier | 1.0000 | 1.0000 | 1.0000 | 1.66s |
| AdaBoostClassifier | 1.0000 | 1.0000 | 1.0000 | 0.02s |
| XGBClassifier | 1.0000 | 1.0000 | 1.0000 | 0.15s |
| LogisticRegression | 1.0000 | 0.9985 | 0.9986 | 0.02s |

Ver análisis completo, hiperparámetros y matriz de confusión en [RESULTADOS.md](RESULTADOS.md).

---

## Licencia

Proyecto educativo — Henry Data Science Bootcamp.
