# MLOps Pipeline — Credit Risk Scoring & Model Monitoring

> **Predicción de pagos en el sector financiero con detección de Data Drift en tiempo real**

**Autor:** Alexis Jacquet · **Programa:** Henry Data Science Bootcamp · M5  
**Versión:** 3.0.0 · **Fecha:** Febrero 2026  

---

## Caso de Negocio

Las instituciones financieras enfrentan riesgo crediticio significativo al otorgar préstamos. La morosidad deteriora la cartera, incrementa los costos operativos y afecta la liquidez. Este proyecto aborda ese problema con un sistema de **scoring automatizado** y **monitoreo continuo**:

- Predice la probabilidad de pago atrasado **antes** de aprobar cada crédito
- Detecta **data drift** para alertar cuando el perfil de clientes se aleja del baseline de entrenamiento
- Proporciona un **dashboard interactivo** para que el equipo de riesgo opere con visibilidad total

**Impacto estimado:** reducción de mora hasta 40% con predicción temprana.

---

## Resultados

| Modelo | ROC-AUC | F1-Score | Accuracy | Tiempo |
|--------|---------|----------|----------|--------|
| **DecisionTreeClassifier** ⭐ | 1.0000 | 1.0000 | 1.0000 | 0.04s |
| RandomForestClassifier | 1.0000 | 1.0000 | 1.0000 | 0.21s |
| GradientBoostingClassifier | 1.0000 | 1.0000 | 1.0000 | 1.66s |
| AdaBoostClassifier | 1.0000 | 1.0000 | 1.0000 | 0.02s |
| XGBClassifier | 1.0000 | 1.0000 | 1.0000 | 0.15s |

**11 modelos evaluados** · **35 features engineered** · **10,763 registros procesados**

> Análisis técnico completo en [RESULTADOS.md](RESULTADOS.md)

---

## Quick Start

### 1. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 2. Ejecutar pipeline de entrenamiento

```bash
python run_pipeline.py
```

**Salida en `results/`:** gráficas comparativas, curvas ROC, matrices de confusión, reporte de evaluación.  
**Tiempo estimado:** ~8 segundos.

### 3. Lanzar el Dashboard de Monitoreo

```bash
streamlit run app_streamlit.py
```

**URL:** `http://localhost:8501`

---

## Dashboard — MLOps Monitor

La aplicación Streamlit ofrece **5 secciones** de análisis en tiempo real:

| Tab | Contenido |
|-----|-----------|
| **Dashboard General** | Estado del sistema, gauges PSI/KS/JS, información del dataset |
| **Análisis de Features** | Tabla filtrable con métricas por feature, heatmap de drift |
| **Distribuciones** | Comparación Baseline vs Current con Plotly interactivo |
| **Análisis Temporal** | Evolución del drift en el tiempo, tendencias |
| **Recomendaciones** | Plan de acción contextualizado según nivel de alerta |

### Sistema de Alertas (4 niveles)

```
🟢 GREEN   → Drift < 10% features    Sin acción requerida
🟡 YELLOW  → Drift 10–20% features   Monitoreo aumentado
🟠 ORANGE  → Drift 20–40% features   Investigación requerida
🔴 RED     → Drift > 40% features    Acción inmediata / reentrenamiento
```

### Métricas estadísticas implementadas

| Métrica | Aplicación | Umbral default |
|---------|-----------|----------------|
| **KS Test** | Variables numéricas | p-value < 0.05 |
| **PSI** | Variables numéricas | ≥ 0.10 moderado / ≥ 0.20 crítico |
| **Jensen-Shannon Divergence** | Variables numéricas | ≥ 0.10 |
| **Chi² Test** | Variables categóricas | p-value < 0.05 |

---

## Estructura del Proyecto

```
mlops-credit-scoring/
│
├── README.md                          # Este archivo
├── RESULTADOS.md                      # Análisis técnico completo de modelos
├── requirements.txt                   # Dependencias Python
├── set_up.bat                         # Instalación con un clic (Windows)
├── ejecutar_dashboard.bat             # Lanzar dashboard (Windows)
│
├── app_streamlit.py                   # 🚀 Dashboard de Monitoreo
├── run_pipeline.py                    # ⚡ Pipeline de entrenamiento
├── main.py                            # Menú integrado (train + dashboard)
│
├── data/
│   └── Base_de_datos.csv              # Dataset financiero (10,763 registros, 23 cols)
│
├── mlops_pipeline/src/
│   ├── ft_engineering.py              # Feature Engineering Pipeline
│   ├── model_training_evaluation.py   # Entrenamiento y evaluación de 11 modelos
│   ├── model_monitoring.py            # Data Drift Detection System
│   ├── model_deploy.py                # Utilidades de despliegue
│   ├── comprension_eda.ipynb          # Análisis exploratorio de datos
│   ├── Cargar_datos.ipynb             # Exploración inicial del dataset
│   └── Analisis_Resultados_Modelos.ipynb  # Análisis de resultados
│
└── results/
    ├── model_comparison.png           # Comparación visual de todos los modelos
    ├── roc_curves.png                 # Curvas ROC Top 5
    ├── confusion_matrices.png         # Matrices de confusión
    ├── evaluation_report.txt          # Reporte textual detallado
    └── model_results.csv              # Tabla de métricas completa
```

---

## Pipeline Técnico

### Feature Engineering (`ft_engineering.py`)

Pipeline modular con `ColumnTransformer` de scikit-learn:

```
Entrada: 23 columnas raw → Salida: 35 features procesadas
```

| Grupo | Cantidad | Transformación |
|-------|----------|----------------|
| Numéricas base | 19 | Imputación median + StandardScaler |
| Categóricas nominales | 1 | OneHotEncoder (`tipo_laboral`) |
| Categóricas ordinales | 1 | OrdinalEncoder (`tendencia_ingresos`) |
| Features de fecha | 4 | Extracción temporal (mes, día semana, trimestre) |
| Features financieras | 10 | Ratios calculados (deuda/ingreso, cuota/salario, ...) |

### Model Training (`model_training_evaluation.py`)

11 algoritmos evaluados con validación cruzada y métricas de clasificación binaria:

```python
modelos = [
    LogisticRegression, DecisionTreeClassifier, RandomForestClassifier,
    GradientBoostingClassifier, AdaBoostClassifier, XGBClassifier,
    LGBMClassifier, ExtraTreesClassifier, SVC, KNeighborsClassifier, GaussianNB
]
```

Criterio de selección: **ROC-AUC → F1-Score → Tiempo de entrenamiento**

### Data Drift Detection (`model_monitoring.py`)

```python
from model_monitoring import DataDriftDetector

detector = DataDriftDetector(ks_threshold=0.05, psi_threshold=0.1, js_threshold=0.1)
detector.fit(X_train_df, feature_names)
results = detector.detect_drift(X_test_df)
alert = detector.generate_alert_message(results)
```

---

## Git Workflow

```
feature/* → developer → certification (QA) → main (producción)
```

| Rama | Propósito |
|------|-----------|
| `developer` | Desarrollo activo y experimentos |
| `certification` | QA, testing, validación de resultados |
| `main` | Producción — código aprobado |

---

## Tecnologías

`Python 3.10+` · `scikit-learn` · `XGBoost` · `LightGBM` · `Streamlit` · `Plotly` · `Pandas` · `NumPy` · `SciPy`

---

*Proyecto académico desarrollado en el Bootcamp de Data Science de Henry — Módulo 5*
