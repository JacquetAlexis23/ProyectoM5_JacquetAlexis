# 🎯 MLOps Pipeline - Predicción de Pagos

Sistema automatizado de Machine Learning para predecir la probabilidad de pago atrasado en clientes bancarios.

**Autor:** Alexis Jacquet  
**Programa:** Henry - Módulo 5  
**Fecha:** Febrero 2026

---

## 🚀 Quick Start

```bash
# 1. Instalar dependencias
pip install -r requirements.txt

# 2. Ejecutar pipeline completo
python run_pipeline.py
```

**Tiempo de ejecución:** ~8 segundos  
**Salida:** Visualizaciones y reportes en carpeta `results/`

---

## 📊 Resultados Principales

| Modelo | ROC-AUC | F1-Score | Accuracy | Tiempo |
|--------|---------|----------|----------|--------|
| **DecisionTreeClassifier** ⭐ | 1.0000 | 1.0000 | 1.0000 | 0.04s |
| RandomForestClassifier | 1.0000 | 1.0000 | 1.0000 | 0.21s |
| GradientBoostingClassifier | 1.0000 | 1.0000 | 1.0000 | 1.66s |
| AdaBoostClassifier | 1.0000 | 1.0000 | 1.0000 | 0.02s |
| XGBClassifier | 1.0000 | 1.0000 | 1.0000 | 0.15s |

**11 modelos evaluados** | **35 features engineered** | **10,763 registros procesados**

> 📄 Ver detalles completos en [RESULTADOS.md](RESULTADOS.md)

---

## 📁 Estructura del Proyecto

```
ProyectoM5_JacquetAlexis/
│
├── data/Base_de_datos.csv              # Dataset (10,763 registros)
├── requirements.txt               # Dependencias Python
├── set_up.bat                     # Script de instalación Windows
├── run_pipeline.py                # ⚡ Ejecutar todo el pipeline
│
├── mlops_pipeline/                # 🔧 Pipeline principal
│   └── src/
│       ├── ft_engineering.py              # Feature Engineering
│       ├── model_training_evaluation.py   # Training & Evaluation
│       ├── model_deploy.py                # (Avance 3 - API REST)
│       ├── model_monitoring.py            # (Avance 4 - Monitoreo)
│       ├── Cargar_datos.ipynb             # Análisis de carga
│       ├── comprension_eda.ipynb          # Exploratory Data Analysis
│       └── Analisis_Resultados_Modelos.ipynb  # Análisis de resultados
│
└── results/                       # 📈 Outputs generados
    ├── model_comparison.png       # Comparación visual de modelos
    ├── roc_curves.png             # Curvas ROC Top 5
    ├── confusion_matrices.png     # Matrices de confusión
    ├── evaluation_report.txt      # Reporte detallado
    └── model_results.csv          # Tabla de resultados
```

---

## 🔀 Flujo de Ramas (Git Workflow)

Este proyecto sigue un flujo de trabajo profesional con ramas para controlar la calidad y el ciclo de vida del código.

### Ramas Principales

| Rama | Propósito | Responsable | Estado |
|------|-----------|-------------|--------|
| **developer** | Desarrollo activo y experimentos | Equipo de desarrollo | Código en progreso |
| **certification** | QA, testing y certificación final | Equipo QA/Auditor | Código estable para pruebas |
| **main** | Producción y releases finales | DevOps/Lead técnico | Código aprobado y desplegado |

### Proceso de Trabajo

```
developer (desarrollo) → PR → certification (QA/testing) → PR → main (producción)
                                      ↑
                                      ↓ (si falla, vuelta a developer)
```

#### **Paso 1: Desarrollo en `developer`**
- Trabaja en features nuevas y mejoras.
- Realiza commits locales y pruebas básicas.
- Cuando esté listo un avance, prepara un Pull Request.

#### **Paso 2: Pull Request a `certification`**
- **Crea PR**: Desde `developer` → `certification`.
- **Auditoría**: Asigna un compañero como reviewer para revisión de código y funcionalidad.
- **Pruebas**: Ejecuta QA completa, testing y validación de resultados.
- **Aprobación**: Si pasa, se mergea; si no, se devuelven cambios.

#### **Paso 3: Pull Request a `main`**
- **Crea PR**: Desde `certification` → `main`.
- **Revisión final**: Lead técnico valida compliance y estabilidad.
- **Deploy**: Merge aprobado activa el código en producción.

### Roles y Responsabilidades

- **Desarrollador**: Crea código en `developer`, responde a feedback.
- **Auditor/QA**: Revisa PRs, ejecuta pruebas, asegura calidad.
- **Lead Técnico**: Aprueba merges finales, supervisa el proceso.

### Comandos Básicos

```bash
# Cambiar rama
git checkout <rama>

# Crear y push branch
git checkout -b feature/nueva
git push origin feature/nueva

# Crear PR (desde GitHub/GitLab interface)
# Asignar reviewer y esperar aprobación
```

---

## 🛠️ Pipeline Implementado

### 1️⃣ Feature Engineering (`ft_engineering.py`)

Procesamiento automático de datos con **ColumnTransformer**:

- **Numeric Features (19):** Imputación + Escalado estándar
- **Categorical Nominal (1):** OneHotEncoder para `tipo_laboral`
- **Categorical Ordinal (1):** OrdinalEncoder para `tendencia_ingresos`
- **Date Features (4):** Extracción de mes, día, trimestre, días desde época
- **Financial Features (10):** Ratios financieros, niveles de endeudamiento, ingresos disponibles

**Total:** 35 features para entrenamiento

### 2️⃣ Model Training & Evaluation (`model_training_evaluation.py`)

Entrenamiento y comparación de **11 algoritmos**:

- Logistic Regression
- Decision Tree ⭐
- Random Forest
- Gradient Boosting
- XGBoost
- LightGBM
- AdaBoost
- Extra Trees
- Support Vector Machine
- K-Nearest Neighbors
- Gaussian Naive Bayes

**Métricas evaluadas:** ROC-AUC, F1-Score, Accuracy, Precision, Recall, Training Time

### 3️⃣ Visualizaciones Generadas

- **Comparación de modelos:** 4 gráficos (métricas, ROC-AUC, F1 vs tiempo, heatmap)
- **Curvas ROC:** Top 5 modelos con estilos distintivos
- **Matrices de confusión:** Top 4 modelos

---

## 💻 Uso Avanzado

### Ejecutar Feature Engineering únicamente

```python
from mlops_pipeline.src.ft_engineering import load_and_prepare_data

data = load_and_prepare_data('data/Base_de_datos.csv')
print(f"Shape X_train: {data['X_train'].shape}")
print(f"Shape X_test: {data['X_test'].shape}")
```

### Entrenar modelos específicos

```python
from mlops_pipeline.src.model_training_evaluation import train_multiple_models

models, results, best_model = train_multiple_models(
    X_train=data['X_train'],
    y_train=data['y_train'],
    X_test=data['X_test'],
    y_test=data['y_test'],
    results_dir='results'
)

print(f"Mejor modelo: {best_model}")
```

---

## 🔧 Tecnologías

| Categoría | Tecnologías |
|-----------|-------------|
| **Core** | Python 3.x, NumPy, Pandas |
| **ML** | scikit-learn, XGBoost, LightGBM |
| **Visualización** | Matplotlib, Seaborn |
| **Notebooks** | Jupyter |

**Versiones completas:** Ver [requirements.txt](requirements.txt)

---

## 📦 Instalación Detallada

### Opción 1: pip (Recomendado)

```bash
pip install -r requirements.txt
```

### Opción 2: Script automatizado (Windows)

```bash
set_up.bat
```

### Opción 3: Conda

```bash
conda create -n mlops python=3.11
conda activate mlops
pip install -r requirements.txt
```

---

## 📈 Detalles del Dataset

- **Registros:** 10,763
- **Features originales:** 21
- **Target:** `Pago_atiempo` (binario: 0=atrasado, 1=a tiempo)
- **Desbalanceo:** 95.3% clase 1 / 4.7% clase 0 (~1:20)
- **Split:** 80% train (8,610) / 20% test (2,153)
- **Estratificación:** Aplicada para mantener proporción de clases

---

## ✅ Validación y Testing

Todas las visualizaciones y reportes son generados automáticamente:

```bash
python run_pipeline.py
```

**Verifica la salida:**
- ✅ 3 archivos PNG en `results/`
- ✅ `evaluation_report.txt` con análisis completo
- ✅ `model_results.csv` con tabla de métricas
- ✅ Sin errores en consola

---

## 🎯 Roadmap

### ✅ Completado
- [x] Análisis exploratorio de datos (EDA)
- [x] Feature Engineering automatizado
- [x] Training pipeline con 11 modelos
- [x] Evaluación comparativa
- [x] Visualizaciones profesionales

### 🔜 Próximos avances
- [ ] **Avance 3:** Model Deployment (API REST)
- [ ] **Avance 4:** Model Monitoring (métricas en producción)
- [ ] **Futuro:** Containerización con Docker

---

## 📞 Contacto

**Alexis Jacquet**  
Proyecto Integrador M5 - Henry  

---

## 📄 Licencia

Proyecto educativo - Henry Bootcamp  
© 2026 Alexis Jacquet

---

**🎉 Proyecto completado exitosamente**
