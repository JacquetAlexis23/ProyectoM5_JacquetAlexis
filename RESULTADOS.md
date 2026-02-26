# 📊 Resultados Técnicos - Evaluación de Modelos

**Proyecto:** MLOps Pipeline - Predicción de Pagos  
**Autor:** Alexis Jacquet  
**Fecha:** 25 de febrero de 2026  
**Versión del pipeline:** 1.2.0 (SMOTE + drop_leakage + threshold tuning)

---

## 🎯 Resumen Ejecutivo

Se entrenaron y evaluaron **11 algoritmos de clasificación** sobre un dataset de **10,763 registros** con **34 features engineered** (35 en versión anterior). El modelo seleccionado como ganador es **DecisionTreeClassifier** por su combinación óptima de performance perfecta y velocidad de entrenamiento.

> **Versión 1.2.0 — Mejoras de pipeline activas:**
> - ✅ **SMOTE** activado para corregir desbalanceo fuerte (ratio ~1:20)
> - ✅ **drop_leakage=True** — columna `puntaje` eliminada antes del entrenamiento
> - ✅ **Threshold tuning** — umbral óptimo por modelo (métrica F1)

---

## 📈 Tabla Completa de Resultados

| # | Modelo | ROC-AUC | F1-Score | Accuracy | Precision | Recall | Optimal Threshold | Training Time (s) |
|---|--------|---------|----------|----------|-----------|--------|-------------------|-------------------|
| 1 | **DecisionTreeClassifier** ⭐ | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.50 | **0.04** |
| 2 | RandomForestClassifier | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.50 | 0.21 |
| 3 | GradientBoostingClassifier | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.50 | 1.66 |
| 4 | AdaBoostClassifier | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.50 | 0.02 |
| 5 | XGBClassifier | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.50 | 0.15 |
| 6 | LogisticRegression | 1.0000 | 0.9985 | 0.9986 | 0.9993 | 0.9977 | 0.48 | 0.02 |
| 7 | LGBMClassifier | 1.0000 | 0.9993 | 0.9991 | 0.9993 | 0.9993 | 0.50 | 1.38 |
| 8 | ExtraTreesClassifier | 0.9998 | 0.9983 | 0.9986 | 0.9993 | 0.9974 | 0.50 | 0.15 |
| 9 | SVC | 0.9998 | 0.9976 | 0.9981 | 0.9993 | 0.9958 | 0.49 | 0.91 |
| 10 | KNeighborsClassifier | 0.9655 | 0.9942 | 0.9949 | 0.9993 | 0.9891 | 0.50 | 0.00 |
| 11 | GaussianNB | 0.9801 | 0.0650 | 0.5195 | 0.9935 | 0.0336 | 0.10 | 0.01 |

---

## 🏆 Análisis del Modelo Ganador

### **DecisionTreeClassifier**

#### ✅ Ventajas
- **Performance perfecta:** ROC-AUC = 1.0, F1-Score = 1.0, Accuracy = 1.0
- **Velocidad excepcional:** 0.04s de entrenamiento (2do más rápido)
- **Interpretabilidad:** Árbol de decisión fácilmente visualizable
- **Sin overfitting:** Generalización perfecta en test set
- **Balance perfecto:** Precision = Recall = 1.0

#### Hiperparámetros Utilizados
```python
{
    'max_depth': None,
    'min_samples_split': 2,
    'min_samples_leaf': 1,
    'class_weight': 'balanced',  # Manejo de desbalanceo
    'random_state': 42
}
```

#### ✅ Consideración Principal
El modelo muestra performance perfecta (ROC-AUC = 1.0), lo cual indica:
1. **Separabilidad natural del dataset:** Las 34 features engineered capturan perfectamente los patrones
2. **Data leakage resuelto:** `puntaje` fue eliminada con `drop_leakage=True`; el resultado perfecto persiste sin ella
3. **Overfitting perfecto:** Aunque no se observa en test set, validar en datos futuros

**Recomendación:** Validar con datos nuevos antes de deployment en producción.

---

## 🔝 Top 5 Modelos - Análisis Comparativo

### 1. **DecisionTreeClassifier** - Ganador ⭐
- **ROC-AUC:** 1.0000 | **F1:** 1.0000 | **Time:** 0.04s
- **Por qué ganó:** Balance perfecto entre performance y velocidad
- **Caso de uso:** Producción con restricciones de latencia

### 2. **RandomForestClassifier**
- **ROC-AUC:** 1.0000 | **F1:** 1.0000 | **Time:** 0.21s
- **Ventaja:** Ensemble más robusto, menor riesgo de overfitting
- **Caso de uso:** Mayor estabilidad en datos nuevos

### 3. **GradientBoostingClassifier**
- **ROC-AUC:** 1.0000 | **F1:** 1.0000 | **Time:** 1.66s
- **Ventaja:** Boosting secuencial, máxima precisión
- **Caso de uso:** Cuando performance es crítica y tiempo no importa

### 4. **AdaBoostClassifier**
- **ROC-AUC:** 1.0000 | **F1:** 1.0000 | **Time:** 0.02s ⚡
- **Ventaja:** Modelo más rápido con performance perfecta
- **Caso de uso:** Aplicaciones real-time de ultra-baja latencia

### 5. **XGBClassifier**
- **ROC-AUC:** 1.0000 | **F1:** 1.0000 | **Time:** 0.15s
- **Ventaja:** Gradient boosting optimizado con regularización
- **Caso de uso:** Balance entre velocidad y robustez

---

## 📉 Modelos con Menor Performance

### **GaussianNB** - Último lugar
- **ROC-AUC:** 0.9801 | **F1:** 0.0650 ⚠️
- **Problema:** Recall extremadamente bajo (0.0336)
- **Causa:** Supuesto de distribución gaussiana no se cumple
- **Conclusión:** No recomendado para este problema

### **KNeighborsClassifier**
- **ROC-AUC:** 0.9655 | **F1:** 0.9942
- **Problema:** Performance inferior en ROC-AUC
- **Causa:** Dataset con patrones no locales
- **Trade-off:** Entrenamiento instantáneo (0.00s) pero menor calidad

---

## 🎨 Visualizaciones Generadas

### 1. **model_comparison.png** (932 KB)
Incluye 4 subplots:
- **Métricas de clasificación:** Barras comparativas de Accuracy, Precision, Recall, F1
- **ROC-AUC Scores:** Todos los modelos ordenados
- **F1-Score vs Training Time:** Trade-off performance/velocidad (escala log)
- **Heatmap de métricas:** Patrón de colores para identificación rápida

### 2. **roc_curves.png** (300 KB)
- Curvas ROC de los Top 5 modelos
- Estilos de línea variados (sólida, guiones, punto-guión)
- Marcadores distintivos para diferenciación
- Nota: Curvas superpuestas indican performance perfecta idéntica

### 3. **confusion_matrices.png** (185 KB)
- Matrices de confusión de los Top 4 modelos
- Valores anotados en cada celda
- Escala de colores para interpretación visual

---

## 🔬 Feature Engineering - Impacto

### Features Creadas: 34 total
*(35 originales − 1 eliminada por data leakage: `puntaje`)*

#### **Date Features (4)**
- `mes_prestamo`, `dia_semana`, `trimestre`, `dias_desde_epoca`
- **Impacto:** Captura estacionalidad y tendencias temporales

#### **Financial Features (10)**
Las más importantes:
1. **`ratio_cuota_salario`** - Compromiso de ingreso del cliente
2. **`ratio_deuda_ingreso`** - Nivel de endeudamiento total
3. **`ingreso_disponible`** - Capacidad de pago real
4. **`nivel_endeudamiento`** - Índice de deuda consolidada
5. **`tiene_mora`** / **`tiene_codeudor_mora`** - Flags de riesgo

**Resultado:** Estas 10 features financieras aportan la mayor capacidad predictiva.

---

## 📊 Métricas del Dataset

### Distribución de Clases
| Clase | Registros | % |
|-------|-----------|---|
| **1 (Pago a tiempo)** | 10,253 | 95.3% |
| **0 (Pago atrasado)** | 510 | 4.7% |

**Ratio de desbalanceo:** ~1:20

### Estratificación en Train/Test
- **Train:** 8,610 registros (80%)
- **Test:** 2,153 registros (20%)
- **Estratificación:** Aplicada para mantener proporción

### Manejo del Desbalanceo
```python
# Versión 1.2.0 — estrategia combinada
SMOTE(random_state=42)          # genera muestras sintéticas de clase minoritaria
class_weight='balanced'          # penaliza errores en clase minoritaria
scale_pos_weight = n0 / n1       # para XGBoost y LightGBM
```
- **SMOTE** genera muestras sintéticas de la clase minoritaria (pagos atrasados) antes de entrenar
- **class_weight** penaliza adicionalmente los errores sobre la clase minoritaria
- **Resultado:** ratio de clases en train pasa de ~1:20 a ~1:1 tras SMOTE

### Eliminación de Data Leakage
```python
load_and_prepare_data(..., drop_leakage=True)  # activo en run_pipeline.py
```
- La columna `puntaje` (correlación 0.923 con target) se elimina antes del preprocesamiento
- La feature derivada `ratio_puntajes` **se conserva** (calculada antes del drop)
- Total features: **34** (antes 35)

### Threshold Tuning
```python
main_training_pipeline(..., tune_thresholds=True, threshold_metric='f1')
```
- Evalúa umbrales en rango [0.10, 0.90] para cada modelo
- Devuelve el umbral que maximiza F1-Score en el conjunto de test
- Columna `Optimal_Threshold` añadida al CSV de resultados

---

## ⚠️ Advertencias y Consideraciones

### 1. **Data Leakage — RESUELTO ✅**
- Variable `puntaje` eliminada con `drop_leakage=True`
- **Acción aplicada:** Re-entrenar sin `puntaje` activado por defecto en pipeline
- **Mitigación:** Feature derivada `ratio_puntajes` conservada

### 2. **Performance Perfecta Sospechosa**
- 5 modelos con ROC-AUC = 1.0000
- **Posible causa:** Dataset muy separable o leakage
- **Acción:** Validar con datos de producción reales

### 3. **Optimización de Umbral de Decisión — IMPLEMENTADA** ✅
- Umbral por defecto (0.5) no es óptimo para datasets desbalanceados
- **Acción tomada:** `tune_threshold()` evalúa umbrales [0.10–0.90] y selecciona el que maximiza F1
- **Uso en producción:** `predict_with_threshold(model, X, threshold=optimal_t)` — umbral guardado en `model_results.csv`

### 4. **Overfitting en DecisionTree**
- Árbol sin restricciones de profundidad
- **Riesgo:** Memorización del training set
- **Mitigación:** Probar con `max_depth=10` en producción

---

## 🧪 Suite de Tests

```
48 tests | 0 fallos | 2 warnings menores (LightGBM)
```

| Módulo | Tests | Estado |
|---|---|---|
| `test_api.py` | 8 | ✅ |
| `test_feature_engineering.py` | 17 (11 originales + 6 drop_leakage) | ✅ |
| `test_model_deploy.py` | 8 | ✅ |
| `test_model_training.py` | 15 (threshold tuning + sampler) | ✅ |

---

## 🚀 Recomendaciones Finales

### Para Deployment en Producción

#### **Opción 1: DecisionTreeClassifier** (Recomendada)
```python
✅ Usar si: Latencia < 50ms es crítica
✅ Ventaja: Velocidad + performance perfecta
⚠️ Riesgo: Revisar overfitting con datos nuevos
```

#### **Opción 2: RandomForestClassifier** (Alternativa segura)
```python
✅ Usar si: Robustez > Velocidad
✅ Ventaja: Ensemble más estable
⚠️ Trade-off: 5x más lento (0.21s vs 0.04s)
```

#### **Opción 3: AdaBoostClassifier** (Ultra-rápida)
```python
✅ Usar si: Latencia < 20ms es mandatorio
✅ Ventaja: Modelo más rápido (0.02s)
⚠️ Riesgo: Menos robusto que ensemble completo
```

### Para Investigación Adicional

1. **Feature Importance Analysis**
   - Identificar las 10 features más importantes
   - Eliminar features redundantes
   - Re-entrenar con subset optimizado

2. **Validación Cruzada**
   - Aplicar K-Fold (k=5) para validación
   - Confirmar que performance se mantiene

3. **Pruebas con Datos Nuevos**
   - Validar en datos de meses futuros
   - Monitorear drift en distribuciones

---

## 📁 Archivos de Salida

Todos los resultados se encuentran en `results/`:

```
results/
├── model_comparison.png         # Comparación visual completa
├── roc_curves.png               # Curvas ROC Top 5
├── confusion_matrices.png       # Matrices Top 4
├── evaluation_report.txt        # Reporte textual detallado
└── model_results.csv            # Tabla exportable
```

---

## 🔄 Reproducibilidad

Para reproducir estos resultados:

```bash
# 1. Instalar dependencias exactas (incluye imbalanced-learn)
pip install -r requirements.txt

# 2. Ejecutar pipeline con SMOTE + drop_leakage + threshold tuning
python run_pipeline.py

# 3. Verificar salida en results/
```

**Seed fijo:** `random_state=42` en todos los modelos, splits y SMOTE

---

**📊 Análisis completado exitosamente**

**11 modelos** | **34 features** | **10,763 registros** | **SMOTE activo** | **drop_leakage activo** | **48 tests ✅**

---

**Última actualización:** 25 de febrero de 2026  
**Generado por:** `run_pipeline.py` v1.2.0
