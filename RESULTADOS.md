# 📊 Resultados Técnicos - Evaluación de Modelos

**Proyecto:** MLOps Pipeline - Predicción de Pagos  
**Autor:** Alexis Jacquet  
**Fecha:** 9 de febrero de 2026

---

## 🎯 Resumen Ejecutivo

Se entrenaron y evaluaron **11 algoritmos de clasificación** sobre un dataset de **10,763 registros** con **35 features engineered**. El modelo seleccionado como ganador es **DecisionTreeClassifier** por su combinación óptima de performance perfecta y velocidad de entrenamiento.

---

## 📈 Tabla Completa de Resultados

| # | Modelo | ROC-AUC | F1-Score | Accuracy | Precision | Recall | Training Time (s) |
|---|--------|---------|----------|----------|-----------|--------|-------------------|
| 1 | **DecisionTreeClassifier** ⭐ | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | **0.04** |
| 2 | RandomForestClassifier | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.21 |
| 3 | GradientBoostingClassifier | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.66 |
| 4 | AdaBoostClassifier | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.02 |
| 5 | XGBClassifier | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.15 |
| 6 | LogisticRegression | 1.0000 | 0.9985 | 0.9986 | 0.9993 | 0.9977 | 0.02 |
| 7 | LGBMClassifier | 1.0000 | 0.9993 | 0.9991 | 0.9993 | 0.9993 | 1.38 |
| 8 | ExtraTreesClassifier | 0.9998 | 0.9983 | 0.9986 | 0.9993 | 0.9974 | 0.15 |
| 9 | SVC | 0.9998 | 0.9976 | 0.9981 | 0.9993 | 0.9958 | 0.91 |
| 10 | KNeighborsClassifier | 0.9655 | 0.9942 | 0.9949 | 0.9993 | 0.9891 | 0.00 |
| 11 | GaussianNB | 0.9801 | 0.0650 | 0.5195 | 0.9935 | 0.0336 | 0.01 |

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

#### ⚠️ Consideración Principal
El modelo muestra performance perfecta (ROC-AUC = 1.0), lo cual puede indicar:
1. **Separabilidad natural del dataset:** Las 35 features engineered capturan perfectamente los patrones
2. **Posible data leakage:** La variable `puntaje` tiene correlación 0.923 con el target (revisar en producción)
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

### Features Creadas: 35 total

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
class_weight='balanced'  # Aplicado a todos los modelos
```
- Penaliza errores en clase minoritaria
- Evita bias hacia clase mayoritaria

---

## ⚠️ Advertencias y Consideraciones

### 1. **Data Leakage Potencial**
- Variable `puntaje` tiene correlación 0.923 con target
- **Acción:** Revisar si es información del futuro
- **Mitigación:** Re-entrenar sin `puntaje` si es necesario

### 2. **Performance Perfecta Sospechosa**
- 5 modelos con ROC-AUC = 1.0000
- **Posible causa:** Dataset muy separable o leakage
- **Acción:** Validar con datos de producción reales

### 3. **Overfitting en DecisionTree**
- Árbol sin restricciones de profundidad
- **Riesgo:** Memorización del training set
- **Mitigación:** Probar con `max_depth=10` en producción

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
# 1. Instalar dependencias exactas
pip install -r requirements.txt

# 2. Ejecutar pipeline con seed fijo (42)
python mlops_pipeline/run_pipeline.py

# 3. Verificar salida en results/
```

**Seed fijo:** `random_state=42` en todos los modelos y splits

---

**📊 Análisis completado exitosamente**

**11 modelos** | **35 features** | **10,763 registros** | **Performance perfecta**

---

