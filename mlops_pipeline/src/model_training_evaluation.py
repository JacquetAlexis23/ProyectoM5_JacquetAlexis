"""
Model Training & Evaluation Pipeline - Versión 1.0.1
====================================================
Pipeline completo para entrenamiento y evaluación de modelos de clasificación:
- Entrenamiento de múltiples algoritmos
- Evaluación exhaustiva con métricas clave
- Visualizaciones comparativas
- Funciones reutilizables para experimentación
- Manejo de clases desbalanceadas
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Usar backend no interactivo
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier, 
                               AdaBoostClassifier, ExtraTreesClassifier)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                              f1_score, roc_auc_score, confusion_matrix, 
                              classification_report, roc_curve, precision_recall_curve)
import time
import warnings
warnings.filterwarnings('ignore')


def summarize_classification(y_true, y_pred, y_pred_proba=None, model_name="Model"):
    """
    Genera un resumen completo de métricas de clasificación
    
    Args:
        y_true: Etiquetas verdaderas
        y_pred: Predicciones del modelo
        y_pred_proba: Probabilidades predichas (opcional)
        model_name: Nombre del modelo
        
    Returns:
        dict: Diccionario con todas las métricas
    """
    metrics = {}
    
    # Métricas básicas
    metrics['Model'] = model_name
    metrics['Accuracy'] = accuracy_score(y_true, y_pred)
    metrics['Precision'] = precision_score(y_true, y_pred, zero_division=0)
    metrics['Recall'] = recall_score(y_true, y_pred, zero_division=0)
    metrics['F1-Score'] = f1_score(y_true, y_pred, zero_division=0)
    metrics['Specificity'] = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
    
    # ROC-AUC si hay probabilidades
    if y_pred_proba is not None:
        metrics['ROC-AUC'] = roc_auc_score(y_true, y_pred_proba)
    else:
        metrics['ROC-AUC'] = np.nan
    
    # Matriz de confusión
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metrics['TN'] = tn
        metrics['FP'] = fp
        metrics['FN'] = fn
        metrics['TP'] = tp
    
    return metrics


def build_model(model_class, X_train, y_train, X_test, y_test, **model_params):
    """
    Construye, entrena y evalúa un modelo de clasificación
    
    Args:
        model_class: Clase del modelo sklearn
        X_train: Features de entrenamiento
        y_train: Target de entrenamiento
        X_test: Features de prueba
        y_test: Target de prueba
        **model_params: Parámetros adicionales del modelo
        
    Returns:
        tuple: (model, metrics, training_time)
    """
    # Inicializar modelo
    model = model_class(**model_params)
    model_name = model.__class__.__name__
    
    print(f"  🔹 Entrenando {model_name}...", end=" ")
    
    # Entrenar
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    # Predecir
    y_pred = model.predict(X_test)
    
    # Obtener probabilidades si el modelo lo soporta
    y_pred_proba = None
    if hasattr(model, 'predict_proba'):
        y_pred_proba = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, 'decision_function'):
        y_pred_proba = model.decision_function(X_test)
    
    # Calcular métricas
    metrics = summarize_classification(y_test, y_pred, y_pred_proba, model_name)
    metrics['Training_Time'] = training_time
    
    print(f"✓ (F1: {metrics['F1-Score']:.4f}, ROC-AUC: {metrics['ROC-AUC']:.4f}, Tiempo: {training_time:.2f}s)")
    
    return model, metrics, training_time


def train_multiple_models(X_train, y_train, X_test, y_test, use_class_weight=True):
    """
    Entrena múltiples modelos de clasificación y compara su desempeño
    
    Args:
        X_train: Features de entrenamiento
        y_train: Target de entrenamiento
        X_test: Features de prueba
        y_test: Target de prueba
        use_class_weight: Si usar balanceo de clases
        
    Returns:
        tuple: (models_dict, results_df)
    """
    print("\n" + "="*80)
    print("🚀 ENTRENAMIENTO DE MODELOS - VERSIÓN 1.0.1")
    print("="*80)
    
    models_dict = {}
    results_list = []
    
    # Configuración de modelos
    class_weight = 'balanced' if use_class_weight else None
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum() if use_class_weight else 1
    
    print("\n📚 Modelos a entrenar:")
    print("  • Logistic Regression\n  • Decision Tree\n  • Random Forest")
    print("  • Gradient Boosting\n  • XGBoost\n  • LightGBM")
    print("  • AdaBoost\n  • Extra Trees\n  • SVM\n  • K-Nearest Neighbors\n  • Naive Bayes")
    
    print("\n⚙️ Iniciando entrenamiento...\n")
    
    # 1. Logistic Regression
    model, metrics, time_taken = build_model(
        LogisticRegression,
        X_train, y_train, X_test, y_test,
        max_iter=1000,
        class_weight=class_weight,
        random_state=42
    )
    models_dict[metrics['Model']] = model
    results_list.append(metrics)
    
    # 2. Decision Tree
    model, metrics, time_taken = build_model(
        DecisionTreeClassifier,
        X_train, y_train, X_test, y_test,
        max_depth=10,
        min_samples_split=20,
        min_samples_leaf=10,
        class_weight=class_weight,
        random_state=42
    )
    models_dict[metrics['Model']] = model
    results_list.append(metrics)
    
    # 3. Random Forest
    model, metrics, time_taken = build_model(
        RandomForestClassifier,
        X_train, y_train, X_test, y_test,
        n_estimators=100,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight=class_weight,
        random_state=42,
        n_jobs=-1
    )
    models_dict[metrics['Model']] = model
    results_list.append(metrics)
    
    # 4. Gradient Boosting
    model, metrics, time_taken = build_model(
        GradientBoostingClassifier,
        X_train, y_train, X_test, y_test,
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )
    models_dict[metrics['Model']] = model
    results_list.append(metrics)
    
    # 5. XGBoost
    model, metrics, time_taken = build_model(
        XGBClassifier,
        X_train, y_train, X_test, y_test,
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric='logloss'
    )
    models_dict[metrics['Model']] = model
    results_list.append(metrics)
    
    # 6. LightGBM
    model, metrics, time_taken = build_model(
        LGBMClassifier,
        X_train, y_train, X_test, y_test,
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        verbose=-1
    )
    models_dict[metrics['Model']] = model
    results_list.append(metrics)
    
    # 7. AdaBoost
    model, metrics, time_taken = build_model(
        AdaBoostClassifier,
        X_train, y_train, X_test, y_test,
        n_estimators=100,
        learning_rate=0.1,
        random_state=42
    )
    models_dict[metrics['Model']] = model
    results_list.append(metrics)
    
    # 8. Extra Trees
    model, metrics, time_taken = build_model(
        ExtraTreesClassifier,
        X_train, y_train, X_test, y_test,
        n_estimators=100,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight=class_weight,
        random_state=42,
        n_jobs=-1
    )
    models_dict[metrics['Model']] = model
    results_list.append(metrics)
    
    # 9. SVM (con probabilidades)
    print("  🔹 Entrenando SVC... (puede tomar más tiempo)", end=" ")
    start_time = time.time()
    model = SVC(
        kernel='rbf',
        C=1.0,
        class_weight=class_weight,
        probability=True,
        random_state=42
    )
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    metrics = summarize_classification(y_test, y_pred, y_pred_proba, 'SVC')
    metrics['Training_Time'] = training_time
    print(f"✓ (F1: {metrics['F1-Score']:.4f}, ROC-AUC: {metrics['ROC-AUC']:.4f}, Tiempo: {training_time:.2f}s)")
    models_dict[metrics['Model']] = model
    results_list.append(metrics)
    
    # 10. K-Nearest Neighbors
    model, metrics, time_taken = build_model(
        KNeighborsClassifier,
        X_train, y_train, X_test, y_test,
        n_neighbors=5,
        weights='distance',
        n_jobs=-1
    )
    models_dict[metrics['Model']] = model
    results_list.append(metrics)
    
    # 11. Naive Bayes
    model, metrics, time_taken = build_model(
        GaussianNB,
        X_train, y_train, X_test, y_test
    )
    models_dict[metrics['Model']] = model
    results_list.append(metrics)
    
    # Crear DataFrame de resultados
    results_df = pd.DataFrame(results_list)
    
    # Ordenar por F1-Score y ROC-AUC
    results_df = results_df.sort_values('ROC-AUC', ascending=False).reset_index(drop=True)
    
    print("\n✅ Entrenamiento completado!")
    print("="*80)
    
    return models_dict, results_df


def plot_model_comparison(results_df, save_path=None):
    """
    Genera visualizaciones comparativas de los modelos
    
    Args:
        results_df: DataFrame con resultados de modelos
        save_path: Ruta para guardar el gráfico (opcional)
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Configurar estilo
    sns.set_style("whitegrid")
    colors = sns.color_palette("husl", len(results_df))
    
    # 1. Comparación de métricas principales
    ax1 = axes[0, 0]
    metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    x = np.arange(len(results_df))
    width = 0.2
    
    for i, metric in enumerate(metrics_to_plot):
        ax1.bar(x + i*width, results_df[metric], width, label=metric, alpha=0.8)
    
    ax1.set_xlabel('Modelos', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax1.set_title('Comparación de Métricas de Clasificación', fontsize=14, fontweight='bold')
    ax1.set_xticks(x + width * 1.5)
    ax1.set_xticklabels(results_df['Model'], rotation=45, ha='right')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # 2. ROC-AUC Score
    ax2 = axes[0, 1]
    bars = ax2.barh(results_df['Model'], results_df['ROC-AUC'], color=colors)
    ax2.set_xlabel('ROC-AUC Score', fontsize=12, fontweight='bold')
    ax2.set_title('ROC-AUC por Modelo', fontsize=14, fontweight='bold')
    ax2.set_xlim([0, 1])
    
    # Añadir valores en las barras
    for i, (bar, value) in enumerate(zip(bars, results_df['ROC-AUC'])):
        ax2.text(value + 0.01, i, f'{value:.4f}', va='center', fontweight='bold')
    
    ax2.grid(axis='x', alpha=0.3)
    
    # 3. F1-Score vs Training Time (con escala logarítmica en X)
    ax3 = axes[1, 0]
    
    # Usar escala logarítmica para mejor separación visual
    scatter = ax3.scatter(results_df['Training_Time'], results_df['F1-Score'], 
                         s=300, c=results_df['ROC-AUC'], cmap='viridis', 
                         alpha=0.7, edgecolors='black', linewidth=2)
    
    # Añadir etiquetas con mejor posicionamiento
    for idx, row in results_df.iterrows():
        # Añadir jitter vertical para evitar superposición
        jitter = (idx % 3 - 1) * 0.003  # Pequeño desplazamiento vertical
        # Posición dinámica basada en índice para evitar solapamiento
        offset_x = 8 if idx % 2 == 0 else -8
        offset_y = 5 + (idx % 5) * 3
        ha_align = 'left' if idx % 2 == 0 else 'right'
        
        ax3.annotate(row['Model'], 
                    (row['Training_Time'], row['F1-Score'] + jitter),
                    xytext=(offset_x, offset_y), textcoords='offset points',
                    fontsize=7, fontweight='bold', ha=ha_align,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5, edgecolor='gray', linewidth=0.5),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.2', color='gray', lw=0.5))
    
    ax3.set_xlabel('Tiempo de Entrenamiento (segundos) - Escala Log', fontsize=12, fontweight='bold')
    ax3.set_ylabel('F1-Score', fontsize=12, fontweight='bold')
    ax3.set_title('F1-Score vs Tiempo de Entrenamiento', fontsize=14, fontweight='bold')
    ax3.set_xscale('log')  # Escala logarítmica para mejor visualización
    ax3.set_ylim([max(0, results_df['F1-Score'].min() - 0.05), 1.02])  # Mejor rango
    plt.colorbar(scatter, ax=ax3, label='ROC-AUC')
    ax3.grid(True, which='both', alpha=0.3, linestyle='--')
    
    # 4. Heatmap de métricas normalizadas
    ax4 = axes[1, 1]
    metrics_for_heatmap = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC', 'Specificity']
    heatmap_data = results_df[['Model'] + metrics_for_heatmap].set_index('Model')
    
    sns.heatmap(heatmap_data.T, annot=True, fmt='.3f', cmap='RdYlGn', 
                ax=ax4, cbar_kws={'label': 'Score'}, linewidths=0.5)
    ax4.set_title('Heatmap de Métricas por Modelo', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Modelos', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Métricas', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n📊 Gráfico guardado en: {save_path}")
    
    plt.close()  # Cerrar figura sin mostrar


def plot_top_models_roc_curves(models_dict, X_test, y_test, top_n=5, results_df=None, save_path=None):
    """
    Grafica las curvas ROC de los mejores modelos
    
    Args:
        models_dict: Diccionario con los modelos entrenados
        X_test: Features de prueba
        y_test: Target de prueba
        top_n: Número de mejores modelos a graficar
        results_df: DataFrame con resultados
        save_path: Ruta para guardar el gráfico
    """
    plt.figure(figsize=(12, 9))
    
    # Obtener los mejores modelos
    if results_df is not None:
        top_models = results_df.nlargest(top_n, 'ROC-AUC')['Model'].tolist()
    else:
        top_models = list(models_dict.keys())[:top_n]
    
    # Usar colores distintivos y estilos de línea variados
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
    linestyles = ['-', '--', '-.', ':', '-']
    markers = ['o', 's', '^', 'D', 'v']
    
    # Contador para curvas únicas
    curves_plotted = []
    
    for i, model_name in enumerate(top_models):
        model = models_dict[model_name]
        
        # Obtener probabilidades
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, 'decision_function'):
            y_pred_proba = model.decision_function(X_test)
        else:
            continue
        
        # Calcular curva ROC
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        auc_score = roc_auc_score(y_test, y_pred_proba)
        
        # Crear identificador de curva para detectar duplicados
        curve_id = f"{fpr.sum():.6f}_{tpr.sum():.6f}"
        
        # Si es una curva duplicada (performance perfecta), agregar desplazamiento visual
        if curve_id in curves_plotted:
            # Agregar marcadores para diferenciar visualmente
            plt.plot(fpr, tpr, color=colors[i % len(colors)], 
                    linestyle=linestyles[i % len(linestyles)], 
                    lw=3, alpha=0.7,
                    marker=markers[i % len(markers)], markevery=50,
                    markersize=8, markerfacecolor='white',
                    label=f'{model_name} (AUC = {auc_score:.4f})')
        else:
            plt.plot(fpr, tpr, color=colors[i % len(colors)], 
                    linestyle=linestyles[i % len(linestyles)], lw=3, 
                    label=f'{model_name} (AUC = {auc_score:.4f})')
            curves_plotted.append(curve_id)
    
    # Línea diagonal
    plt.plot([0, 1], [0, 1], 'k--', lw=2, alpha=0.5, label='Random Classifier (AUC = 0.5000)')
    
    # Zona de clasificador excelente (sombreado)
    plt.fill_between([0, 0, 0.2], [0.8, 1, 1], alpha=0.1, color='green', label='Zona Excelente')
    
    plt.xlabel('False Positive Rate', fontsize=13, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=13, fontweight='bold')
    plt.title(f'Curvas ROC - Top {top_n} Modelos\n(Líneas superpuestas indican performance similar)', 
              fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=9, framealpha=0.9, edgecolor='black')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    
    # Añadir texto informativo
    if len(curves_plotted) < len(top_models):
        plt.text(0.5, 0.05, 'Nota: Algunas curvas están superpuestas debido a performance perfecta (AUC=1.0)',
                ha='center', fontsize=9, style='italic', 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Curva ROC guardada en: {save_path}")
    
    plt.close()  # Cerrar figura sin mostrar


def plot_confusion_matrices(models_dict, X_test, y_test, top_n=4, results_df=None, save_path=None):
    """
    Grafica matrices de confusión de los mejores modelos
    
    Args:
        models_dict: Diccionario con los modelos entrenados
        X_test: Features de prueba
        y_test: Target de prueba
        top_n: Número de mejores modelos
        results_df: DataFrame con resultados
        save_path: Ruta para guardar el gráfico
    """
    # Obtener los mejores modelos
    if results_df is not None:
        top_models = results_df.nlargest(top_n, 'ROC-AUC')['Model'].tolist()
    else:
        top_models = list(models_dict.keys())[:top_n]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()
    
    for i, model_name in enumerate(top_models[:4]):
        model = models_dict[model_name]
        y_pred = model.predict(X_test)
        
        cm = confusion_matrix(y_test, y_pred)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                   cbar_kws={'label': 'Count'})
        axes[i].set_title(f'{model_name}', fontsize=12, fontweight='bold')
        axes[i].set_xlabel('Predicción', fontsize=10, fontweight='bold')
        axes[i].set_ylabel('Real', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Matrices de confusión guardadas en: {save_path}")
    
    plt.close()  # Cerrar figura sin mostrar


def generate_evaluation_report(results_df, save_path=None):
    """
    Genera un reporte detallado de evaluación de modelos
    
    Args:
        results_df: DataFrame con resultados
        save_path: Ruta para guardar el reporte
        
    Returns:
        str: Reporte formateado
    """
    report = []
    report.append("\n" + "="*80)
    report.append("📊 REPORTE DE EVALUACIÓN DE MODELOS")
    report.append("="*80)
    
    # Tabla resumen
    report.append("\n📋 TABLA RESUMEN DE MÉTRICAS:")
    report.append("-"*80)
    
    # Formatear DataFrame para impresión
    display_df = results_df.copy()
    numeric_cols = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC', 'Specificity']
    for col in numeric_cols:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}")
    
    display_df['Training_Time'] = display_df['Training_Time'].apply(lambda x: f"{x:.2f}s")
    
    report.append(display_df.to_string(index=False))
    
    # Mejor modelo
    report.append("\n" + "="*80)
    report.append("🏆 MEJOR MODELO POR MÉTRICA:")
    report.append("="*80)
    
    best_metrics = {
        'ROC-AUC': results_df.loc[results_df['ROC-AUC'].idxmax()],
        'F1-Score': results_df.loc[results_df['F1-Score'].idxmax()],
        'Recall': results_df.loc[results_df['Recall'].idxmax()],
        'Precision': results_df.loc[results_df['Precision'].idxmax()]
    }
    
    for metric, row in best_metrics.items():
        report.append(f"\n🥇 Mejor {metric}: {row['Model']} ({row[metric]:.4f})")
    
    # Top 3 modelos
    report.append("\n" + "="*80)
    report.append("🎯 TOP 3 MODELOS (por ROC-AUC):")
    report.append("="*80)
    
    for idx, row in results_df.head(3).iterrows():
        report.append(f"\n{idx+1}. {row['Model']}")
        report.append(f"   • ROC-AUC: {row['ROC-AUC']:.4f}")
        report.append(f"   • F1-Score: {row['F1-Score']:.4f}")
        report.append(f"   • Precisión: {row['Precision']:.4f}")
        report.append(f"   • Recall: {row['Recall']:.4f}")
        report.append(f"   • Tiempo: {row['Training_Time']:.2f}s")
    
    # Análisis de trade-offs
    report.append("\n" + "="*80)
    report.append("⚖️ ANÁLISIS DE TRADE-OFFS:")
    report.append("="*80)
    
    fastest = results_df.loc[results_df['Training_Time'].idxmin()]
    report.append(f"\n⚡ Modelo más rápido: {fastest['Model']} ({fastest['Training_Time']:.2f}s)")
    report.append(f"   • ROC-AUC: {fastest['ROC-AUC']:.4f}")
    
    best_recall = results_df.loc[results_df['Recall'].idxmax()]
    report.append(f"\n🎯 Mejor Recall (minimiza falsos negativos): {best_recall['Model']} ({best_recall['Recall']:.4f})")
    
    best_precision = results_df.loc[results_df['Precision'].idxmax()]
    report.append(f"\n🎯 Mejor Precisión (minimiza falsos positivos): {best_precision['Model']} ({best_precision['Precision']:.4f})")
    
    # Recomendación final
    report.append("\n" + "="*80)
    report.append("💡 RECOMENDACIÓN FINAL:")
    report.append("="*80)
    
    best_model = results_df.iloc[0]
    report.append(f"\nEl modelo RECOMENDADO es: **{best_model['Model']}**")
    report.append(f"\nRazones:")
    report.append(f"  ✓ Mayor ROC-AUC: {best_model['ROC-AUC']:.4f}")
    report.append(f"  ✓ F1-Score balanceado: {best_model['F1-Score']:.4f}")
    report.append(f"  ✓ Buen balance entre Precision ({best_model['Precision']:.4f}) y Recall ({best_model['Recall']:.4f})")
    report.append(f"  ✓ Tiempo de entrenamiento aceptable: {best_model['Training_Time']:.2f}s")
    
    report.append("\n" + "="*80)
    
    report_text = "\n".join(report)
    
    # Guardar reporte
    if save_path:
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        print(f"\n📄 Reporte guardado en: {save_path}")
    
    print(report_text)
    
    return report_text


def main_training_pipeline(X_train, y_train, X_test, y_test, output_dir='../../results/'):
    """
    Pipeline principal de entrenamiento y evaluación
    
    Args:
        X_train: Features de entrenamiento
        y_train: Target de entrenamiento
        X_test: Features de prueba
        y_test: Target de prueba
        output_dir: Directorio para guardar resultados
        
    Returns:
        tuple: (models_dict, results_df, best_model_name)
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Entrenar modelos
    models_dict, results_df = train_multiple_models(X_train, y_train, X_test, y_test)
    
    # Generar visualizaciones
    print("\n📊 Generando visualizaciones...")
    plot_model_comparison(results_df, save_path=f'{output_dir}model_comparison.png')
    plot_top_models_roc_curves(models_dict, X_test, y_test, top_n=5, results_df=results_df, 
                               save_path=f'{output_dir}roc_curves.png')
    plot_confusion_matrices(models_dict, X_test, y_test, top_n=4, results_df=results_df,
                           save_path=f'{output_dir}confusion_matrices.png')
    
    # Generar reporte
    generate_evaluation_report(results_df, save_path=f'{output_dir}evaluation_report.txt')
    
    # Guardar tabla de resultados
    results_df.to_csv(f'{output_dir}model_results.csv', index=False)
    print(f"\n📊 Tabla de resultados guardada en: {output_dir}model_results.csv")
    
    # Obtener mejor modelo
    best_model_name = results_df.iloc[0]['Model']
    
    return models_dict, results_df, best_model_name


if __name__ == "__main__":
    # Este script debe ejecutarse después de ft_engineering.py
    print("⚠️ Este módulo debe ejecutarse después de cargar los datos con ft_engineering.py")
    print("Ejemplo de uso:")
    print("""
    from ft_engineering import load_and_prepare_data
    from model_training_evaluation import main_training_pipeline
    
    # Cargar y preparar datos
    data = load_and_prepare_data('../../data/Base_de_datos.csv')
    
    # Entrenar y evaluar modelos
    models, results, best_model = main_training_pipeline(
        data['X_train'], data['y_train'],
        data['X_test'], data['y_test']
    )
    """)
