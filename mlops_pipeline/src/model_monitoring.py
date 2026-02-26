"""
Model Monitoring & Data Drift Detection - Versión 1.0.0
=======================================================
Sistema avanzado de monitoreo de modelos de Machine Learning con detección
de data drift utilizando múltiples métricas estadísticas.

Funcionalidades:
- Detección de data drift con KS test, PSI, Jensen-Shannon divergence
- Análisis temporal de drift
- Sistema de alertas automáticas
- Visualización de distribuciones
- Almacenamiento histórico de métricas

Autor: Alexis Jacquet
Proyecto: M5 - Henry - Avance 3
Fecha: Febrero 2026
"""

import pandas as pd
import numpy as np
import warnings
from datetime import datetime
from scipy import stats
from scipy.spatial.distance import jensenshannon
from typing import Dict, List, Tuple, Optional, Any
import json
import os
warnings.filterwarnings('ignore')


class DataDriftDetector:
    """
    Detector avanzado de Data Drift con múltiples métricas estadísticas
    """
    
    def __init__(self, 
                 ks_threshold: float = 0.05,
                 psi_threshold: float = 0.1,
                 js_threshold: float = 0.1,
                 chi2_threshold: float = 0.05):
        """
        Args:
            ks_threshold: Umbral para p-value del test Kolmogorov-Smirnov
            psi_threshold: Umbral para Population Stability Index
            js_threshold: Umbral para Jensen-Shannon divergence
            chi2_threshold: Umbral para test Chi-cuadrado
        """
        self.ks_threshold = ks_threshold
        self.psi_threshold = psi_threshold
        self.js_threshold = js_threshold
        self.chi2_threshold = chi2_threshold
        
        self.baseline_data = None
        self.feature_names = None
        self.numeric_features = []
        self.categorical_features = []
        self.drift_history = []
        
    def fit(self, baseline_data: pd.DataFrame, feature_names: List[str] = None):
        """
        Establece los datos de referencia (baseline) para comparaciones futuras
        
        Args:
            baseline_data: DataFrame con los datos de referencia (ej: train set)
            feature_names: Lista de nombres de features (opcional)
        """
        self.baseline_data = baseline_data.copy()
        
        if feature_names is not None:
            self.feature_names = feature_names
        else:
            self.feature_names = list(baseline_data.columns)
        
        # Identificar features numéricas y categóricas
        for col in self.feature_names:
            if baseline_data[col].dtype in ['int64', 'float64']:
                self.numeric_features.append(col)
            else:
                self.categorical_features.append(col)
        
        print(f"✓ Baseline establecido: {len(self.feature_names)} features")
        print(f"  - Numéricas: {len(self.numeric_features)}")
        print(f"  - Categóricas: {len(self.categorical_features)}")
        
    def kolmogorov_smirnov_test(self, baseline_col: pd.Series, current_col: pd.Series) -> Dict:
        """
        Aplica el test de Kolmogorov-Smirnov para detectar diferencias en distribuciones
        
        Args:
            baseline_col: Serie de datos de referencia
            current_col: Serie de datos actuales
            
        Returns:
            Dict con estadístico KS, p-value y si hay drift
        """
        # Remover NaN
        baseline_clean = baseline_col.dropna()
        current_clean = current_col.dropna()
        
        if len(baseline_clean) == 0 or len(current_clean) == 0:
            return {'ks_statistic': np.nan, 'p_value': np.nan, 'drift_detected': False}
        
        ks_statistic, p_value = stats.ks_2samp(baseline_clean, current_clean)
        
        return {
            'ks_statistic': ks_statistic,
            'p_value': p_value,
            'drift_detected': p_value < self.ks_threshold
        }
    
    def population_stability_index(self, baseline_col: pd.Series, current_col: pd.Series, 
                                   bins: int = 10) -> Dict:
        """
        Calcula el Population Stability Index (PSI)
        
        PSI < 0.1: Sin cambio significativo
        0.1 <= PSI < 0.2: Cambio moderado
        PSI >= 0.2: Cambio significativo (requiere investigación)
        
        Args:
            baseline_col: Serie de datos de referencia
            current_col: Serie de datos actuales
            bins: Número de bins para discretización
            
        Returns:
            Dict con PSI y nivel de alerta
        """
        # Remover NaN
        baseline_clean = baseline_col.dropna()
        current_clean = current_col.dropna()
        
        if len(baseline_clean) == 0 or len(current_clean) == 0:
            return {'psi': np.nan, 'alert_level': 'unknown'}
        
        # Crear bins basados en baseline
        try:
            _, bin_edges = np.histogram(baseline_clean, bins=bins)
            
            # Calcular distribuciones
            baseline_dist, _ = np.histogram(baseline_clean, bins=bin_edges)
            current_dist, _ = np.histogram(current_clean, bins=bin_edges)
            
            # Normalizar (añadir pequeño valor para evitar división por cero)
            baseline_dist = baseline_dist / len(baseline_clean) + 1e-10
            current_dist = current_dist / len(current_clean) + 1e-10
            
            # Calcular PSI
            psi = np.sum((current_dist - baseline_dist) * np.log(current_dist / baseline_dist))
            
            # Determinar nivel de alerta
            if psi < 0.1:
                alert_level = 'green'
            elif psi < 0.2:
                alert_level = 'yellow'
            else:
                alert_level = 'red'
            
            return {
                'psi': psi,
                'alert_level': alert_level,
                'drift_detected': psi >= self.psi_threshold
            }
        except Exception as e:
            return {'psi': np.nan, 'alert_level': 'error', 'drift_detected': False}
    
    def jensen_shannon_divergence(self, baseline_col: pd.Series, current_col: pd.Series,
                                   bins: int = 50) -> Dict:
        """
        Calcula la divergencia de Jensen-Shannon entre dos distribuciones
        
        Args:
            baseline_col: Serie de datos de referencia
            current_col: Serie de datos actuales
            bins: Número de bins para discretización
            
        Returns:
            Dict con JS divergence y si hay drift
        """
        # Remover NaN
        baseline_clean = baseline_col.dropna()
        current_clean = current_col.dropna()
        
        if len(baseline_clean) == 0 or len(current_clean) == 0:
            return {'js_divergence': np.nan, 'drift_detected': False}
        
        try:
            # Crear bins basados en el rango completo
            min_val = min(baseline_clean.min(), current_clean.min())
            max_val = max(baseline_clean.max(), current_clean.max())
            bin_edges = np.linspace(min_val, max_val, bins + 1)
            
            # Calcular distribuciones
            baseline_dist, _ = np.histogram(baseline_clean, bins=bin_edges, density=True)
            current_dist, _ = np.histogram(current_clean, bins=bin_edges, density=True)
            
            # Normalizar
            baseline_dist = baseline_dist / (baseline_dist.sum() + 1e-10)
            current_dist = current_dist / (current_dist.sum() + 1e-10)
            
            # Calcular JS divergence
            js_div = jensenshannon(baseline_dist, current_dist)
            
            return {
                'js_divergence': js_div,
                'drift_detected': js_div >= self.js_threshold
            }
        except Exception as e:
            return {'js_divergence': np.nan, 'drift_detected': False}
    
    def chi_square_test(self, baseline_col: pd.Series, current_col: pd.Series) -> Dict:
        """
        Aplica test Chi-cuadrado para variables categóricas
        
        Args:
            baseline_col: Serie de datos de referencia
            current_col: Serie de datos actuales
            
        Returns:
            Dict con estadístico chi2, p-value y si hay drift
        """
        # Remover NaN
        baseline_clean = baseline_col.dropna()
        current_clean = current_col.dropna()
        
        if len(baseline_clean) == 0 or len(current_clean) == 0:
            return {'chi2_statistic': np.nan, 'p_value': np.nan, 'drift_detected': False}
        
        try:
            # Obtener categorías únicas
            all_categories = list(set(baseline_clean.unique()) | set(current_clean.unique()))
            
            # Calcular frecuencias observadas
            baseline_counts = baseline_clean.value_counts().reindex(all_categories, fill_value=0)
            current_counts = current_clean.value_counts().reindex(all_categories, fill_value=0)
            
            # Test chi-cuadrado
            chi2_statistic, p_value = stats.chisquare(
                f_obs=current_counts.values + 1,  # +1 para evitar ceros
                f_exp=baseline_counts.values + 1
            )
            
            return {
                'chi2_statistic': chi2_statistic,
                'p_value': p_value,
                'drift_detected': p_value < self.chi2_threshold
            }
        except Exception as e:
            return {'chi2_statistic': np.nan, 'p_value': np.nan, 'drift_detected': False}
    
    def detect_drift(self, current_data: pd.DataFrame, 
                     timestamp: Optional[str] = None) -> Dict[str, Any]:
        """
        Detecta data drift comparando datos actuales con baseline
        
        Args:
            current_data: DataFrame con datos actuales
            timestamp: Timestamp opcional para tracking temporal
            
        Returns:
            Dict con resultados completos del análisis de drift
        """
        if self.baseline_data is None:
            raise ValueError("Debe llamar a fit() primero para establecer baseline")
        
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        results = {
            'timestamp': timestamp,
            'n_features': len(self.feature_names),
            'n_samples_baseline': len(self.baseline_data),
            'n_samples_current': len(current_data),
            'features': {},
            'summary': {}
        }
        
        # Analizar cada feature
        for feature in self.feature_names:
            if feature not in current_data.columns:
                continue
            
            feature_results = {'feature_name': feature}
            
            if feature in self.numeric_features:
                # Aplicar tests para variables numéricas
                ks_result = self.kolmogorov_smirnov_test(
                    self.baseline_data[feature], 
                    current_data[feature]
                )
                psi_result = self.population_stability_index(
                    self.baseline_data[feature], 
                    current_data[feature]
                )
                js_result = self.jensen_shannon_divergence(
                    self.baseline_data[feature], 
                    current_data[feature]
                )
                
                feature_results.update({
                    'type': 'numeric',
                    'ks_test': ks_result,
                    'psi': psi_result,
                    'js_divergence': js_result,
                    'drift_detected': (
                        ks_result['drift_detected'] or 
                        psi_result['drift_detected'] or 
                        js_result['drift_detected']
                    )
                })
                
            elif feature in self.categorical_features:
                # Aplicar test para variables categóricas
                chi2_result = self.chi_square_test(
                    self.baseline_data[feature], 
                    current_data[feature]
                )
                
                feature_results.update({
                    'type': 'categorical',
                    'chi2_test': chi2_result,
                    'drift_detected': chi2_result['drift_detected']
                })
            
            results['features'][feature] = feature_results
        
        # Resumen general
        total_drifted = sum(1 for f in results['features'].values() if f.get('drift_detected', False))
        results['summary'] = {
            'total_features': len(results['features']),
            'features_with_drift': total_drifted,
            'drift_percentage': (total_drifted / len(results['features']) * 100) if results['features'] else 0,
            'overall_drift_detected': total_drifted > 0,
            'alert_level': self._determine_alert_level(total_drifted, len(results['features']))
        }
        
        # Almacenar en historial
        self.drift_history.append(results)
        
        return results
    
    def _determine_alert_level(self, drifted_count: int, total_count: int) -> str:
        """Determina nivel de alerta basado en porcentaje de features con drift"""
        if total_count == 0:
            return 'unknown'
        
        drift_pct = (drifted_count / total_count) * 100
        
        if drift_pct == 0:
            return 'green'
        elif drift_pct < 10:
            return 'yellow'
        elif drift_pct < 30:
            return 'orange'
        else:
            return 'red'
    
    def generate_alert_message(self, drift_results: Dict) -> str:
        """
        Genera mensaje de alerta basado en resultados de drift
        
        Args:
            drift_results: Resultados del análisis de drift
            
        Returns:
            Mensaje de alerta formateado
        """
        summary = drift_results['summary']
        alert_level = summary['alert_level']
        
        messages = {
            'green': "✅ ESTADO NORMAL: No se detectó data drift significativo.",
            'yellow': "⚠️ ALERTA MENOR: Se detectó drift en algunas features. Monitorear.",
            'orange': "🟠 ALERTA MODERADA: Drift significativo detectado. Considerar reentrenamiento.",
            'red': "🚨 ALERTA CRÍTICA: Drift severo detectado. Reentrenamiento urgente recomendado."
        }
        
        message = messages.get(alert_level, "⚪ ESTADO DESCONOCIDO")
        message += f"\n\n📊 Resumen:\n"
        message += f"   • Features analizadas: {summary['total_features']}\n"
        message += f"   • Features con drift: {summary['features_with_drift']}\n"
        message += f"   • Porcentaje de drift: {summary['drift_percentage']:.1f}%\n"
        
        if summary['features_with_drift'] > 0:
            message += f"\n🔍 Features afectadas:\n"
            for feature, results in drift_results['features'].items():
                if results.get('drift_detected', False):
                    message += f"   • {feature}\n"
        
        # Recomendaciones
        message += f"\n💡 Recomendaciones:\n"
        if alert_level == 'green':
            message += "   • Continuar con monitoreo regular\n"
        elif alert_level == 'yellow':
            message += "   • Incrementar frecuencia de monitoreo\n"
            message += "   • Revisar features afectadas\n"
        elif alert_level == 'orange':
            message += "   • Planificar reentrenamiento del modelo\n"
            message += "   • Analizar causas del drift\n"
            message += "   • Revisar calidad de datos de entrada\n"
        else:  # red
            message += "   • ACCIÓN INMEDIATA: Reentrenar modelo\n"
            message += "   • Investigar cambios en fuente de datos\n"
            message += "   • Considerar pausar predicciones hasta ajustes\n"
        
        return message
    
    def save_results(self, drift_results: Dict, output_dir: str = "results/monitoring"):
        """
        Guarda resultados de drift en archivo JSON
        
        Args:
            drift_results: Resultados del análisis
            output_dir: Directorio de salida
        """
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = drift_results['timestamp'].replace(':', '-').replace(' ', '_')
        filename = f"drift_report_{timestamp}.json"
        filepath = os.path.join(output_dir, filename)
        
        # Convertir a formato serializable
        serializable_results = self._make_serializable(drift_results)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Resultados guardados en: {filepath}")
    
    def _make_serializable(self, obj: Any) -> Any:
        """Convierte objetos numpy/pandas a tipos serializables"""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif pd.isna(obj):
            return None
        else:
            return obj
    
    def get_drift_summary_dataframe(self, drift_results: Dict) -> pd.DataFrame:
        """
        Convierte resultados de drift a DataFrame para análisis
        
        Args:
            drift_results: Resultados del análisis
            
        Returns:
            DataFrame con resumen por feature
        """
        rows = []
        
        for feature, results in drift_results['features'].items():
            row = {
                'feature': feature,
                'type': results['type'],
                'drift_detected': results['drift_detected']
            }
            
            if results['type'] == 'numeric':
                row.update({
                    'ks_statistic': results['ks_test']['ks_statistic'],
                    'ks_pvalue': results['ks_test']['p_value'],
                    'psi': results['psi']['psi'],
                    'psi_alert': results['psi']['alert_level'],
                    'js_divergence': results['js_divergence']['js_divergence']
                })
            else:
                row.update({
                    'chi2_statistic': results['chi2_test']['chi2_statistic'],
                    'chi2_pvalue': results['chi2_test']['p_value']
                })
            
            rows.append(row)
        
        df = pd.DataFrame(rows)
        return df.sort_values('drift_detected', ascending=False)


def monitor_model_predictions(model, X_baseline: pd.DataFrame, y_baseline: pd.Series,
                               X_current: pd.DataFrame, y_current: pd.Series = None,
                               feature_names: List[str] = None) -> Dict:
    """
    Función principal de monitoreo que combina drift detection con análisis de predicciones
    
    Args:
        model: Modelo entrenado
        X_baseline: Features de referencia (train set)
        y_baseline: Target de referencia
        X_current: Features actuales (new data)
        y_current: Target actual (opcional, si está disponible)
        feature_names: Nombres de features
        
    Returns:
        Dict con análisis completo de monitoreo
    """
    print("\n" + "="*80)
    print("🔍 INICIANDO MONITOREO DE MODELO Y DATA DRIFT")
    print("="*80)
    
    # Crear detector y establecer baseline
    detector = DataDriftDetector()
    detector.fit(X_baseline, feature_names)
    
    # Detectar drift
    print("\n📊 Analizando data drift...")
    drift_results = detector.detect_drift(X_current)
    
    # Generar predicciones
    print("\n🤖 Generando predicciones...")
    baseline_predictions = model.predict(X_baseline)
    current_predictions = model.predict(X_current)
    
    # Análisis de predicciones
    prediction_analysis = {
        'baseline_mean': float(baseline_predictions.mean()),
        'current_mean': float(current_predictions.mean()),
        'prediction_drift': abs(baseline_predictions.mean() - current_predictions.mean()),
        'baseline_distribution': {
            'class_0': int((baseline_predictions == 0).sum()),
            'class_1': int((baseline_predictions == 1).sum())
        },
        'current_distribution': {
            'class_0': int((current_predictions == 0).sum()),
            'class_1': int((current_predictions == 1).sum())
        }
    }
    
    # Si hay labels actuales, calcular métricas
    if y_current is not None:
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        prediction_analysis['performance'] = {
            'accuracy': float(accuracy_score(y_current, current_predictions)),
            'precision': float(precision_score(y_current, current_predictions, zero_division=0)),
            'recall': float(recall_score(y_current, current_predictions, zero_division=0)),
            'f1_score': float(f1_score(y_current, current_predictions, zero_division=0))
        }
    
    # Combinar resultados
    monitoring_results = {
        'drift_analysis': drift_results,
        'prediction_analysis': prediction_analysis,
        'alert_message': detector.generate_alert_message(drift_results)
    }
    
    # Mostrar alerta
    print("\n" + "="*80)
    print(monitoring_results['alert_message'])
    print("="*80)
    
    return monitoring_results, detector


if __name__ == "__main__":
    """
    Ejemplo de uso del sistema de monitoreo
    """
    print("\n🔬 Model Monitoring & Data Drift Detection System")
    print("=" * 80)
    print("Este módulo debe ser importado en el pipeline principal")
    print("Ver app_streamlit.py para interfaz interactiva")
    print("=" * 80)
